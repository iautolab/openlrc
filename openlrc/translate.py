#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import json
import os
import uuid
from abc import ABC, abstractmethod
from itertools import zip_longest
from pathlib import Path

import requests

from openlrc.agents import ChunkedTranslatorAgent, ContextReviewerAgent
from openlrc.chatbot import ChatBot
from openlrc.context import TranslateInfo, TranslationContext
from openlrc.exceptions import ChatBotException
from openlrc.logger import logger
from openlrc.prompter import AtomicTranslatePrompter
from openlrc.utils import get_text_token_number


class Translator(ABC):
    @abstractmethod
    def translate(self, texts: str | list[str], src_lang: str, target_lang: str, info: TranslateInfo) -> list[str]:
        pass


class LLMTranslator(Translator):
    """
    Translator using Large Language Models for translation.
    This class implements a sophisticated translation process using chunking,
    context-aware translation, and fallback mechanisms.
    """

    CHUNK_SIZE = 30
    RETRY_STREAK = 10  # Number of consecutive chunks to use retry model after a failure
    LINE_MISMATCH_RETRIES = 2  # Max attempts per agent for line-count mismatch recovery
    MIN_SPLIT_SIZE = 3  # Minimum chunk size for binary-split retry; below this, use atomic
    MAX_CHUNK_TOKENS = 1000  # Token budget per chunk (text content only, excludes prompt overhead)
    SCENE_THRESHOLD = 30.0  # Seconds of silence that indicates a scene boundary

    def __init__(
        self,
        *,
        chatbot: ChatBot,
        retry_chatbot: ChatBot | None = None,
        chunk_size: int = CHUNK_SIZE,
        intercept_line: int | None = None,
        chunked_guideline: bool = False,
        timestamps: list[tuple[float, float | None]] | None = None,
    ):
        """
        Initialize the LLMTranslator with given parameters.

        Args:
            chatbot: Primary ChatBot instance for translation.
            retry_chatbot: Optional ChatBot instance for retry attempts.
            chunk_size: Maximum lines per chunk for processing.
            intercept_line: Line number to intercept translation, useful for debugging.
            chunked_guideline: Enable chunked guideline generation for long texts. Default: False.
            timestamps: Optional list of (start, end) per line for scene-aware chunking.
                When provided, chunk splitting respects scene boundaries (time gaps > SCENE_THRESHOLD).
                When None, only token budget and line-count limits apply.
        """
        self.chatbot = chatbot
        self.retry_chatbot = retry_chatbot
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.chunked_guideline = chunked_guideline
        self.timestamps = timestamps
        self.use_retry_cnt = 0

    @staticmethod
    def make_chunks(texts: list[str], chunk_size: int = 30) -> list[list[tuple[int, str]]]:
        """
        Split the text into chunks of specified size for efficient processing.

        Args:
            texts (List[str]): List of texts to be chunked.
            chunk_size (int): Maximum size of each chunk.

        Returns:
            List[List[Tuple[int, str]]]: List of chunks, each chunk is a list of (line_number, text) tuples.
        """
        chunks = []
        start = 1
        for i in range(0, len(texts), chunk_size):
            chunk = [(start + j, text) for j, text in enumerate(texts[i : i + chunk_size])]
            start += len(chunk)
            chunks.append(chunk)

        # Merge the last chunk if it's too small
        if len(chunks) >= 2 and len(chunks[-1]) < chunk_size / 2:
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        return chunks

    def make_chunks_by_tokens(self, texts: list[str]) -> list[list[tuple[int, str]]]:
        """Split texts into chunks using token budget, scene boundaries, and line-count limits.

        Splitting strategy (applied in priority order):
        1. **Scene boundary (hard split):** A time gap > ``SCENE_THRESHOLD`` between
           adjacent lines forces a chunk break regardless of token budget.
           Only active when ``self.timestamps`` is set.
        2. **Token budget:** Accumulated text tokens must not exceed ``MAX_CHUNK_TOKENS``.
           When the budget is exceeded, the chunk is split at the largest time gap
           within the current chunk (if timestamps are available), otherwise at the
           current position.
        3. **Line-count cap:** A chunk never exceeds ``chunk_size`` lines.

        A small trailing chunk (< chunk_size/2 lines *and* < MAX_CHUNK_TOKENS/2 tokens)
        is merged into the previous chunk, unless a scene boundary separates them.

        Args:
            texts: List of subtitle texts.

        Returns:
            List of chunks, each chunk a list of ``(line_number, text)`` tuples.
        """
        if not texts:
            return []

        timestamps = self.timestamps
        # Validate timestamps length if provided.
        if timestamps is not None and len(timestamps) != len(texts):
            logger.warning(
                f"Timestamps length ({len(timestamps)}) != texts length ({len(texts)}), ignoring timestamps."
            )
            timestamps = None

        # Pre-compute token counts for each line.
        token_counts = [get_text_token_number(t) for t in texts]

        chunks: list[list[tuple[int, str]]] = []
        current_chunk: list[tuple[int, str]] = []
        current_tokens = 0

        for idx, text in enumerate(texts):
            line_number = idx + 1
            line_tokens = token_counts[idx]

            # Check scene boundary: gap between previous line's end and current line's start.
            if timestamps and current_chunk and idx > 0:
                prev_end = timestamps[idx - 1][1]
                cur_start = timestamps[idx][0]
                if prev_end is not None and (cur_start - prev_end) > self.SCENE_THRESHOLD:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0

            # Check token budget or line-count cap — need to flush before adding.
            if current_chunk and (
                current_tokens + line_tokens > self.MAX_CHUNK_TOKENS or len(current_chunk) >= self.chunk_size
            ):
                # Try to split at the largest time gap within the current chunk.
                split_idx = self._find_best_split(current_chunk, timestamps)
                if split_idx is not None and split_idx > 0:
                    chunks.append(current_chunk[:split_idx])
                    leftover = current_chunk[split_idx:]
                    current_chunk = leftover
                    current_tokens = sum(token_counts[c[0] - 1] for c in current_chunk)
                else:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append((line_number, text))
            current_tokens += line_tokens

        if current_chunk:
            chunks.append(current_chunk)

        # Merge small trailing chunk into the previous one, unless a scene boundary separates them.
        if len(chunks) >= 2:
            tail = chunks[-1]
            is_small_tail = (
                len(tail) < self.chunk_size / 2
                and sum(token_counts[c[0] - 1] for c in tail) < self.MAX_CHUNK_TOKENS / 2
            )

            # Check for scene boundary between the two chunks.
            scene_break = False
            if timestamps and is_small_tail:
                prev_last = chunks[-2][-1][0] - 1  # 0-based index of last line in previous chunk
                tail_first = tail[0][0] - 1  # 0-based index of first line in tail
                prev_end = timestamps[prev_last][1]
                cur_start = timestamps[tail_first][0]
                if prev_end is not None and (cur_start - prev_end) > self.SCENE_THRESHOLD:
                    scene_break = True

            if is_small_tail and not scene_break:
                chunks[-2].extend(tail)
                chunks.pop()

        return chunks

    @staticmethod
    def _find_best_split(
        chunk: list[tuple[int, str]], timestamps: list[tuple[float, float | None]] | None
    ) -> int | None:
        """Find the index within *chunk* that has the largest time gap before it.

        Returns the chunk-local index (1..len-1) of the best split point,
        or *None* if timestamps are unavailable or the chunk has fewer than 2 lines.
        """
        if not timestamps or len(chunk) < 2:
            return None

        best_gap = -1.0
        best_idx = None
        for i in range(1, len(chunk)):
            prev_global = chunk[i - 1][0] - 1  # 0-based index into timestamps
            cur_global = chunk[i][0] - 1
            prev_end = timestamps[prev_global][1]
            cur_start = timestamps[cur_global][0]
            if prev_end is not None:
                gap = cur_start - prev_end
                if gap > best_gap:
                    best_gap = gap
                    best_idx = i

        return best_idx

    @staticmethod
    def _is_valid_translation(translated: list[str] | None, expected_len: int) -> bool:
        """Check whether a translation result is usable (non-empty and correct line count)."""
        return translated is not None and len(translated) == expected_len

    def _translate_chunk(
        self,
        translator_agent: ChunkedTranslatorAgent,
        chunk: list[tuple[int, str]],
        context: TranslationContext,
        chunk_id: int,
        retry_agent: ChunkedTranslatorAgent | None = None,
    ) -> tuple[list[str], TranslationContext]:
        """
        Translate a single chunk of text, with retry mechanism.

        Tries the primary agent first, then optionally falls back to a retry agent.
        Each agent attempt includes a glossary-removal retry when the line count
        is inconsistent.

        Returns the best available result. The caller should check whether
        ``len(translated) == len(chunk)`` to decide if further fallback
        (e.g. split or atomic translation) is needed.
        """
        expected = len(chunk)
        max_attempts = 1 if retry_agent else self.LINE_MISMATCH_RETRIES

        def _try_agent(agent: ChunkedTranslatorAgent) -> tuple[list[str] | None, TranslationContext | None]:
            """Try an agent up to max_attempts times for line-count mismatch recovery.

            Each attempt includes a glossary-removal sub-retry when applicable.
            Returns the best available result (may still have mismatched line count).
            """
            trans: list[str] | None = None
            ctx: TranslationContext | None = None

            for attempt in range(max_attempts):
                try:
                    trans, ctx = agent.translate_chunk(chunk_id, chunk, context)
                except ChatBotException:
                    logger.error(f"Failed to translate chunk {chunk_id}.")
                    return None, None

                if self._is_valid_translation(trans, expected):
                    return trans, ctx

                # Line count mismatch — retry without glossary if applicable.
                if trans is not None and agent.info.glossary:
                    logger.warning(
                        f"Agent {agent}: Removing glossary for chunk {chunk_id} due to inconsistent translation."
                    )
                    try:
                        trans, ctx = agent.translate_chunk(chunk_id, chunk, context, use_glossary=False)
                    except ChatBotException:
                        logger.error(f"Failed to translate chunk {chunk_id}.")

                if self._is_valid_translation(trans, expected):
                    return trans, ctx

                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Chunk {chunk_id} line count mismatch ({len(trans) if trans else 0} vs {expected}),"
                        f" retrying ({attempt + 1}/{max_attempts})."
                    )

            return trans, ctx

        translated: list[str] | None = None
        updated_ctx: TranslationContext | None = None

        # Step 1: Try primary or retry agent based on retry streak.
        if self.use_retry_cnt == 0 or not retry_agent:
            translated, updated_ctx = _try_agent(translator_agent)
        else:
            logger.info(f"Using retry agent for chunk {chunk_id}, remaining retries: {self.use_retry_cnt}")
            translated, updated_ctx = _try_agent(retry_agent)
            self.use_retry_cnt -= 1

        # Step 2: If primary failed and retry agent is available, switch to it.
        if not self._is_valid_translation(translated, expected) and retry_agent and self.use_retry_cnt == 0:
            self.use_retry_cnt = self.RETRY_STREAK
            logger.warning(
                f"Using retry agent {retry_agent} for chunk {chunk_id}, and next {self.use_retry_cnt} chunks."
            )
            translated, updated_ctx = _try_agent(retry_agent)

            # Retry agent also failed — reset streak so next chunk tries primary first.
            if not self._is_valid_translation(translated, expected):
                logger.warning(f"Retry agent also failed for chunk {chunk_id}, resetting retry streak.")
                self.use_retry_cnt = 0

        if not translated:
            raise ChatBotException(f"Failed to translate chunk {chunk_id}.")

        return translated, updated_ctx or context

    def _split_and_translate(
        self,
        translator_agent: ChunkedTranslatorAgent,
        chunk: list[tuple[int, str]],
        context: TranslationContext,
        chunk_id: int,
        src_lang: str,
        target_lang: str,
        retry_agent: ChunkedTranslatorAgent | None = None,
    ) -> list[str]:
        """Recursively split a chunk in half and translate each half separately.

        Used as a fallback when ``_translate_chunk`` returns a line-count mismatch.
        Splits at the midpoint, translates each half via ``_translate_chunk``, and
        merges the results.  Halves that still mismatch are split again recursively
        until ``MIN_SPLIT_SIZE`` is reached, at which point ``atomic_translate`` is
        used as the final fallback.

        Returns:
            List of translated strings with ``len == len(chunk)``.
        """
        if len(chunk) <= self.MIN_SPLIT_SIZE:
            if not chunk:
                return []
            logger.info(f"Chunk {chunk_id} below MIN_SPLIT_SIZE ({self.MIN_SPLIT_SIZE}), using atomic translation.")
            return self.atomic_translate(self.chatbot, [c[1] for c in chunk], src_lang, target_lang)

        mid = len(chunk) // 2
        left, right = chunk[:mid], chunk[mid:]
        logger.info(f"Splitting chunk {chunk_id} into {len(left)}+{len(right)} lines for retry.")

        results: list[str] = []
        for half in (left, right):
            try:
                translated, _ = self._translate_chunk(
                    translator_agent, half, context, chunk_id, retry_agent=retry_agent
                )
            except ChatBotException:
                translated = None

            if self._is_valid_translation(translated, len(half)):
                results.extend(translated)
            else:
                # Recurse on the failing half.
                results.extend(
                    self._split_and_translate(
                        translator_agent, half, context, chunk_id, src_lang, target_lang, retry_agent
                    )
                )

        return results

    def translate(
        self,
        texts: str | list[str],
        src_lang: str,
        target_lang: str,
        info: TranslateInfo | None = None,
        compare_path: Path = Path("translate_intermediate.json"),
    ) -> list[str]:
        """
        Translate a list of texts from source language to target language.

        This method implements the main translation process:
        1. Initialize translation agents and chunk the input texts.
        2. Build or load a translation guideline.
        3. Translate each chunk, maintaining context between chunks.
        4. Handle translation failures with retry mechanisms and atomic translation.
        5. Save intermediate results for potential resumption.

        Args:
            texts (Union[str, List[str]]): Text or list of texts to translate.
            src_lang (str): Source language code.
            target_lang (str): Target language code.
            info (TranslateInfo): Additional translation information like title and glossary.
            compare_path (Path): Path to save intermediate results for potential resumption.

        Returns:
            List[str]: List of translated texts.
        """
        if info is None:
            info = TranslateInfo()

        if not isinstance(texts, list):
            texts = [texts]

        if not texts:
            return []

        translator_agent = ChunkedTranslatorAgent(src_lang, target_lang, info, chatbot=self.chatbot)

        retry_agent = (
            ChunkedTranslatorAgent(src_lang, target_lang, info, chatbot=self.retry_chatbot)
            if self.retry_chatbot
            else None
        )

        chunks = self.make_chunks_by_tokens(texts)
        logger.info(f"Translating {info.title}: {len(chunks)} chunks, {len(texts)} lines in total.")

        translations, summaries, compare_list, start_chunk, guideline = self._resume_translation(compare_path)
        if not guideline:
            logger.info("Building translation guideline.")
            context_reviewer = ContextReviewerAgent(
                src_lang,
                target_lang,
                info,
                chatbot=self.chatbot,
                retry_chatbot=self.retry_chatbot,
                chunked_guideline=self.chunked_guideline,
            )
            guideline = context_reviewer.build_context(
                texts, title=info.title or "", glossary=info.glossary, forced_glossary=info.forced_glossary
            )
            logger.debug(f"Translation Guideline:\n{guideline}")

        context = TranslationContext(guideline=guideline, previous_summaries=summaries)
        for i, chunk in list(enumerate(chunks, start=1))[start_chunk:]:
            atomic = False
            translated, context = self._translate_chunk(translator_agent, chunk, context, i, retry_agent=retry_agent)

            if not self._is_valid_translation(translated, len(chunk)):
                logger.warning(
                    f"Chunk {i} translation length inconsistent: {len(translated)} vs {len(chunk)},"
                    f" attempting binary-split retry."
                )
                translated = self._split_and_translate(
                    translator_agent, chunk, context, i, src_lang, target_lang, retry_agent=retry_agent
                )
                atomic = True

            translations.extend(translated)
            summaries.append(context.summary or "")
            logger.info(f"Translated {info.title}: {i}/{len(chunks)}")
            logger.debug(f"Summary: {context.summary}")
            logger.debug(f"Scene: {context.scene}")

            compare_list.extend(self._generate_compare_list(chunk, translated, i, atomic, context))
            self._save_intermediate_results(compare_path, compare_list, summaries, context.scene or "", guideline)
            context.previous_summaries = summaries

        self.api_fee += translator_agent.cost + (retry_agent.cost if retry_agent else 0)

        logger.info(f"Translation complete for {info.title}. Fee: {self.api_fee:.4f} USD")

        return translations

    def _resume_translation(self, compare_path: Path) -> tuple[list[str], list[str], list[dict], int, str]:
        """
        Resume translation from a saved state.

        This method allows the translation process to be resumed from a previous point,
        which is useful for long translations or in case of interruptions.

        Args:
            compare_path (Path): Path to the saved translation state.

        Returns:
            Tuple[List[str], List[str], List[dict], int, str]: Tuple containing:
                - translations: List of already translated texts.
                - summaries: List of translation summaries.
                - compare_list: List of dictionaries for comparison.
                - start_chunk: The chunk number to resume from.
                - guideline: The translation guideline.
        """
        translations, summaries, compare_list, start_chunk, guideline = [], [], [], 0, ""

        if compare_path.exists():
            logger.info(f"Resuming translation from {compare_path}")
            with open(compare_path, encoding="utf-8") as f:
                compare_results = json.load(f)
            compare_list = compare_results["compare"]
            summaries = compare_results["summaries"]
            translations = [item["output"] for item in compare_list]
            start_chunk = compare_list[-1]["chunk"]
            guideline = compare_results["guideline"]
            logger.info(f"Resuming translation from chunk {start_chunk}")

        return translations, summaries, compare_list, start_chunk, guideline

    def _generate_compare_list(
        self,
        chunk: list[tuple[int, str]],
        translated: list[str],
        chunk_id: int,
        atomic: bool,
        context: TranslationContext,
    ) -> list[dict]:
        """
        Generate a comparison list for the translated chunk.

        This method creates a detailed record of each translation, including the original text,
        translated text, and metadata about the translation process.

        Args:
            chunk (List[Tuple[int, str]]): Original chunk of text.
            translated (List[str]): Translated texts.
            chunk_id (int): ID of the current chunk.
            atomic (bool): Whether atomic translation was used.
            context (TranslationContext): Current translation context.

        Returns:
            List[dict]: List of dictionaries containing comparison information.
        """
        return [
            {
                "chunk": chunk_id,
                "idx": item[0] if item else "N/A",
                "method": "atomic" if atomic else "chunked",
                "model": str(context.model),
                "input": item[1] if item else "N/A",
                "output": trans if trans else "N/A",
            }
            for (item, trans) in zip_longest(chunk, translated)
        ]

    def _save_intermediate_results(
        self, compare_path: Path, compare_list: list[dict], summaries: list[str], scene: str, guideline: str
    ):
        """
        Save intermediate translation results to a file.

        This method saves the current state of the translation process, allowing for
        potential resumption of the translation task later.

        Args:
            compare_path (Path): Path to save the results.
            compare_list (List[dict]): List of comparison dictionaries.
            summaries (List[str]): List of translation summaries.
            scene (str): Current scene description.
            guideline (str): Translation guideline.
        """
        compare_results = {"compare": compare_list, "summaries": summaries, "scene": scene, "guideline": guideline}
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(compare_results, f, indent=4, ensure_ascii=False)

    def atomic_translate(self, chatbot: ChatBot, texts: list[str], src_lang: str, target_lang: str) -> list[str]:
        """
        Perform atomic translation for each text individually.

        This method is used as a fallback when chunk translation fails. It translates
        each text separately, which can be slower but more reliable for problematic texts.

        Args:
            chatbot (ChatBot): ChatBot instance to use for translation.
            texts (List[str]): List of texts to translate.
            src_lang (str): Source language code.
            target_lang (str): Target language code.

        Returns:
            List[str]: List of translated texts.

        Raises:
            AssertionError: If the number of translated texts doesn't match the input.
        """
        from openlrc.agents import ChunkedTranslatorAgent as _CTA

        prompter = AtomicTranslatePrompter(src_lang, target_lang)
        message_lists = [[{"role": "user", "content": prompter.user(text)}] for text in texts]

        responses = chatbot.message(message_lists, output_checker=prompter.check_format, temperature=_CTA.TEMPERATURE)
        self.api_fee += sum(chatbot.api_fees[-(len(texts)) :])
        translated = list(map(chatbot.get_content, responses))

        if len(translated) != len(texts):
            raise ChatBotException(
                f"Atomic translation failed: expected {len(texts)} translations, got {len(translated)}"
            )

        return translated


class MSTranslator(Translator):
    """
    Translator using Microsoft Translator API.
    This class provides an alternative translation method using Microsoft's services.
    """

    def __init__(self):
        """
        Initialize the Microsoft Translator with API key and endpoint.
        The API key is expected to be set in the environment variables.
        """
        self.key = os.environ["MS_TRANSLATOR_KEY"]
        self.endpoint = "https://api.cognitive.microsofttranslator.com"
        self.location = "eastasia"
        self.path = "/translate"
        self.constructed_url = self.endpoint + self.path

        self.headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Ocp-Apim-Subscription-Region": self.location,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        }

    def translate(self, texts: str | list[str], src_lang, target_lang, info=None):  # type: ignore[override]
        params = {"api-version": "3.0", "from": src_lang, "to": target_lang}

        body = [{"text": text} for text in texts]

        try:
            request = requests.post(self.constructed_url, params=params, headers=self.headers, json=body, timeout=20)
        except TimeoutError:
            raise RuntimeError("Failed to connect to Microsoft Translator API.") from None
        response = request.json()

        return json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(",", ": "))
