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

    def __init__(
        self,
        *,
        chatbot: ChatBot,
        retry_chatbot: ChatBot | None = None,
        chunk_size: int = CHUNK_SIZE,
        intercept_line: int | None = None,
        chunked_guideline: bool = False,
    ):
        """
        Initialize the LLMTranslator with given parameters.

        Args:
            chatbot: Primary ChatBot instance for translation.
            retry_chatbot: Optional ChatBot instance for retry attempts.
            chunk_size: Size of text chunks for processing, balancing efficiency and context.
            intercept_line: Line number to intercept translation, useful for debugging.
            chunked_guideline: Enable chunked guideline generation for long texts. Default: False.
        """
        self.chatbot = chatbot
        self.retry_chatbot = retry_chatbot
        self.chunk_size = chunk_size
        self.api_fee = 0
        self.intercept_line = intercept_line
        self.chunked_guideline = chunked_guideline
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

        def _try_agent(agent: ChunkedTranslatorAgent) -> tuple[list[str] | None, TranslationContext | None]:
            """Single agent attempt: translate, then retry without glossary if line count mismatches."""
            trans: list[str] | None = None
            ctx: TranslationContext | None = None
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

        translator_agent = ChunkedTranslatorAgent(src_lang, target_lang, info, chatbot=self.chatbot)

        retry_agent = (
            ChunkedTranslatorAgent(src_lang, target_lang, info, chatbot=self.retry_chatbot)
            if self.retry_chatbot
            else None
        )

        # proofreader = ProofreaderAgent(src_lang, target_lang, info, chatbot=self.chatbot)

        chunks = self.make_chunks(texts, chunk_size=self.chunk_size)
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
            chunk_texts = [c[1] for c in chunk]

            if not self._is_valid_translation(translated, len(chunk)):
                logger.warning(
                    f"Chunk {i} translation length inconsistent: {len(translated)} vs {len(chunk)},"
                    f" Attempting atomic translation."
                )
                translated = self.atomic_translate(self.chatbot, chunk_texts, src_lang, target_lang)
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
