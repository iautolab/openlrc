#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

import os
import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from openlrc.agents import ChunkedTranslatorAgent, ContextReviewerAgent, TranslationContext
from openlrc.context import TranslateInfo
from openlrc.models import ModelConfig, ModelProvider
from openlrc.prompter import ChunkedTranslatePrompter

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_CHEAP_MODEL = ModelConfig(
    provider=ModelProvider.OPENAI,
    name="google/gemini-2.5-flash-lite",
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)
LIVE_API = os.environ.get("OPENLRC_TEST_LIVE_API", "").lower() in ("1", "true", "yes")


class DummyMessage(BaseModel):
    content: str


class DummyChoice(BaseModel):
    message: DummyMessage


class DummyResponse(BaseModel):
    choices: list[DummyChoice]


class TestTranslatorAgent(unittest.TestCase):
    @patch(
        "openlrc.chatbot.GPTBot.message",
        MagicMock(
            return_value=[
                DummyResponse(
                    choices=[
                        DummyChoice(
                            message=DummyMessage(
                                content="<summary>Example Summary</summary>\n<scene>Example Scene</scene>\n#1\nOriginal>xxx\nTranslation>\nBonjour, comment ça va?\n#2\nOriginal>xxx\nTranslation>\nJe vais bien, merci.\n"
                            )
                        )
                    ]
                )
            ]
        ),
    )
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_translate_chunk_success(self):
        agent = ChunkedTranslatorAgent(
            src_lang="en",
            target_lang="fr",
            info=TranslateInfo(title="Example Title", audio_type="Book", glossary={"hello": "bonjour"}),
        )
        agent.chatbot.api_fees = [0.00035]
        translations, context = agent.translate_chunk(
            chunk_id=1,
            chunk=[(1, "Hello, how are you?"), (2, "I am fine, thank you.")],
            context=TranslationContext(
                summary="Example Summary", previous_summaries=["s1", "s2"], scene="Example Scene"
            ),
        )

        self.assertListEqual(translations, ["Bonjour, comment ça va?", "Je vais bien, merci."])
        self.assertEqual(context.summary, "Example Summary")
        self.assertEqual(context.scene, "Example Scene")

    #  Handle invalid chatbot model names gracefully
    def test_invalid_chatbot_model(self):
        with self.assertRaises(ValueError):
            ChunkedTranslatorAgent(src_lang="en", target_lang="fr", info=TranslateInfo(), chatbot_model="invalid-model")

    @patch(
        "openlrc.chatbot.GPTBot.get_content",
        MagicMock(
            return_value="<summary>Example Summary</summary>\n<scene>Example Scene</scene>\n#1\nOriginal>xxx\nTranslation>\nBonjour, comment ça va?\n#2\nOriginal>xxx\nTranslation>\nJe vais bien, merci.\n"
        ),
    )
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_parse_response_success(self):
        agent = ChunkedTranslatorAgent(src_lang="en", target_lang="fr")
        translations, summary, scene = agent._parse_responses("dummy_response")

        self.assertListEqual(translations, ["Bonjour, comment ça va?", "Je vais bien, merci."])
        self.assertEqual(summary, "Example Summary")
        self.assertEqual(scene, "Example Scene")

    #  Properly format texts for translation
    def test_format_texts_success(self):
        texts = [(1, "Hello, how are you?"), (2, "I am fine, thank you.")]
        formatted_text = ChunkedTranslatePrompter.format_texts(texts)

        expected_output = (
            "#1\nOriginal>\nHello, how are you?\nTranslation>\n\n#2\nOriginal>\nI am fine, thank you.\nTranslation>\n"
        )
        self.assertEqual(formatted_text, expected_output)

    #  Use glossary terms in translations when provided
    def test_use_glossary_terms_success(self):
        glossary = {"hello": "bonjour", "how are you": "comment ça va"}
        prompter = ChunkedTranslatePrompter(src_lang="en", target_lang="fr", context=TranslateInfo(glossary=glossary))

        formatted_glossary = prompter.formatted_glossary

        expected_output = "\n# Glossary\nUse the following glossary to ensure consistency in your translations:\n<preferred-translation>\nhello: bonjour\nhow are you: comment ça va\n</preferred-translation>\n"
        self.assertEqual(formatted_glossary, expected_output)


@unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1 and valid API keys")
class TestContextReviewerAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OPENROUTER_API_KEY:
            raise unittest.SkipTest("OPENROUTER_API_KEY is required for LLM integration tests.")

    def test_generates_valid_context(self):
        texts = [
            "John and Sarah discuss their plan to locate a suspect",
            "John: 'As a 10 years experienced detector, my advice is we should start our search in the uptown area.'",
            "Sarah: 'Agreed. Let's gather more information before we move.'",
            "Then, they prepare to start their investigation.",
        ]
        title = "The Detectors"
        glossary = {"suspect": "嫌疑人", "uptown": "市中心"}

        agent = ContextReviewerAgent("en", "zh", chatbot_model=OPENROUTER_CHEAP_MODEL)
        context = agent.build_context(texts, title, glossary)

        self.assertIsNotNone(context)
        self.assertIsInstance(context, str)
        self.assertIn("Glossary", context)
        self.assertIn("Characters", context)
        self.assertIn("Summary", context)
        self.assertIn("Tone and Style", context)
        self.assertIn("Target Audience", context)


VALID_GUIDELINE = (
    "### Glossary:\n- suspect: 嫌疑人\n\n"
    "### Characters:\n- John: 约翰\n\n"
    "### Summary:\nA detective story.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)

PARTIAL_GUIDELINE_1 = (
    "### Glossary:\n- suspect: 嫌疑人\n\n"
    "### Characters:\n- John: 约翰\n\n"
    "### Summary:\nJohn investigates.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)

PARTIAL_GUIDELINE_2 = (
    "### Glossary:\n- uptown: 市中心\n\n"
    "### Characters:\n- Sarah: 萨拉\n\n"
    "### Summary:\nSarah joins the investigation.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)

MERGED_GUIDELINE = (
    "### Glossary:\n- suspect: 嫌疑人\n- uptown: 市中心\n\n"
    "### Characters:\n- John: 约翰\n- Sarah: 萨拉\n\n"
    "### Summary:\nJohn and Sarah investigate together.\n\n"
    "### Tone and Style:\nFormal.\n\n"
    "### Target Audience:\nAdult viewers."
)


def _make_dummy_response(content: str) -> DummyResponse:
    return DummyResponse(choices=[DummyChoice(message=DummyMessage(content=content))])


class TestContextReviewerChunking(unittest.TestCase):
    """Mock tests for chunked guideline generation."""

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_short_text_no_chunking(self, mock_message):
        """Short text should use single-pass, message called once."""
        mock_message.return_value = [_make_dummy_response(VALID_GUIDELINE)]

        agent = ContextReviewerAgent("en", "zh")
        # Large context window: no chunking needed.
        agent.chatbot.model_info.context_window = 100000
        result = agent.build_context(["Hello", "World"], title="Test")

        self.assertEqual(mock_message.call_count, 1)
        self.assertIn("Glossary", result)
        self.assertIn("Summary", result)

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_long_text_triggers_chunking(self, mock_message):
        """When context window is small and chunked_guideline=True, should split and merge."""
        mock_message.return_value = [_make_dummy_response(MERGED_GUIDELINE)]

        agent = ContextReviewerAgent("en", "zh", chunked_guideline=True)
        agent.chatbot.model_info.context_window = 2500
        agent.chatbot.model_info.max_tokens = 1024
        texts = [f"Line {i}: Some subtitle text here that is a bit longer." for i in range(200)]
        result = agent.build_context(texts, title="Test")

        # Should have called message multiple times: N chunks + 1 merge.
        self.assertGreaterEqual(mock_message.call_count, 3)
        self.assertIn("Glossary", result)

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_long_text_without_flag_returns_empty(self, mock_message):
        """When chunked_guideline=False (default), long text should return empty without calling LLM."""
        mock_message.return_value = [_make_dummy_response(VALID_GUIDELINE)]

        agent = ContextReviewerAgent("en", "zh")
        agent.chatbot.model_info.context_window = 2500
        agent.chatbot.model_info.max_tokens = 1024
        texts = [f"Line {i}: Some subtitle text here that is a bit longer." for i in range(200)]
        result = agent.build_context(texts, title="Test")

        # Should not call LLM at all, return empty guideline.
        self.assertEqual(mock_message.call_count, 0)
        self.assertEqual(result, "")

    @patch("openlrc.chatbot.GPTBot.message")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-dummy"})
    def test_merge_failure_raises(self, mock_message):
        """If all merge retries fail, should raise RuntimeError."""
        from openlrc.exceptions import ChatBotException

        def side_effect_fn(*args, **kwargs):
            system_content = args[0][0]["content"] if args else ""
            if "merge" in system_content.lower():
                raise ChatBotException("merge failed")
            return [_make_dummy_response(PARTIAL_GUIDELINE_1)]

        mock_message.side_effect = side_effect_fn

        agent = ContextReviewerAgent("en", "zh", chunked_guideline=True)
        agent.chatbot.model_info.context_window = 2500
        agent.chatbot.model_info.max_tokens = 1024
        texts = [f"Line {i}: Some subtitle text here that is a bit longer." for i in range(200)]

        with self.assertRaises(RuntimeError):
            agent.build_context(texts, title="Test")

    def test_split_texts_by_tokens(self):
        """Verify token-based splitting produces chunks within budget."""
        texts = [f"Word{i} " * 10 for i in range(20)]  # ~10 tokens each
        chunks = ContextReviewerAgent._split_texts_by_tokens(texts, max_text_tokens=50)

        self.assertGreater(len(chunks), 1)
        # All original lines should be preserved.
        flat = [line for chunk in chunks for line in chunk]
        self.assertEqual(flat, texts)
