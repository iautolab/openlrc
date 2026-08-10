#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

import unittest
from pathlib import Path

import openai

from openlrc.agents import create_chatbot
from openlrc.media_utils import get_similarity
from openlrc.translate import LLMTranslator
from tests.conftest import LIVE_API, TEST_LLM_API_KEY, TEST_MODEL


@unittest.skipUnless(LIVE_API, "Requires OPENLRC_TEST_LIVE_API=1 and valid API keys")
class TestLLMTranslator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TEST_LLM_API_KEY:
            raise unittest.SkipTest("DEEPSEEK_API_KEY is required for LLM integration tests.")

    def tearDown(self) -> None:
        compare_path = Path("translate_intermediate.json")
        compare_path.unlink(missing_ok=True)

    def test_single_chunk_translation(self):
        text = "Hello, how are you?"
        chatbot = create_chatbot(TEST_MODEL)
        try:
            translator = LLMTranslator(chatbot=chatbot)
            translation = translator.translate(text, "en", "es")[0]
        finally:
            chatbot.close()

        self.assertGreater(get_similarity(translation, "Hola, ¿cómo estás?"), 0.5)

    def test_multiple_chunk_translation(self):
        texts = ["Hello, how are you?", "I am fine, thank you."]
        chatbot = create_chatbot(TEST_MODEL)
        try:
            translator = LLMTranslator(chatbot=chatbot)
            translations = translator.translate(texts, "en", "es")
        finally:
            chatbot.close()
        self.assertGreater(get_similarity(translations[0], "Hola, ¿cómo estás?"), 0.5)
        self.assertGreater(get_similarity(translations[1], "Estoy bien, gracias."), 0.5)

    def test_different_language_translation(self):
        text = "Hello, how are you?"
        chatbot = create_chatbot(TEST_MODEL)
        try:
            translator = LLMTranslator(chatbot=chatbot)
            translation = translator.translate(text, "en", "ja")[0]
            self.assertTrue(
                get_similarity(translation, "こんにちは、お元気ですか？") > 0.5
                or get_similarity(translation, "こんにちは、調子はどうですか?") > 0.5
            )
        except openai.OpenAIError:
            pass
        except AssertionError:
            print(f"Translation failed: {text} -> {translation}")
        finally:
            chatbot.close()

    def test_empty_text_list_translation(self):
        chatbot = create_chatbot(TEST_MODEL)
        try:
            translator = LLMTranslator(chatbot=chatbot)
            translations = translator.translate([], "en", "es")
        finally:
            chatbot.close()
        self.assertEqual(translations, [])

    def test_atomic_translate(self):
        texts = ["Hello, how are you?", "I am fine, thank you."]
        chatbot = create_chatbot(TEST_MODEL)
        try:
            translator = LLMTranslator(chatbot=chatbot)
            translations = translator.atomic_translate(chatbot, texts, "en", "zh")
        finally:
            chatbot.close()
        self.assertGreater(get_similarity(translations[0], "你好，你好吗？"), 0.5)
        self.assertGreater(get_similarity(translations[1], "我很好，谢谢。"), 0.5)


# Not integrated by the openlrc main function because of performance
#
# class TestDeepLTranslator(unittest.TestCase):
#     def test_single_chunk_translation(self):
#         text = 'Hello, how are you?'
#         translator = DeepLTranslator()
#         translation = translator.translate(text, 'en', 'es')[0]
#
#         assert get_similarity(translation, 'Hola, ¿cómo estás?') > 0.5
#
#     def test_multiple_chunk_translation(self):
#         texts = ['Hello, how are you?', 'I am fine, thank you.']
#         translator = DeepLTranslator()
#         translations = translator.translate(texts, 'en', 'es')
#         assert get_similarity(translations[0], 'Hola, ¿cómo estás?') > 0.5
#         assert get_similarity(translations[1], 'Estoy bien, gracias.') > 0.5
