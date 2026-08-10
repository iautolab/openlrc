#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Shared test configuration.

Live integration tests use a single OpenAI-compatible DeepSeek model. The
endpoint and model name remain configurable for compatible local test servers,
but credentials always come from ``DEEPSEEK_API_KEY``.

Environment variables:
    DEEPSEEK_API_KEY             DeepSeek API key
    OPENLRC_TEST_LLM_BASE_URL    OpenAI-compatible API base URL
    OPENLRC_TEST_MODEL           Model name (default: deepseek-v4-flash)
    OPENLRC_TEST_LIVE_API        Set to "1" to enable live API tests
    OPENLRC_TEST_STRESS          Set to "1" to enable stress tests
"""

import os

from openlrc.models import ModelConfig, ModelProvider

TEST_LLM_BASE_URL = os.environ.get("OPENLRC_TEST_LLM_BASE_URL", "https://api.deepseek.com")
TEST_LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
TEST_MODEL = ModelConfig(
    provider=ModelProvider.OPENAI,
    name=os.environ.get("OPENLRC_TEST_MODEL", "deepseek-v4-flash"),
    base_url=TEST_LLM_BASE_URL,
    api_key=TEST_LLM_API_KEY,
)

LIVE_API = os.environ.get("OPENLRC_TEST_LIVE_API", "").lower() in ("1", "true", "yes")
STRESS_TEST = os.environ.get("OPENLRC_TEST_STRESS", "").lower() in ("1", "true", "yes")
