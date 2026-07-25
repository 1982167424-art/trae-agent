# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
This test file is used to test the Ollama client. This test program is expected to verify basic functionalities and check if the results match the expected output.

Currently, we only test init, chat, and set chat history.

WARNING: This Ollama test should not be used in the GitHub Actions workflow, as using Ollama for testing consumes too much time due to installation.
"""

import os
import unittest

from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.llm_clients.ollama_client import OllamaClient

TEST_MODEL = "qwen3:4b"


@unittest.skipIf(
    os.getenv("SKIP_OLLAMA_TEST", "").lower() == "true",
    "Ollama tests skipped due to SKIP_OLLAMA_TEST environment variable",
)
class TestOllamaClient(unittest.TestCase):
    def test_OllamaClient_init(self):
        """
        Test ollama client provides a test case for initialize the ollama client
        It should not be used to check any configiguration based on BaseLLMClient instead we should just check the parameters
        that will change during the init process.
        """
        model_config = ModelConfig(
            TEST_MODEL,
            model_provider=ModelProvider(
                provider="ollama",
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                api_version=None,
            ),
            max_tokens=1000,
            temperature=0.8,
            top_p=7.0,
            top_k=8,
            parallel_tool_calls=False,
            max_retries=1,
        )
        ollama_client = OllamaClient(model_config)
        self.assertEqual(ollama_client.api_key, "ollama")
        self.assertEqual(ollama_client.base_url, "http://localhost:11434/v1")

    def test_ollama_set_chat_history(self):
        """
        There is nothing we have to assert for this test case just see if it can run
        """
        model_config = ModelConfig(
            TEST_MODEL,
            model_provider=ModelProvider(
                provider="ollama",
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                api_version=None,
            ),
            max_tokens=1000,
            temperature=0.8,
            top_p=7.0,
            top_k=8,
            parallel_tool_calls=False,
            max_retries=1,
        )
        ollama_client = OllamaClient(model_config)
        message = LLMMessage("user", "this is a test message")
        ollama_client.set_chat_history(messages=[message])
        self.assertTrue(True)  # runnable

    def test_ollama_chat(self):
        """
        There is nothing we have to assert for this test case just see if it can run
        """
        model_config = ModelConfig(
            TEST_MODEL,
            model_provider=ModelProvider(
                provider="ollama",
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                api_version=None,
            ),
            max_tokens=1000,
            temperature=0.8,
            top_p=7.0,
            top_k=8,
            parallel_tool_calls=False,
            max_retries=1,
        )
        ollama_client = OllamaClient(model_config)
        message = LLMMessage("user", "this is a test message")
        ollama_client.chat(messages=[message], model_config=model_config)
        self.assertTrue(True)  # runnable

    def test_supports_tool_calling(self):
        """
        A test case to check the support tool calling function
        """
        model_config = ModelConfig(
            TEST_MODEL,
            model_provider=ModelProvider(
                provider="ollama",
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                api_version=None,
            ),
            max_tokens=1000,
            temperature=0.8,
            top_p=7.0,
            top_k=8,
            parallel_tool_calls=False,
            max_retries=1,
        )
        ollama_client = OllamaClient(model_config)
        self.assertEqual(ollama_client.supports_tool_calling(model_config), True)
        model_config.model = "no such model"
        self.assertEqual(ollama_client.supports_tool_calling(model_config), False)


class TestOllamaMultimodalParsing(unittest.TestCase):
    """Verify multimodal (vision) message parsing for kimi-vl-a3b.

    These tests exercise ``OllamaClient.parse_messages`` only — no running
    Ollama daemon is required, so they are NOT gated on SKIP_OLLAMA_TEST.
    They guard against the regression where the ``images`` field was silently
    dropped, which was the explicit multimodal block that prevented
    kimi-vl-a3b from receiving image input.
    """

    # NOTE: intentionally not decorated with the SKIP_OLLAMA_TEST skip —
    # parse_messages is a pure function and does not contact the Ollama daemon.

    def _make_client(self) -> OllamaClient:
        model_config = ModelConfig(
            "kimi-vl-a3b",
            model_provider=ModelProvider(
                provider="ollama",
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                api_version=None,
            ),
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            top_k=0,
            parallel_tool_calls=False,
            max_retries=1,
        )
        return OllamaClient(model_config)

    def test_user_message_without_images_has_no_images_key(self):
        """Pure-text user messages must not carry an ``images`` field."""
        client = self._make_client()
        parsed = client.parse_messages(
            [LLMMessage(role="user", content="describe this")]
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["role"], "user")
        self.assertEqual(parsed[0]["content"], "describe this")
        self.assertNotIn("images", parsed[0])

    def test_user_message_with_images_forwards_images(self):
        """Vision user messages must forward the ``images`` payload verbatim."""
        client = self._make_client()
        img_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes"
        parsed = client.parse_messages(
            [
                LLMMessage(
                    role="user",
                    content="what is in this image?",
                    images=[img_bytes],
                )
            ]
        )
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["role"], "user")
        self.assertEqual(parsed[0]["content"], "what is in this image?")
        self.assertIn("images", parsed[0])
        self.assertEqual(parsed[0]["images"], [img_bytes])

    def test_multiple_images_preserved_in_order(self):
        """Multiple --image flags must be preserved in the order supplied."""
        client = self._make_client()
        img_a = b"image-a-bytes"
        img_b = b"image-b-bytes"
        parsed = client.parse_messages(
            [
                LLMMessage(
                    role="user",
                    content="compare these two",
                    images=[img_a, img_b],
                )
            ]
        )
        self.assertEqual(parsed[0]["images"], [img_a, img_b])

    def test_system_and_assistant_messages_unaffected_by_images_field(self):
        """Non-user roles must remain text-only even when images exist elsewhere."""
        client = self._make_client()
        parsed = client.parse_messages(
            [
                LLMMessage(role="system", content="you are a vision model"),
                LLMMessage(role="user", content="look", images=[b"img"]),
                LLMMessage(role="assistant", content="ok"),
            ]
        )
        self.assertNotIn("images", parsed[0])
        self.assertNotIn("images", parsed[2])
        self.assertIn("images", parsed[1])


if __name__ == "__main__":
    unittest.main()
