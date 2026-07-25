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
    """Pure unit tests for multimodal (vision) message parsing.

    These tests do NOT require a running Ollama server — they only exercise
    ``parse_messages`` / ``set_chat_history`` and therefore run in CI even
    when ``SKIP_OLLAMA_TEST=true`` is set (the network-dependent
    ``TestOllamaClient`` class is the one that gets skipped).
    """

    def _make_client(self) -> OllamaClient:
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
        return OllamaClient(model_config)

    def test_parse_messages_forwards_images_for_user(self):
        """Multimodal user messages must carry the ``images`` field so that
        local vision-language models (e.g. ``kimi-vl-a3b``) can actually see
        images."""
        ollama_client = self._make_client()
        image_bytes = b"\x89PNG\r\n\x1a\n"  # PNG header bytes, enough for the parser
        messages = [
            LLMMessage(role="system", content="You are a vision assistant."),
            LLMMessage(
                role="user",
                content="What is in this image?",
                images=[image_bytes, "/tmp/fake.png"],
            ),
        ]
        parsed = ollama_client.parse_messages(messages)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["role"], "system")
        self.assertNotIn("images", parsed[0])

        self.assertEqual(parsed[1]["role"], "user")
        self.assertEqual(parsed[1]["content"], "What is in this image?")
        self.assertIn("images", parsed[1])
        self.assertEqual(parsed[1]["images"], [image_bytes, "/tmp/fake.png"])

    def test_parse_messages_omits_images_when_absent(self):
        """Backward compatibility: a plain text user message must NOT carry an
        ``images`` key, otherwise some Ollama backends reject the payload."""
        ollama_client = self._make_client()
        parsed = ollama_client.parse_messages([LLMMessage(role="user", content="plain text only")])
        self.assertEqual(parsed[0]["role"], "user")
        self.assertEqual(parsed[0]["content"], "plain text only")
        self.assertNotIn("images", parsed[0])

    def test_parse_messages_does_not_attach_images_to_system_or_assistant(self):
        """Only user messages should carry images; system/assistant messages
        must remain text-only even if an ``images`` value is supplied."""
        ollama_client = self._make_client()
        parsed = ollama_client.parse_messages(
            [
                LLMMessage(role="system", content="sys", images=[b"\x89PNG\r\n\x1a\n"]),
                LLMMessage(role="assistant", content="asst", images=[b"\x89PNG\r\n\x1a\n"]),
            ]
        )
        self.assertEqual(parsed[0]["role"], "system")
        self.assertNotIn("images", parsed[0])
        self.assertEqual(parsed[1]["role"], "assistant")
        self.assertNotIn("images", parsed[1])

    def test_set_chat_history_preserves_images(self):
        """``set_chat_history`` must round-trip image-bearing messages so that
        subsequent ``chat()`` calls can forward them to Ollama."""
        ollama_client = self._make_client()
        ollama_client.set_chat_history(
            [
                LLMMessage(
                    role="user",
                    content="describe this",
                    images=[b"\x89PNG\r\n\x1a\n"],
                )
            ]
        )
        self.assertEqual(len(ollama_client.message_history), 1)
        self.assertIn("images", ollama_client.message_history[0])


if __name__ == "__main__":
    unittest.main()
