# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Multimodal (vision) message parsing tests for every provider client.

These tests exercise each client's ``parse_messages`` only — no running
daemon or API key is required. They guard against the regression where the
``LLMMessage.images`` field was silently dropped, which was the explicit
multimodal block that prevented vision-language models (kimi-vl-a3b on
Ollama, GPT-4o, Claude 3, Gemini, Doubao-vision, etc.) from receiving
image input.

Provider-specific format expectations verified here:
  - Ollama:             user msg carries ``images: [bytes, ...]``
  - OpenAI-compatible:  user msg content is a list of
                        {type: text | image_url, ...} blocks
  - OpenAI Responses:   user msg content is a list of
                        {type: input_text | input_image, ...} blocks
  - Anthropic:          user msg content is a list of
                        {type: text | image, source: {type: base64, ...}}
  - Google Gemini:      user Content has multiple Parts, image Part
                        built via Part.from_bytes
"""

import base64
import unittest

from trae_agent.utils.config import ModelConfig, ModelProvider
from trae_agent.utils.llm_clients.anthropic_client import AnthropicClient
from trae_agent.utils.llm_clients.google_client import GoogleClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.llm_clients.ollama_client import OllamaClient
from trae_agent.utils.llm_clients.openai_client import OpenAIClient
from trae_agent.utils.llm_clients.openrouter_client import OpenRouterClient

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_BYTES = PNG_MAGIC + b"fake-png-body"


def _make_config(provider: str, model: str, base_url: str = "http://localhost/v1") -> ModelConfig:
    return ModelConfig(
        model=model,
        model_provider=ModelProvider(
            provider=provider,
            api_key="test-key",
            base_url=base_url,
            api_version=None,
        ),
        max_tokens=8192,
        temperature=0.5,
        top_p=0.95,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=1,
    )


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------


class TestOllamaMultimodalParsing(unittest.TestCase):
    def _client(self) -> OllamaClient:
        return OllamaClient(_make_config("ollama", "kimi-vl-a3b", "http://localhost:11434/v1"))

    def test_user_message_without_images_has_no_images_key(self):
        parsed = self._client().parse_messages([LLMMessage(role="user", content="hi")])
        self.assertNotIn("images", parsed[0])

    def test_user_message_with_images_forwards_bytes_verbatim(self):
        parsed = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=[PNG_BYTES])]
        )
        self.assertEqual(parsed[0]["images"], [PNG_BYTES])

    def test_multiple_images_preserved_in_order(self):
        parsed = self._client().parse_messages(
            [LLMMessage(role="user", content="cmp", images=[PNG_BYTES, b"img-b"])]
        )
        self.assertEqual(parsed[0]["images"], [PNG_BYTES, b"img-b"])

    def test_system_and_assistant_messages_unaffected(self):
        parsed = self._client().parse_messages(
            [
                LLMMessage(role="system", content="sys"),
                LLMMessage(role="user", content="look", images=[PNG_BYTES]),
                LLMMessage(role="assistant", content="ok"),
            ]
        )
        self.assertNotIn("images", parsed[0])
        self.assertNotIn("images", parsed[2])
        self.assertIn("images", parsed[1])


# --------------------------------------------------------------------------
# OpenAI-compatible (OpenRouter as concrete instance — no API call made)
# --------------------------------------------------------------------------


class TestOpenAICompatibleMultimodalParsing(unittest.TestCase):
    def _client(self) -> OpenRouterClient:
        return OpenRouterClient(_make_config("openrouter", "gpt-4o"))

    def test_user_message_without_images_uses_plain_string_content(self):
        parsed = self._client().parse_messages([LLMMessage(role="user", content="hi")])
        self.assertEqual(parsed[0]["content"], "hi")

    def test_user_message_with_images_uses_content_block_list(self):
        parsed = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=[PNG_BYTES])]
        )
        content = parsed[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "look"})
        self.assertEqual(content[1]["type"], "image_url")
        url = content[1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        # The encoded payload must round-trip back to the original bytes.
        decoded = base64.b64decode(url.split(",", 1)[1])
        self.assertEqual(decoded, PNG_BYTES)

    def test_string_image_is_passed_through_as_url(self):
        parsed = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=["https://example.com/a.png"])]
        )
        content = parsed[0]["content"]
        self.assertEqual(content[1], {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}})

    def test_multiple_images_preserved_in_order(self):
        parsed = self._client().parse_messages(
            [LLMMessage(role="user", content="cmp", images=[PNG_BYTES, b"\xff\xd8\xffjpeg"])]
        )
        content = parsed[0]["content"]
        self.assertEqual(len(content), 3)  # text + 2 images
        self.assertEqual(content[1]["image_url"]["url"].split(";", 1)[0], "data:image/png")
        self.assertEqual(content[2]["image_url"]["url"].split(";", 1)[0], "data:image/jpeg")


# --------------------------------------------------------------------------
# OpenAI Responses API
# --------------------------------------------------------------------------


class TestOpenAIResponsesMultimodalParsing(unittest.TestCase):
    def _client(self) -> OpenAIClient:
        return OpenAIClient(_make_config("openai", "gpt-4o", "https://api.openai.com/v1"))

    def test_user_message_without_images_uses_plain_string_content(self):
        parsed = self._client().parse_messages([LLMMessage(role="user", content="hi")])
        self.assertEqual(parsed[0], {"role": "user", "content": "hi"})

    def test_user_message_with_images_uses_input_text_input_image_blocks(self):
        parsed = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=[PNG_BYTES])]
        )
        content = parsed[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "input_text", "text": "look"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))

    def test_string_image_passed_through(self):
        parsed = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=["https://example.com/a.png"])]
        )
        content = parsed[0]["content"]
        self.assertEqual(content[1], {"type": "input_image", "image_url": "https://example.com/a.png"})


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class TestAnthropicMultimodalParsing(unittest.TestCase):
    def _client(self) -> AnthropicClient:
        return AnthropicClient(_make_config("anthropic", "claude-3-5-sonnet"))

    def test_user_message_without_images_uses_plain_string_content(self):
        parsed, _ = self._client().parse_messages([LLMMessage(role="user", content="hi")])
        self.assertEqual(parsed[0]["content"], "hi")

    def test_user_message_with_images_uses_text_plus_image_blocks(self):
        parsed, _ = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=[PNG_BYTES])]
        )
        content = parsed[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "look"})
        self.assertEqual(content[1]["type"], "image")
        source = content[1]["source"]
        self.assertEqual(source["type"], "base64")
        self.assertEqual(source["media_type"], "image/png")
        self.assertEqual(base64.b64decode(source["data"]), PNG_BYTES)

    def test_assistant_role_does_not_carry_images(self):
        # images on assistant messages are nonsensical; verify they're ignored.
        parsed, _ = self._client().parse_messages(
            [
                LLMMessage(role="user", content="look", images=[PNG_BYTES]),
                LLMMessage(role="assistant", content="ok", images=[PNG_BYTES]),
            ]
        )
        # assistant msg content stays a plain string
        self.assertEqual(parsed[1]["content"], "ok")

    def test_data_url_string_image_strips_prefix(self):
        data_url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
        parsed, _ = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=[data_url])]
        )
        source = parsed[0]["content"][1]["source"]
        # data must be raw base64 without the data: prefix
        self.assertEqual(base64.b64decode(source["data"]), PNG_BYTES)


# --------------------------------------------------------------------------
# Google Gemini
# --------------------------------------------------------------------------


class TestGoogleMultimodalParsing(unittest.TestCase):
    def _client(self) -> GoogleClient:
        return GoogleClient(_make_config("google", "gemini-2.0-flash"))

    def test_user_message_without_images_has_single_text_part(self):
        parsed, _ = self._client().parse_messages([LLMMessage(role="user", content="hi")])
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].role, "user")
        self.assertEqual(len(parsed[0].parts), 1)
        self.assertEqual(parsed[0].parts[0].text, "hi")

    def test_user_message_with_images_adds_inline_data_parts(self):
        parsed, _ = self._client().parse_messages(
            [LLMMessage(role="user", content="look", images=[PNG_BYTES])]
        )
        parts = parsed[0].parts
        # text part + 1 image part
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].text, "look")
        # image part carries the raw bytes back
        self.assertEqual(parts[1].inline_data.data, PNG_BYTES)
        self.assertEqual(parts[1].inline_data.mime_type, "image/png")

    def test_multiple_images_preserved_in_order(self):
        parsed, _ = self._client().parse_messages(
            [LLMMessage(role="user", content="cmp", images=[PNG_BYTES, b"\xff\xd8\xffjpeg"])]
        )
        parts = parsed[0].parts
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[1].inline_data.mime_type, "image/png")
        self.assertEqual(parts[2].inline_data.mime_type, "image/jpeg")


if __name__ == "__main__":
    unittest.main()
