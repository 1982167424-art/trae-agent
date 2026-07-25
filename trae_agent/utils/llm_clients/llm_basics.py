# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from dataclasses import dataclass

from trae_agent.tools.base import ToolCall, ToolResult


@dataclass
class LLMMessage:
    """Standard message format.

    ``images`` is an optional list of image payloads for multimodal (vision)
    messages. Each entry may be raw image bytes, a base64-encoded string, or a
    filesystem path to an image file. Providers that do not support vision
    input will simply ignore this field. Currently the Ollama client forwards
    ``images`` to the underlying ``ollama.chat()`` call so that local
    vision-language models (e.g. ``kimi-vl-a3b``) can receive image input.
    """

    role: str
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    images: list[bytes | str] | None = None


@dataclass
class LLMUsage:
    """LLM usage format."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def __str__(self) -> str:
        return f"LLMUsage(input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, cache_creation_input_tokens={self.cache_creation_input_tokens}, cache_read_input_tokens={self.cache_read_input_tokens}, reasoning_tokens={self.reasoning_tokens})"


@dataclass
class LLMResponse:
    """Standard LLM response format."""

    content: str
    usage: LLMUsage | None = None
    model: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None
