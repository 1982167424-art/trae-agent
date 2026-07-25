# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from dataclasses import dataclass

from trae_agent.tools.base import ToolCall, ToolResult


@dataclass
class LLMMessage:
    """Standard message format.

    ``images`` 为多模态(视觉)消息的可选图片载荷列表,每个元素可以是图片
    原始 bytes、base64 编码字符串或图片文件路径。不支持视觉输入的 provider
    会忽略该字段。目前 Ollama client 会把 ``images`` 透传给底层
    ``ollama.chat()`` 调用,使本地视觉语言模型(如 ``kimi-vl-a3b`` 配合
    mmproj projector)能够接收图片输入。
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
