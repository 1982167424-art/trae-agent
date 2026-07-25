# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


import base64
from dataclasses import dataclass

from trae_agent.tools.base import ToolCall, ToolResult


@dataclass
class LLMMessage:
    """Standard message format.

    ``images`` 为多模态(视觉)消息的可选图片载荷列表,每个元素可以是图片
    原始 bytes、base64 编码字符串或图片文件路径。不支持视觉输入的 provider
    会忽略该字段。各 client 会按本 provider 的 API 规范把 images 转换成
    对应的 content block / part 结构(Ollama 透传 bytes;OpenAI /
    OpenAI-compatible / Anthropic 转 base64 data URL / source;Google
    转 Part.from_bytes)。
    """

    role: str
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    images: list[bytes | str] | None = None


# --- 多模态共用工具 -------------------------------------------------
# 各 provider 的 API 接收图片的格式各不相同,但底层的「识别 MIME」和「拼 data URL」
# 是公共逻辑,集中在这里避免散落多处。


_IMAGE_MAGIC_TO_MIME: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    # WebP: RIFF....WEBP
    (b"RIFF", "image/webp"),
    (b"\x42\x50\x47\xfb", "image/bpg"),
]


def detect_image_mime(data: bytes) -> str:
    """Sniff the MIME type of an image from its magic bytes.

    Falls back to ``image/png`` (the most permissive choice across providers)
    when the bytes don't match a known signature — passing an unknown image
    to a provider still beats dropping it silently.
    """
    for magic, mime in _IMAGE_MAGIC_TO_MIME:
        if data.startswith(magic):
            # WebP 还需校验偏移 8-12 是 'WEBP'
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return "image/png"


def bytes_to_data_url(data: bytes, mime: str | None = None) -> str:
    """Wrap raw image bytes as a base64 ``data:`` URL.

    Used for OpenAI Chat Completions / Responses ``image_url`` and for
    Anthropic ``image`` source blocks.
    """
    resolved_mime = mime or detect_image_mime(data)
    return f"data:{resolved_mime};base64,{base64.b64encode(data).decode('ascii')}"


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
