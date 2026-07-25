# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from dataclasses import dataclass

from trae_agent.tools.base import ToolCall, ToolResult


@dataclass
class LLMMessage:
    """Standard message format."""

    role: str
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


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


@dataclass
class TextContent:
    """A plain-text block inside a multimodal message."""

    text: str = ""
    type: str = "text"


@dataclass
class ImageContent:
    """A single image block for multimodal messages.

    Provide EITHER a remote/Data-URI ``url`` OR raw ``data``
    (base64 string) + ``media_type``. ``detail`` is only honored by
    providers that support it (OpenAI chat completions).
    """

    url: str | None = None
    data: str | None = None
    media_type: str = "image/png"
    detail: str = "auto"
    type: str = "image"


# ``LLMMessage.content`` may be ``str`` (legacy text), a ``list`` of
# ``TextContent`` / ``ImageContent`` (multimodal), or ``None``.
MultimodalPart = "TextContent | ImageContent"


def is_multimodal(content) -> bool:
    """Return True if ``content`` is a multimodal (image-bearing) payload."""
    return isinstance(content, list) and any(
        isinstance(p, ImageContent)
        or (isinstance(p, dict) and p.get("type") in ("image", "image_url"))
        for p in content
    )


def _coerce_part(p, supports_multimodal: bool) -> "MultimodalPart | None":
    """Coerce a raw list element into a typed part, or None to drop."""
    if isinstance(p, (TextContent, ImageContent)):
        if isinstance(p, ImageContent) and not supports_multimodal:
            return None
        return p
    if isinstance(p, dict):
        ptype = p.get("type")
        if ptype in ("image", "image_url"):
            if not supports_multimodal:
                return None
            return ImageContent(
                url=p.get("url"),
                data=p.get("data"),
                media_type=p.get("media_type", "image/png"),
                detail=p.get("detail", "auto"),
            )
        # Anything else is treated as text.
        return TextContent(text=p.get("text", ""))
    # Fallback: stringify.
    return TextContent(text=str(p))


def normalize_content(
    content: str | list | None, supports_multimodal: bool
) -> list["MultimodalPart"]:
    """Normalize a message ``content`` into a list of typed parts.

    - ``None`` → ``[]``
    - ``str`` → ``[TextContent(text)]``
    - ``list`` → coerced parts; image parts are DROPPED when
      ``supports_multimodal`` is False (graceful degradation so we
      never send images to a text-only model).
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [TextContent(text=content)]
    parts: list[MultimodalPart] = []
    for p in content:
        coerced = _coerce_part(p, supports_multimodal)
        if coerced is not None:
            parts.append(coerced)
    return parts


def to_provider_content(
    content: str | list | None,
    supports_multimodal: bool,
    converter: "callable[[list[MultimodalPart]], object]",
) -> "str | object":
    """Return provider-ready message content from a message ``content``.

    - Legacy ``str`` (or a single text part) is returned as-is, so
      existing single-string message handling is byte-for-byte unchanged.
    - A multimodal ``list`` of parts is passed to ``converter``
      (which builds the provider-specific content structure).
    - Image parts are dropped when ``supports_multimodal`` is False
      (see :func:`normalize_content`), so text-only models never
      receive images.
    """
    parts = normalize_content(content, supports_multimodal)
    if len(parts) == 1 and isinstance(parts[0], TextContent):
        return parts[0].text
    if not parts:
        return content if isinstance(content, str) else ""
    return converter(parts)
