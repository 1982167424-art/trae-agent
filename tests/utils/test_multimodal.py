"""Tests for the multimodal image pipeline.

Covers:
- llm_basics: normalize_content / to_provider_content (incl. graceful degrade)
- multimodal: load_image / load_images
- per-client content converters (openai-compatible, openai responses,
  anthropic, gemini, ollama)
- trajectory recorder image redaction
"""

import base64

import pytest

from trae_agent.utils.llm_clients.llm_basics import (
    ImageContent,
    LLMMessage,
    TextContent,
    normalize_content,
    to_provider_content,
)
from trae_agent.utils.multimodal import load_image, load_images

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
B64 = base64.b64encode(PNG_BYTES).decode()


# ---------------------------------------------------------------------------
# normalize_content / to_provider_content
# ---------------------------------------------------------------------------


class TestNormalizeContent:
    def test_plain_string(self):
        parts = normalize_content("hello", supports_multimodal=True)
        assert len(parts) == 1
        assert isinstance(parts[0], TextContent)
        assert parts[0].text == "hello"

    def test_none_content(self):
        parts = normalize_content(None, supports_multimodal=True)
        assert parts == [] or all(isinstance(p, TextContent) for p in parts)

    def test_mixed_content_kept_when_multimodal(self):
        content = [TextContent(text="look"), ImageContent(data=B64)]
        parts = normalize_content(content, supports_multimodal=True)
        assert any(isinstance(p, ImageContent) for p in parts)
        assert any(isinstance(p, TextContent) for p in parts)

    def test_images_dropped_when_not_multimodal(self):
        """Graceful degradation: text-only models never see image parts."""
        content = [TextContent(text="look"), ImageContent(data=B64)]
        parts = normalize_content(content, supports_multimodal=False)
        assert all(not isinstance(p, ImageContent) for p in parts)
        assert any(isinstance(p, TextContent) and p.text == "look" for p in parts)


class TestToProviderContent:
    def test_single_text_returns_plain_str(self):
        """Legacy behavior: pure-text messages stay plain strings."""
        result = to_provider_content("hello", True, lambda parts: parts)
        assert result == "hello"

    def test_multimodal_calls_converter(self):
        content = [TextContent(text="a"), ImageContent(data=B64)]
        called = {}

        def conv(parts):
            called["parts"] = parts
            return ["converted"]

        result = to_provider_content(content, True, conv)
        assert result == ["converted"]
        assert len(called["parts"]) == 2

    def test_degrade_to_plain_str(self):
        """When model lacks multimodal, images drop and result is plain text."""
        content = [TextContent(text="only text"), ImageContent(data=B64)]
        result = to_provider_content(content, False, lambda parts: parts)
        assert result == "only text"


# ---------------------------------------------------------------------------
# load_image / load_images
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_load_png(self, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(PNG_BYTES)
        img = load_image(str(p))
        assert isinstance(img, ImageContent)
        assert img.media_type == "image/png"
        assert base64.b64decode(img.data) == PNG_BYTES

    def test_load_jpeg_media_type(self, tmp_path):
        p = tmp_path / "photo.JPG"
        p.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        img = load_image(str(p))
        assert img.media_type == "image/jpeg"

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/img.png")

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("not an image")
        with pytest.raises(ValueError):
            load_image(str(p))

    def test_load_images_multiple(self, tmp_path):
        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.webp"
        p1.write_bytes(PNG_BYTES)
        p2.write_bytes(b"RIFFfakewebp")
        imgs = load_images([str(p1), str(p2)])
        assert len(imgs) == 2
        assert imgs[0].media_type == "image/png"
        assert imgs[1].media_type == "image/webp"


# ---------------------------------------------------------------------------
# Per-client converters
# ---------------------------------------------------------------------------


class TestClientConverters:
    def _parts(self):
        return [TextContent(text="hi"), ImageContent(data=B64, media_type="image/png")]

    def test_openai_compatible(self):
        from trae_agent.utils.llm_clients.openai_compatible_base import (
            _convert_oa_content,
        )

        out = _convert_oa_content(self._parts())
        assert out[0] == {"type": "text", "text": "hi"}
        assert out[1]["type"] == "image_url"
        assert out[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_openai_responses(self):
        from trae_agent.utils.llm_clients.openai_client import OpenAIClient

        out = OpenAIClient._convert_oa_responses(self._parts())
        assert out[0] == {"type": "input_text", "text": "hi"}
        assert out[1]["type"] == "input_image"
        assert out[1]["image_url"].startswith("data:image/png;base64,")

    def test_anthropic(self):
        from trae_agent.utils.llm_clients.anthropic_client import (
            _convert_anthropic_content,
        )

        out = _convert_anthropic_content(self._parts())
        assert out[0] == {"type": "text", "text": "hi"}
        assert out[1]["type"] == "image"
        assert out[1]["source"]["type"] == "base64"
        assert out[1]["source"]["media_type"] == "image/png"
        assert out[1]["source"]["data"] == B64

    def test_gemini(self):
        pytest.importorskip("google.genai")
        from trae_agent.utils.llm_clients.google_client import _convert_gemini_content

        parts = _convert_gemini_content(self._parts())
        # first part text, second part inline_data blob
        assert parts[0].text == "hi"
        assert parts[1].inline_data is not None
        assert parts[1].inline_data.mime_type == "image/png"
        assert parts[1].inline_data.data == PNG_BYTES

    def test_gemini_remote_url_rejected(self):
        pytest.importorskip("google.genai")
        from trae_agent.utils.llm_clients.google_client import _convert_gemini_content

        with pytest.raises(ValueError):
            _convert_gemini_content([ImageContent(url="https://example.com/x.png")])

    def test_ollama(self):
        from trae_agent.utils.llm_clients.ollama_client import _convert_ollama_content

        out = _convert_ollama_content(self._parts())
        assert out[0] == {"type": "text", "text": "hi"}
        assert out[1]["type"] == "image"
        assert out[1]["image"] == B64


# ---------------------------------------------------------------------------
# Trajectory redaction
# ---------------------------------------------------------------------------


class TestTrajectoryRedaction:
    def test_image_redacted(self, tmp_path):
        from trae_agent.utils.trajectory_recorder import TrajectoryRecorder

        recorder = TrajectoryRecorder(str(tmp_path / "traj.json"))
        msg = LLMMessage(
            role="user",
            content=[TextContent(text="see image"), ImageContent(data=B64)],
        )
        data = recorder._serialize_message(msg)
        assert data["role"] == "user"
        text_parts = [p for p in data["content"] if p.get("type") == "text"]
        image_parts = [p for p in data["content"] if p.get("type") == "image"]
        assert text_parts[0]["text"] == "see image"
        assert image_parts[0]["data"] == "[image data redacted from trajectory]"
        assert B64 not in str(data)

    def test_plain_string_untouched(self, tmp_path):
        from trae_agent.utils.trajectory_recorder import TrajectoryRecorder

        recorder = TrajectoryRecorder(str(tmp_path / "traj.json"))
        msg = LLMMessage(role="assistant", content="plain answer")
        data = recorder._serialize_message(msg)
        assert data["content"] == "plain answer"
