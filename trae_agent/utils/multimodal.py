# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Helpers for loading local image files into multimodal message content.

Images are read from disk and base64-encoded so they can be embedded
directly in a model's ``content`` array (no extra upload step).
"""

import base64
from pathlib import Path

from trae_agent.utils.llm_clients.llm_basics import ImageContent

# Map file extension -> RFC-683 media (MIME) type.
EXT_MEDIA: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def load_image(path: str | Path) -> ImageContent:
    """Load a single image file into an :class:`ImageContent` block.

    The file is read and base64-encoded; the media type is inferred
    from the file extension.

    Raises:
        FileNotFoundError: if the path does not exist or is not a file.
        ValueError: if the file extension is not a supported image type.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    media_type = EXT_MEDIA.get(p.suffix.lower())
    if media_type is None:
        supported = ", ".join(sorted(EXT_MEDIA))
        raise ValueError(
            f"Unsupported image type '{p.suffix}' for {path}. "
            f"Supported: {supported}"
        )
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return ImageContent(data=data, media_type=media_type)


def load_images(paths: list[str | Path]) -> list[ImageContent]:
    """Load multiple image files; failures raise immediately (fail-fast)."""
    return [load_image(p) for p in paths]
