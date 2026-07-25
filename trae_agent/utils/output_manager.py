"""Output directory manager — organizes generated files by project on Desktop.

All generated files (images, videos, documents) are placed in:
    ~/Desktop/trae-agent-outputs/<project-name>/

Project name is derived from:
1. The --working-dir CLI argument (basename)
2. Or the current working directory (basename)
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def get_output_dir(project_name: str | None = None, working_dir: str | None = None) -> Path:
    """Get the output directory for generated files.

    Returns:
        ~/Desktop/trae-agent-outputs/<project-name>/
    """
    if not project_name:
        if working_dir:
            project_name = Path(working_dir).name
        else:
            project_name = Path.cwd().name

    # Sanitize project name for filesystem
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
    safe_name = safe_name.strip("_") or "default"

    desktop = Path.home() / "Desktop"
    output_dir = desktop / "trae-agent-outputs" / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def generate_output_path(
    filename: str | None = None,
    extension: str = ".png",
    project_name: str | None = None,
    working_dir: str | None = None,
) -> Path:
    """Generate a full output path for a new file.

    Args:
        filename: Optional base name (without extension). Auto-generated if None.
        extension: File extension including dot (e.g. ".png", ".mp4", ".pdf").
        project_name: Project folder name. Auto-detected if None.
        working_dir: Working directory for project name detection.

    Returns:
        Full path like ~/Desktop/trae-agent-outputs/my-project/generated_20260725_120000.png
    """
    output_dir = get_output_dir(project_name, working_dir)

    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{ts}"

    # Ensure extension starts with dot
    if not extension.startswith("."):
        extension = "." + extension

    # Remove extension from filename if present
    if filename.endswith(extension):
        filename = filename[: -len(extension)]

    return output_dir / f"{filename}{extension}"
