"""Output directory manager — organizes generated files by project on Desktop.

Directory structure:
    ~/Desktop/trae-agent-outputs/<project-name>/
    ├── run_20260725_120000/          # 每次运行一个子文件夹
    │   ├── generated_image.png
    │   └── trajectories/
    │       └── trajectory_*.json
    └── run_20260725_130000/
        └── ...

Project name is derived from --working-dir or current directory.
Each run creates a timestamped subfolder.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# Module-level session ID, set once per process
_session_id: str | None = None


def _get_session_id() -> str:
    """Get or create a session ID for this agent run."""
    global _session_id
    if _session_id is None:
        _session_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return _session_id


def reset_session():
    """Reset session ID (call at start of each new task)."""
    global _session_id
    _session_id = None


def get_output_dir(project_name: str | None = None, working_dir: str | None = None) -> Path:
    """Get the output directory for the current run.

    Returns:
        ~/Desktop/trae-agent-outputs/<project-name>/<session-id>/
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
    session_id = _get_session_id()
    output_dir = desktop / "trae-agent-outputs" / safe_name / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_project_dir(project_name: str | None = None, working_dir: str | None = None) -> Path:
    """Get the project-level output directory (without session subfolder).

    Returns:
        ~/Desktop/trae-agent-outputs/<project-name>/
    """
    if not project_name:
        if working_dir:
            project_name = Path(working_dir).name
        else:
            project_name = Path.cwd().name

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
    safe_name = safe_name.strip("_") or "default"

    desktop = Path.home() / "Desktop"
    project_dir = desktop / "trae-agent-outputs" / safe_name
    project_dir.mkdir(parents=True, exist_ok=True)

    return project_dir


def generate_output_path(
    filename: str | None = None,
    extension: str = ".png",
    project_name: str | None = None,
    working_dir: str | None = None,
) -> Path:
    """Generate a full output path for a new file.

    Returns:
        ~/Desktop/trae-agent-outputs/<project>/run_<ts>/<filename>.<ext>
    """
    output_dir = get_output_dir(project_name, working_dir)

    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{ts}"

    if not extension.startswith("."):
        extension = "." + extension

    if filename.endswith(extension):
        filename = filename[: -len(extension)]

    return output_dir / f"{filename}{extension}"
