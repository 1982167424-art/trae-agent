# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Video generation tool — supports the Doubao Seedance backend via Volcengine Ark.

The Ark video generation API is asynchronous:
  1. POST  /api/v3/contents/generations/tasks   -> returns a task id
  2. GET   /api/v3/contents/generations/tasks/{id}  (poll) -> status + video_url
  3. download the resulting video_url to output_path

Usage:
    VIDEO_GEN_PROVIDER=doubao trae-cli run "用 seedance 生成一段橘猫视频"
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing_extensions import override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter
from trae_agent.utils.output_manager import generate_output_path


class ToolError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# Volcengine Ark video generation base (Beijing). Adjust region if needed.
ARK_VIDEO_BASE = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"

# Default model: Doubao Seedance 1.0 Pro Fast (pre-set inference endpoint, call by model id).
DEFAULT_MODEL = "doubao-seedance-1-0-pro-fast-251015"

# How long to keep polling for the async task before giving up (seconds).
POLL_TIMEOUT = 600
POLL_INTERVAL = 10


class VideoGenTool(Tool):
    """Generate videos from text prompts using Doubao Seedance via Volcengine Ark."""

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)
        self._api_key: str = ""
        self._model: str = ""

    def _ensure_config(self):
        if self._api_key:
            return

        provider = os.environ.get("VIDEO_GEN_PROVIDER", "doubao").lower()
        if provider != "doubao":
            raise ToolError(
                f"Unknown VIDEO_GEN_PROVIDER: {provider}. Only 'doubao' (Volcengine Ark Seedance) is supported."
            )

        api_key = os.environ.get("DOUBAO_API_KEY", "")
        if not api_key:
            try:
                from trae_agent.utils.config import Config

                config = Config.create(config_file="trae_config.yaml")
                if config.trae_agent:
                    api_key = config.trae_agent.model.model_provider.api_key
            except Exception:
                pass
        if not api_key:
            raise ToolError(
                "Doubao video generation requires DOUBAO_API_KEY env var "
                "(or doubao.api_key in trae_config.yaml)."
            )
        self._api_key = api_key
        self._model = os.environ.get("VIDEO_GEN_MODEL", DEFAULT_MODEL)

    @override  # type: ignore[misc]
    def get_name(self) -> str:
        return "video_gen"

    @override  # type: ignore[misc]
    def get_description(self) -> str:
        return """Generate a video from a text description using a text-to-video model.
The generated video is saved to a file (mp4) and its path is returned.
Use this tool when the user asks to create, generate, make, or produce a video / clip / animation.
Examples: "生成一段橘猫玩耍的短视频", "create a 5-second video of a sunset", "用 seedance 做一个猫咪动画" """

    @override  # type: ignore[misc]
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="prompt",
                type="string",
                description="Detailed text description of the video to generate.",
                required=True,
            ),
            ToolParameter(
                name="output_path",
                type="string",
                description="File path to save the generated video (.mp4). Defaults to ./generated_<timestamp>.mp4",
                required=False,
            ),
            ToolParameter(
                name="duration",
                type="integer",
                description="Video duration in seconds. Default: 5",
                required=False,
            ),
            ToolParameter(
                name="ratio",
                type="string",
                description="Aspect ratio. Options: '16:9', '9:16', '1:1'. Default: '16:9'",
                required=False,
            ),
        ]

    # ---- internal HTTP helpers (urllib, no extra deps) ----

    def _post_task(self, prompt: str, duration: int, ratio: str) -> str:
        body = {
            "model": self._model,
            "content": [{"type": "text", "text": prompt}],
            "ratio": ratio,
            "duration": duration,
            "watermark": False,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            ARK_VIDEO_BASE,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        task_id = payload.get("id")
        if not task_id:
            raise ToolError(f"No task id in response: {payload}")
        return task_id

    def _poll_task(self, task_id: str) -> str:
        url = f"{ARK_VIDEO_BASE}/{task_id}"
        deadline = time.time() + POLL_TIMEOUT
        while time.time() < deadline:
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            status = payload.get("status")
            if status == "succeeded":
                content = payload.get("content") or {}
                video_url = content.get("video_url") or content.get("file_url")
                if not video_url:
                    raise ToolError(f"Task succeeded but no video_url: {payload}")
                return video_url
            if status == "failed":
                err = payload.get("error")
                raise ToolError(f"Video generation failed: {err}")
            if status in ("expired",):
                raise ToolError(f"Video generation expired: {payload}")
            # queued / running -> wait and retry
            time.sleep(POLL_INTERVAL)
        raise ToolError(f"Timed out waiting for video task {task_id} after {POLL_TIMEOUT}s")

    @override  # type: ignore[misc]
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        prompt = str(arguments.get("prompt", ""))
        if not prompt:
            return ToolExecResult(error="prompt is required", error_code=-1)

        output_path = str(arguments.get("output_path", ""))
        try:
            duration = int(arguments.get("duration", 5))
        except (TypeError, ValueError):
            duration = 5
        ratio = str(arguments.get("ratio", "16:9"))

        try:
            self._ensure_config()

            task_id = self._post_task(prompt, duration, ratio)
            video_url = self._poll_task(task_id)

            if output_path:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
            else:
                out = generate_output_path(extension=".mp4")

            # download the video
            req = urllib.request.Request(video_url, headers={}, method="GET")
            with urllib.request.urlopen(req, timeout=300) as resp:
                out.write_bytes(resp.read())

            provider = os.environ.get("VIDEO_GEN_PROVIDER", "doubao")
            return ToolExecResult(
                output=(
                    f"Video saved to: {out.resolve()}\n"
                    f"Prompt: {prompt}\nDuration: {duration}s\nRatio: {ratio}\n"
                    f"Provider: {provider}\nModel: {self._model}"
                )
            )

        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            return ToolExecResult(error=f"Video generation HTTP error {e.code}: {detail}", error_code=-1)
        except ToolError as e:
            return ToolExecResult(error=str(e), error_code=-1)
        except Exception as e:  # noqa: BLE001
            return ToolExecResult(error=f"Video generation error: {e}", error_code=-1)
