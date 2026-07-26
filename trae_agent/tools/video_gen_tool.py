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

import asyncio
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

    def __init__(
        self,
        model_provider: str | None = None,
        working_dir: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model_provider)
        # #10 修复:支持构造函数注入 api_key,避免被迫从 CWD 读 trae_config.yaml。
        self._api_key: str = api_key or ""
        self._model: str = ""
        # #6 修复:沙箱根目录;设置后 output_path 必须落在其内。
        self._working_dir: str | None = working_dir

    def _ensure_config(self):
        if self._api_key:
            self._model = os.environ.get("VIDEO_GEN_MODEL", DEFAULT_MODEL)
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
                "(or doubao.api_key in trae_config.yaml, or pass api_key=...)."
            )
        self._api_key = api_key
        self._model = os.environ.get("VIDEO_GEN_MODEL", DEFAULT_MODEL)

    def _resolve_output_path(self, output_path: str) -> Path:
        """Resolve and sandbox-check the output path.

        #6 修复:绝对路径必须落在 working_dir 之内;相对路径按 CWD 解析。
        working_dir 未设置(本地可信运行)时不限制,但仍解析为绝对路径。
        """
        out = Path(output_path)
        if not out.is_absolute():
            out = Path.cwd() / out
        out = out.resolve()
        if self._working_dir is not None:
            wd = Path(self._working_dir).resolve()
            if out != wd and not out.is_relative_to(wd):
                raise ToolError(
                    f"output_path {out} escapes the allowed working directory "
                    f"{wd}. Refusing to write outside the workspace."
                )
        return out

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
    # These are blocking and are run off the event loop via asyncio.to_thread.
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

    def _get_task(self, task_id: str) -> dict:
        url = f"{ARK_VIDEO_BASE}/{task_id}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def _poll_task(self, task_id: str) -> str:
        # #7 修复:轮询循环改为 async,每次 HTTP GET 用 to_thread 跑在
        # 线程里,等待用 await asyncio.sleep,从而不冻结事件循环
        # (原 time.sleep + 阻塞 urllib 在 async 函数里会卡死 UI / 无法 Ctrl-C)。
        deadline = time.time() + POLL_TIMEOUT
        while time.time() < deadline:
            payload = await asyncio.to_thread(self._get_task, task_id)
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
            await asyncio.sleep(POLL_INTERVAL)
        raise ToolError(f"Timed out waiting for video task {task_id} after {POLL_TIMEOUT}s")

    def _download(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={}, method="GET")
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()

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

            task_id = await asyncio.to_thread(self._post_task, prompt, duration, ratio)
            video_url = await self._poll_task(task_id)

            if output_path:
                out = self._resolve_output_path(output_path)
            else:
                out = generate_output_path(extension=".mp4")

            out.parent.mkdir(parents=True, exist_ok=True)
            data = await asyncio.to_thread(self._download, video_url)
            out.write_bytes(data)

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
