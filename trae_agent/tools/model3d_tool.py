# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""3D model generation tool — supports the Hyper3D backend via Volcengine Ark.

The Ark 3D generation API is asynchronous (same shape as Seedance video):
  1. POST  /api/v3/contents/generations/tasks   -> returns a task id
  2. GET   /api/v3/contents/generations/tasks/{id}  (poll) -> status + file_url
  3. download the resulting file_url (.glb) to output_path

Usage:
    MODEL3D_PROVIDER=doubao trae-cli run "用 3D 模型生成一个橘猫"
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


# Volcengine Ark async generation base (Beijing).
ARK_3D_BASE = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"

# Default model: Hyper3D (pre-set inference endpoint, call by endpoint id).
DEFAULT_MODEL = "ep-20260725205234-k8bkc"

# How long to keep polling for the async task before giving up (seconds).
POLL_TIMEOUT = 600
POLL_INTERVAL = 10


class Model3DTool(Tool):
    """Generate 3D models from text prompts using Hyper3D via Volcengine Ark."""

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)
        self._api_key: str = ""
        self._model: str = ""

    def _ensure_config(self):
        if self._api_key:
            return

        provider = os.environ.get("MODEL3D_PROVIDER", "doubao").lower()
        if provider != "doubao":
            raise ToolError(
                f"Unknown MODEL3D_PROVIDER: {provider}. Only 'doubao' (Volcengine Ark Hyper3D) is supported."
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
                "3D generation requires DOUBAO_API_KEY env var "
                "(or doubao.api_key in trae_config.yaml)."
            )
        self._api_key = api_key
        self._model = os.environ.get("MODEL3D_MODEL", DEFAULT_MODEL)

    @override  # type: ignore[misc]
    def get_name(self) -> str:
        return "model3d"

    @override  # type: ignore[misc]
    def get_description(self) -> str:
        return """Generate a 3D model (.glb) from a text description using a text-to-3D model (Hyper3D).
The generated 3D model file is saved and its path is returned.
Use this tool when the user asks to create, generate, make, or produce a 3D model / 3D asset / 3D object.
Examples: "生成一个橘猫的3D模型", "create a 3D model of a small chair", "用 3D 模型做一个杯子" """

    @override  # type: ignore[misc]
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="prompt",
                type="string",
                description="Detailed text description of the 3D object to generate.",
                required=True,
            ),
            ToolParameter(
                name="output_path",
                type="string",
                description="File path to save the generated 3D model (.glb). Defaults to ./generated_<timestamp>.glb",
                required=False,
            ),
        ]

    # ---- internal HTTP helpers (urllib, no extra deps) ----

    def _post_task(self, prompt: str) -> str:
        body = {
            "model": self._model,
            "content": [{"type": "text", "text": prompt}],
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            ARK_3D_BASE,
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
        url = f"{ARK_3D_BASE}/{task_id}"
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
                file_url = content.get("file_url") or content.get("url")
                if not file_url:
                    raise ToolError(f"Task succeeded but no file_url: {payload}")
                return file_url
            if status == "failed":
                err = payload.get("error")
                raise ToolError(f"3D generation failed: {err}")
            if status in ("expired",):
                raise ToolError(f"3D generation expired: {payload}")
            # queued / running -> wait and retry
            time.sleep(POLL_INTERVAL)
        raise ToolError(f"Timed out waiting for 3D task {task_id} after {POLL_TIMEOUT}s")

    @override  # type: ignore[misc]
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        prompt = str(arguments.get("prompt", ""))
        if not prompt:
            return ToolExecResult(error="prompt is required", error_code=-1)

        output_path = str(arguments.get("output_path", ""))

        try:
            self._ensure_config()

            task_id = self._post_task(prompt)
            file_url = self._poll_task(task_id)

            if output_path:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
            else:
                out = generate_output_path(extension=".glb")

            # download the 3D model file
            req = urllib.request.Request(file_url, headers={}, method="GET")
            with urllib.request.urlopen(req, timeout=300) as resp:
                out.write_bytes(resp.read())

            provider = os.environ.get("MODEL3D_PROVIDER", "doubao")
            return ToolExecResult(
                output=(
                    f"3D model saved to: {out.resolve()}\n"
                    f"Prompt: {prompt}\n"
                    f"Provider: {provider}\nModel: {self._model}"
                )
            )

        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            return ToolExecResult(error=f"3D generation HTTP error {e.code}: {detail}", error_code=-1)
        except ToolError as e:
            return ToolExecResult(error=str(e), error_code=-1)
        except Exception as e:  # noqa: BLE001
            return ToolExecResult(error=f"3D generation error: {e}", error_code=-1)
