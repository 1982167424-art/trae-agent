# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Image generation tool — supports multiple backends (SiliconFlow, OpenAI, Doubao)."""

import asyncio
import base64
import os
from pathlib import Path

import openai
from typing_extensions import override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter
from trae_agent.utils.output_manager import generate_output_path


class ToolError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ImageGenTool(Tool):
    """Generate images from text prompts.

    Supported backends (set via IMAGE_GEN_PROVIDER env var):
    - siliconflow (default): SiliconFlow 免费图片生成, 支持 Flux/Stable Diffusion
    - openai: OpenAI DALL-E 3
    - doubao: 火山引擎 Doubao Seedream (需要单独开通图片生成 endpoint)

    Usage:
        trae-cli run "画一只猫"
        IMAGE_GEN_PROVIDER=openai trae-cli run "生成风景图"
    """

    def __init__(
        self,
        model_provider: str | None = None,
        working_dir: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model_provider)
        self._client: openai.OpenAI | None = None
        self._model: str = ""
        # #6 修复:沙箱根目录;设置后 output_path 必须落在其内。
        self._working_dir: str | None = working_dir
        # #10 修复:支持构造函数注入 api_key。
        self._injected_api_key: str = api_key or ""

    def _ensure_client(self):
        if self._client is not None:
            return

        provider = os.environ.get("IMAGE_GEN_PROVIDER", "siliconflow").lower()

        if provider == "siliconflow":
            api_key = os.environ.get("SILICONFLOW_API_KEY", "")
            if not api_key:
                raise ToolError(
                    "SiliconFlow requires SILICONFLOW_API_KEY env var. "
                    "Get free key at: https://cloud.siliconflow.cn/"
                )
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.siliconflow.cn/v1",
            )
            self._model = os.environ.get("IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-schnell")

        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ToolError("OpenAI requires OPENAI_API_KEY for DALL-E image generation.")
            self._client = openai.OpenAI(api_key=api_key)
            self._model = os.environ.get("IMAGE_GEN_MODEL", "dall-e-3")

        elif provider == "doubao":
            # #10 修复:优先用构造函数注入的 api_key,否则回退到 env / CWD config。
            api_key = self._injected_api_key or os.environ.get("DOUBAO_API_KEY", "")
            base_url = "https://ark.cn-beijing.volces.com/api/v3"
            if not api_key:
                try:
                    from trae_agent.utils.config import Config
                    config = Config.create(config_file="trae_config.yaml")
                    if config.trae_agent:
                        api_key = config.trae_agent.model.model_provider.api_key
                        base_url = config.trae_agent.model.model_provider.base_url or base_url
                except Exception:
                    pass
            if not api_key:
                raise ToolError(
                    "Doubao requires DOUBAO_API_KEY env var, api_key=... injection, "
                    "or config in trae_config.yaml."
                )
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self._model = os.environ.get("IMAGE_GEN_MODEL", "doubao-seedream-4-0-250828")
        else:
            raise ToolError(f"Unknown IMAGE_GEN_PROVIDER: {provider}. Use: siliconflow, openai, doubao")

    @override
    def get_name(self) -> str:
        return "image_gen"

    @override
    def get_description(self) -> str:
        return """Generate an image from a text description.
The generated image is saved to a file and its path is returned.
Use this tool when the user asks to create, generate, draw, or design an image.
Examples: "画一个登录页面的线框图", "生成一张日落风景图", "create a logo for my startup" """

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="prompt",
                type="string",
                description="Detailed text description of the image to generate.",
                required=True,
            ),
            ToolParameter(
                name="output_path",
                type="string",
                description="File path to save the generated image. Defaults to ./generated_<timestamp>.png",
                required=False,
            ),
            ToolParameter(
                name="size",
                type="string",
                description="Image size. Options: '1024x1024', '1792x1024', '1024x1792'. Default: '1024x1024'",
                required=False,
            ),
        ]

    def _resolve_output_path(self, output_path: str) -> Path:
        """Resolve and sandbox-check the output path (see VideoGenTool)."""
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

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        prompt = str(arguments.get("prompt", ""))
        if not prompt:
            return ToolExecResult(error="prompt is required", error_code=-1)

        output_path = str(arguments.get("output_path", ""))
        size = str(arguments.get("size", "1024x1024"))

        try:
            self._ensure_client()

            # #7 修复:把阻塞的 SDK 调用与下载移出事件循环,
            # 避免 async 函数里长时间冻结 UI / 无法响应中断。
            response = await asyncio.to_thread(
                self._client.images.generate,
                model=self._model,
                prompt=prompt,
                size=size,
                n=1,
            )

            image_data = response.data[0]
            image_url = getattr(image_data, "url", None)
            image_b64 = getattr(image_data, "b64_json", None)

            # Use output manager to organize files by project on Desktop
            if output_path:
                out = self._resolve_output_path(output_path)
            else:
                out = generate_output_path(extension=".png")

            if image_b64:
                out.write_bytes(base64.b64decode(image_b64))
            elif image_url:
                import urllib.request

                await asyncio.to_thread(urllib.request.urlretrieve, image_url, str(out))
            else:
                return ToolExecResult(error="No image data in API response", error_code=-1)

            provider = os.environ.get("IMAGE_GEN_PROVIDER", "siliconflow")
            return ToolExecResult(
                output=f"Image saved to: {out.resolve()}\nPrompt: {prompt}\nSize: {size}\nProvider: {provider}"
            )

        except openai.BadRequestError as e:
            return ToolExecResult(error=f"Image generation failed: {e}", error_code=-1)
        except ToolError as e:
            return ToolExecResult(error=str(e), error_code=-1)
        except Exception as e:
            return ToolExecResult(error=f"Image generation error: {e}", error_code=-1)
