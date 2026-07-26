# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Translation tool — supports the Doubao Seed-Translation backend via Volcengine Ark.

The Ark translation model (doubao-seed-translation-250915) does NOT support the
chat/completions API. It is invoked through the Responses API:

  POST https://ark.cn-beijing.volces.com/api/v3/responses

with a `translation_options` block attached to each input text item. The source
language is auto-detected when omitted.

Usage:
    TRANSLATION_MODEL=ep-20260725205127-jkwmn trae-cli run "把这段翻译成中文"
"""

import json
import os
import urllib.error
import urllib.request

from typing_extensions import override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class ToolError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# Volcengine Ark Responses API base (Beijing).
ARK_RESPONSES_BASE = "https://ark.cn-beijing.volces.com/api/v3/responses"

# Default endpoint: Doubao Seed-Translation (pre-set inference endpoint, call by id).
DEFAULT_MODEL = "ep-20260725205127-jkwmn"

# Common language hints surfaced in the parameter description.
_LANG_HINT = "e.g. 'en', 'zh', 'ja', 'ko', 'fr', 'de', 'ru', 'es', 'pt', 'it', 'th', 'vi', 'ar'"


class TranslationTool(Tool):
    """Translate text between languages using Doubao Seed-Translation via Volcengine Ark."""

    def __init__(self, model_provider: str | None = None, api_key: str | None = None):
        super().__init__(model_provider)
        self._api_key: str = api_key or ""
        self._model: str = ""

    def _ensure_config(self):
        if self._api_key:
            self._model = os.environ.get("TRANSLATION_MODEL", DEFAULT_MODEL)
            return

        # #10 修复:优先用构造函数注入的 api_key,避免被迫从 CWD 读
        # trae_config.yaml(docker 模式 / CWD 无该文件时拿不到 key)。
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
                "Doubao translation requires DOUBAO_API_KEY env var "
                "(or doubao.api_key in trae_config.yaml, or pass api_key=...)."
            )
        self._api_key = api_key
        self._model = os.environ.get("TRANSLATION_MODEL", DEFAULT_MODEL)
        if not api_key:
            raise ToolError(
                "Doubao translation requires DOUBAO_API_KEY env var "
                "(or doubao.api_key in trae_config.yaml)."
            )
        self._api_key = api_key
        self._model = os.environ.get("TRANSLATION_MODEL", DEFAULT_MODEL)

    @override  # type: ignore[misc]
    def get_name(self) -> str:
        return "translation"

    @override  # type: ignore[misc]
    def get_description(self) -> str:
        return """Translate text from one language to another using a dedicated translation model.
Use this tool when the user asks to translate, 翻译, 翻訳, 번역, or convert text between languages.
Provide the source text, an optional source language (auto-detected if omitted),
and a target language. Returns the translated text.
Examples: "把 'Hello world' 翻译成中文", "translate this paragraph to Japanese",
"将以下内容翻译为英文: ..." """

    @override  # type: ignore[misc]
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="string",
                description="The text to translate.",
                required=True,
            ),
            ToolParameter(
                name="target_language",
                type="string",
                description=f"Target language code. {_LANG_HINT}. Default: 'en'.",
                required=False,
            ),
            ToolParameter(
                name="source_language",
                type="string",
                description=f"Source language code. Omit to auto-detect. {_LANG_HINT}.",
                required=False,
            ),
        ]

    def _translate(self, text: str, source: str, target: str) -> str:
        translation_options: dict[str, str] = {"target_language": target}
        if source:
            translation_options["source_language"] = source

        body = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text,
                            "translation_options": translation_options,
                        }
                    ],
                }
            ],
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            ARK_RESPONSES_BASE,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        outputs = payload.get("output") or []
        for out in outputs:
            if out.get("type") != "message":
                continue
            for item in out.get("content") or []:
                if item.get("type") == "output_text":
                    translated = item.get("text")
                    if translated:
                        return translated
        raise ToolError(f"No translated text in response: {payload}")

    @override  # type: ignore[misc]
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        text = str(arguments.get("text", ""))
        if not text:
            return ToolExecResult(error="text is required", error_code=-1)
        target = str(arguments.get("target_language", "en")) or "en"
        source = str(arguments.get("source_language", ""))

        try:
            self._ensure_config()
            translated = self._translate(text, source, target)
            return ToolExecResult(
                output=(
                    f"Translation (->{target}):\n{translated}\n"
                    f"Model: {self._model}"
                )
            )
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            return ToolExecResult(error=f"Translation HTTP error {e.code}: {detail}", error_code=-1)
        except ToolError as e:
            return ToolExecResult(error=str(e), error_code=-1)
        except Exception as e:  # noqa: BLE001
            return ToolExecResult(error=f"Translation error: {e}", error_code=-1)
