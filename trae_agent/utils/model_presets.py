# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Vendor model presets for multi-provider support.

Referenced from community coding-agent repos (opencode, MiMo-Code, kimi-code,
qwen-code, grok-build, CodeGeeX4, Seed-Coder, DeepSeek-R1, Hy3): every one
exposes "connect to any OpenAI-compatible endpoint + multi-vendor". This module
provides ready-made presets so users can switch providers with one config entry
instead of hand-writing base_url / api_key / env plumbing.

The returned dicts are compatible with `trae_agent.utils.config.ModelProvider`
and `ModelConfig` field names, so they can be merged into a YAML config.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelPreset:
    id: str
    name: str
    provider: str  # used as ModelProvider.provider -> env <PROVIDER>_API_KEY
    base_url: str
    default_model: str
    # Recommended sampling params (vendor best-practice)
    temperature: float = 0.7
    top_p: float = 1.0
    # How the model is hosted / obtained
    host: str = "cloud"  # cloud | local
    reasoning: bool = False  # supports chain-of-thought / reasoning_effort
    note: str = ""


# fmt: off
MODEL_PRESETS: dict[str, ModelPreset] = {
    "deepseek": ModelPreset(
        id="deepseek", name="DeepSeek", provider="deepseek",
        base_url="https://api.deepseek.com/v1", default_model="deepseek-chat",
        temperature=0.6, top_p=1.0, reasoning=True,
        note="推理用 deepseek-reasoner；R1 建议 temperature=0.6、避免系统提示",
    ),
    "qwen": ModelPreset(
        id="qwen", name="阿里通义千问 Qwen", provider="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus", temperature=0.7, top_p=0.8,
        note="Qwen-Agent / qwen-code 同源；长上下文与工具调用稳定",
    ),
    "kimi": ModelPreset(
        id="kimi", name="Kimi (Moonshot)", provider="moonshot",
        base_url="https://api.moonshot.cn/v1", default_model="moonshot-v1-8k",
        temperature=0.7, top_p=0.9, reasoning=True,
        note="kimi-code 使用的 Kimi 模型；支持长上下文",
    ),
    "hy3": ModelPreset(
        id="hy3", name="腾讯混元 Hy3", provider="hy3",
        base_url="http://localhost:8000/v1", default_model="hy3",
        temperature=0.9, top_p=1.0, host="local", reasoning=True,
        note="需本地 vLLM/SGLang 部署(295B MoE)；reasoning_effort=no_think/low/high",
    ),
    "seed_coder": ModelPreset(
        id="seed_coder", name="字节 Seed-Coder", provider="seed_coder",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="ep-Seed-Coder", temperature=0.6, top_p=0.8, reasoning=True,
        note="ByteDance Seed-Coder 8B 家族(Base/Instruct/Reasoning)；火山方舟 endpoint",
    ),
    "codegeex": ModelPreset(
        id="codegeex", name="智谱 CodeGeeX4", provider="codegeex",
        base_url="http://localhost:8000/v1", default_model="codegeex4-all-9b",
        temperature=0.95, top_p=0.9, host="local",
        note="本地 Ollama/vLLM 部署 9B 全场景代码模型, 支持 Function Call",
    ),
    "grok": ModelPreset(
        id="grok", name="xAI Grok", provider="grok",
        base_url="https://api.x.ai/v1", default_model="grok-4",
        temperature=0.7, top_p=0.9, reasoning=True,
        note="grok-build 使用的 Grok 模型",
    ),
    "doubao": ModelPreset(
        id="doubao", name="豆包 / 火山方舟", provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="your-doubao-endpoint", temperature=0.7, top_p=0.9,
        note="已接入的豆包 endpoint(ep-...)；视频生成也用此 Key",
    ),
}
# fmt: on


def list_presets() -> list[ModelPreset]:
    """Return all available vendor presets."""
    return list(MODEL_PRESETS.values())


def get_preset(preset_id: str) -> Optional[ModelPreset]:
    """Return a single preset by id, or None if unknown."""
    return MODEL_PRESETS.get(preset_id)


def to_provider_dict(preset_id: str, *, api_key: str = "") -> dict:
    """Return a ModelProvider-compatible dict (api_key, provider, base_url)."""
    preset = MODEL_PRESETS.get(preset_id)
    if preset is None:
        raise KeyError(f"Unknown preset: {preset_id!r}")
    return {
        "api_key": api_key,
        "provider": preset.provider,
        "base_url": preset.base_url,
    }


def to_model_dict(preset_id: str, *, model: Optional[str] = None) -> dict:
    """Return a ModelConfig-compatible dict for the preset's default model."""
    preset = MODEL_PRESETS.get(preset_id)
    if preset is None:
        raise KeyError(f"Unknown preset: {preset_id!r}")
    return {
        "model": model or preset.default_model,
        "model_provider": preset.provider,
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": 0,
        "parallel_tool_calls": True,
        "max_retries": 3,
    }
