# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tests for the auto model router."""

from types import SimpleNamespace

import pytest

from trae_agent.utils.config import ModelRoutingConfig
from trae_agent.utils.model_router import ModelRouter


def _routing():
    return ModelRoutingConfig(
        vision="trae_agent_model",
        strong="trae_agent_model",
        fast="qwen2_5_coder_7b",
        tiny="qwen2_5_coder_0_5b",
    )


def _mc(*, multimodal: bool):
    return SimpleNamespace(supports_multimodal=multimodal)


def test_select_initial_image_uses_vision():
    r = ModelRouter(_routing(), {})
    assert r.select_initial("随便说点什么", has_image=True) == "trae_agent_model"


def test_select_initial_strong_keywords():
    r = ModelRouter(_routing(), {})
    # 架构/分析/复杂/推理 → strong (doubao)
    for task in ["分析这个模块的架构", "重构这段复杂逻辑", "排查 root cause"]:
        assert r.select_initial(task) == "trae_agent_model", task


def test_select_initial_fast_keywords():
    r = ModelRouter(_routing(), {})
    # 写/改/加/修复 → fast (本地 qwen)
    for task in ["写个快速脚本", "修复 bug", "给函数加注释"]:
        assert r.select_initial(task) == "qwen2_5_coder_7b", task


def test_select_initial_ambiguous_defaults_strong():
    r = ModelRouter(_routing(), {})
    # 模糊任务 → 质量优先 (strong)
    assert r.select_initial("帮我处理一下项目") == "trae_agent_model"


def test_select_initial_tiny_fallback():
    # tiny 未配置时应回退到 fast
    r = ModelRouter(
        ModelRoutingConfig(vision="v", strong="s", fast="f", tiny=None), {}
    )
    assert r.select_initial("重命名这个变量", has_image=False) == "f"


def test_constraints_switch_to_vision_when_image_and_not_multimodal():
    r = ModelRouter(_routing(), {})
    current = _mc(multimodal=False)
    assert r.select_for_constraints(True, current) == "trae_agent_model"


def test_constraints_no_switch_when_no_image():
    r = ModelRouter(_routing(), {})
    current = _mc(multimodal=False)
    assert r.select_for_constraints(False, current) is None


def test_constraints_no_switch_when_multimodal_already():
    r = ModelRouter(_routing(), {})
    current = _mc(multimodal=True)
    # 当前模型已支持多模态,即使有图也不切(避免频繁换)
    assert r.select_for_constraints(True, current) is None


def test_sticky_does_not_thrash_on_soft_signals():
    """Per-step only switches on hard constraints, never on phrasing."""
    r = ModelRouter(_routing(), {})
    # 当前是 fast(本地,非多模态),但本步没有图片 → 不切
    current = _mc(multimodal=False)
    assert r.select_for_constraints(False, current) is None


def _mini_model(*, provider: str, model: str, multimodal: bool = False):
    from trae_agent.utils.config import ModelConfig, ModelProvider

    return ModelConfig(
        model=model,
        model_provider=ModelProvider(api_key="test", provider=provider),
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        parallel_tool_calls=False,
        max_retries=1,
        supports_multimodal=multimodal,
    )


def _mini_routing_config():
    from trae_agent.utils.config import TraeAgentConfig

    strong = _mini_model(provider="doubao", model="ep-strong", multimodal=True)
    fast = _mini_model(provider="ollama", model="qwen2.5-coder:7b")
    routing = ModelRoutingConfig(vision="strong", strong="strong", fast="fast", tiny=None)
    return TraeAgentConfig(
        model=strong,
        tools=["bash"],
        max_steps=5,
        model_routing=routing,
        all_models={"strong": strong, "fast": fast},
        allow_mcp_servers=None,
        mcp_servers_config={},
    )


def test_routing_defers_client_and_accepts_recorder():
    """Regression: set_trajectory_recorder must not crash in routing mode
    where the LLM client is created lazily (was None -> AttributeError)."""
    from trae_agent.agent.trae_agent import TraeAgent
    from trae_agent.utils.trajectory_recorder import TrajectoryRecorder

    agent = TraeAgent(trae_agent_config=_mini_routing_config())
    # routing defers client creation
    assert agent._llm_client is None
    assert agent._model_config is None
    # this used to raise AttributeError ('NoneType' has no attribute ...)
    rec = TrajectoryRecorder(trajectory_path="/tmp/_routing_rec_smoke.json")
    agent.set_trajectory_recorder(rec)
    assert agent._trajectory_recorder is rec
    # resolving a model builds the client and binds the recorder
    agent.new_task("写个快速脚本", extra_args={"project_path": "/tmp"})
    assert agent._llm_client is not None
