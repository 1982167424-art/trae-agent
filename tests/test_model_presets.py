# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Regression tests for vendor model presets."""

import pytest

from trae_agent.utils.model_presets import (
    MODEL_PRESETS,
    get_preset,
    list_presets,
    to_model_dict,
    to_provider_dict,
)


def test_list_presets_non_empty():
    presets = list_presets()
    assert len(presets) >= 7
    for p in presets:
        assert p.id and p.name and p.base_url and p.default_model


def test_to_provider_dict_shape():
    d = to_provider_dict("deepseek", api_key="sk-test")
    assert d["provider"] == "deepseek"
    assert d["base_url"] == "https://api.deepseek.com/v1"
    assert d["api_key"] == "sk-test"


def test_to_model_dict_shape():
    d = to_model_dict("deepseek")
    assert d["model"] == "deepseek-chat"
    assert d["model_provider"] == "deepseek"
    assert d["parallel_tool_calls"] is True


def test_to_model_dict_override_model():
    d = to_model_dict("deepseek", model="deepseek-reasoner")
    assert d["model"] == "deepseek-reasoner"


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        to_provider_dict("no-such-vendor")
    with pytest.raises(KeyError):
        to_model_dict("no-such-vendor")
    assert get_preset("no-such-vendor") is None


def test_all_presets_have_env_friendly_provider():
    for pid, p in MODEL_PRESETS.items():
        assert pid == p.id
        assert p.provider.isidentifier() or "_" in p.provider
