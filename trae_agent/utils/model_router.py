# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Auto model routing for trae-agent.

Picks a model per task (and re-checks hard constraints per step) from a
tiered pool, balancing quality (strong/cloud) against speed/cost (fast/
local). Routing is opt-in via the ``model_routing`` config block.

Design notes
-------------
* The router only ever returns a **model name** (a key into the
  ``models`` config). The agent resolves that name to a concrete
  :class:`~trae_agent.utils.config.ModelConfig` and (re)builds its
  LLM client.
* Per-step switching is intentionally *sticky*: :meth:`ModelRouter.select_for_constraints`
  returns a model **only** when a hard constraint is violated (e.g. an
  image is present but the current model can't handle multimodal). Soft
  signals (task phrasing) are ignored at step level so the model is not
  thrashing every step.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trae_agent.utils.config import ModelConfig, ModelRoutingConfig

# --- keyword heuristics (lowercase substring match on the task text) ---

# Tasks that should drop to the tiniest/fastest local model.
_TINY_KEYWORDS = (
    "重命名",
    "改名",
    "格式化",
    "重排",
    "rename",
    "format",
    "trivial",
    "simple fix",
)

# Tasks that need the strong / high-quality model (reasoning, design,
# debugging root-cause, architecture).
_STRONG_KEYWORDS = (
    "架构",
    "设计",
    "分析",
    "复杂",
    "推理",
    "重构",
    "优化",
    "为什么",
    "原理",
    "如何实现",
    "根因",
    "排查",
    "architecture",
    "analyze",
    "complex",
    "refactor",
    "optimize",
    "why",
    "how does",
    "reason",
    "debug",
    "root cause",
)

# Mechanical coding tasks that are fine on a fast local coder.
_FAST_KEYWORDS = (
    "写",
    "改",
    "加",
    "修复",
    "注释",
    "实现",
    "创建",
    "新增",
    "更新",
    "修复bug",
    "write",
    "add",
    "fix",
    "implement",
    "create",
    "edit",
    "update",
)


class ModelRouter:
    """Selects a model name from a tiered pool.

    Args:
        routing: parsed ``model_routing`` config (tier → model name).
        models: full ``models`` dict (name → ModelConfig), used only for
            validation/lookups if needed.
    """

    def __init__(
        self,
        routing: "ModelRoutingConfig",
        models: "dict[str, ModelConfig] | None" = None,
    ):
        self.routing = routing
        self.models = models or {}

    def select_initial(self, task: str, *, has_image: bool = False) -> str:
        """Pick the model name for a new task.

        Images always win (→ vision tier). Otherwise classify the task text
        into a tier and return that tier's model name, falling back down the
        chain (fast → strong) if a tier is unset.
        """
        if has_image:
            return self.routing.vision
        tier = self._classify(task)
        name = getattr(self.routing, tier, None)
        if name:
            return name
        # Fallbacks if a tier was left empty.
        if tier == "tiny" and self.routing.fast:
            return self.routing.fast
        return self.routing.strong

    def select_for_constraints(
        self, has_image: bool, current: "ModelConfig | None"
    ) -> "str | None":
        """Per-step sticky check.

        Returns a model name **only** when a hard constraint is violated:
        an image is present but the current model is not multimodal-capable.
        Otherwise returns ``None`` (keep the current model — no thrashing).
        """
        if has_image and current is not None and not current.supports_multimodal:
            return self.routing.vision
        return None

    def _classify(self, task: str) -> str:
        """Return one of ``"tiny"`` / ``"strong"`` / ``"fast"``.

        Order matters: trivial keywords → tiny, otherwise strong beats fast
        (quality-first), fast only when clearly a mechanical coding task,
        and an ambiguous task defaults to strong (quality safe side).
        """
        t = (task or "").lower()
        if any(k.lower() in t for k in _TINY_KEYWORDS):
            return "tiny"
        if any(k.lower() in t for k in _STRONG_KEYWORDS):
            return "strong"
        if any(k.lower() in t for k in _FAST_KEYWORDS):
            return "fast"
        return "strong"
