# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import tempfile
import unittest
from pathlib import Path

from trae_agent.utils.llm_clients.llm_basics import LLMMessage
from trae_agent.utils.trajectory_loader import load_trajectory
from trae_agent.utils.trajectory_recorder import TrajectoryRecorder


class TestTrajectorySerialization(unittest.TestCase):
    """Regression tests for bug #12 (empty llm_messages -> []) and bug #7
    (must_patch persisted and recovered by `trae resume`)."""

    def test_empty_llm_messages_serialized_as_empty_list(self):
        rec = TrajectoryRecorder(trajectory_path=Path(tempfile.mktemp(suffix=".json")))
        rec.record_agent_step(step_number=1, state="test", llm_messages=[])
        step = rec.trajectory_data["agent_steps"][-1]
        # Bug #12: an explicit empty list must NOT become JSON null.
        self.assertEqual(step["llm_messages"], [])

    def test_non_empty_llm_messages_serialized(self):
        rec = TrajectoryRecorder(trajectory_path=Path(tempfile.mktemp(suffix=".json")))
        msgs = [LLMMessage(role="user", content="hello")]
        rec.record_agent_step(step_number=1, state="test", llm_messages=msgs)
        step = rec.trajectory_data["agent_steps"][-1]
        self.assertEqual(len(step["llm_messages"]), 1)
        self.assertEqual(step["llm_messages"][0]["content"], "hello")

    def test_must_patch_persisted_and_loaded(self):
        f = Path(tempfile.mktemp(suffix=".json"))
        rec = TrajectoryRecorder(trajectory_path=f)
        rec.trajectory_data["must_patch"] = "true"
        # Record a complete step so the saved trajectory is resumable.
        rec.record_agent_step(
            step_number=1,
            state="test",
            llm_messages=[LLMMessage(role="user", content="hi")],
        )
        rec.save_trajectory()
        loaded = load_trajectory(f)
        # Bug #7: original --must-patch state must survive a resume.
        self.assertEqual(loaded.must_patch, "true")
        f.unlink(missing_ok=True)
