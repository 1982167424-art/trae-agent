# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for the VideoGenTool (Doubao Seedance backend)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from trae_agent.tools.video_gen_tool import VideoGenTool


class TestVideoGenToolBasics(unittest.TestCase):
    def test_get_name(self):
        self.assertEqual(VideoGenTool().get_name(), "video_gen")

    def test_get_description(self):
        desc = VideoGenTool().get_description().lower()
        self.assertIn("video", desc)

    def test_parameters(self):
        names = {p.name for p in VideoGenTool().get_parameters()}
        self.assertEqual(names, {"prompt", "output_path", "duration", "ratio"})


class TestVideoGenToolExecute(unittest.IsolatedAsyncioTestCase):
    async def test_missing_prompt(self):
        res = await VideoGenTool().execute({"prompt": ""})
        self.assertIsNotNone(res.error)
        self.assertEqual(res.error_code, -1)

    async def test_happy_path(self):
        tool = VideoGenTool()
        tool._api_key = "test-key"
        tool._model = "doubao-seedance-1-0-pro-fast-251015"

        fake_mp4 = b"\x00\x00\x00\x18ftypmp42"
        fake_resp = MagicMock()
        fake_resp.read.return_value = fake_mp4
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.mp4")
            with patch.object(tool, "_post_task", return_value="task-abc"), patch.object(
                tool, "_poll_task", return_value="http://x/v.mp4"
            ), patch(
                "trae_agent.tools.video_gen_tool.urllib.request.urlopen",
                return_value=fake_resp,
            ):
                res = await tool.execute(
                    {"prompt": "a cat running", "output_path": out_path}
                )
            self.assertIsNone(res.error, msg=res.error)
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("Video saved to:", res.output)
            self.assertIn(out_path, res.output)

    async def test_unknown_provider(self):
        with patch.dict(os.environ, {"VIDEO_GEN_PROVIDER": "foo"}, clear=True):
            res = await VideoGenTool().execute({"prompt": "a cat"})
        self.assertIsNotNone(res.error)
        self.assertIn("Unknown VIDEO_GEN_PROVIDER", res.error)

    async def test_poll_failure_propagates(self):
        tool = VideoGenTool()
        tool._api_key = "test-key"
        tool._model = "m"
        with patch.object(tool, "_post_task", return_value="task-abc"), patch.object(
            tool, "_poll_task", side_effect=RuntimeError("boom")
        ):
            res = await tool.execute({"prompt": "a cat", "output_path": "/tmp/x.mp4"})
        self.assertIsNotNone(res.error)
        self.assertIn("boom", res.error)
