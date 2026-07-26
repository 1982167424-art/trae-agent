# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import tempfile
from unittest.mock import MagicMock, patch

from trae_agent.tools.model3d_tool import Model3DTool, ToolError


class TestModel3DToolMeta:
    def test_name(self):
        assert Model3DTool().get_name() == "model3d"

    def test_parameters(self):
        params = {p.name: p for p in Model3DTool().get_parameters()}
        assert "prompt" in params
        assert params["prompt"].required is True


class TestModel3DToolExecute:
    async def test_happy_path(self):
        tool = Model3DTool()
        tool._api_key = "test-key"
        tool._model = "ep-20260725205234-k8bkc"

        fake_glb = b"glTF\x02\x00\x00\x00"
        fake_resp = MagicMock()
        fake_resp.read.return_value = fake_glb
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.glb")
            with patch.object(
                tool, "_post_task", return_value="task-xyz"
            ), patch.object(
                tool, "_poll_task", return_value="http://x/m.glb"
            ), patch(
                "trae_agent.tools.model3d_tool.urllib.request.urlopen",
                return_value=fake_resp,
            ):
                res = await tool.execute(
                    {"prompt": "a small chair", "output_path": out_path}
                )

            assert res.error is None, res.error
            assert os.path.exists(out_path)
            assert out_path in res.output

    async def test_missing_prompt(self):
        tool = Model3DTool()
        res = await tool.execute({})
        assert res.error is not None

    async def test_poll_failed(self):
        tool = Model3DTool()
        tool._api_key = "test-key"
        tool._model = "m"
        with patch.object(
            tool, "_post_task", return_value="task-1"
        ), patch.object(
            tool, "_poll_task", side_effect=ToolError("boom")
        ):
            res = await tool.execute({"prompt": "x"})
        assert res.error is not None
        assert "boom" in res.error
