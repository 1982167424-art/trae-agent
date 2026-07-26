# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for the ImageGenTool (SiliconFlow / OpenAI / Doubao backends)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from trae_agent.tools.image_gen_tool import ImageGenTool


class TestImageGenToolBasics(unittest.TestCase):
    def test_get_name(self):
        self.assertEqual(ImageGenTool().get_name(), "image_gen")

    def test_parameters(self):
        names = {p.name for p in ImageGenTool().get_parameters()}
        self.assertEqual(names, {"prompt", "output_path", "size"})


class TestImageGenToolConfig(unittest.TestCase):
    def test_doubao_default_model_is_user_provisioned(self):
        # The user enabled doubao-seedream-4-0-250828; the tool must default to it
        # (not the old 3.0 name that 404s).
        with patch.dict(
            os.environ,
            {"DOUBAO_API_KEY": "k", "IMAGE_GEN_PROVIDER": "doubao"},
            clear=True,
        ), patch("trae_agent.tools.image_gen_tool.openai.OpenAI") as MockOpenAI:
            tool = ImageGenTool()
            tool._ensure_client()
            self.assertEqual(tool._model, "doubao-seedream-4-0-250828")
            MockOpenAI.assert_called_once()


class TestImageGenToolExecute(unittest.IsolatedAsyncioTestCase):
    async def test_missing_prompt(self):
        res = await ImageGenTool().execute({"prompt": ""})
        self.assertIsNotNone(res.error)
        self.assertEqual(res.error_code, -1)

    async def test_happy_path_url(self):
        tool = ImageGenTool()
        tool._client = MagicMock()
        tool._model = "doubao-seedream-4-0-250828"

        fake_png = b"\x89PNG\r\n\x1a\n"
        data = MagicMock()
        data.url = "http://x/i.png"
        data.b64_json = None
        resp = MagicMock()
        resp.data = [data]
        tool._client.images.generate.return_value = resp

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "img.png")
            with patch("urllib.request.urlretrieve") as mocked:
                mocked.side_effect = lambda url, path: open(path, "wb").write(fake_png)
                res = await tool.execute({"prompt": "a cat", "output_path": out})
            self.assertIsNone(res.error, msg=res.error)
            self.assertTrue(os.path.exists(out))
            self.assertIn("Image saved to:", res.output)

    async def test_happy_path_b64(self):
        tool = ImageGenTool()
        tool._client = MagicMock()
        tool._model = "m"

        import base64

        payload = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        data = MagicMock()
        data.url = None
        data.b64_json = payload
        resp = MagicMock()
        resp.data = [data]
        tool._client.images.generate.return_value = resp

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "img2.png")
            res = await tool.execute({"prompt": "a cat", "output_path": out})
            self.assertIsNone(res.error, msg=res.error)
            self.assertTrue(os.path.exists(out))
