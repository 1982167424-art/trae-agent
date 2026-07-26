# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Unit tests for the TranslationTool (Doubao Seed-Translation backend)."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from trae_agent.tools.translation_tool import TranslationTool


def _make_fake_resp(payload: dict) -> MagicMock:
    fake = MagicMock()
    fake.read.return_value = json.dumps(payload).encode("utf-8")
    fake.__enter__.return_value = fake
    fake.__exit__.return_value = False
    return fake


_HAPPY_PAYLOAD = {
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello, world!"}],
        }
    ],
    "status": "completed",
}


class TestTranslationToolBasics(unittest.TestCase):
    def test_get_name(self):
        self.assertEqual(TranslationTool().get_name(), "translation")

    def test_get_description(self):
        desc = TranslationTool().get_description().lower()
        self.assertIn("translate", desc)

    def test_parameters(self):
        names = {p.name for p in TranslationTool().get_parameters()}
        self.assertEqual(names, {"text", "target_language", "source_language"})


class TestTranslationToolExecute(unittest.IsolatedAsyncioTestCase):
    async def test_missing_text(self):
        res = await TranslationTool().execute({"text": ""})
        self.assertIsNotNone(res.error)
        self.assertEqual(res.error_code, -1)

    async def test_happy_path(self):
        tool = TranslationTool()
        tool._api_key = "test-key"
        tool._model = "ep-20260725205127-jkwmn"

        with patch(
            "trae_agent.tools.translation_tool.urllib.request.urlopen",
            return_value=_make_fake_resp(_HAPPY_PAYLOAD),
        ):
            res = await tool.execute(
                {"text": "你好，世界", "target_language": "en"}
            )
        self.assertIsNone(res.error, msg=res.error)
        self.assertIn("Hello, world!", res.output)
        self.assertIn("ep-20260725205127-jkwmn", res.output)

    async def test_source_omitted_autodetect(self):
        tool = TranslationTool()
        tool._api_key = "test-key"
        tool._model = "ep-20260725205127-jkwmn"

        captured: dict = {}

        def _fake_urlopen(req, timeout=None):
            captured["data"] = req.data
            return _make_fake_resp(_HAPPY_PAYLOAD)

        with patch(
            "trae_agent.tools.translation_tool.urllib.request.urlopen",
            side_effect=_fake_urlopen,
        ):
            res = await tool.execute({"text": "hi", "target_language": "ja"})
        self.assertIsNone(res.error, msg=res.error)

        data = json.loads(captured["data"])
        opts = data["input"][0]["content"][0]["translation_options"]
        self.assertNotIn("source_language", opts)
        self.assertEqual(opts["target_language"], "ja")

    async def test_source_included_when_provided(self):
        tool = TranslationTool()
        tool._api_key = "test-key"
        tool._model = "ep-20260725205127-jkwmn"

        captured: dict = {}

        def _fake_urlopen(req, timeout=None):
            captured["data"] = req.data
            return _make_fake_resp(_HAPPY_PAYLOAD)

        with patch(
            "trae_agent.tools.translation_tool.urllib.request.urlopen",
            side_effect=_fake_urlopen,
        ):
            res = await tool.execute(
                {"text": "hi", "target_language": "en", "source_language": "fr"}
            )
        self.assertIsNone(res.error, msg=res.error)

        data = json.loads(captured["data"])
        opts = data["input"][0]["content"][0]["translation_options"]
        self.assertEqual(opts["source_language"], "fr")
        self.assertEqual(opts["target_language"], "en")

    async def test_http_error(self):
        tool = TranslationTool()
        tool._api_key = "test-key"
        tool._model = "ep-20260725205127-jkwmn"

        def _fake_urlopen_err(req, timeout=None):
            raise HTTPError(
                url="x",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"invalid model"}'),
            )

        with patch(
            "trae_agent.tools.translation_tool.urllib.request.urlopen",
            side_effect=_fake_urlopen_err,
        ):
            res = await tool.execute({"text": "hi", "target_language": "en"})
        self.assertIsNotNone(res.error)
        self.assertIn("HTTP error 400", res.error)

    async def test_no_translated_text(self):
        tool = TranslationTool()
        tool._api_key = "test-key"
        tool._model = "ep-20260725205127-jkwmn"

        empty_payload = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                }
            ]
        }
        with patch(
            "trae_agent.tools.translation_tool.urllib.request.urlopen",
            return_value=_make_fake_resp(empty_payload),
        ):
            res = await tool.execute({"text": "hi", "target_language": "en"})
        self.assertIsNotNone(res.error)
        self.assertIn("No translated text", res.error)
