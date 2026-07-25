# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import unittest

from trae_agent.utils.config import resolve_config_value


class TestResolveConfigValue(unittest.TestCase):
    """Regression tests for bug #6: empty-string CLI value must not win.

    resolve_config_value resolves with priority CLI > ENV > Config.
    An explicit empty string from the CLI (e.g. ``--api-key ""``) should be
    treated as "not provided" and fall back to env / config instead of
    returning the empty string.
    """

    def test_cli_value_takes_priority(self):
        self.assertEqual(
            resolve_config_value(cli_value="cli", config_value="cfg", env_var="NOPE_X"),
            "cli",
        )

    def test_empty_string_falls_back_to_env(self):
        os.environ["TEST_RESOLVE_ENV"] = "from_env"
        try:
            self.assertEqual(
                resolve_config_value(
                    cli_value="", config_value="cfg", env_var="TEST_RESOLVE_ENV"
                ),
                "from_env",
            )
        finally:
            del os.environ["TEST_RESOLVE_ENV"]

    def test_empty_string_falls_back_to_config(self):
        self.assertEqual(
            resolve_config_value(cli_value="", config_value="cfg", env_var="NOPE_X"),
            "cfg",
        )

    def test_none_cli_falls_back_to_config(self):
        self.assertEqual(
            resolve_config_value(cli_value=None, config_value="cfg", env_var="NOPE_X"),
            "cfg",
        )

    def test_falsy_but_valid_cli_value_is_kept(self):
        # 0 / False must NOT be treated as "not provided".
        self.assertEqual(
            resolve_config_value(cli_value=0, config_value=5, env_var="NOPE_X"),
            0,
        )
