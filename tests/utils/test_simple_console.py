# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Interaction-consistency tests for the simple CLI console."""

import os
import unittest
from unittest.mock import patch

from trae_agent.utils.cli.cli_console import ConsoleMode
from trae_agent.utils.cli.simple_console import SimpleCLIConsole


class TestSimpleConsoleWorkingDir(unittest.TestCase):
    def _console(self) -> SimpleCLIConsole:
        return SimpleCLIConsole(mode=ConsoleMode.INTERACTIVE)

    def test_empty_input_returns_cwd(self):
        c = self._console()
        with patch("builtins.input", return_value=""):
            self.assertEqual(c.get_working_dir_input(), os.getcwd())

    def test_explicit_path_returned(self):
        c = self._console()
        with patch("builtins.input", return_value="/tmp/foo"):
            self.assertEqual(c.get_working_dir_input(), "/tmp/foo")

    def test_interrupt_returns_cwd(self):
        c = self._console()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            self.assertEqual(c.get_working_dir_input(), os.getcwd())

    def test_non_interactive_returns_empty(self):
        c = SimpleCLIConsole(mode=ConsoleMode.RUN)
        self.assertEqual(c.get_working_dir_input(), "")
