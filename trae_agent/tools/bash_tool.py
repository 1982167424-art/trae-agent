# Copyright (c) 2023 Anthropic
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 13 June 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/anthropics/anthropic-quickstarts/blob/main/LICENSE
#
# This modified file is released under the same license.

import asyncio
import os
import re
from typing_extensions import override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolError, ToolExecResult, ToolParameter


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _timed_out: bool

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = ",,,,bash-command-exit-__ERROR_CODE__-banner,,,,"  # `__ERROR_CODE__` will be replaced by `$?` or `!errorlevel!` later

    def __init__(self) -> None:
        self._started = False
        self._timed_out = False
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._started:
            return

        # Windows compatibility: os.setsid not available

        if os.name != "nt":  # Unix-like systems
            self._process = await asyncio.create_subprocess_shell(
                self.command,
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        else:
            self._process = await asyncio.create_subprocess_shell(
                "cmd.exe /v:on",  # enable delayed expansion to allow `echo !errorlevel!`
                shell=True,
                bufsize=0,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        self._started = True

    async def stop(self) -> None:
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process is None:
            return
        if self._process.returncode is not None:
            return
        try:
            self._process.terminate()

            # Wait until the process has truly terminated.
            stdout, stderr = await asyncio.wait_for(self._process.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.kill()
            try:
                # Set a shorter timeout for the cleanup process
                stdout, stderr = await asyncio.wait_for(self._process.communicate(), timeout=2.0)
            except asyncio.TimeoutError:
                # If it still timeout, return None.
                return None
        except Exception:
            return None

    async def run(self, command: str) -> ToolExecResult:
        """Execute a command in the bash shell."""
        if not self._started or self._process is None:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolExecResult(
                error=f"bash has exited with returncode {self._process.returncode}. tool must be restarted.",
                error_code=-1,
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        error_code = 0

        sentinel_before, pivot, sentinel_after = self._sentinel.partition("__ERROR_CODE__")
        assert pivot == "__ERROR_CODE__"

        errcode_retriever = "!errorlevel!" if os.name == "nt" else "$?"
        command_sep = "&" if os.name == "nt" else ";"

        # send command to the process
        # Capture real exit code: run command, wait for it, record $?.
        sentinel_replaced = self._sentinel.replace('__ERROR_CODE__', errcode_retriever)
        if os.name == "nt":
            # Windows: run command, wait, capture errorlevel
            bg_command = command
            if not command.endswith("&"):
                bg_command = command + " &"
            shell_cmd = f"(\n{bg_command}\n){command_sep} {sentinel_replaced}\nexit\n"
        else:
            # Unix: run in subshell, wait for background process, capture exit code
            shell_cmd = f"(\n{command}\n); __exit_code=$?; echo {sentinel_replaced.replace(errcode_retriever, '$__exit_code')}\nexit\n"
        self._process.stdin.write(shell_cmd.encode())
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        buffer = ""
        output = ""
        try:
            async with asyncio.timeout(self._timeout):
                # Read until we find the sentinel in the stdout stream
                while True:
                    # Read a line from stdout (sentinel is on its own line)
                    line = await self._process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode()
                    if sentinel_before in line_str:
                        # Extract exit banner from this line
                        exit_banner = line_str.strip()
                        # Get error code inside banner
                        error_code_str, pivot, _ = exit_banner.partition(sentinel_after)
                        if not pivot or not error_code_str.isdecimal():
                            continue
                        error_code = int(error_code_str)
                        # The actual output is everything before this line
                        output = buffer
                        break
                    # Add line to buffer if it's not the sentinel
                    buffer += line_str
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            output = output[:-1]  # pyright: ignore[reportUnknownVariableType]

        error: str = ""
        try:
            error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        except (AttributeError, ValueError):
            pass
        if error.endswith("\n"):  # pyright: ignore[reportUnknownMemberType]
            error = error[:-1]  # pyright: ignore[reportUnknownVariableType]

        # clear the buffers so that the next output can be read correctly
        try:
            self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
            self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        except AttributeError:
            pass

        return ToolExecResult(output=output, error=error, error_code=error_code)  # pyright: ignore[reportUnknownArgumentType]


class BashTool(Tool):
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)
        self._session: _BashSession | None = None

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return "bash"

    @override
    def get_description(self) -> str:
        return """Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
"""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        # For OpenAI models, all parameters must be required=True
        # For other providers, optional parameters can have required=False
        restart_required = self.model_provider == "openai"

        return [
            ToolParameter(
                name="command",
                type="string",
                description="The bash command to run.",
                required=True,
            ),
            ToolParameter(
                name="restart",
                type="boolean",
                description="Set to true to restart the bash session.",
                required=restart_required,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        if arguments.get("restart"):
            if self._session:
                await self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolExecResult(output="tool has been restarted.")

        if self._session is None:
            try:
                self._session = _BashSession()
                await self._session.start()
            except Exception as e:
                return ToolExecResult(error=f"Error starting bash session: {e}", error_code=-1)

        command = str(arguments["command"]) if "command" in arguments else None
        if command is None:
            return ToolExecResult(
                error=f"No command provided for the {self.get_name()} tool",
                error_code=-1,
            )
        try:
            return await self._session.run(command)
        except Exception as e:
            return ToolExecResult(error=f"Error running bash command: {e}", error_code=-1)

    @override
    async def close(self):
        """Properly close self._process."""
        if self._session:
            ret = await self._session.stop()
            self._session = None
            return ret


class ReadOnlyBashTool(BashTool):
    """A read-only variant of BashTool, used in Plan mode.

    The execution surface is identical to BashTool (same `bash` tool name, same
    parameters), but mutating commands are physically rejected *before* they
    reach the shell, by static analysis. This is enforcement, not guidance: the
    prompt-level "don't run rm" rule can be ignored by the LLM, but a ToolError
    raised here cannot be bypassed without changing the code.

    Blocked categories:
        - File deletion (rm, rmdir, unlink)
        - File mutation (>, >>, sed -i, perl -i, awk redirect, tee)
        - Permissions / ownership (chmod, chown)
        - Package management (pip/npm/brew/apt/uv install)
        - Git writes (push, commit, reset, stash drop)
        - Network mutation (curl -X POST/PUT/DELETE/PATCH)
        - System state (shutdown, reboot, kill -9, mkfs, dd of=)
        - Service control (systemctl start/stop, killall)

    The blocklist is intentionally conservative — false positives are better
    than a write slipping through in plan mode.
    """

    # (human-readable reason, regex). Order matters: more specific first.
    BLOCKED_PATTERNS: list[tuple[str, str]] = [
        # === File deletion / destruction ===
        ("rm", r"\brm\s+(-\w+\s+)*[^|;&]*"),
        ("rmdir", r"\brmdir\s+"),
        ("unlink", r"\bunlink\s+"),
        # === File / dir mutation ===
        # Match `>` or `>>` to a non-control target. Negative lookahead lets
        # `>/dev/null`, `>&1`, `2>&1` through (those are control flow, not writes).
        ("file redirect (>)", r"(?<![&0-9])>{1,2}\s*(?!/dev/null\b|&\d?\b)[^\s|&;]+"),
        ("mv", r"\bmv\s+"),
        ("cp", r"\bcp\s+(?:-[rRa-zA-Z]+\s+)*[^|;&]*"),  # cp is also a write
        ("mkdir", r"\bmkdir\s+"),
        ("touch", r"\btouch\s+"),
        ("ln -s", r"\bln\s+(-\w+\s+)*-s\b"),
        ("truncate", r"\btruncate\s+"),
        ("sed -i", r"\bsed\s+-i\w*\b"),
        ("perl -i", r"\bperl\s+-i\w*\b"),
        ("tee", r"\btee\s+[^|]"),
        # === Permissions / ownership ===
        ("chmod", r"\bchmod\s+"),
        ("chown", r"\bchown\s+"),
        # === Package management ===
        ("pip install", r"\bpip3?\s+install\b"),
        ("pip uninstall", r"\bpip3?\s+uninstall\b"),
        ("npm install/add", r"\bnpm\s+(install|i|add)\b"),
        ("brew install", r"\bbrew\s+install\b"),
        ("apt install", r"\bapt(-get)?\s+install\b"),
        ("uv pip install", r"\buv\s+pip\s+(install|add)\b"),
        # === Git writes ===
        ("git push", r"\bgit\s+push\b"),
        ("git commit", r"\bgit\s+commit\b"),
        ("git add", r"\bgit\s+add\b"),
        ("git reset", r"\bgit\s+reset\b"),
        ("git stash", r"\bgit\s+stash\s+(push|apply|drop)\b"),
        ("git checkout (file)", r"\bgit\s+checkout\s+(?![\w./-]+$)[^|&;]+"),
        ("git merge", r"\bgit\s+merge\b"),
        ("git rebase", r"\bgit\s+rebase\b"),
        ("git tag", r"\bgit\s+tag\b"),
        # === Network mutation ===
        ("curl POST/PUT/DELETE", r"\bcurl\s+[^|;&]*-X\s*(POST|PUT|DELETE|PATCH)\b"),
        ("wget POST/PUT", r"\bwget\s+[^|;&]*--post"),
        # === System state ===
        ("shutdown/reboot", r"\b(shutdown|reboot|halt|poweroff)\b"),
        ("kill -9", r"\bkill\s+-9\b"),
        ("mkfs", r"\bmkfs\b"),
        ("dd of=...", r"\bdd\s+.*\bof\s*="),
        # === Service / process control ===
        ("systemctl start/stop", r"\bsystemctl\s+(start|stop|restart|enable|disable)\b"),
        ("killall", r"\bkillall\s+"),
        # === Container / virtualenv writes ===
        ("docker rm", r"\bdocker\s+(rm|rmi|system\s+prune)\b"),
        # === Inline code executors (bypass shell redirect checks) ===
        # `python3 -c "open('x','w').write(...)"` 不经过 shell 重定向,
        # 原黑名单无法拦截。这类命令在 plan 模式下完全禁止。
        ("python3 -c", r"\bpython3?\s+-c\b"),
        ("node -e/--eval", r"\bnode\s+(--eval|-e)\b"),
        ("deno eval", r"\bdeno\s+eval\b"),
        # === Git mutations ===
        # `git branch -D` 删除分支(原 SAFE_PREFIXES 里 "git branch" 允许了它)
        ("git branch -d/-D/-m", r"\bgit\s+branch\s+-(?:[dDmM]|-delete|-move)\b"),
        ("git clean", r"\bgit\s+clean\b"),
    ]

    # Whitelisted safe operations (always allowed regardless of patterns above).
    # NOTE: `echo`, `printf`, `python3`, `node` are intentionally NOT here —
    # they can write files via `> file` redirects and should go through the
    # redirect check.
    SAFE_PREFIXES: tuple[str, ...] = (
        "ls", "cat", "head", "tail", "less", "more", "wc", "file",
        "find", "grep", "rg", "ag", "ack",
        "true", "false", "test", "[", "[[",
        "cd", "pwd", "env", "printenv", "which", "type", "command -v",
        "git status", "git log", "git diff", "git show", "git blame",
        "git ls-files", "git remote -v",
        "ps", "top -l 1", "lsof", "netstat", "ss",
        "date", "cal", "uname", "whoami", "id", "hostname",
        "tree", "du", "df",
    )

    @override
    def get_description(self) -> str:
        return """Read-only bash (PLAN MODE).

* Identical surface to `bash`, but mutating commands are physically rejected
  before execution: file writes, deletes, package installs, git commits/pushes,
  curl POSTs, system changes all return a [PLAN-MODE] ToolError.
* Use this to explore the codebase: read files, search, run tests, inspect git.
* For any actual change, switch to build mode (`trae run`).
"""

    def check_command(self, command: str) -> tuple[bool, str]:
        """Return (allowed, reason). reason is empty when allowed.

        Blacklist is checked FIRST against every segment of a pipeline.
        Safe-prefix short-circuit only applies to the FIRST command in the
        pipeline (before any &&, ||, ;, | operators) AND only when no
        blacklist pattern matched.
        """
        # 1) Check blacklist against the ENTIRE command (catches cd /tmp && rm -rf ~)
        for desc, pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return False, desc

        # 2) Only if no blacklist match, apply safe-prefix on the first command
        cmd_stripped = command.strip()
        # Split on shell operators to get the first command
        first_cmd = re.split(r"\s*(?:&&|\|\||;|\|)\s*", cmd_stripped, maxsplit=1)[0].strip()
        for safe in self.SAFE_PREFIXES:
            if first_cmd.startswith(safe + " ") or first_cmd == safe:
                return True, ""

        return True, ""

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        # Skip check for the restart command — it's safe (just resets session).
        if arguments.get("restart"):
            return await super().execute(arguments)

        command = str(arguments.get("command", "") or "")
        if command:
            allowed, reason = self.check_command(command)
            if not allowed:
                return ToolExecResult(
                    error=(
                        f"[PLAN-MODE BLOCKED] {reason}\n"
                        f"  Command: {command!r}\n"
                        f"  Plan mode is read-only. To make this change, "
                        f"re-run with `trae run` (build mode)."
                    ),
                    error_code=-1,
                )
        return await super().execute(arguments)
