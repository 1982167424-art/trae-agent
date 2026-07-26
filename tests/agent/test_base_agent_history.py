# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from trae_agent.agent.agent_basics import (
    AgentExecution,
    AgentStep,
    AgentStepState,
)
from trae_agent.agent.trae_agent import TraeAgent
from trae_agent.tools.base import ToolCall, ToolResult
from trae_agent.utils.config import Config
from trae_agent.utils.legacy.legacy_config import LegacyConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse


class TestBaseAgentHistory(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        test_config = {
            "default_provider": "anthropic",
            "max_steps": 20,
            "model_providers": {
                "anthropic": {
                    "model": "claude-sonnet-4-20250514",
                    "api_key": "test-dummy-api-key",  # dummy api key
                    "max_tokens": 4096,
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 0,
                    "parallel_tool_calls": False,
                    "max_retries": 10,
                }
            },
        }
        self.config = Config.create_from_legacy_config(
            legacy_config=LegacyConfig(test_config)
        )
        # Avoid creating a real LLMClient instance (no API calls).
        self.llm_patcher = patch("trae_agent.agent.base_agent.LLMClient")
        self.llm_patcher.start()
        self.agent = TraeAgent(self.config.trae_agent)
        # Replace the real tool executor with a mock so no real tools run
        # and we control what the tool-call handler returns.
        self.agent._tool_caller = MagicMock()
        self.agent._tool_caller.sequential_tool_call = AsyncMock(
            return_value=[
                ToolResult(
                    call_id="call_1", name="bash", success=True, result="ok"
                )
            ]
        )

    def tearDown(self):
        self.llm_patcher.stop()

    async def test_run_llm_step_records_assistant_turn(self):
        """The agent must write the assistant turn (with its tool_call) back
        into the returned messages.

        Before the fix, only tool_result/reflection messages were appended,
        so from the 2nd step on the history was missing the assistant turn.
        That makes Anthropic reject (consecutive user msgs), OpenAI reject
        orphan tool outputs, and Google/Ollama mis-alternate — every task
        needing >=2 tool calls crashed at step 2.
        """
        tool_call = ToolCall(
            name="bash", call_id="call_1", arguments={"command": "echo hi"}
        )
        self.agent._llm_client.chat.return_value = LLMResponse(
            content="", tool_calls=[tool_call]
        )

        step = AgentStep(step_number=1, state=AgentStepState.THINKING)
        execution = AgentExecution(task="do it", steps=[])
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content="sys"),
            LLMMessage(role="user", content="do it"),
        ]

        new_messages = await self.agent._run_llm_step(step, messages, execution)

        # First returned message MUST be the assistant turn carrying the tool call.
        self.assertEqual(new_messages[0].role, "assistant")
        self.assertIsNotNone(new_messages[0].tool_call)
        self.assertEqual(new_messages[0].tool_call.call_id, "call_1")
        # The assistant turn must precede the tool_result user message.
        self.assertEqual(new_messages[1].role, "user")
        self.assertIsNotNone(new_messages[1].tool_result)

    async def test_chat_called_with_reuse_history_false(self):
        """The agent owns the full message history, so it must tell each
        client to treat the incoming `messages` as the complete history
        (reuse_history=False). Otherwise OpenAI-compatible clients
        accumulate `message_history += parsed_messages` and double the
        conversation (duplicate system prompt, misplaced tool messages).
        """
        tool_call = ToolCall(
            name="bash", call_id="call_1", arguments={"command": "echo hi"}
        )
        self.agent._llm_client.chat.return_value = LLMResponse(
            content="", tool_calls=[tool_call]
        )

        step = AgentStep(step_number=1, state=AgentStepState.THINKING)
        execution = AgentExecution(task="do it", steps=[])
        messages: list[LLMMessage] = [LLMMessage(role="user", content="do it")]

        await self.agent._run_llm_step(step, messages, execution)

        self.agent._llm_client.chat.assert_called_once()
        _, kwargs = self.agent._llm_client.chat.call_args
        self.assertFalse(kwargs.get("reuse_history", True))


if __name__ == "__main__":
    unittest.main()
