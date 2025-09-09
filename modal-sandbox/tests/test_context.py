import copy
import os
from pathlib import Path
from uuid import UUID

import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from relace_agent.agents import BuildAgent, UserAgent
from relace_agent.agents import TestAgent as _TestAgent  # avoids pytest confusion
from relace_agent.context import (
    ChatHistory,
    ChatItem,
    Context,
    ToolUse,
    UserMessage,
)
from relace_agent.repo import Repo

MESSAGES: list[ChatItem] = [
    UserMessage(content="What's the weather like today?"),
    ToolUse(
        message=ChatCompletionMessage(
            role="assistant",
            content="I'll check the weather in your location:",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    type="function",
                    id="tool_call_1",
                    function=Function(
                        name="weather",
                        arguments='{"latitude": 1, "longitude": 2}',
                    ),
                )
            ],
        ),
        results={"tool_call_1": "The weather is sunny with a high of 75°F."},
    ),
]


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(
        repo_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        root_path=tmp_path / "repo",
        bundle_path=tmp_path / "repo.bundle",
    )


class TestChatHistory:
    def test_append(self) -> None:
        chat_history = ChatHistory()
        for message in MESSAGES:
            chat_history.append(message)

        assert list(chat_history.messages) == MESSAGES

    def test_as_params(self) -> None:
        chat_history = ChatHistory()
        for message in MESSAGES:
            chat_history.append(message)

        params = chat_history.as_params()
        assert params == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather like today?",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "I'll check the weather in your location:",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "arguments": '{"latitude": 1, "longitude": 2}',
                            "name": "weather",
                        },
                        "id": "tool_call_1",
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "text",
                        "text": "The weather is sunny with a high of 75°F.",
                    }
                ],
                "tool_call_id": "tool_call_1",
            },
        ]

    def test_persistent_write(self, tmp_path: Path) -> None:
        path = tmp_path / "chat_history.json"
        with ChatHistory.persistent(path) as chat_history:
            for message in MESSAGES:
                chat_history.append(message)
        assert path.exists()
        assert path.read_text()

    def test_persistent_read_after_write(self, tmp_path: Path) -> None:
        path = tmp_path / "chat_history.json"
        with ChatHistory.persistent(path) as chat_history:
            for message in MESSAGES:
                chat_history.append(message)
            chat_history_old = copy.deepcopy(chat_history.messages)

        with ChatHistory.persistent(path) as chat_history:
            chat_history_new = copy.deepcopy(chat_history.messages)

        assert chat_history_old == chat_history_new

    def test_persistent_read_invalid(self, tmp_path: Path) -> None:
        path = tmp_path / "chat_history.json"
        path.write_text("invalid json")
        with ChatHistory.persistent(path) as chat_history:
            assert not chat_history.messages


class TestContext:
    @pytest.fixture
    def context(self, repo: Repo) -> Context:
        os.environ["OPENAI_API_KEY"] = "openai-test-key"
        return Context(repo=repo)

    def test_build_agent_init(self, context: Context) -> None:
        agent = BuildAgent.from_context(context)
        assert agent.tools

    def test_test_agent_init(self, context: Context) -> None:
        agent = _TestAgent.from_context(context)
        assert agent.tools

    def test_user_agents(self, context: Context) -> None:
        for name in context.config.user_agents:
            agent = UserAgent.from_context(context, name=name)
            assert agent.name == name
            assert agent.tools
