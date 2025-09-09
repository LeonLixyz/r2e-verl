import os
from pathlib import Path
from uuid import UUID

import pytest

from relace_agent.agents import Agent
from relace_agent.config import AgentConfig
from relace_agent.context import Context, UserMessage
from relace_agent.repo import Repo


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    return Repo(
        repo_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        root_path=repo_path,
        bundle_path=tmp_path / "repo.bundle",
    )


@pytest.fixture
def context(repo: Repo) -> Context:
    return Context(
        repo,
        inputs={
            "system_prompt": "My system prompt",
            "user_prompt": "My user prompt",
        },
    )


class TestAgent:
    @pytest.fixture
    def agent(self) -> Agent:
        os.environ["OPENAI_API_KEY"] = "openai-test-key"
        return Agent(
            config=AgentConfig(
                system="System: ${{ inputs.system_prompt }}",
                user="User: ${{ inputs.user_prompt }}",
                tools=[],
            ),
            name="test-agent",
        )

    def test_format_system_prompt_qwen(self, agent: Agent, context: Context) -> None:
        agent.config.model_name = "qwen3/qwen3-coder"
        assert agent.format_system_prompt(context) == {
            "content": [
                {
                    "text": "System: My system prompt",
                    "type": "text",
                },
            ],
            "role": "system",
        }

    def test_format_system_prompt_claude(self, agent: Agent, context: Context) -> None:
        agent.config.model_name = "anthropic/claude-sonnet-4"
        assert agent.format_system_prompt(context) == {
            "content": [
                {
                    "text": "System: My system prompt",
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "role": "system",
        }

    def test_format_history_empty(self, agent: Agent, context: Context) -> None:
        assert agent.format_history(context) == []

    def test_format_history_qwen(self, agent: Agent, context: Context) -> None:
        agent.config.model_name = "qwen/qwen3-coder"
        context.history.append(UserMessage(content="User message"))
        assert agent.format_history(context) == [
            {
                "content": [
                    {
                        "text": "User message",
                        "type": "text",
                    },
                ],
                "role": "user",
            },
        ]

    def test_format_history_claude(self, agent: Agent, context: Context) -> None:
        agent.config.model_name = "anthropic/claude-sonnet-4"
        context.history.append(UserMessage(content="User message"))
        assert agent.format_history(context) == [
            {
                "content": [
                    {
                        "text": "User message",
                        "type": "text",
                        # Adds cache_control marker
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                "role": "user",
            },
        ]
