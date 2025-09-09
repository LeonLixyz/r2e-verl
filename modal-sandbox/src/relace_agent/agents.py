from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Self

from langfuse import get_client
from langfuse.openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)

from relace_agent.config import AgentConfig
from relace_agent.context import (
    AssistantMessage,
    Context,
    ToolUse,
    UserMessage,
)
from relace_agent.errors import AgentError, AgentStop, BuildError, TestError, ToolError
from relace_agent.server.types import (
    AgentEvent,
    BaseEvent,
    BuildEvent,
    CommittedEvent,
    DeployedEvent,
    TestEvent,
)
from relace_agent.tools import Tool, ToolSchema

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig, name: str) -> None:
        self.config: AgentConfig = config
        self.name: str = name
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            timeout=config.prompt_timeout,
            max_retries=config.prompt_retries,
        )
        self.tools: dict[str, Tool[ToolSchema]] = {
            name: Tool.from_name(name) for name in config.tools
        }

    def format_tool_definitions(self) -> list[ChatCompletionToolParam]:
        tools = [tool.schema.openai() for tool in self.tools.values()]
        if tools and self.config.model_name.startswith("anthropic/"):
            tools[-1]["cache_control"] = {"type": "ephemeral"}  # type: ignore
        return tools

    def format_system_prompt(
        self, context: Context
    ) -> ChatCompletionSystemMessageParam:
        prompt: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": context.interpolate(self.config.system),
                }
            ],
        }
        if self.config.model_name.startswith("anthropic/"):
            prompt["content"][-1]["cache_control"] = {"type": "ephemeral"}  # type: ignore
        return prompt

    def format_history(self, context: Context) -> list[ChatCompletionMessageParam]:
        messages = context.history.as_params()
        if self.config.model_name.startswith("anthropic/"):
            # Add cache_control to last item in history
            for message in reversed(messages):
                content = message.get("content")
                if content and isinstance(content, list):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                    break
        return messages

    async def generate_response(self, context: Context) -> ChatCompletion:
        """Generate a response from the model."""
        response = await self.client.chat.completions.create(
            user=f"repo_{context.repo.repo_id}",  # results in sticky routing by repo
            messages=[
                self.format_system_prompt(context),
                *self.format_history(context),
            ],
            model=self.config.model_name,
            tools=self.format_tool_definitions(),
            max_tokens=self.config.max_tokens,
            extra_body={
                "provider": {
                    "data_collection": "deny",
                    "require_parameters": True,
                },
            },
        )
        if not response.choices:
            raise RuntimeError("Received empty choices")
        # Handle OpenRouter API errors
        if response.choices[0].finish_reason == "error":  # type: ignore
            error = getattr(response.choices[0], "error", "Unknown error")
            raise RuntimeError(f"API error: {error}")
        return response

    async def handle_response(
        self,
        context: Context,
        response: ChatCompletion,
    ) -> AsyncIterator[BaseEvent]:
        logger.info("Agent response: %s", response)
        context.history.record_usage(response.usage)

        # Configured for one choice only
        choice = response.choices[0]
        if choice.message.tool_calls:
            logger.info("Agent performing tool calls")
            if choice.message.content:
                yield AgentEvent(name=self.name, content=choice.message.content)

            tool_results: dict[str, str] = {}
            for tool_call in choice.message.tool_calls:
                try:
                    tool = self.get_tool(tool_call.function.name)
                    tool_input = tool.schema.from_json_str(tool_call.function.arguments)
                    yield tool_input.event(context)
                    tool_results[tool_call.id] = await tool.execute(tool_input, context)
                except ToolError as error:
                    logger.warning("Tool error: %s", error)
                    tool_results[tool_call.id] = f"Error: {error!s}"

            context.history.append(
                ToolUse(message=choice.message, results=tool_results)
            )
        elif choice.finish_reason == "stop":
            logger.info("Agent stopped naturally: %s", choice.finish_reason)
            if choice.message.content:
                yield AgentEvent(name=self.name, content=choice.message.content)
                context.history.append(AssistantMessage(content=choice.message.content))
            raise AgentStop(choice.finish_reason)
        else:
            logger.error("Agent stopped unexpectedly: %s", choice.finish_reason)
            raise AgentError(f"Agent stopped unexpectedly: {choice.finish_reason}")

    def get_tool(self, tool_name: str) -> Tool[ToolSchema]:
        try:
            return self.tools[tool_name]
        except KeyError as e:
            raise ToolError(f"Unknown tool: {tool_name}") from e

    async def run(self, context: Context) -> AsyncIterator[BaseEvent]:
        """Run the agent with the given query."""
        langfuse = get_client()
        with langfuse.start_as_current_span(
            name=f"{self.__class__.__name__}.run", input=context
        ) as langfuse_span:
            # Run the conversation loop
            context.history.append(
                UserMessage(content=context.interpolate(self.config.user))
            )
            turn = 0
            while turn < self.config.max_turns:
                turn += 1
                response = await self.generate_response(context)

                try:
                    async for event in self.handle_response(context, response):
                        yield event
                except AgentStop:
                    break
            else:
                logger.warning("Agent reached turn limit: %s", self.config.max_turns)

            # Commit and push changes
            with langfuse_span.start_as_current_span(name="commit") as commit_span:
                commit = await context.repo.commit(f"Agent: {self.name}")
                if commit:
                    await context.repo.push_bundle()
                    commit_span.update(output=commit)

                    yield CommittedEvent(repo_head=commit)
                else:
                    logger.info("No changes committed; skipping deployment")
                    return

            # Build test and deploy
            with langfuse_span.start_as_current_span(name="build"):
                try:
                    yield BuildEvent(event="start")
                    await context.run_build()
                    yield BuildEvent(event="pass")
                except BuildError:
                    yield BuildEvent(event="fail")
                    raise

            with langfuse_span.start_as_current_span(name="test"):
                try:
                    yield TestEvent(event="start")
                    await context.run_test()
                    yield TestEvent(event="pass")
                except TestError:
                    yield TestEvent(event="fail")
                    raise

            with langfuse_span.start_as_current_span(name="deploy"):
                await context.run_deploy()
                yield DeployedEvent(
                    url=f"https://{context.repo.repo_id}.dev-server.relace.run"
                )


class UserAgent(Agent):
    @classmethod
    def from_context(
        cls,
        context: Context,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> Self:
        config = context.config.user_agents.get(name)
        if config is None:
            raise ValueError(f"Unknown user agent: {name!r}")
        if overrides:
            logger.info("Applying config overrides: %s", overrides)
            config = AgentConfig.model_validate(
                config.model_dump() | overrides,
                strict=True,
            )
        return cls(config, name)

    async def run(self, context: Context) -> AsyncIterator[BaseEvent]:
        try:
            async for event in super().run(context):
                yield event
        except BuildError as error:
            context.error = error
            async for event in BuildAgent.from_context(context).run(context):
                yield event
        except TestError as error:
            context.error = error
            async for event in TestAgent.from_context(context).run(context):
                yield event


class BuildAgent(Agent):
    @classmethod
    def from_context(cls, context: Context, name: str = "build_agent") -> Self:
        return cls(context.config.build_agent, name)

    async def run(self, context: Context) -> AsyncIterator[BaseEvent]:
        try:
            async for event in super().run(context):
                yield event
        except TestError as error:
            context.error = error
            async for event in TestAgent.from_context(context).run(context):
                yield event


class TestAgent(Agent):
    @classmethod
    def from_context(cls, context: Context, name: str = "test_agent") -> Self:
        return cls(context.config.test_agent, name)
