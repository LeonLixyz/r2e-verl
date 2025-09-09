from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Self, cast, override

from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from playwright.async_api import ConsoleMessage, async_playwright
from playwright.async_api import Error as PageError
from pydantic import BaseModel

from relace_agent.config import RelaceConfig
from relace_agent.errors import BuildError, DeployError, TestError
from relace_agent.repo import Repo
from relace_agent.server.deployment import deploy_to_cloudflare, repo_dist

logger = logging.getLogger(__name__)


@dataclass
class ShellOutput:
    exit_code: int
    stdout: str
    stderr: str


class BaseChatItem(BaseModel, metaclass=ABCMeta):
    @abstractmethod
    def as_message_params(self) -> list[ChatCompletionMessageParam]: ...


class UserMessage(BaseChatItem):
    role: Literal["user"] = "user"
    content: str

    @override
    def as_message_params(self) -> list[ChatCompletionMessageParam]:
        return [
            {
                "role": self.role,
                "content": [
                    {
                        "type": "text",
                        "text": self.content,
                    }
                ],
            }
        ]


class AssistantMessage(BaseChatItem):
    role: Literal["assistant"] = "assistant"
    content: str

    @override
    def as_message_params(self) -> list[ChatCompletionMessageParam]:
        return [
            {
                "role": self.role,
                "content": self.content,
            }
        ]


class ToolUse(BaseChatItem):
    message: ChatCompletionMessage
    results: dict[str, str]

    @override
    def as_message_params(self) -> list[ChatCompletionMessageParam]:
        results = [
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": [
                    {
                        "type": "text",
                        "text": tool_result,
                    }
                ],
            }
            for tool_id, tool_result in self.results.items()
        ]
        return [
            self.message.model_dump(mode="json", exclude_none=True),  # type: ignore
            *results,  # type: ignore
        ]


# Discriminated union needed for serialization to work correctly
ChatItem = UserMessage | AssistantMessage | ToolUse


class ChatHistoryJson(BaseModel):
    messages: list[ChatItem]

    @classmethod
    def load_messages(cls, path: Path) -> list[ChatItem]:
        return cls.model_validate_json(path.read_text()).messages

    @classmethod
    def dump_messages(cls, path: Path, messages: Iterable[ChatItem]) -> None:
        path.write_text(cls(messages=list(messages)).model_dump_json())


# TODO: Intelligent file context (one copy at a time, not written to disk)
# TODO: Intelligent truncation/summarization of history
# TODO: Encapsulate tool definitions and system prompt
class ChatHistory:
    def __init__(
        self,
        max_messages: int | None = 100,
        max_tokens: int = 140_000,
    ) -> None:
        self.messages: deque[ChatItem] = deque(maxlen=max_messages)
        self.max_tokens: int = max_tokens
        self.last_tokens: int = 0

    @classmethod
    @contextmanager
    def persistent(cls, path: Path) -> Iterator[Self]:
        instance = cls()
        if path.exists():
            try:
                loaded = ChatHistoryJson.load_messages(path)
                logger.info("Loaded %s messages from %s", len(loaded), path)
                instance.messages.clear()
                instance.messages.extend(loaded)
            except Exception as e:
                logger.warning("Failed to load chat history: %s", e)

        try:
            yield instance
        finally:
            try:
                ChatHistoryJson.dump_messages(path, instance.messages)
                logger.info("Dumped %s messages to %s", len(instance.messages), path)
            except Exception as e:
                logger.warning("Failed to save chat history: %s", e)

    def append(self, item: ChatItem) -> None:
        self.messages.append(item)

    def as_params(self) -> list[ChatCompletionMessageParam]:
        return [
            param for message in self.messages for param in message.as_message_params()
        ]

    def record_usage(
        self, usage: CompletionUsage | None, auto_clean: bool = True
    ) -> None:
        if usage is not None:
            self.last_tokens = usage.total_tokens
        # Truncate history if we're over our limit
        if auto_clean and self.last_tokens > self.max_tokens:
            self.clean()

    # TODO (eborgnia): This is the simplest solution, but we should consider more sophisticated approaches.
    # - Remove unecessary file reads that pollute the context window.
    # - Delete the center messages instead to not lose the original user intent.
    # - Maintain only summary messages or some kind of markdown log of the changes.
    def clean(self) -> None:
        """
        Simple conversation cleaning by removing the oldest non-user message.
        This helps manage token limits by gradually removing old context.
        """
        logger.info("Cleaning conversation history")
        messages_list = list(self.messages)
        for i, message in enumerate(messages_list[:]):
            if not isinstance(message, UserMessage):
                del messages_list[i]
                self.messages.clear()
                self.messages.extend(messages_list)
                logger.info("Removed message: %s", message)
                break
        else:
            logger.info("No messages to remove from conversation history")

    def tokenize(self) -> str:
        chunks: list[str] = []
        for message in self.messages:
            if isinstance(message, UserMessage):
                chunks.append(f"<user>{message.content}</user>")
            elif isinstance(message, AssistantMessage):
                chunks.append(f"<assistant>{message.content}</assistant>")
            elif isinstance(message, ToolUse) and message.message.tool_calls:
                chunks.extend(
                    f"<tool_call>{tool_call.function.name}</tool_call>"
                    for tool_call in message.message.tool_calls
                )
        return "\n".join(chunks)


@dataclass
class Context:
    repo: Repo
    inputs: dict[str, str] = field(default_factory=dict)
    history: ChatHistory = field(default_factory=ChatHistory)
    config: RelaceConfig = field(default_factory=RelaceConfig.load_default)
    shell: ShellOutput | None = field(default=None, init=False)
    error: Exception | None = field(default=None, init=False)

    _interpolate_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"\$\{\{ (?P<namespace>\w+)(\.(?P<key>\w+))? \}\}"
    )

    def interpolate(self, template: str) -> str:
        def handle_match(match: re.Match[str]) -> str:
            namespace = match.group("namespace")
            key = match.group("key")
            match (namespace, key):
                case ("repo", "root_path"):
                    return str(self.repo.root_path)
                case ("repo", "file_tree"):
                    return "\n".join(
                        f"- {path.relative_to(self.repo.root_path)}"
                        for path in self.repo.list_tracked_files()
                    )
                case ("inputs", input_key):
                    return self.inputs[input_key]
                case ("shell", ("stdout" | "stderr" | "exit_code") as shell_key):
                    if self.shell is None:
                        raise ValueError("Invalid shell substitution context")
                    return cast(str, getattr(self.shell, shell_key))
                case ("error", None):
                    if self.error is None:
                        raise ValueError("Invalid error substitution context")
                    return str(self.error)
                case _:
                    raise ValueError(f"Invalid substitution: {namespace}.{key}")

        return self._interpolate_pattern.sub(handle_match, template)

    async def run_build(self) -> None:
        async def run_cmd(args: list[str]) -> None:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=self.repo.root_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.build.timeout,
                )
            except TimeoutError as e:
                raise BuildError("Build command timed out") from e

            if proc.returncode != 0:
                self.shell = ShellOutput(
                    exit_code=proc.returncode or -1,
                    stdout=stdout.decode(),
                    stderr=stderr.decode(),
                )
                raise BuildError(self.shell.stderr.strip())

        await run_cmd(self.config.build.install)
        await run_cmd(self.config.build.build)

    async def run_test(self) -> None:
        errors: list[str] = []

        def handle_error(error: PageError) -> None:
            errors.append(f"[PageError] {error.message}")

        def handle_console(message: ConsoleMessage) -> None:
            if message.type == "error":
                errors.append(f"[ConsoleError] {message.text}")

        def kill_process_group(pid: int) -> None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                # Process may already be dead
                pass

        async with AsyncExitStack() as stack:
            # Start server in a process group so we can kill it and all children
            server = await asyncio.create_subprocess_exec(
                *self.config.test.serve,
                cwd=self.repo.root_path,
                preexec_fn=os.setsid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            server_pid = os.getpgid(server.pid)
            stack.callback(kill_process_group, server_pid)

            playwright = await stack.enter_async_context(async_playwright())
            browser = await playwright.chromium.launch(headless=True)
            stack.push_async_callback(browser.close)

            # Load and check for errors
            page = await browser.new_page()
            page.on("pageerror", handle_error)
            page.on("console", handle_console)
            await page.goto(
                self.config.test.serve_url, timeout=self.config.test.wait_load
            )
            await page.wait_for_timeout(self.config.test.wait_error)
            if errors:
                raise TestError("\n".join(errors))

            # Check that page is not empty
            text_content = await page.text_content("body")
            if text_content is None:
                raise TestError("Page has no text content")
            if not text_content.strip():
                raise TestError("Page text content is empty")

            # Check that page has visible elements
            visible_elements = await page.query_selector_all(
                "body *:not(script):not(style):not(meta):not(link)"
            )
            if not visible_elements:
                raise TestError("Page has no visible elements")

    async def run_deploy(self) -> None:
        success = await deploy_to_cloudflare(
            local_dist_path=self.repo.root_path / self.config.deploy.dist_path,
            prefix=repo_dist(self.repo.repo_id),
        )
        if not success:
            raise DeployError("Deployment failed")
