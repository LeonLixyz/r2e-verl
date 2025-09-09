from __future__ import annotations

import asyncio
import csv
import functools
import io
import itertools
import logging
import random
import re
import tarfile
import tempfile
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, cast

import modal
from datasets import load_dataset
from langfuse import get_client, observe
from langfuse.openai import AsyncOpenAI
from modal.stream_type import StreamType
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from playwright.async_api import ConsoleMessage, async_playwright
from playwright.async_api import Error as PageError
from pydantic import BaseModel, ConfigDict

from relace_agent.config import (
    AgentConfig,
    BaseConfig,
    BuildConfig,
    DeployConfig,
    FileConfig,
    TestConfig,
)
from relace_agent.context import (
    AssistantMessage,
    ChatHistory,
    ShellOutput,
    ToolUse,
    UserMessage,
)
from relace_agent.errors import AgentError, AgentStop, BuildError, TestError, ToolError
from relace_agent.logging import setup_logging
from relace_agent.modal.agent_toolbox import (
    SANDBOX_TIMEOUT,
    TestResponse,
    create_sandbox,
    parse_test_output,
)
from relace_agent.server.deployment import deploy_to_cloudflare
from relace_agent.server.toolbox import BASH_TIMEOUT, SandboxTool, SandboxToolContext
from relace_agent.tools import ToolSchema

logger = logging.getLogger(__name__)

app = modal.App(name="Agent-Toolbox-Eval")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync()
    .run_commands("playwright install --with-deps --only-shell")
    .add_local_file("./relace-r2e.yaml", "/relace-r2e.yaml")
    .add_local_file("./relace-ui.yaml", "/relace-ui.yaml")
)


ui_volume = modal.Volume.from_name(name="relace-agent-ui-evals")
ui_volume_mnt = Path("/volumes/ui")

SANDBOX_UI_PORT = 4173


async def create_ui_sandbox(template_repo_bundle: Path) -> modal.Sandbox:
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install("git", "ripgrep", "curl")
        .run_commands(
            # Install Node.js 22 using official Node.js repository
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
            "npm install -g vite",
            # Configure git
            "git config --global user.name 'Relace Agent'",
            "git config --global user.email 'noreply@relace.ai'",
        )
        .add_local_file(template_repo_bundle, "/repo.bundle", copy=True)
        .workdir("/repo")
        .run_commands(
            "git clone /repo.bundle ./",
            "rm /repo.bundle",
        )
        # NOTE: This allows vite preview sites to be accessed through the sandbox tunnel
        # This results in the site being exposed to the public internet. There does not
        # seem to be a reasonable way to test a sandboxed server in modal without this
        .env({"__VITE_ADDITIONAL_SERVER_ALLOWED_HOSTS": ".modal.host"})
    )
    with modal.enable_output():
        return await modal.Sandbox.create.aio(
            image=image,
            app=app,
            encrypted_ports=[SANDBOX_UI_PORT],
            timeout=SANDBOX_TIMEOUT,
        )


def retry[T, **P](
    exc_types: tuple[type[BaseException], ...] = (Exception,),
    *,
    max_retries: int = 3,
    base_delay: float = 1,  # initial delay in seconds
    max_delay: float = 60,  # maximum delay in seconds
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Retries an async function with exponential backoff."""

    def decorator(f: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(f)
        async def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            retries = 0
            while True:
                try:
                    return await f(*args, **kwargs)
                except exc_types as exc:
                    retries += 1
                    if retries > max_retries:
                        logger.error("Retries for %s exhausted", f.__name__)
                        raise
                    else:
                        delay = min(base_delay * (2**retries), max_delay)
                        logger.warning(
                            "Retrying %s after %s seconds (%s of %s): %s: %s",
                            f.__name__,
                            delay,
                            retries,
                            max_retries,
                            exc.__class__.__name__,
                            exc,
                        )
                        await asyncio.sleep(delay)

        return inner

    return decorator


class EvalParameters(BaseConfig):
    agent_name: str
    overrides: dict[str, Any] | None = None


class EvalConfig(FileConfig):
    # Allow extra fields to enable references
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    sandbox_agents: dict[str, AgentConfig]
    matrix: dict[str, EvalParameters]
    ui_build: BuildConfig | None = None
    ui_test: TestConfig | None = None
    ui_deploy: DeployConfig | None = None

    @classmethod
    def load_r2e(cls) -> Self:
        return cls.load(Path(__file__).parent.parent.parent.parent / "relace-r2e.yaml")

    @classmethod
    def load_ui(cls) -> Self:
        return cls.load(Path(__file__).parent.parent.parent.parent / "relace-ui.yaml")

    def get_parameters(self, matrix_key: str) -> EvalParameters:
        params = self.matrix.get(matrix_key)
        if not params:
            raise ValueError(f"Unknown matrix key: {matrix_key}")
        return params

    def get_agent(
        self,
        name: str,
        overrides: dict[str, Any] | None = None,
    ) -> EvalAgent:
        config = self.sandbox_agents.get(name)
        if not config:
            raise ValueError(f"Unknown sandbox agent: {name}")
        if overrides:
            logger.info("Applying config overrides: %s", overrides)
            config = AgentConfig.model_validate(
                config.model_dump() | overrides,
                strict=True,
            )
        return EvalAgent(name=name, config=config)


@dataclass
class SandboxContext:
    sandbox: modal.Sandbox
    session_id: uuid.UUID
    config: EvalConfig
    inputs: dict[str, str]
    history: ChatHistory
    repo_root: str | None = None
    shell: ShellOutput | None = field(default=None, init=False)
    error: Exception | None = field(default=None, init=False)

    _interpolate_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"\$\{\{ (?P<namespace>\w+)(\.(?P<key>\w+))? \}\}"
    )

    # TODO: Pass full context in instead
    def get_tool_context(self) -> SandboxToolContext:
        return SandboxToolContext(history=self.history)

    def interpolate(self, template: str) -> str:
        def handle_match(match: re.Match[str]) -> str:
            namespace = match.group("namespace")
            key = match.group("key")
            match (namespace, key):
                case ("repo", "root_path"):
                    if self.repo_root is None:
                        raise ValueError("repo_root is not configured")
                    return str(self.repo_root)
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

    @observe
    async def run_tests(self) -> TestResponse:
        test_runner = await self.sandbox.exec.aio(
            "bash", "-c", "./run_tests.sh", timeout=BASH_TIMEOUT
        )
        exit_code = await test_runner.wait.aio()
        if exit_code == -1:
            raise TimeoutError("Test execution timed out")

        # Parse test output
        test_stdout = await test_runner.stdout.read.aio()
        test_stderr = await test_runner.stderr.read.aio()
        logger.info("Test output:\nSTDOUT:\n%s\nSTDERR:\n%s", test_stdout, test_stderr)
        return parse_test_output(test_stdout, exit_code)

    @observe
    async def run_ui_build(self, config: BuildConfig) -> None:
        logger.info("Running build: %s", config)

        async def run_cmd(args: list[str]) -> None:
            logger.info("Running `%s`", " ".join(args))
            build_runner = await self.sandbox.exec.aio(
                *args,
                workdir=self.repo_root,
                text=True,
                timeout=config.timeout,
            )
            exit_code = await build_runner.wait.aio()
            if exit_code == -1:
                raise TimeoutError("UI build timed out")
            elif exit_code != 0:
                self.shell = ShellOutput(
                    exit_code=exit_code,
                    stdout=await build_runner.stdout.read.aio(),
                    stderr=await build_runner.stderr.read.aio(),
                )
                raise BuildError(self.shell.stderr)

        await run_cmd(config.install)
        await run_cmd(config.build)

    @observe
    async def run_ui_test(self, config: TestConfig) -> None:
        logger.info("Running tests: %s", config)

        server_tunnel = (await self.sandbox.tunnels.aio())[SANDBOX_UI_PORT]
        server = await self.sandbox.exec.aio(
            *config.serve,
            workdir=self.repo_root,
            stdout=StreamType.DEVNULL,
            stderr=StreamType.DEVNULL,
            timeout=(config.wait_load + config.wait_error) + 1,
        )

        errors: list[str] = []

        def handle_error(error: PageError) -> None:
            errors.append(f"[PageError] {error.message}")

        def handle_console(message: ConsoleMessage) -> None:
            if message.type == "error":
                errors.append(f"[ConsoleError] {message.text}")

        async with AsyncExitStack() as stack:
            playwright = await stack.enter_async_context(async_playwright())
            browser = await playwright.chromium.launch(headless=True)
            stack.push_async_callback(browser.close)

            # Load and check for errors
            page = await browser.new_page()
            page.on("pageerror", handle_error)
            page.on("console", handle_console)
            await page.goto(server_tunnel.url, timeout=config.wait_load)
            await page.wait_for_timeout(config.wait_error)
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
            logger.info("Tests passed!")

    @observe
    async def run_ui_deploy(self, config: DeployConfig) -> None:
        logger.info("Running deploy: %s", config)

        logger.info("Extracting dist from sandbox")
        tar_stream = await self.sandbox.exec.aio(
            "tar",
            "-czf",
            "-",
            config.dist_path,
            workdir=self.repo_root,
            text=False,
        )
        tar_bytes = await tar_stream.stdout.read.aio()

        # Extract tarfile to temporary directory
        with tempfile.TemporaryDirectory(prefix="dist-") as temp_dir:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                await asyncio.to_thread(tar.extractall, path=temp_dir)

            slug = f"eval-{self.session_id}"
            await deploy_to_cloudflare(
                Path(temp_dir) / config.dist_path,
                prefix=f"{slug}/dist/",
            )
            logger.info("Deployed to https://%s.dev-server.relace.run", slug)

    async def run_sequence(
        self,
        agent_name: str,
        fix_build_with: str | None = None,
        fix_test_with: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        agent = self.config.get_agent(agent_name, overrides)
        await agent.run(self)

        try:
            if self.config.ui_build:
                await self.run_ui_build(self.config.ui_build)
            if self.config.ui_test:
                await self.run_ui_test(self.config.ui_test)
            if self.config.ui_deploy:
                await self.run_ui_deploy(self.config.ui_deploy)
        except BuildError as error:
            logger.error("Error during build", exc_info=True)
            if fix_build_with:
                logger.info("Fixing build error with %s", fix_build_with)
                self.error = error
                await self.run_sequence(
                    agent_name=fix_build_with,
                    fix_test_with=fix_test_with,
                    overrides=overrides,
                )
            else:
                raise
        except TestError as error:
            logger.error("Error during test", exc_info=True)
            if fix_test_with:
                logger.info("Fixing test error with %s", fix_test_with)
                self.error = error
                await self.run_sequence(
                    agent_name=fix_test_with,
                    overrides=overrides,
                )
            else:
                raise


class EvalAgent:
    def __init__(self, config: AgentConfig, name: str) -> None:
        self.config: AgentConfig = config
        self.name: str = name
        self.client: AsyncOpenAI = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            timeout=config.prompt_timeout,
            max_retries=config.prompt_retries,
        )
        self.tools: dict[str, type[ToolSchema]] = {
            name: ToolSchema.from_name(name) for name in config.tools
        }

    def format_tool_definitions(self) -> list[ChatCompletionToolParam]:
        tools = [tool_schema.openai() for tool_schema in self.tools.values()]
        if tools and self.config.model_name.startswith("anthropic/"):
            tools[-1]["cache_control"] = {"type": "ephemeral"}  # type: ignore
        return tools

    def format_system_prompt(
        self, context: SandboxContext
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

    def format_history(
        self, context: SandboxContext
    ) -> list[ChatCompletionMessageParam]:
        messages = context.history.as_params()
        if self.config.model_name.startswith("anthropic/"):
            # Add cache_control to last item in history
            for message in reversed(messages):
                content = message.get("content")
                if content and isinstance(content, list):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                    break
        return messages

    @retry()
    async def generate_response(self, context: SandboxContext) -> ChatCompletion:
        """Generate a response from the model."""
        response = await self.client.chat.completions.create(
            user=context.sandbox.object_id,  # results in sticky routing by repo
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
                    "only": [
                        "alibaba/plus",
                        "novita/fp8",
                        "deepinfra/fp8",
                        "google-vertex",
                        "anthropic",
                        "openai",
                    ],
                },
            },
            extra_headers={
                "HTTP-Referer": "https://relace.ai",
                "X-Title": "Relace Evaluation Framework",
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
        context: SandboxContext,
        response: ChatCompletion,
    ) -> None:
        logger.info("Agent response: %s", response)
        context.history.record_usage(response.usage)

        # Configured for one choice only
        choice = response.choices[0]
        if choice.message.tool_calls:
            logger.info("Agent performing tool calls")

            tool_results: dict[str, str] = {}
            for tool_call in choice.message.tool_calls:
                try:
                    if tool_call.function.name not in self.tools:
                        raise ToolError(f"Unknown tool: {tool_call.function.name}")
                    tool = SandboxTool.from_name(
                        tool_call.function.name, context.sandbox
                    )
                    tool_input = tool.schema.from_json_str(tool_call.function.arguments)
                    tool_results[tool_call.id] = await tool.execute(
                        tool_input,
                        context.get_tool_context(),
                    )
                    await tool.commit()
                except ToolError as error:
                    logger.warning("Tool error: %s", error)
                    tool_results[tool_call.id] = f"Error: {error!s}"

            context.history.append(
                ToolUse(message=choice.message, results=tool_results)
            )
        elif choice.finish_reason == "stop":
            logger.info("Agent stopped naturally: %s", choice.finish_reason)
            if choice.message.content:
                context.history.append(AssistantMessage(content=choice.message.content))
            raise AgentStop(choice.finish_reason)
        else:
            logger.error("Agent stopped unexpectedly: %s", choice.finish_reason)
            raise AgentError(f"Agent stopped unexpectedly: {choice.finish_reason}")

    async def run(self, context: SandboxContext) -> None:
        """Run the agent with the given query."""
        langfuse = get_client()
        with langfuse.start_as_current_span(
            name=f"{self.__class__.__name__}.run", input=context
        ):
            # Run the conversation loop
            context.history.append(
                UserMessage(content=context.interpolate(self.config.user))
            )
            turn = 0
            while turn < self.config.max_turns:
                turn += 1
                logger.info("Starting turn %s", turn)
                response = await self.generate_response(context)

                try:
                    await self.handle_response(context, response)
                except AgentStop:
                    break
            else:
                logger.warning("Agent reached turn limit: %s", self.config.max_turns)


class TestCase(BaseModel):
    image_tag: str
    problem_statement: str


TestCaseOutcome = Literal["success", "partial_success", "failure"]


class UiTestCase(BaseModel):
    user_prompt: str
    template_repo_bundle: Path
    matrix_key: str


UiTestCaseOutcome = Literal["success", "test_failure", "build_failure", "error"]


@app.cls(
    image=image,
    secrets=[
        modal.Secret.from_name("relace-toolbox-langfuse"),
        modal.Secret.from_name("relace-toolbox-eval-open-router"),
        modal.Secret.from_name("relace-key-toolbox-eval"),
        modal.Secret.from_name("cloudflare-secret"),
    ],
    volumes={
        ui_volume_mnt.as_posix(): ui_volume,
    },
    max_containers=1,
    timeout=60 * 60 * 24,
)
@modal.concurrent(max_inputs=32)
class TestCaseRunner:
    @modal.enter()
    def init_logging(self) -> None:
        setup_logging(logging.INFO, extra_modules=["httpx"])

    @modal.method()
    @observe
    async def eval_test_case(
        self,
        test_case: TestCase,
        agent_name: str,
    ) -> TestCaseOutcome:
        session_id = uuid.uuid4()
        logger.info("Running test case: %s", test_case)
        logger.info("Starting session: %s", session_id)
        sandbox = await create_sandbox(
            user_id=agent_name,
            session_id=session_id,
            image_tag=test_case.image_tag,
            parent_app=app,
        )
        try:
            langfuse = get_client()
            langfuse.update_current_trace(
                user_id=agent_name,
                session_id=str(session_id),
                metadata={
                    "sandbox_id": sandbox.object_id,
                    "image_tag": test_case.image_tag,
                    "agent": agent_name,
                },
                tags=["toolbox-eval"],
            )
            context = SandboxContext(
                sandbox=sandbox,
                session_id=session_id,
                inputs={"problem_statement": test_case.problem_statement},
                config=EvalConfig.load_r2e(),
                history=ChatHistory(max_messages=None),
            )
            test_start = await context.run_tests()
            await context.run_sequence(agent_name)
            test_end = await context.run_tests()

            # Determine outcome
            test_delta = Counter(test_end.counts)
            test_delta.subtract(test_start.counts)
            if test_delta["PASSED"] > 0 and test_end.counts.get("FAILED", 0) == 0:
                return "success"
            elif test_delta["PASSED"] > 0:
                return "partial_success"
            else:
                return "failure"
        except Exception:
            logger.error("Test case failed: %s", test_case, exc_info=True)
            return "failure"
        finally:
            logger.info("Ending session: %s", session_id)
            sandbox.terminate()

    @modal.method()
    @observe
    async def eval_ui_test_case(self, test_case: UiTestCase) -> UiTestCaseOutcome:
        config = EvalConfig.load_ui()
        params = config.get_parameters(test_case.matrix_key)
        session_id = uuid.uuid4()
        logger.info("Running test case: %s", test_case)
        logger.info("Starting session: %s", session_id)
        sandbox = await create_ui_sandbox(test_case.template_repo_bundle)
        try:
            langfuse = get_client()
            langfuse.update_current_trace(
                user_id=f"{params.agent_name}-{test_case.matrix_key}",
                session_id=str(session_id),
                metadata={
                    "sandbox_id": sandbox.object_id,
                },
                tags=["toolbox-ui-eval"],
            )
            context = SandboxContext(
                sandbox=sandbox,
                session_id=session_id,
                inputs={"user_prompt": test_case.user_prompt},
                config=EvalConfig.load_ui(),
                history=ChatHistory(max_messages=None),
                repo_root="/repo",
            )
            await context.run_sequence(
                "ui_agent",
                fix_build_with="build_agent",
                fix_test_with="test_agent",
                overrides=params.overrides,
            )
            return "success"
        except TestError:
            return "test_failure"
        except BuildError:
            return "build_failure"
        except Exception:
            logger.error("Test case failed: %s", test_case, exc_info=True)
            return "error"
        finally:
            logger.info("Ending session: %s", session_id)
            sandbox.terminate()


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    timeout=60 * 60 * 24,
)
def eval_r2e_dataset(
    agent_name: str = "qwen-replace-and-apply",
    dataset: str = "R2E-Gym/R2E-Gym-Subset",
    split: str = "train",
    count: int = 1,
    seed: int = 0,
) -> None:
    setup_logging(logging.INFO, extra_modules=["httpx"])

    def iter_test_cases() -> Iterator[TestCase]:
        logger.info("Loading dataset %s", dataset)
        for example in load_dataset(dataset, split=split, streaming=True):
            try:
                yield TestCase(
                    image_tag=example["docker_image"],
                    problem_statement=example["problem_statement"],
                )
            except Exception as e:
                logger.error("Error parsing example: %s", example, exc_info=e)

    tests = random.Random(seed).sample(list(iter_test_cases()), count)
    logger.info("Running %d test cases", len(tests))

    TestCaseRunner().eval_test_case.for_each(
        tests,
        kwargs={"agent_name": agent_name},
        ignore_exceptions=True,
    )


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    timeout=60 * 60 * 24,
)
def eval_r2e_subset(
    agent_name: str = "qwen-replace-and-apply",
    image_dataset: str = "relace/r2e-filtered",
    source_dataset: str = "R2E-Gym/R2E-Gym-Subset",
    count: int | None = None,
) -> None:
    setup_logging(logging.INFO, extra_modules=["httpx"])

    logger.info("Loading image dataset %s", image_dataset)
    images: set[str] = set()
    for item in itertools.islice(
        load_dataset(image_dataset, split="train", streaming=True),
        count,
    ):
        images.add(item["docker_image"])

    def iter_test_cases() -> Iterator[TestCase]:
        logger.info("Loading source dataset %s", source_dataset)
        for example in load_dataset(
            source_dataset,
            split="train",
            streaming=True,
        ):
            if example["docker_image"] not in images:
                continue

            try:
                yield TestCase(
                    image_tag=example["docker_image"],
                    problem_statement=example["problem_statement"],
                )
            except Exception as e:
                logger.error("Error parsing example: %s", example, exc_info=e)

    tests = list(iter_test_cases())
    logger.info("Running %d test cases", len(tests))

    TestCaseRunner().eval_test_case.for_each(
        tests,
        kwargs={"agent_name": agent_name},
        ignore_exceptions=True,
    )


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    timeout=60 * 60 * 24,
)
def eval_r2e_cases(
    image_tags: str,
    agent_name: str = "qwen-replace-and-apply",
    source_dataset: str = "R2E-Gym/R2E-Gym-Subset",
) -> None:
    setup_logging(logging.INFO, extra_modules=["httpx"])
    images = set(image_tags.split(","))

    def iter_test_cases() -> Iterator[TestCase]:
        logger.info("Loading source dataset %s", source_dataset)
        for example in load_dataset(
            source_dataset,
            split="train",
            streaming=True,
        ):
            if example["docker_image"] in images:
                yield TestCase(
                    image_tag=example["docker_image"],
                    problem_statement=example["problem_statement"],
                )

    tests = list(iter_test_cases())
    logger.info("Running %d test cases", len(tests))

    TestCaseRunner().eval_test_case.for_each(
        tests,
        kwargs={"agent_name": agent_name},
        ignore_exceptions=True,
    )


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("relace-toolbox-langfuse"),
    ],
    timeout=60 * 60 * 24,
)
def rerun_traces(
    agent_name: str = "qwen-replace-and-apply",
    output: str | None = None,
    dry_run: bool = False,
) -> None:
    setup_logging(logging.INFO, extra_modules=["httpx"])
    langfuse = get_client()

    page: int | None = None
    tests: dict[str, TestCase] = {}
    while True:
        traces = langfuse.api.trace.list(
            name="eval_test_case",
            user_id=agent_name,
            page=page,
        )
        for trace in traces.data:
            if output and trace.output != output:
                continue
            if not trace.input:
                continue

            test_case = TestCase(**trace.input["args"][0])
            # Ensure we do not duplicate any test cases
            if test_case.image_tag not in tests:
                tests[test_case.image_tag] = test_case
                logger.info("Added test case: %s", test_case.image_tag)

        page = traces.meta.page + 1
        if page > traces.meta.total_pages:
            break

    logger.info("Running %d test cases", len(tests))
    if not dry_run:
        TestCaseRunner().eval_test_case.for_each(
            tests.values(),
            kwargs={"agent_name": agent_name},
            ignore_exceptions=True,
        )


@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={
        ui_volume_mnt.as_posix(): ui_volume,
    },
)
def eval_ui_csv(
    matrix: str = "anthropic/claude-sonnet-4,qwen/qwen3-coder",
    csv_path: str = "92_prompts.csv",
    template_path: str = "vite-base.bundle",
    count: int | None = None,
) -> None:
    setup_logging(logging.INFO, extra_modules=["httpx"])

    csv_mount = ui_volume_mnt / csv_path
    if not csv_mount.is_file():
        raise ValueError("csv_path is invalid")
    template_mount = ui_volume_mnt / template_path
    if not template_mount.is_file():
        raise ValueError("template_path is invalid")
    matrix_keys = matrix.split(",")

    def iter_test_cases() -> Iterator[UiTestCase]:
        logger.info("Loading test cases: %s", csv_path)
        with csv_mount.open("r") as csv_file:
            for row in itertools.islice(csv.DictReader(csv_file), count):
                for matrix_key in matrix_keys:
                    yield UiTestCase(
                        user_prompt=row["prompt"],
                        template_repo_bundle=template_mount,
                        matrix_key=matrix_key,
                    )

    TestCaseRunner().eval_ui_test_case.for_each(
        iter_test_cases(),
        ignore_exceptions=True,
    )


@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={
        ui_volume_mnt.as_posix(): ui_volume,
    },
)
def eval_ui_prompt(
    prompt: str,
    matrix: str = "anthropic/claude-sonnet-4,qwen/qwen3-coder",
    template_path: str = "vite-base.bundle",
    repeats: int = 1,
) -> None:
    setup_logging(logging.INFO, extra_modules=["httpx"])

    template_mount = ui_volume_mnt / template_path
    if not template_mount.is_file():
        raise ValueError("template_path is invalid")
    matrix_keys = matrix.split(",")

    def iter_test_cases() -> Iterator[UiTestCase]:
        for matrix_key in matrix_keys:
            for _ in range(repeats):
                yield UiTestCase(
                    user_prompt=prompt,
                    template_repo_bundle=template_mount,
                    matrix_key=matrix_key,
                )

    TestCaseRunner().eval_ui_test_case.for_each(
        iter_test_cases(),
        ignore_exceptions=True,
    )
