import functools
import logging
import re
import uuid
from collections import Counter
from typing import Annotated, Any, Literal, TypeAlias, cast

import modal
from async_lru import alru_cache
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langfuse import get_client, observe
from pydantic import BaseModel

from relace_agent.errors import ToolError
from relace_agent.logging import setup_logging
from relace_agent.server.database import Database
from relace_agent.server.toolbox import BASH_TIMEOUT, SandboxTool, SandboxToolContext
from relace_agent.server.types import RelaceUserId

logger = logging.getLogger(__name__)

# App configuration
APP_NAME = "Relace-Toolbox"
APP_DOMAIN = "toolbox.modal-origin.relace.run"

SANDBOX_TIMEOUT = 60 * 60  # 1 hour

app = modal.App(APP_NAME)
app_api = FastAPI(
    title="Relace Toolbox",
    description="Server for running sandboxed AI agent tool calls",
)

security = HTTPBearer(
    description="Relace API key",
)


@functools.cache
def get_db() -> Database:
    return Database.from_env()


DependsDb: TypeAlias = Annotated[Database, Depends(get_db)]


@alru_cache
async def get_user_by_api_key(relace_api_key: str) -> str:
    user = await get_db().get_user_id(relace_api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    return await get_user_by_api_key(credentials.credentials)


DependsUser: TypeAlias = Annotated[RelaceUserId, Depends(get_user)]

SessionId: TypeAlias = uuid.UUID


GITIGNORE_GLOBAL = """
__pycache__/
*.py[cod]
.vite/
.cache/
node_modules/
build/
dist/
""".strip().replace("\n", "\\n")


async def create_sandbox(
    user_id: RelaceUserId,
    session_id: SessionId,
    image_tag: str,
    parent_app: modal.App = app,
) -> modal.Sandbox:
    if image_tag == "default":
        image = (
            modal.Image.debian_slim(python_version="3.13")
            .apt_install("git", "ripgrep")
            .workdir("/repo")
            .run_commands(
                f"printf '{GITIGNORE_GLOBAL}' > /root/.gitignore_global",
                "git config --global core.excludesfile /root/.gitignore_global",
                "git config --global user.name 'Relace Agent'",
                "git config --global user.email 'noreply@relace.ai'",
                "git init",
            )
        )
    else:
        # See https://github.com/R2E-Gym/R2E-Gym/blob/68092c5b1e5e23b2615f56978c8de22b15b070ce/src/r2egym/agenthub/runtime/docker.py#L538
        image = (
            modal.Image.from_registry(image_tag)
            .apt_install("ripgrep")
            .workdir("/testbed")
            .run_commands("chmod +x ./run_tests.sh")
            .run_commands("ln -s /r2e_tests /testbed/r2e_tests")
        )

    with modal.enable_output():
        sandbox = await modal.Sandbox.create.aio(
            image=image,
            app=parent_app,
            timeout=SANDBOX_TIMEOUT,
        )
        sandbox.set_tags(
            {
                "user_id": user_id,
                "session_id": str(session_id),
                "image": image_tag,
            }
        )
    return sandbox


async def get_sandbox(
    user_id: RelaceUserId, session_id: SessionId
) -> modal.Sandbox | None:
    async for sandbox in modal.Sandbox.list.aio(
        app_id=app.app_id,
        tags={"user_id": user_id, "session_id": str(session_id)},
    ):
        return sandbox
    return None


class CreateSessionRequest(BaseModel):
    image_tag: str = "default"


class CreateSessionResponse(BaseModel):
    session_id: SessionId


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCallResponse(BaseModel):
    output: str


class PythonExecRequest(BaseModel):
    path: str
    input: str | None = None


class PythonExecResponse(BaseModel):
    exit_code: int
    stdout: str
    stderr: str


TestResult = Literal["PASSED", "FAILED", "ERROR", "SKIPPED"]


class TestResponse(BaseModel):
    exit_code: int
    tests: dict[str, TestResult]
    counts: dict[TestResult, int]


@app_api.post("/session")
@observe()
async def create_session(
    user_id: DependsUser,
    request: CreateSessionRequest | None = None,
) -> CreateSessionResponse:
    if request is None:
        request = CreateSessionRequest()

    session_id: SessionId = uuid.uuid4()
    sandbox = await create_sandbox(user_id, session_id, request.image_tag)
    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=str(session_id),
        user_id=user_id,
        metadata={
            "sandbox_id": sandbox.object_id,
        },
    )

    return CreateSessionResponse(session_id=session_id)


@app_api.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
@observe()
async def terminate_session(
    user_id: DependsUser,
    session_id: SessionId,
) -> None:
    sandbox = await get_sandbox(user_id, session_id)
    if sandbox is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid session_id: {session_id}",
        )
    await sandbox.terminate.aio()

    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=str(session_id),
        user_id=user_id,
        metadata={
            "sandbox_id": sandbox.object_id,
        },
    )


@app_api.post("/session/{session_id}/tool")
@observe()
async def tool_call(
    user_id: DependsUser,
    session_id: SessionId,
    request: ToolCallRequest,
) -> ToolCallResponse:
    sandbox = await get_sandbox(user_id, session_id)
    if sandbox is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid session_id: {session_id}",
        )

    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=str(session_id),
        user_id=user_id,
        metadata={"sandbox_id": sandbox.object_id},
    )
    tool_context = SandboxToolContext()
    try:
        tool = SandboxTool.from_name(request.name, sandbox)
        tool_input = tool.schema.from_json_dict(request.arguments)
        tool_output = await tool.execute(tool_input, tool_context)
        await tool.commit()
    except ToolError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    return ToolCallResponse(output=tool_output)


@app_api.post("/session/{session_id}/exec")
@observe()
async def exec_python_file(
    user_id: DependsUser,
    session_id: SessionId,
    request: PythonExecRequest,
) -> PythonExecResponse:
    sandbox = await get_sandbox(user_id, session_id)
    if sandbox is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid session_id: {session_id}",
        )

    langfuse = get_client()
    langfuse.update_current_trace(
        session_id=str(session_id),
        user_id=user_id,
        metadata={"sandbox_id": sandbox.object_id},
    )

    process = await sandbox.exec.aio(
        "python",
        "-B",
        request.path,
        timeout=BASH_TIMEOUT,
    )
    if request.input:
        process.stdin.write(request.input)
        process.stdin.write_eof()
        await process.stdin.drain.aio()

    exit_code = await process.wait.aio()
    if exit_code == -1:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Python execution timed out",
        )

    return PythonExecResponse(
        exit_code=exit_code,
        stdout=await process.stdout.read.aio(),
        stderr=await process.stderr.read.aio(),
    )


_TEST_PATTERN = re.compile(r"(?P<result>PASSED|FAILED|ERROR|SKIPPED)\s(?P<name>.*)")
_ANSI_COLOR_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def parse_test_output(stdout: str, exit_code: int) -> TestResponse:
    tests: dict[str, TestResult] = {}
    counts: Counter[TestResult] = Counter()
    for line in stdout.splitlines():
        line = _ANSI_COLOR_PATTERN.sub("", line)
        match = _TEST_PATTERN.match(line)
        if match:
            test_name = match.group("name")
            test_result = cast(TestResult, match.group("result"))
            tests[test_name] = test_result
            counts[test_result] += 1

    return TestResponse(
        exit_code=exit_code,
        tests=tests,
        counts=counts,
    )


@app_api.post("/session/{session_id}/test")
@observe()
async def run_tests(
    user_id: DependsUser,
    session_id: SessionId,
) -> TestResponse:
    sandbox = await get_sandbox(user_id, session_id)
    if sandbox is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid session_id: {session_id}",
        )

    test_runner = await sandbox.exec.aio(
        "bash", "-c", "./run_tests.sh", timeout=BASH_TIMEOUT
    )
    exit_code = await test_runner.wait.aio()
    if exit_code == -1:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Test execution timed out",
        )

    # Parse test output
    test_stdout = await test_runner.stdout.read.aio()
    test_stderr = await test_runner.stderr.read.aio()
    logger.info("Test output:\nSTDOUT:\n%s\nSTDERR:\n%s", test_stdout, test_stderr)
    return parse_test_output(test_stdout, exit_code)


@app.function(
    image=modal.Image.debian_slim(python_version="3.13").uv_sync(),
    secrets=[
        modal.Secret.from_name("relace-toolbox-langfuse"),
        modal.Secret.from_name("relace-key-toolbox"),
        modal.Secret.from_name("relace-agent-supabase"),
    ],
    timeout=60 * 15,
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app(custom_domains=[APP_DOMAIN])
def fastapi_app() -> FastAPI:
    setup_logging()
    return app_api
