import asyncio
import functools
import json
import logging
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import time
from typing import Annotated, Literal, TypeAlias

import aiofile
import fastapi
import modal
import pygit2
from async_lru import alru_cache
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langfuse import get_client
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sse_starlette import EventSourceResponse, ServerSentEvent

from relace_agent.agents import UserAgent
from relace_agent.context import ChatHistory, Context
from relace_agent.errors import BuildError, TestError
from relace_agent.logging import setup_logging
from relace_agent.repo import Repo
from relace_agent.retrieval import (
    FileChunk,
    RepoEmbedder,
    RepoRetriever,
)
from relace_agent.server.database import Database
from relace_agent.server.deployment import clear_cloudflare_bucket, repo_dist
from relace_agent.server.storage import STORAGE
from relace_agent.server.types import (
    BaseEvent,
    CommittedEvent,
    DoneEvent,
    PagedResponse,
    PromptErrorEvent,
    PromptId,
    RelaceUserId,
    RepoAgentRequest,
    RepoAskRequest,
    RepoAskResponse,
    RepoCheckoutRequest,
    RepoClonedFile,
    RepoCloneResponse,
    RepoCreateFilesSource,
    RepoCreateGitSource,
    RepoCreateLegacyRequest,
    RepoCreateRequest,
    RepoGitRequest,
    RepoId,
    RepoInfo,
    RepoLogItem,
    RepoMetadata,
    RepoPullRequest,
    RepoRetrieveRequest,
    RepoRetrieveResponse,
    RepoUpdateDiff,
    RepoUpdateFiles,
    RepoUpdateGit,
    RepoUpdateRequest,
    StartedEvent,
)

logger = logging.getLogger(__name__)

# App configuration
APP_NAME = "Relace-Agent"
APP_DOMAIN = "agent.modal-origin.relace.run"

REPO_LOCK = modal.Dict.from_name("repo-lock", create_if_missing=True)
MAX_TIMEOUT = 60 * 60  # 1 hour


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    STORAGE.setup_cache()
    yield


app = modal.App(APP_NAME)
app_api = FastAPI(
    title="Relace Agent",
    description="Configurable AI code agent",
    lifespan=lifespan,
)


security = HTTPBearer(
    description="Relace API key",
)


@functools.cache
def get_db() -> Database:
    return Database.from_env()


DependsDb: TypeAlias = Annotated[Database, Depends(get_db)]


@alru_cache(maxsize=512)
async def get_user_by_api_key(relace_api_key: str) -> RelaceUserId:
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
) -> RelaceUserId:
    return await get_user_by_api_key(credentials.credentials)


DependsUser: TypeAlias = Annotated[RelaceUserId, Depends(get_user)]


@alru_cache(maxsize=512)
async def validate_repo(user_id: RelaceUserId, repo_id: RepoId) -> None:
    valid = await get_db().check_repo(user_id, repo_id)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repo {repo_id} is inaccessible",
        )


async def get_repo_id(
    user_id: DependsUser,
    repo_id: Annotated[
        RepoId,
        fastapi.Path(title="Repo UUID"),
    ],
) -> RepoId:
    await validate_repo(user_id, repo_id)
    return repo_id


DependsRepoId: TypeAlias = Annotated[RepoId, Depends(get_repo_id)]


@asynccontextmanager
async def checkout_repo(
    repo_id: RepoId,
    release_after: int = MAX_TIMEOUT,
) -> AsyncIterator[Repo]:
    locked_until = await REPO_LOCK.get.aio(repo_id)
    if locked_until and locked_until > time():
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Repo {repo_id} is currently being accessed by another request.",
        )

    # NOTE: This is currently prone to races between get and put. This can be fixed by
    # putting some unique ID with the timestamp, and then checking that the ID matches
    # after the put. This should be very unlikely, so it's probably not worth the
    # added latency and complexity right now.
    await REPO_LOCK.put.aio(repo_id, time() + release_after)

    try:
        logger.info("Acquired lock for repo %s", repo_id)
        repo = Repo.from_repo_id(repo_id)
        repo_head = await get_db().get_repo_head(repo_id)
        await repo.init_cache(head=repo_head)
        yield repo
    finally:
        await REPO_LOCK.pop.aio(repo_id)
        logger.info("Released lock for repo %s", repo_id)


async def get_repo(repo_id: DependsRepoId) -> AsyncIterator[Repo]:
    async with checkout_repo(repo_id) as repo:
        yield repo


DependsRepo: TypeAlias = Annotated[Repo, Depends(get_repo)]
DependsRepoRetriever: TypeAlias = Annotated[RepoRetriever, Depends(RepoRetriever)]
DependsRepoEmbedder: TypeAlias = Annotated[RepoEmbedder, Depends(RepoEmbedder)]


@app_api.post("/repo")
async def create_repo(
    db: DependsDb,
    user_id: DependsUser,
    repo_embedder: DependsRepoEmbedder,
    request: RepoCreateRequest | RepoCreateLegacyRequest,
    background_tasks: BackgroundTasks,
) -> RepoInfo:
    """Create a new repository from the provided template."""
    repo_id: RepoId = uuid.uuid4()
    logger.info("Creating new repo: %s, %s", repo_id, request)
    repo = Repo.from_repo_id(repo_id)
    # Convert Swayable legacy format to new format
    if isinstance(request, RepoCreateLegacyRequest):
        request = request.convert()

    remote: str | None = None
    remote_branch: str | None = None
    repo_chunks = []

    # TODO: Switch to exhaustive match on source type
    if isinstance(request.source, RepoCreateGitSource):
        remote = str(request.source.url)
        remote_branch = request.source.branch
        await repo.clone(
            user_id=user_id,
            url=str(request.source.url),
            branch=request.source.branch,
        )
        repo_chunks = [chunk async for chunk in FileChunk.iter_repo(repo)]
    elif isinstance(request.source, RepoCreateFilesSource):
        await repo.init()
        for file in request.source.files:
            file_path = repo.root_path / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofile.async_open(file_path, "w") as writer:
                await writer.write(file.content)
            repo_chunks.extend(
                FileChunk.iter_text(
                    file.filename.as_posix(),
                    file.content,
                )
            )
            repo_create_file_size.record(len(file.content))
        repo_create_file_count.add(len(request.source.files))
        await repo.commit("Initial commit", allow_empty=True)
    elif request.source is None:
        await repo.init()
        await repo.commit("Initial commit", allow_empty=True)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid repository source format",
        )

    # Store artifacts
    repo_info = repo.info()
    await db.insert_repo(
        repo_id,
        user_id,
        metadata=request.metadata,
        remote=remote,
        remote_branch=remote_branch,
        head=repo_info.repo_head,
    )
    repo.bundle_path.parent.mkdir()
    await repo.push_bundle()

    # Process embeddings
    await repo_embedder.upsert_chunks(repo, repo_chunks)
    background_tasks.add_task(repo_embedder.embed_chunks, repo, repo_chunks)

    return repo_info


@app_api.get("/repo", response_model_exclude_none=True)
async def list_repo_metadata(
    db: DependsDb,
    user_id: DependsUser,
    filter_metadata: Annotated[str | None, Query()] = None,
    created_after: Annotated[datetime | None, Query()] = None,
    created_before: Annotated[datetime | None, Query()] = None,
    order_by: Annotated[Literal["created_at", "updated_at"], Query()] = "created_at",
    order_descending: Annotated[bool, Query()] = False,
    page_start: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=100)] = 100,
) -> PagedResponse[RepoMetadata]:
    """Get metadata for all repositories owned by the user."""
    parsed_metadata: dict[str, str] | None = None
    if filter_metadata:
        try:
            parsed_metadata = json.loads(filter_metadata)
        except json.JSONDecodeError:
            pass
        if not isinstance(parsed_metadata, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="filter_metadata must be a JSON object",
            )
    return await db.list_repos(
        user_id,
        filter_metadata=parsed_metadata,
        created_before=created_before,
        created_after=created_after,
        order_by=order_by,
        order_descending=order_descending,
        page_start=page_start,
        page_size=page_size,
    )


@app_api.get("/repo/{repo_id}", response_model_exclude_none=True)
async def get_repo_metadata(
    db: DependsDb,
    repo_id: DependsRepoId,
) -> RepoMetadata:
    """Get metadata for a single repository."""
    return await db.get_repo_metadata(repo_id)


@app_api.post("/repo/{repo_id}/update")
async def update_repo_contents(
    db: DependsDb,
    repo: DependsRepo,
    repo_embedder: DependsRepoEmbedder,
    request: RepoUpdateRequest,
    background_tasks: BackgroundTasks,
) -> RepoInfo:
    """Update the contents of a repository."""
    # TODO: Throw error if any filenames appear more than once in the request
    file_deletes: set[Path] = set()
    file_writes: dict[Path, str] = {}
    file_renames: dict[Path, Path] = {}

    # Extract file changes from the various request formats
    if isinstance(request.source, RepoUpdateFiles):
        stale_files = {
            path.relative_to(repo.root_path)
            for path in repo.list_tracked_files(ignore_tracked=())
        }
        for f in request.source.files:
            stale_files.discard(f.filename)
            file_writes[f.filename] = f.content
        file_deletes.update(stale_files)
    elif isinstance(request.source, RepoUpdateDiff):
        for operation in request.source.operations:
            if operation.type == "write":
                file_writes[operation.filename] = operation.content
            elif operation.type == "delete":
                file_deletes.add(operation.filename)
            elif operation.type == "rename":
                file_renames[operation.old_filename] = operation.new_filename
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid operation type: {operation.type}",
                )
    elif isinstance(request.source, RepoUpdateGit):
        # TODO: Support git updates
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Git updates are not yet supported",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request type",
        )

    # Process deletions
    for path in file_deletes:
        logger.info("Deleting %s", path)
        file_path = repo.root_path / path
        if file_path.is_dir():
            shutil.rmtree(file_path)
        elif file_path.is_file():
            file_path.unlink()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {path} does not exist",
            )
    # Process writes
    write_chunks: list[FileChunk] = []
    for path, content in file_writes.items():
        logger.info("Writing %s", path)
        file_path = repo.root_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofile.async_open(file_path, "w") as writer:
            await writer.write(content)
        write_chunks.extend(FileChunk.iter_text(path.as_posix(), content))
    # Process renames
    for path_old, path_new in file_renames.items():
        logger.info("Renaming %s to %s", path_old, path_new)
        old_file_path = repo.root_path / path_old
        new_file_path = repo.root_path / path_new
        if not old_file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {path_old} does not exist",
            )
        old_file_path.rename(new_file_path)

    # Process embeddings
    await repo_embedder.delete_paths(repo, file_deletes)
    await repo_embedder.rename_paths(repo, file_renames)
    await repo_embedder.upsert_chunks(repo, write_chunks)
    background_tasks.add_task(repo_embedder.embed_chunks, repo, write_chunks)

    if head := await repo.commit("Update repository contents"):
        await repo.push_bundle()
        await db.update_repo_head(repo.repo_id, head)
    return repo.info()


@app_api.delete("/repo/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_repo(
    db: DependsDb,
    repo_id: DependsRepoId,
    repo_embedder: DependsRepoEmbedder,
) -> None:
    """Delete a repository and its associated data."""
    await repo_embedder.delete_all(repo_id)
    # NOTE: We need to delete individual files in persistent storage, as we are using
    # S3 as the backend, which does not support directory deletion. Eventually we should
    # use the S3 API to properly delete all files under the repo_id prefix.
    for repo_file in [
        STORAGE.repo_bundle(repo_id),
        STORAGE.repo_history(repo_id),
    ]:
        logger.info("Deleting %s", repo_file)
        await asyncio.to_thread(repo_file.unlink, missing_ok=True)

    cache = STORAGE.repo_cache(repo_id)
    if cache.is_dir():
        logger.info("Deleting cache %s", cache)
        await asyncio.to_thread(shutil.rmtree, cache)

    await clear_cloudflare_bucket(repo_dist(repo_id))
    await db.delete_repo(repo_id)


@app_api.put(
    "/repo/{repo_id}/file/{file_path:path}",
    responses={
        status.HTTP_200_OK: {"model": RepoInfo},
        status.HTTP_201_CREATED: {"model": RepoInfo},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "required": True,
        }
    },
)
async def upload_file(
    db: DependsDb,
    repo: DependsRepo,
    repo_embedder: DependsRepoEmbedder,
    file_path: str,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
) -> RepoInfo:
    """Write a file to a repository.

    Automatically commits the change and returns the repo info with the updated head.
    """
    local_path = repo.root_path / file_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Indicate whether file is created or modified
    if not local_path.exists():
        response.status_code = status.HTTP_201_CREATED

    # Write content from stream
    async with aiofile.async_open(local_path, "wb") as writer:
        async for chunk in request.stream():
            await writer.write(chunk)

    # Process embeddings
    repo_chunks = [
        chunk async for chunk in FileChunk.iter_file(local_path, repo.root_path)
    ]
    await repo_embedder.upsert_chunks(repo, repo_chunks)
    background_tasks.add_task(repo_embedder.embed_chunks, repo, repo_chunks)

    # TODO: Validate checksum
    if head := await repo.commit(f"Write {file_path}", [file_path]):
        await repo.push_bundle()
        await db.update_repo_head(repo.repo_id, head)
    return repo.info()


@app_api.get("/repo/{repo_id}/file/{file_path:path}")
def download_file(
    repo: DependsRepo,
    file_path: str,
) -> FileResponse:
    """Read a file from a repository."""
    local_path = repo.root_path / file_path
    if not local_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_path} does not exist",
        )
    return FileResponse(path=local_path, filename=file_path)


@app_api.delete("/repo/{repo_id}/file/{file_path:path}")
async def delete_file(
    db: DependsDb,
    repo: DependsRepo,
    repo_embedder: DependsRepoEmbedder,
    file_path: str,
) -> RepoInfo:
    """Delete a file from a repository.

    Automatically commits the change and returns the repo info with the updated head.
    """
    local_path = repo.root_path / file_path
    if local_path.is_dir():
        shutil.rmtree(local_path)
    elif local_path.is_file():
        local_path.unlink()
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_path} does not exist",
        )
    await repo_embedder.delete_paths(repo, [local_path])

    if head := await repo.commit(f"Delete {file_path}", [file_path]):
        await repo.push_bundle()
        await db.update_repo_head(repo.repo_id, head)

    return repo.info()


# TODO: Merge into the update route
@app_api.patch("/repo/{repo_id}/pull", include_in_schema=False)
async def pull_remote(
    db: DependsDb,
    repo: DependsRepo,
    user_id: DependsUser,
    request: RepoPullRequest,
) -> RepoInfo:
    """Pull changes from the remote repository and merge into the current branch."""
    remote = await db.get_repo_remote(repo.repo_id)
    if not remote.remote:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Repository has no remote configured",
        )
    await repo.pull_remote(
        user_id=user_id,
        remote=remote.remote,
        remote_branch=remote.remote_branch,
        new_branch=request.new_branch,
    )
    repo_info = repo.info()
    await repo.push_bundle()
    await db.update_repo_head(repo.repo_id, repo_info.repo_head)
    return repo_info


@app_api.patch("/repo/{repo_id}/checkout", include_in_schema=False)
async def checkout_commit(
    db: DependsDb,
    repo: DependsRepo,
    request: RepoCheckoutRequest,
) -> RepoInfo:
    """Checkout a particular commit hash.

    Returns the repo info with the updated head.
    """
    old_head = repo.get_head()
    await repo.checkout(request.repo_head)
    new_info = repo.info()
    if new_info.repo_head == old_head:
        logger.info("Repo head is unchanged: %s", new_info.repo_head)
        return new_info

    await repo.push_bundle()
    await db.update_repo_head(repo.repo_id, new_info.repo_head)

    context = Context(repo=repo)
    try:
        await context.run_build()
        await context.run_test()
        await context.run_deploy()
    except BuildError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Build failed, skipping deployment",
        ) from e
    except TestError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tests failed, skipping deployment",
        ) from e

    return new_info


@app_api.get("/repo/{repo_id}/site", include_in_schema=False)
def visit_deployed_site(
    repo_id: DependsRepoId,
) -> RedirectResponse:
    return RedirectResponse(f"https://{repo_id}.dev-server.relace.run")


@app_api.post("/repo/{repo_id}/agent", include_in_schema=False)
async def run_agent(
    db: DependsDb,
    user_id: DependsUser,
    repo_id: DependsRepoId,
    request: RepoAgentRequest,
) -> EventSourceResponse:
    """Run a user defined agent from the template repo configuration.

    Updates (e.g. agent outputs) are surfaced in real time as Server-Sent Events.
    """
    prompt_id: PromptId = uuid.uuid4()
    logger.info("User ID: %s", user_id)
    logger.info("Repo ID: %s", repo_id)
    logger.info("Prompt ID: %s", prompt_id)
    logger.info("Request: %s", request)

    stack = AsyncExitStack()
    # Checkout repo immediately to acquire the lock
    repo = await stack.enter_async_context(checkout_repo(repo_id))

    async def event_stream() -> AsyncIterator[BaseEvent]:
        async with stack:
            await db.insert_prompt(prompt_id, repo_id, request)
            langfuse = get_client()
            langfuse_span = stack.enter_context(
                langfuse.start_as_current_span(
                    name=run_agent.__name__,
                    metadata={"prompt_id": prompt_id},
                    input=request,
                )
            )
            langfuse_span.update_trace(
                user_id=user_id,
                session_id=str(repo_id),
            )

            try:
                context = Context(
                    inputs=request.agent_inputs,
                    repo=repo,
                    history=stack.enter_context(
                        ChatHistory.persistent(STORAGE.repo_history(repo_id))
                    ),
                )
                agent = UserAgent.from_context(
                    context,
                    request.agent_name,
                    request.overrides,
                )
                yield StartedEvent(prompt_id=prompt_id)
                async for event in agent.run(context):
                    yield event
            except GeneratorExit:
                await db.update_prompt_status(prompt_id, status="cancelled")
                await db.insert_event(
                    prompt_id, PromptErrorEvent(error="Client disconnected")
                )
                langfuse_span.update(output="cancelled")
                raise
            except Exception as exc:
                await db.update_prompt_status(prompt_id, status="failed")
                langfuse_span.update(output=exc)
                yield PromptErrorEvent.from_exc(exc)
                yield DoneEvent()
                raise
            else:
                await db.update_prompt_status(prompt_id, status="succeeded")
                langfuse_span.update(output="succeeded")
                yield DoneEvent()

    async def event_stream_sse() -> AsyncIterator[ServerSentEvent]:
        async for event in event_stream():
            # TODO: Move this to be closer to the bundle push so the logic is easier
            # to trace. We should probably move the commit/build/test logic out of the
            # Agent class to accomplish this.
            if isinstance(event, CommittedEvent):
                await db.update_repo_head(repo.repo_id, event.repo_head)

            await db.insert_event(prompt_id, event)
            yield event.to_sse()

    return EventSourceResponse(event_stream_sse())


@app_api.get(
    "/repo/{repo_id}/chat",
    response_model_exclude_none=True,
    include_in_schema=False,
)
async def get_chat_log(
    db: DependsDb,
    repo_id: DependsRepoId,
    created_after: Annotated[datetime | None, Query()] = None,
    created_before: Annotated[datetime | None, Query()] = None,
    page_start: Annotated[int, Query(ge=0, description="Page start index")] = 0,
    page_size: Annotated[
        int, Query(ge=1, le=100, description="Maximum results per page")
    ] = 100,
) -> PagedResponse[RepoLogItem]:
    """Retrieve the event log for a repository, including user prompts."""
    return await db.list_events(
        repo_id,
        created_before=created_before,
        created_after=created_after,
        page_start=page_start,
        page_size=page_size,
    )


@app_api.post(
    "/repo/{repo_id}/git",
    include_in_schema=False,
    response_class=PlainTextResponse,
)
async def git(
    repo: DependsRepo,
    request: RepoGitRequest,
) -> str:
    result = await repo.git(*request.args, check=False)
    return (
        f">>>CODE: {result.returncode}\n"
        f">>>STDOUT:\n{result.stdout}\n"
        f">>>STDERR:\n{result.stderr}\n"
    )


@app_api.post("/repo/{repo_id}/retrieve")
async def retrieve_relevant_content(
    repo: DependsRepo,
    repo_retriever: DependsRepoRetriever,
    request: RepoRetrieveRequest,
) -> RepoRetrieveResponse:
    """Retrieve relevant content from a repository based on a query."""
    return RepoRetrieveResponse(
        results=await repo_retriever.retrieve_files(
            repo,
            query=request.query,
            score_threshold=request.score_threshold,
            token_limit=request.token_limit,
        ),
    )


@app_api.post("/repo/{repo_id}/ask")
async def ask_question(
    repo: DependsRepo,
    repo_retriever: DependsRepoRetriever,
    request: RepoAskRequest,
) -> RepoAskResponse:
    """Ask a question about the repository and receive a natural language response."""
    return RepoAskResponse(
        answer=await repo_retriever.ask_question(repo, request.query),
    )


@app_api.get("/repo/{repo_id}/clone", response_model=RepoCloneResponse)
async def clone_repo(
    repo: DependsRepo,
    commit: str | None = None,
    as_bundle: bool = False,
) -> RepoCloneResponse | FileResponse:
    """Return all readable tracked files in a repository.

    If a `commit` is provided, read file contents from that commit; otherwise
    read from the working directory.
    """

    # If bundle requested, return the bundle file (ensure it exists)
    if as_bundle:
        # If a specific commit is requested, create a temporary bundle for that commit
        if commit:
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bundle")
                tmp_path = tmp.name
                await repo.git("bundle", "create", tmp_path, commit)
                return FileResponse(
                    path=tmp_path, filename=f"{repo.repo_id}-{commit}.bundle"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to prepare bundle for commit {commit}",
                ) from e

        # Otherwise, return the persistent repo bundle
        return FileResponse(path=repo.bundle_path, filename=str(repo.repo_id))

    files: list[dict[str, str]] = []

    if commit:
        # Read files from the specified commit using pygit2 tree traversal
        try:
            git_repo = repo._repo()
            obj_any = git_repo.revparse_single(commit)
            if isinstance(obj_any, pygit2.Commit):
                commit_obj = obj_any
            else:
                peeled_any = obj_any.peel(None)
                if not isinstance(peeled_any, pygit2.Commit):
                    raise ValueError("Provided reference is not a commit")
                commit_obj = peeled_any
            tree = commit_obj.tree
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Invalid commit: {commit}",
            ) from e

        def walk_tree(prefix: str, tree_obj: "pygit2.Tree") -> None:
            for entry in tree_obj:
                name = entry.name
                path = f"{prefix}{name}" if prefix == "" else f"{prefix}/{name}"
                try:
                    obj_any = git_repo[entry.id]
                except Exception:
                    continue
                # Recurse into subtrees; add blobs
                if isinstance(obj_any, pygit2.Tree):
                    walk_tree(path, obj_any)
                elif isinstance(obj_any, pygit2.Blob):
                    try:
                        content = obj_any.data.decode("utf-8")
                    except UnicodeDecodeError:
                        # Skip non-text/binary files
                        continue
                    files.append({"filepath": path, "content": content})

        walk_tree("", tree)
    else:
        # Read files from working directory using tracked files list
        for abs_path in repo.list_tracked_files():
            try:
                if not abs_path.is_file():
                    continue

                rel_path = abs_path.relative_to(repo.root_path)
                content = abs_path.read_text(encoding="utf-8")
                files.append({"filepath": str(rel_path), "content": content})

            except UnicodeDecodeError:
                # Skip non-text/binary files
                continue
            except Exception:
                # Best-effort: skip files that error during read
                continue

    return RepoCloneResponse(
        files=[
            RepoClonedFile(
                filename=f["filepath"],
                content=f["content"],
            )
            for f in files
        ],
    )


image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "build-essential", "clang", "curl")
    .run_commands(
        # Install Node.js 22 using official Node.js repository
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g vite",
        # Configure git
        "git config --global user.name 'Relace Agent'",
        "git config --global user.email 'noreply@relace.ai'",
    )
    .uv_sync(extras=["agent"])
    .run_commands("playwright install --with-deps --only-shell")
    .add_local_file("./relace.yaml", "/relace.yaml")
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("langfuse-secret"),
        modal.Secret.from_name("relace-key-agent"),
        modal.Secret.from_name("relace-agent-supabase"),
        modal.Secret.from_name("relace-agent-github-app"),
        modal.Secret.from_name("relace-agent-open-router"),
        modal.Secret.from_name("cloudflare-secret"),
        modal.Secret.from_name("turbopuffer-secret"),
        modal.Secret.from_name("grafana-otel-secret"),
    ],
    volumes=STORAGE.mounts(),
    timeout=MAX_TIMEOUT,
    scaledown_window=60 * 15,  # 15 minutes
    cloud="aws",
    region="us-east-1",
)
@modal.concurrent(max_inputs=32)
@modal.asgi_app(custom_domains=[APP_DOMAIN])
def fastapi_app() -> FastAPI:
    setup_logging(extra_modules=["httpx"])
    setup_telemetry()
    return app_api


def setup_telemetry() -> None:
    # Set up metrics
    metric_exporter = OTLPMetricExporter()
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Set up tracing
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    FastAPIInstrumentor.instrument_app(app_api)
    HTTPXClientInstrumentor().instrument()
    AsyncioInstrumentor().instrument()


# Define OpenTelemetry metrics
meter = metrics.get_meter(__name__)
repo_create_file_count = meter.create_counter(
    name="repo_create_file_count",
    unit="1",
)
repo_create_file_size = meter.create_histogram(
    name="repo_create_file_size",
    unit="bytes",
)
