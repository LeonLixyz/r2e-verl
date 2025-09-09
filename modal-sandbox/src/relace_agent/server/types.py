from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl
from sse_starlette import ServerSentEvent

RelaceApiKey: TypeAlias = str
RelaceUserId: TypeAlias = str
RepoId: TypeAlias = UUID
PromptId: TypeAlias = UUID
PromptStatus: TypeAlias = Literal["pending", "succeeded", "cancelled", "failed"]


class PagedResponse[T](BaseModel):
    items: list[T]
    total_items: int
    next_page: int | None = None


class File(BaseModel):
    filename: Path
    content: str

    def __repr__(self) -> str:
        # Omit content to reduce log noise
        items = self.model_dump(exclude={"content"})
        items["content"] = f"<{len(self.content)} chars>"
        return f"{self.__class__.__name__}({items})"


class RepoUpdateFiles(BaseModel):
    type: Literal["files"]
    files: list[File]


class FileWriteOperation(BaseModel):
    type: Literal["write"]
    filename: Path
    content: str


class FileDeleteOperation(BaseModel):
    type: Literal["delete"]
    filename: Path


class FileRenameOperation(BaseModel):
    type: Literal["rename"]
    old_filename: Path
    new_filename: Path


class RepoUpdateDiff(BaseModel):
    type: Literal["diff"]
    operations: list[
        Annotated[
            FileWriteOperation | FileDeleteOperation | FileRenameOperation,
            Field(discriminator="type"),
        ]
    ]


class RepoUpdateGit(BaseModel):
    type: Literal["git"]
    url: HttpUrl
    branch: str | None = None


# TODO: Support metadata updates
class RepoUpdateRequest(BaseModel):
    source: Annotated[
        RepoUpdateFiles | RepoUpdateDiff | RepoUpdateGit,
        Field(discriminator="type"),
    ]


# TODO: Remove this after migrating Swayable
class RepoCreateLegacyRequest(BaseModel):
    template: str = "squack-io/vite-base"
    template_branch: str | None = None
    metadata: dict[str, str] | None = None
    description: Annotated[str | None, Field(deprecated="Use metadata instead")] = None

    def convert(self) -> RepoCreateRequest:
        return RepoCreateRequest(
            source=RepoCreateGitSource(
                type="git",
                url=HttpUrl(f"https://github.com/{self.template}"),
                branch=self.template_branch,
            ),
            metadata=self.metadata,
        )


# TODO: Support creating from an existing Relace repo
class RepoCreateRequest(BaseModel):
    source: Annotated[
        RepoCreateGitSource | RepoCreateFilesSource | None,
        Field(discriminator="type"),
    ] = None
    metadata: dict[str, str] | None = None


class RepoCreateGitSource(BaseModel):
    type: Literal["git"]
    url: HttpUrl
    branch: str | None = None


class RepoCreateFilesSource(BaseModel):
    type: Literal["files"]
    files: list[File]


# TODO: Only include repo_id on creation (not on updates)
# TODO: Consider merging this into RepoMetadata now that the head is stored in Supabase
class RepoInfo(BaseModel):
    repo_id: RepoId
    repo_head: str


class RepoMetadata(BaseModel):
    repo_id: RepoId
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, str] | None = None

    @classmethod
    def from_row(
        cls,
        id: RepoId,
        created_at: str,
        updated_at: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        return cls(
            repo_id=id,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
            metadata=metadata,
        )


class RepoRemote(BaseModel):
    remote: str | None = None
    remote_branch: str | None = None


# TODO: Add force/overwrite option
class RepoPullRequest(BaseModel):
    new_branch: str | None = None


class RepoCheckoutRequest(BaseModel):
    repo_head: str


class RepoAgentRequest(BaseModel):
    agent_name: str
    agent_inputs: dict[str, str]
    overrides: dict[str, Any] | None = None


class RepoLogItem(BaseModel):
    timestamp: datetime
    event_type: Literal[
        "user_prompt",
        "prompt_error",
        "started",
        "agent",
        "tool",
        "committed",
        "build",
        "test",
        "deployed",
        "done",
    ]
    event: (
        RepoAgentRequest
        | PromptErrorEvent
        | StartedEvent
        | AgentEvent
        | ToolEvent
        | CommittedEvent
        | BuildEvent
        | TestEvent
        | DeployedEvent
        | DoneEvent
    )

    @classmethod
    def from_row(
        cls,
        event_type: str,
        event_data: dict[str, Any],
        created_at: str,
    ) -> Self:
        timestamp = datetime.fromisoformat(created_at)
        if event_type == "user_prompt":
            return cls(
                timestamp=timestamp,
                event_type=event_type,  # type: ignore
                event=RepoAgentRequest(**event_data),
            )
        for event_class in (
            PromptErrorEvent,
            StartedEvent,
            AgentEvent,
            ToolEvent,
            CommittedEvent,
            BuildEvent,
            TestEvent,
            DeployedEvent,
            DoneEvent,
        ):
            if event_class.event_type == event_type:
                return cls(
                    timestamp=timestamp,
                    event_type=event_type,  # type: ignore
                    event=event_class(**event_data),
                )
        raise ValueError(f"Unexpected event type: {event_type}")


class RepoGitRequest(BaseModel):
    args: list[str]


class RepoRetrieveRequest(BaseModel):
    query: str
    filter: str | None = None
    include_content: bool = False
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    token_limit: int = Field(default=30_000, ge=0)


class RepoRetrieveResult(BaseModel):
    filename: str
    score: float
    content: str | None = None


class RepoRetrieveResponse(BaseModel):
    results: list[RepoRetrieveResult]


class RepoAskRequest(BaseModel):
    query: str


class RepoAskResponse(BaseModel):
    answer: str


class RepoClonedFile(BaseModel):
    """Represents a cloned file with its path and full content."""

    filename: str
    content: str


class RepoCloneResponse(BaseModel):
    """Response containing all readable files in a repository."""

    files: list[RepoClonedFile] = Field(default_factory=list)


# ==========
# SSE Events
# ==========


class BaseEvent(BaseModel):
    event_type: ClassVar[str | None] = None

    def to_sse(self) -> ServerSentEvent:
        return ServerSentEvent(
            event=self.event_type,
            data=self.model_dump_json(exclude_none=True),
        )


# TODO: Rename events for consistency
class PromptErrorEvent(BaseEvent):
    event_type: ClassVar[str | None] = "prompt_error"

    error: str

    @classmethod
    def from_exc(cls, exc: BaseException) -> Self:
        return cls(error=f"{exc.__class__.__name__}: {exc!s}")


class StartedEvent(BaseEvent):
    event_type: ClassVar[str | None] = "started"

    prompt_id: PromptId


class AgentEvent(BaseEvent):
    event_type: ClassVar[str | None] = "agent"

    name: str
    content: str


class ToolEvent(BaseEvent):
    event_type: ClassVar[str | None] = "tool"

    name: str
    path: str | None = None


class CommittedEvent(BaseEvent):
    event_type: ClassVar[str | None] = "committed"

    repo_head: str


class BuildEvent(BaseEvent):
    event_type: ClassVar[str | None] = "build"

    event: Literal["start", "pass", "fail"]


class TestEvent(BaseEvent):
    event_type: ClassVar[str | None] = "test"

    event: Literal["start", "pass", "fail"]


class DeployedEvent(BaseEvent):
    event_type: ClassVar[str | None] = "deployed"

    url: str


class DoneEvent(BaseEvent):
    event_type: ClassVar[str | None] = "done"


class FilePathsRequest(BaseModel):
    file_paths: str | list[str]
