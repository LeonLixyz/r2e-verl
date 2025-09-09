import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, Literal, Self, cast

import supabase
from opentelemetry import trace
from postgrest import CountMethod
from pydantic import BaseModel

from relace_agent.server.types import (
    BaseEvent,
    PagedResponse,
    PromptId,
    PromptStatus,
    RelaceApiKey,
    RelaceUserId,
    RepoId,
    RepoLogItem,
    RepoMetadata,
    RepoRemote,
)

tracer = trace.get_tracer(__name__)


@dataclass
class Database:
    agent_client: supabase.AsyncClient
    user_client: supabase.AsyncClient

    repos_table: ClassVar[str] = "repos"
    prompts_table: ClassVar[str] = "prompts"
    events_table: ClassVar[str] = "events"
    user_table: ClassVar[str] = "api_keys"
    user_info_table: ClassVar[str] = "users"
    chat_view: ClassVar[str] = "chat_log"

    @classmethod
    def from_env(cls) -> Self:
        return cls(
            agent_client=supabase.AsyncClient(
                supabase_url=os.environ["RELACE_AGENT_SUPABASE_URL"],
                supabase_key=os.environ["RELACE_AGENT_SUPABASE_KEY"],
            ),
            user_client=supabase.AsyncClient(
                supabase_url=os.environ["RELACE_SUPABASE_URL"],
                supabase_key=os.environ["RELACE_SUPABASE_KEY"],
            ),
        )

    @tracer.start_as_current_span("Database.get_user_id")
    async def get_user_id(self, relace_api_key: RelaceApiKey) -> RelaceUserId | None:
        result = await (
            self.user_client.table(self.user_table)
            .select("clerk_user_id")
            .eq("api_key", relace_api_key)
            .maybe_single()
            .execute()
        )
        if not result:
            return None
        return RelaceUserId(result.data["clerk_user_id"])

    @tracer.start_as_current_span("Database.get_github_installation_id")
    async def get_github_installation_id(self, user_id: RelaceUserId) -> str | None:
        result = await (
            self.user_client.table(self.user_info_table)
            .select("github_installation_id")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        if not result:
            return None
        return cast(str | None, result.data["github_installation_id"])

    @tracer.start_as_current_span("Database.insert_repo")
    async def insert_repo(
        self,
        repo_id: RepoId,
        user_id: RelaceUserId,
        metadata: dict[str, str] | None,
        remote: str | None,
        remote_branch: str | None,
        head: str,
    ) -> None:
        (
            await self.agent_client.table(self.repos_table)
            .insert(
                {
                    "id": str(repo_id),
                    "user_id": user_id,
                    "metadata": metadata,
                    "remote": remote,
                    "remote_branch": remote_branch,
                    "head_commit_hash": head,
                }
            )
            .execute()
        )

    @tracer.start_as_current_span("Database.delete_repo")
    async def delete_repo(self, repo_id: RepoId) -> None:
        # NOTE: Deletions from the repos table cascade to the prompts table
        await (
            self.agent_client.table(self.repos_table)
            .delete()
            .eq("id", repo_id)
            .execute()
        )

    @tracer.start_as_current_span("Database.check_repo")
    async def check_repo(self, user_id: RelaceUserId, repo_id: RepoId) -> bool:
        """Checks if a particular user_id/repo_id pair is valid."""
        result = await (
            self.agent_client.table(self.repos_table)
            .select("*", count=CountMethod.exact, head=True)
            .eq("id", repo_id)
            .eq("user_id", user_id)
            .execute()
        )
        return result.count is not None and result.count > 0

    @tracer.start_as_current_span("Database.get_repo_metadata")
    async def get_repo_metadata(self, repo_id: RepoId) -> RepoMetadata:
        result = await (
            self.agent_client.table(self.repos_table)
            .select("id", "created_at", "updated_at", "metadata")
            .eq("id", repo_id)
            .single()
            .execute()
        )
        return RepoMetadata.from_row(**result.data)

    @tracer.start_as_current_span("Database.get_repo_remote")
    async def get_repo_remote(self, repo_id: RepoId) -> RepoRemote:
        result = await (
            self.agent_client.table(self.repos_table)
            .select("remote", "remote_branch")
            .eq("id", repo_id)
            .single()
            .execute()
        )
        return RepoRemote(**result.data)

    @tracer.start_as_current_span("Database.get_repo_head")
    async def get_repo_head(self, repo_id: RepoId) -> str | None:
        result = await (
            self.agent_client.table(self.repos_table)
            .select("head_commit_hash")
            .eq("id", repo_id)
            .single()
            .execute()
        )
        # TODO: Update to str (without None) after backfilling existing repos
        return cast(str | None, result.data["head_commit_hash"])

    @tracer.start_as_current_span("Database.update_repo_head")
    async def update_repo_head(self, repo_id: RepoId, head_commit_hash: str) -> None:
        await (
            self.agent_client.table(self.repos_table)
            .update(
                {
                    "head_commit_hash": head_commit_hash,
                    "updated_at": datetime.now(tz=UTC).isoformat(),
                }
            )
            .eq("id", repo_id)
            .execute()
        )

    @tracer.start_as_current_span("Database.list_repos")
    async def list_repos(
        self,
        user_id: RelaceUserId,
        filter_metadata: dict[str, str] | None,
        created_before: datetime | None,
        created_after: datetime | None,
        order_by: Literal["created_at", "updated_at"],
        order_descending: bool,
        page_start: int,
        page_size: int,
    ) -> PagedResponse[RepoMetadata]:
        page_end = page_start + page_size - 1
        query = (
            self.agent_client.table(self.repos_table)
            .select(
                "id",
                "created_at",
                "updated_at",
                "metadata",
                count=CountMethod.exact,
            )
            .eq("user_id", user_id)
            .order(order_by, desc=order_descending)
            .range(page_start, page_end)
        )
        if filter_metadata is not None:
            query = query.contains("metadata", filter_metadata)
        if created_before is not None:
            query = query.lt("created_at", created_before.isoformat())
        if created_after is not None:
            query = query.gt("created_at", created_after.isoformat())

        result = await query.execute()
        total_count = result.count or 0
        return PagedResponse(
            items=[RepoMetadata.from_row(**row) for row in result.data],
            next_page=(page_end + 1) if (page_end + 1) < total_count else None,
            total_items=total_count,
        )

    @tracer.start_as_current_span("Database.insert_prompt")
    async def insert_prompt(
        self,
        prompt_id: PromptId,
        repo_id: RepoId,
        body: BaseModel,
        status: PromptStatus = "pending",
    ) -> None:
        await (
            self.agent_client.table(self.prompts_table)
            .insert(
                {
                    "id": str(prompt_id),
                    "repo_id": str(repo_id),
                    "status": status,
                    "body": body.model_dump(mode="json", exclude_none=True),
                }
            )
            .execute()
        )

    @tracer.start_as_current_span("Database.update_prompt_status")
    async def update_prompt_status(
        self, prompt_id: PromptId, status: PromptStatus
    ) -> None:
        await (
            self.agent_client.table(self.prompts_table)
            .update({"status": status})
            .eq("id", prompt_id)
            .execute()
        )

    @tracer.start_as_current_span("Database.insert_event")
    async def insert_event(self, prompt_id: PromptId, event: BaseEvent) -> None:
        await (
            self.agent_client.table(self.events_table)
            .insert(
                {
                    "prompt_id": str(prompt_id),
                    "event_data": event.model_dump(mode="json", exclude_none=True),
                    "event_type": event.event_type,
                }
            )
            .execute()
        )

    @tracer.start_as_current_span("Database.list_events")
    async def list_events(
        self,
        repo_id: RepoId,
        created_before: datetime | None,
        created_after: datetime | None,
        page_start: int,
        page_size: int,
    ) -> PagedResponse[RepoLogItem]:
        page_end = page_start + page_size - 1
        query = (
            self.agent_client.table(self.chat_view)
            .select("event_type", "event_data", "created_at", count=CountMethod.exact)
            .eq("repo_id", repo_id)
            .order("created_at", desc=True)
            .range(page_start, page_end)
        )
        if created_before is not None:
            query = query.lt("created_at", created_before.isoformat())
        if created_after is not None:
            query = query.gt("created_at", created_after.isoformat())

        result = await query.execute()
        total_count = result.count or 0
        return PagedResponse(
            items=[RepoLogItem.from_row(**row) for row in result.data],
            next_page=(page_end + 1) if (page_end + 1) < total_count else None,
            total_items=total_count,
        )
