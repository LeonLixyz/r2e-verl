import asyncio
import json
import logging
from asyncio import Semaphore
from collections.abc import AsyncIterator

import modal
from httpx import ASGITransport, AsyncClient

from relace_agent.logging import setup_logging
from relace_agent.modal.agent_runner import app_api
from relace_agent.server.storage import STORAGE
from relace_agent.server.types import PagedResponse, RepoId, RepoMetadata

logger = logging.getLogger(__name__)

app = modal.App("Relace-Agent-Cleanup")


class LocalServer:
    """Client to interact with the FastAPI routes directly."""

    def __init__(self, api_key: str) -> None:
        self.client = AsyncClient(
            transport=ASGITransport(app=app_api),
            base_url="http://local",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def iter_repos(
        self,
        filter_metadata: dict[str, str] | None = None,
        page_start: int | None = None,
    ) -> AsyncIterator[RepoMetadata]:
        while True:
            params: dict[str, str | int] = {}
            if page_start is not None:
                params["page_start"] = page_start
            if filter_metadata:
                params["filter_metadata"] = json.dumps(filter_metadata)

            response = await self.client.get("/repo", params=params)
            response_data = PagedResponse[RepoMetadata].model_validate_json(
                response.text
            )
            for repo in response_data.items:
                yield repo

            page_start = response_data.next_page
            if page_start is None:
                break

    async def del_repo(self, repo_id: RepoId) -> None:
        await self.client.delete(f"/repo/{repo_id}")


@app.function(
    image=modal.Image.debian_slim(python_version="3.13").uv_sync(),
    secrets=[
        modal.Secret.from_name("relace-key-agent"),
        modal.Secret.from_name("relace-agent-supabase"),
        modal.Secret.from_name("turbopuffer-secret"),
        modal.Secret.from_name("cloudflare-secret"),
    ],
    volumes=STORAGE.mounts(),
    timeout=60 * 60 * 4,  # 4 hours
)
async def purge_user(
    api_key: str,
    delete: bool = False,
    parallel: int = 1,
    limit: int | None = None,
    with_metadata: str | None = None,
) -> None:
    setup_logging()

    filter_metadata: dict[str, str] | None = None
    if with_metadata:
        filter_metadata = json.loads(with_metadata)

    client = LocalServer(api_key)
    semaphore = Semaphore(parallel)

    async def delete_repo(repo: RepoMetadata) -> None:
        async with semaphore:
            logger.info("Deleting repo %s", repo)
            await client.del_repo(repo.repo_id)

    repos = [repo async for repo in client.iter_repos(filter_metadata=filter_metadata)]
    if limit is not None:
        repos = repos[:limit]
    logger.info("Collected %s repos for deletion", len(repos))

    tasks = []
    for repo in repos:
        if delete:
            tasks.append(asyncio.create_task(delete_repo(repo)))

    if tasks:
        await asyncio.gather(*tasks)
        logger.info("Deleted %s repos", len(tasks))
