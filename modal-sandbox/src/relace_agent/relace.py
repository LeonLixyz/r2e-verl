import asyncio
import difflib
import logging
import os
from collections.abc import AsyncIterator, Iterable
from itertools import batched
from pathlib import Path
from typing import Protocol, final

import httpx
from httpx import AsyncClient
from langfuse import get_client, observe
from pydantic import BaseModel

logger = logging.getLogger(__file__)


DEFAULT_EMBEDDINGS_MODEL = "relace-embed-v1"


# TODO: Replace with async file wrappers
class PathLike(Protocol):
    def exists(self) -> bool: ...
    def read_text(self) -> str: ...
    def write_text(self, data: str) -> int | None: ...
    def __str__(self) -> str: ...


class ApplyRequest(BaseModel):
    initialCode: str
    editSnippet: str
    instructions: str | None = None
    stream: bool | None = None


class ApplyUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ApplyResponse(BaseModel):
    mergedCode: str
    usage: ApplyUsage


class QuickEditRequest(BaseModel):
    initialCode: str
    instructions: str


class QuickEditResponse(BaseModel):
    updatedCode: str
    usage: ApplyUsage


class RankFileInput(BaseModel):
    filename: str
    code: str


class RankFileOutput(BaseModel):
    filename: str
    score: float


class RankRequest(BaseModel):
    query: str
    codebase: list[RankFileInput]
    token_limit: int | None = None


class RankUsage(BaseModel):
    total_tokens: int


class RankResponse(BaseModel):
    results: list[RankFileOutput]
    usage: RankUsage


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]


class EmbedUsage(BaseModel):
    total_tokens: int


class EmbedResult(BaseModel):
    index: int
    embedding: list[float]


class EmbedResponse(BaseModel):
    results: list[EmbedResult]
    # NOTE: Usage is omitted if the origin server returns 0 for input tokens
    # This is likely a bug in the origin server
    usage: EmbedUsage | None = None


@final
class RelaceClient:
    def __init__(
        self,
        api_key: str | None = None,
        relace_origin: str = "endpoint.relace.run",
        rank_subdomain: str = "ranker",
        apply_subdomain: str = "instantapply",
        edit_subdomain: str = "edit",
        embed_subdomain: str = "embeddings-repos",
        timeout: float = 600.0,
        max_connections: int = 32,
    ) -> None:
        self.rank_host: str = f"https://{rank_subdomain}.{relace_origin}"
        self.apply_host: str = f"https://{apply_subdomain}.{relace_origin}"
        self.edit_host: str = f"https://{edit_subdomain}.{relace_origin}"
        self.embed_host: str = f"https://{embed_subdomain}.{relace_origin}"
        self.session: AsyncClient = AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections),
        )

        # Configure API authentication
        api_key = api_key or os.environ.get("RELACE_API_KEY")
        if not api_key:
            raise ValueError("Relace API key is not configured.")
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    @observe(name="Relace.rank", as_type="generation")
    async def rank(self, request: RankRequest) -> RankResponse:
        """Call the Relace reranker API."""
        result = await self.session.post(
            f"{self.rank_host}/v2/code/rank",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        result.raise_for_status()
        result_parsed = RankResponse.model_validate_json(result.text)

        langfuse = get_client()
        langfuse.update_current_generation(
            usage_details={
                "input": result_parsed.usage.total_tokens,
            }
        )
        return result_parsed

    @observe(name="Relace.rank_files")
    async def rank_files(
        self,
        paths: Iterable[PathLike],
        query: str,
        token_limit: int | None = None,
        threshold: float = 0.0,
    ) -> list[Path]:
        """Rank the input files using the Relace reranker API.

        Returns:
            Files with a score above the threshold, ordered from highest score to lowest.
        """
        codebase: list[RankFileInput] = []
        for path in paths:
            try:
                code = path.read_text()
            except UnicodeDecodeError:
                logger.warning("Skipping non-unicode file from ranker: %s", path)
                continue
            codebase.append(RankFileInput(filename=str(path), code=code))

        request = RankRequest(
            query=query,
            codebase=codebase,
            token_limit=token_limit,
        )
        response = await self.rank(request)
        return [
            Path(file.filename) for file in response.results if file.score >= threshold
        ]

    @observe(name="Relace.apply", as_type="generation")
    async def apply(self, request: ApplyRequest) -> ApplyResponse:
        """Call the Relace fast apply API."""
        result = await self.session.post(
            f"{self.apply_host}/v1/code/apply",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        result.raise_for_status()
        result_parsed = ApplyResponse.model_validate_json(result.text)

        langfuse = get_client()
        langfuse.update_current_generation(
            usage_details={
                "input": result_parsed.usage.prompt_tokens,
                "output": result_parsed.usage.completion_tokens,
            },
            metadata={
                "output_size": len(result_parsed.mergedCode),
                "snippet_size": len(request.editSnippet),
                "snippet_ratio": len(request.editSnippet)
                / len(result_parsed.mergedCode)
                if len(result_parsed.mergedCode) > 0
                else 0,
            },
        )
        return result_parsed

    @observe(name="Relace.apply_file")
    async def apply_file(
        self,
        path: PathLike,
        edit_snippet: str,
        instructions: str | None = None,
    ) -> str:
        """Apply an edit snippet to a file using the Relace fast apply API.

        Returns:
            A unified diff of the changes made to the file.
        """
        request = ApplyRequest(
            initialCode=path.read_text(),
            editSnippet=edit_snippet,
            instructions=instructions,
        )
        response = await self.apply(request)
        diff = "".join(
            difflib.unified_diff(
                request.initialCode.splitlines(keepends=True),
                response.mergedCode.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
        )
        if diff:
            path.write_text(response.mergedCode)
        return diff

    @observe(name="Relace.quick_edit", as_type="generation")
    async def quick_edit(self, request: QuickEditRequest) -> QuickEditResponse:
        """Call the Relace fast apply API."""
        result = await self.session.post(
            f"{self.edit_host}/v1/code/edit",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        result.raise_for_status()
        result_parsed = QuickEditResponse.model_validate_json(result.text)

        langfuse = get_client()
        langfuse.update_current_generation(
            usage_details={
                "input": result_parsed.usage.prompt_tokens,
                "output": result_parsed.usage.completion_tokens,
            },
        )
        return result_parsed

    @observe(name="Relace.quick_edit_file")
    async def quick_edit_file(self, path: PathLike, instructions: str) -> str:
        """Edit a file using the Relace quick edit API.

        Returns:
            A unified diff of the changes made to the file.
        """
        request = QuickEditRequest(
            initialCode=path.read_text(),
            instructions=instructions,
        )
        response = await self.quick_edit(request)
        diff = "".join(
            difflib.unified_diff(
                request.initialCode.splitlines(keepends=True),
                response.updatedCode.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
        )
        if diff:
            path.write_text(response.updatedCode)
        return diff

    @observe(name="Relace.embed", as_type="generation")
    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Call the Relace embeddings API."""
        result = await self.session.post(
            f"{self.embed_host}/v1/code/embed",
            json=request.model_dump(mode="json", exclude_none=True),
        )
        result.raise_for_status()
        result_parsed = EmbedResponse.model_validate_json(result.text)

        langfuse = get_client()
        langfuse.update_current_generation(
            usage_details={
                "input": result_parsed.usage.total_tokens if result_parsed.usage else 0,
            },
            metadata={
                "input_size": len(result_parsed.results),
            },
        )
        return result_parsed

    @observe(name="Relace.embed_batches")
    async def embed_batches(
        self,
        inputs: Iterable[str],
        batch_size: int = 32,
        parallel_batches: int = 4,
        model: str = DEFAULT_EMBEDDINGS_MODEL,
    ) -> AsyncIterator[list[float]]:
        """Embed multiple inputs using the Relace embeddings API.

        Args:
            inputs: Iterable of input strings to embed.
            batch_size: Number of inputs per batch.
            parallel_batches: Maximum number of concurrent batches to process.
            model: The model to use for embeddings.

        Yields:
            Embedding for each input (input order preserved).
        """
        semaphore = asyncio.Semaphore(parallel_batches)

        async def process_batch(batch_data: list[str]) -> list[list[float]]:
            async with semaphore:
                request = EmbedRequest(
                    model=model,
                    input=batch_data,
                )
                response = await self.embed(request)
                return [result.embedding for result in response.results]

        # Create all batch tasks
        # TODO: Lazily consume inputs to avoid loading everything into memory
        batch_tasks = []
        for batch in batched(inputs, batch_size, strict=False):
            task = asyncio.create_task(process_batch(list(batch)))
            batch_tasks.append(task)

        # Yield results in order as they complete
        for task in batch_tasks:
            batch_embeddings = await task
            for embedding in batch_embeddings:
                yield embedding

    @observe(name="Relace.embed_text")
    async def embed_text(
        self,
        text: str,
        model: str = DEFAULT_EMBEDDINGS_MODEL,
    ) -> list[float]:
        """Embed a text using the Relace embeddings API.

        Returns:
            A list of embeddings for the input text.
        """
        request = EmbedRequest(
            model=model,
            input=text,
        )
        response = await self.embed(request)
        if len(response.results) != 1:
            raise ValueError(
                f"Expected exactly one embedding result, got {len(response.results)}"
            )

        return response.results[0].embedding
