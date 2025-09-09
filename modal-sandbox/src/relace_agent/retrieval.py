import hashlib
import logging
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import ClassVar, Self
from uuid import UUID

import aiofile
import turbopuffer
from langfuse.openai import AsyncOpenAI
from opentelemetry import trace
from pydantic import BaseModel
from turbopuffer import AsyncTurbopuffer
from turbopuffer.types import QueryParam

from relace_agent.relace import (
    RankFileInput,
    RankRequest,
    RelaceClient,
)
from relace_agent.repo import Repo
from relace_agent.server.types import RepoId, RepoRetrieveResult

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

TURBOPUFFER_MAX_ROWS = 1200

# TODO: Check model against Turbopuffer attributes
EMBEDDING_MODEL = "relace-embed-v1"
EMBEDDING_VECTOR_DIMS = 2560
EMBEDDING_MAX_TOKENS = 32_000


@dataclass
class FileChunk:
    id: UUID
    file_path: str
    content: str
    content_hash: str

    # Convert embedding token limit to character limits
    max_chars: ClassVar[int] = int(EMBEDDING_MAX_TOKENS * 3.5)
    min_chars: ClassVar[int] = int(max_chars * 0.75)
    additional_chars: ClassVar[int] = max_chars - min_chars

    @classmethod
    def from_content(cls, file_path: str, content: str) -> Self:
        return cls(
            file_path=file_path,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            id=uuid.uuid4(),
        )

    @classmethod
    def from_lines(cls, file_path: str, lines: Iterable[str]) -> Self:
        return cls.from_content(file_path, content="".join(lines))

    @classmethod
    def iter_text(cls, file_path: str, text: str) -> Iterator[Self]:
        """Iterate over chunks of text from memory."""
        n = len(text)
        start_idx = 0
        while start_idx < n:
            first_part = min(start_idx + cls.min_chars, n)
            additional_part = min(first_part + cls.additional_chars, n)
            if additional_part == n:
                end_idx = n
            else:
                new_line_idx = text.find("\n", first_part, additional_part)
                if new_line_idx != -1:
                    end_idx = new_line_idx + 1
                else:
                    end_idx = additional_part

            chunk = text[start_idx:end_idx]
            yield cls.from_content(file_path, chunk)
            start_idx = end_idx

    @classmethod
    async def iter_file(cls, file_path: Path, root_path: Path) -> AsyncIterator[Self]:
        """Iterate over chunks of a single file asynchronously."""
        logger.info("Chunking file %s", file_path)
        rel_path = file_path.relative_to(root_path).as_posix()

        buffer: list[str] = []
        left_over: str = ""

        async with aiofile.async_open(file_path, "r", encoding="utf-8") as f:
            while True:
                min_chunk = left_over + await f.read(cls.min_chars - len(left_over))
                if not min_chunk:
                    break

                buffer.append(min_chunk)
                left_over = ""

                additional_chunk = await f.read(cls.additional_chars)

                new_line_idx = additional_chunk.find("\n")

                if new_line_idx != -1:
                    buffer.append(additional_chunk[: new_line_idx + 1])
                    left_over = additional_chunk[new_line_idx + 1 :]
                else:
                    buffer.append(additional_chunk)

                yield cls.from_lines(rel_path, buffer)
                buffer.clear()

    @classmethod
    async def iter_repo(cls, repo: Repo) -> AsyncIterator[Self]:
        """Iterate over all chunks of a repository asynchronously."""
        logger.info("Chunking repo %s", repo.repo_id)
        for file_path in repo.list_tracked_files():
            try:
                async for chunk in cls.iter_file(file_path, repo.root_path):
                    yield chunk
            except UnicodeDecodeError as e:
                logger.warning("Failed to read %s: %s", file_path, e)


@dataclass
class FileChunkEmbedding:
    id: UUID
    embedding: list[float]
    embedding_model: str


class TurbopufferRow(BaseModel):
    """Represents a row in the Turbopuffer database."""

    id: UUID
    vector: list[float]
    file_path: str
    content_hash: str
    embedding_model: str | None
    pending_update: bool

    @classmethod
    def new(cls, chunk: FileChunk) -> Self:
        return cls(
            id=chunk.id,
            vector=[0.0] * EMBEDDING_VECTOR_DIMS,
            file_path=chunk.file_path,
            content_hash=chunk.content_hash,
            embedding_model=None,
            pending_update=True,
        )


# TODO: Extract all attribute keys to constants
class TurbopufferWrapper:
    def __init__(
        self,
        turbopuffer_region: str = "aws-us-east-1",
        turbopuffer_api_key: str | None = None,
    ) -> None:
        self.tpuf_client = AsyncTurbopuffer(
            api_key=turbopuffer_api_key,
            region=turbopuffer_region,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.tpuf_client.close()

    def repo_namespace(self, repo_id: RepoId) -> str:
        return f"repo-{repo_id}"

    @tracer.start_as_current_span("TurbopufferWrapper.query")
    async def query(
        self,
        embedded_query: list[float],
        namespace: str,
        top_k: int = TURBOPUFFER_MAX_ROWS,
    ) -> list[str]:
        logger.info("Querying namespace %s with top_k=%d", namespace, top_k)
        file_path_attr = "file_path"
        tpuf_ns = self.tpuf_client.namespace(namespace)
        tpuf_queries: list[QueryParam] = [
            {
                "filters": ("pending_update", "Eq", True),
                "include_attributes": [file_path_attr],
                "top_k": top_k,
            },
            {
                "rank_by": ("vector", "ANN", embedded_query),
                "filters": ("pending_update", "Eq", False),
                "include_attributes": [file_path_attr],
                "top_k": top_k,
            },
        ]
        try:
            tpuf_response = await tpuf_ns.multi_query(queries=tpuf_queries)
        except turbopuffer.NotFoundError:
            logger.warning("Namespace %s not found", namespace)
            return []

        # Using dict for "ordered set" behavior
        unique_files: dict[str, None] = {
            getattr(row, file_path_attr): None
            for result in tpuf_response.results
            for row in (result.rows or [])
        }
        return list(unique_files.keys())

    # TODO: Avoid re-embedding existing chunks:
    # - Delete stale chunks with hashes that are no longer present
    # - Leave existing chunks with embeddings intact
    # - Write placeholders for new chunks
    # - Return list of new chunks to be embedded
    @tracer.start_as_current_span("TurbopufferWrapper.write_chunk_placeholders")
    async def write_chunk_placeholders(
        self,
        namespace: str,
        chunks: list[FileChunk],
    ) -> None:
        logger.info("Writing %s placeholders to namespace %s", len(chunks), namespace)
        tpuf_ns = self.tpuf_client.namespace(namespace)
        upsert_result = await tpuf_ns.write(
            delete_by_filter=[
                "file_path",
                "In",
                # Use set to deduplicate, convert to list for API compat
                list({chunk.file_path for chunk in chunks}),
            ],
            upsert_rows=[
                TurbopufferRow.new(chunk).model_dump(mode="json") for chunk in chunks
            ],
            schema={"id": "uuid"},
            distance_metric="cosine_distance",
        )
        logger.info("Updated namespace %s: %s", namespace, upsert_result)

    @tracer.start_as_current_span("TurbopufferWrapper.write_chunk_embeddings")
    async def write_chunk_embeddings(
        self,
        namespace: str,
        embeddings: list[FileChunkEmbedding],
    ) -> None:
        logger.info("Writing %s embeddings to namespace %s", len(embeddings), namespace)
        tpuf_ns = self.tpuf_client.namespace(namespace)
        if not await tpuf_ns.exists():
            logger.warning("Namespace %s does not exist, skipping write", namespace)
            return

        # NOTE: This is prone to races if the chunks are deleted after this query
        # This means stale embeddings may be kept alongside fresh ones until the
        # next write/delete happens

        # Handle pagination for large numbers of chunk IDs
        chunk_ids = [str(chunk.id) for chunk in embeddings]
        all_existing_rows = []

        # Process chunk IDs in batches
        batch_size = TURBOPUFFER_MAX_ROWS
        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i : i + batch_size]
            logger.info(
                "Querying batch %d/%d with %d chunk IDs",
                i // batch_size + 1,
                (len(chunk_ids) + batch_size - 1) // batch_size,
                len(batch_ids),
            )

            existing = await tpuf_ns.query(
                filters=("id", "In", batch_ids),
                rank_by=("id", "asc"),
                top_k=batch_size,
                include_attributes=True,
            )
            if existing.rows:
                all_existing_rows.extend(existing.rows)

        if not all_existing_rows:
            logger.info("Chunks not found, skipping update")
            return

        # Update existing rows with new embeddings in batches
        row_lookup = {row.id: row for row in all_existing_rows}
        row_updates = []

        for chunk in embeddings:
            chunk_row = row_lookup.get(str(chunk.id))
            if chunk_row is None:
                logger.info("Chunk %s not found, skipping update", chunk.id)
                continue

            chunk_row.vector = chunk.embedding
            chunk_row.pending_update = False  # type: ignore
            chunk_row.embedding_model = chunk.embedding_model  # type: ignore
            row_updates.append(chunk_row.model_dump(mode="json"))

        # Process updates in batches to avoid request body too large error
        batch_size = TURBOPUFFER_MAX_ROWS
        total_updated = 0

        for i in range(0, len(row_updates), batch_size):
            batch = row_updates[i : i + batch_size]
            logger.info(
                "Updating batch %d/%d with %d rows",
                i // batch_size + 1,
                (len(row_updates) + batch_size - 1) // batch_size,
                len(batch),
            )

            result = await tpuf_ns.write(
                upsert_rows=batch,
                schema={"id": "uuid"},
                distance_metric="cosine_distance",
            )
            total_updated += result.rows_affected
            logger.info("Updated batch: %s", result)

        logger.info(
            "Updated namespace %s: %d total rows updated", namespace, total_updated
        )

    @tracer.start_as_current_span("TurbopufferWrapper.rename_files")
    async def rename_files(self, namespace: str, file_renames: dict[str, str]) -> None:
        logger.info("Renaming files in namespace %s", namespace)
        tpuf_ns = self.tpuf_client.namespace(namespace)
        if not await tpuf_ns.exists():
            logger.warning("Namespace %s does not exist, skipping rename", namespace)
            return

        # Handle pagination for large numbers of file paths
        file_paths = list(file_renames.keys())
        all_existing_rows = []

        # Process file paths in batches
        batch_size = TURBOPUFFER_MAX_ROWS
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i : i + batch_size]
            logger.info(
                "Querying batch %d/%d with %d file paths",
                i // batch_size + 1,
                (len(file_paths) + batch_size - 1) // batch_size,
                len(batch_paths),
            )

            existing = await tpuf_ns.query(
                filters=("file_path", "In", batch_paths),
                top_k=batch_size,
            )
            if existing.rows:
                all_existing_rows.extend(existing.rows)

        if not all_existing_rows:
            logger.info("Chunks not found, skipping update")
            return

        # Update rows with new file paths
        result = await tpuf_ns.write(
            patch_rows=[
                {
                    "id": row.id,
                    "file_path": file_renames[row.file_path],  # type: ignore
                }
                for row in all_existing_rows
            ],
        )
        logger.info("Updated namespace %s: %s", namespace, result)

    @tracer.start_as_current_span("TurbopufferWrapper.delete_files")
    async def delete_files(self, namespace: str, file_paths: list[str]) -> None:
        logger.info("Deleting files from namespace %s", namespace)
        tpuf_ns = self.tpuf_client.namespace(namespace)
        if not await tpuf_ns.exists():
            logger.warning("Namespace %s does not exist, skipping delete", namespace)
            return

        # Handle large numbers of file paths in batches
        batch_size = TURBOPUFFER_MAX_ROWS
        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i : i + batch_size]
            logger.info(
                "Deleting batch %d/%d with %d file paths",
                i // batch_size + 1,
                (len(file_paths) + batch_size - 1) // batch_size,
                len(batch_paths),
            )

            result = await tpuf_ns.write(
                delete_by_filter=["file_path", "In", batch_paths]
            )
            logger.info("Deleted batch from namespace %s: %s", namespace, result)

    @tracer.start_as_current_span("TurbopufferWrapper.delete_namespace")
    async def delete_namespace(self, namespace: str) -> None:
        tpuf_ns = self.tpuf_client.namespace(namespace)
        if not await tpuf_ns.exists():
            logger.info("Namespace %s does not exist, skipping delete", namespace)
            return
        logger.info("Deleting namespace %s", namespace)
        await tpuf_ns.delete_all()


class RepoEmbedder:
    def __init__(self) -> None:
        self.relace = RelaceClient()

    @tracer.start_as_current_span("RepoEmbedder.upsert_chunks")
    async def upsert_chunks(self, repo: Repo, chunks: list[FileChunk]) -> None:
        if not chunks:
            logger.info("Skipping upsert_chunks for %s", repo.repo_id)
            return
        async with TurbopufferWrapper() as tpuf:
            await tpuf.write_chunk_placeholders(
                namespace=tpuf.repo_namespace(repo.repo_id),
                chunks=chunks,
            )

    @tracer.start_as_current_span("RepoEmbedder.embed_chunks")
    async def embed_chunks(self, repo: Repo, chunks: list[FileChunk]) -> None:
        if not chunks:
            logger.info("Skipping embed_chunks for %s", repo.repo_id)
            return
        chunk_embeddings: list[FileChunkEmbedding] = []
        async for embedding in self.relace.embed_batches(
            [chunk.content for chunk in chunks],
            model=EMBEDDING_MODEL,
        ):
            chunk = chunks[len(chunk_embeddings)]
            chunk_embeddings.append(
                FileChunkEmbedding(
                    id=chunk.id,
                    embedding=embedding,
                    embedding_model=EMBEDDING_MODEL,
                )
            )
        async with TurbopufferWrapper() as tpuf:
            await tpuf.write_chunk_embeddings(
                namespace=tpuf.repo_namespace(repo.repo_id),
                embeddings=chunk_embeddings,
            )

    @tracer.start_as_current_span("RepoEmbedder.rename_paths")
    async def rename_paths(self, repo: Repo, renames: dict[Path, Path]) -> None:
        if not renames:
            logger.info("Skipping rename_paths for %s", repo.repo_id)
            return
        async with TurbopufferWrapper() as tpuf:
            await tpuf.rename_files(
                namespace=tpuf.repo_namespace(repo.repo_id),
                file_renames={
                    old.as_posix(): new.as_posix() for old, new in renames.items()
                },
            )

    @tracer.start_as_current_span("RepoEmbedder.delete_paths")
    async def delete_paths(self, repo: Repo, paths: Iterable[Path]) -> None:
        if not paths:
            logger.info("Skipping delete_paths for %s", repo.repo_id)
            return
        async with TurbopufferWrapper() as tpuf:
            await tpuf.delete_files(
                namespace=tpuf.repo_namespace(repo.repo_id),
                file_paths=[path.as_posix() for path in paths],
            )

    @tracer.start_as_current_span("RepoEmbedder.delete_all")
    async def delete_all(self, repo_id: RepoId) -> None:
        async with TurbopufferWrapper() as tpuf:
            await tpuf.delete_namespace(
                namespace=tpuf.repo_namespace(repo_id),
            )


class RepoRetriever:
    def __init__(self) -> None:
        self.relace = RelaceClient()
        self.openai = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
        )

    @tracer.start_as_current_span("RepoRetriever.retrieve_files")
    async def retrieve_files(
        self,
        repo: Repo,
        query: str,
        top_k: int = 100,
        token_limit: int = 30_000,
        score_threshold: float = 0.3,
        max_candidate_files: int = 50,
    ) -> list[RepoRetrieveResult]:
        # Perform vector similarity search to get candidate files
        logger.info("Embedding query (length %d)", len(query))
        query_vector = await self.relace.embed_text(query, model=EMBEDDING_MODEL)
        async with TurbopufferWrapper() as tpuf:
            query_results = await tpuf.query(
                query_vector,
                namespace=tpuf.repo_namespace(repo.repo_id),
                top_k=top_k,
            )
        if not query_results:
            logger.info(
                "Empty query result; listing candidates from %s", repo.root_path
            )
            query_results = [
                p.relative_to(repo.root_path).as_posix()
                for p in repo.list_tracked_files()
            ]

        # Load file contents asynchronously
        logger.info("Loading %d candidate files", len(query_results))
        codebase: dict[str, RankFileInput] = {}
        for path in query_results[:max_candidate_files]:
            async with aiofile.async_open(repo.root_path / path, "r") as f:
                code = await f.read()
            codebase[path] = RankFileInput(filename=path, code=code)

        # Rank the candidate files with Relace API
        logger.info("Ranking %d candidate files", len(codebase))
        ranker_results = await self.relace.rank(
            RankRequest(
                query=query,
                codebase=list(codebase.values()),
                token_limit=token_limit,
            )
        )

        # Return results above the score threshold
        logger.info("Filtering %s ranker results", len(ranker_results.results))
        return [
            RepoRetrieveResult(
                filename=result.filename,
                score=result.score,
                content=codebase[result.filename].code,
            )
            for result in ranker_results.results
            if result.score >= score_threshold
        ]

    @tracer.start_as_current_span("RepoRetriever.ask_question")
    async def ask_question(
        self,
        repo: Repo,
        query: str,
    ) -> str:
        results = await self.retrieve_files(repo, query)
        if not results:
            return "No relevant files found."

        file_lines = []
        for result in results:
            if not result.content:
                raise ValueError("Result content is empty")
            file_lines.append(f"File: {result.filename}")
            file_lines.append(f"Score: {result.score}")
            file_lines.append("Content:")
            for i, line in enumerate(result.content.splitlines(), start=1):
                file_lines.append(f"{i}\t{line}")
            file_lines.append("\n")
        file_contents = "\n".join(file_lines)

        return await self.generate_response(
            user_prompt=ASK_USER_PROMPT.format(
                user_query=query,
                file_contents=file_contents,
            ),
            system_prompt=ASK_SYSTEM_PROMPT,
        )

    async def generate_response(self, user_prompt: str, system_prompt: str) -> str:
        response = await self.openai.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=8192,
            stream=False,
            extra_body={
                "provider": {
                    "order": ["cerebras"],
                    "allow_fallbacks": False,
                }
            },
        )
        if not response.choices:
            raise ValueError("No choices returned from OpenAI API")
        # Handle OpenRouter API errors
        if response.choices[0].finish_reason == "error":  # type: ignore
            error = getattr(response.choices[0], "error", "Unknown error")
            raise RuntimeError(f"API error: {error}")

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty content returned from OpenAI API")
        return content.strip()


ASK_SYSTEM_PROMPT = """
You are an expert code analyst helping with code analysis.
You have been given a user's conversation/request and several relevant code files from
a git repository. You will be given a list of files that are relevant to the user's
request, and you will need to analyze them to provide a detailed explanation of why they
are relevant to the user's request.
""".strip()

ASK_USER_PROMPT = """
INSTRUCTIONS:
You are given the following User Conversation and Relevant Code Files. Search through
the files and find the event types that are most relevant to the user's request.
Provide the specific lines in the code that are relevant, and explain why those lines
are important in fulfilling the user's request.

For each relevant finding, please include:
1. The file name and specific line numbers
2. The actual code snippet from those lines
3. The specific event type that is relevant to the user's request
4. A clear explanation of why this code is relevant to the user's request
5. How this relates to event tracking or data that could be queried

Finally, include a summary table at the end that shows the relevant files, line numbers, event types, and what it represents. Do not include irrelevant files in the summary table.
Format your response clearly with file citations and line numbers.

USER'S CONVERSATION:
{user_query}

RELEVANT CODE FILES:
{file_contents}
""".strip()
