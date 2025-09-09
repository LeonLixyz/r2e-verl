from __future__ import annotations

import asyncio
import difflib
import io
import tarfile
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, cast, final, override

import aiofile
import modal
from langfuse import get_client, observe
from pydantic import Field

from relace_agent.context import ChatHistory
from relace_agent.errors import ToolError, ToolInputError
from relace_agent.relace import PathLike, RelaceClient
from relace_agent.tools import (
    BashToolSchema,
    DirectoryViewToolSchema,
    FileDeleteToolSchema,
    FileEditToolSchema,
    FileSearchToolSchema,
    FileViewToolSchema,
    FileWriteToolSchema,
    ToolSchema,
)

BASH_TIMEOUT = 60 * 2  # 2 minutes

logger = getLogger(__name__)


# TODO: Unify this with SandboxContext in modal scripts
@dataclass
class SandboxToolContext:
    history: ChatHistory = field(default_factory=ChatHistory)


class SandboxTool[T: ToolSchema](ABC):
    schema: ClassVar[type[T]]  # type: ignore

    _tools: ClassVar[dict[str, type[SandboxTool[ToolSchema]]]] = {}

    def __init__(self, sandbox: modal.Sandbox) -> None:
        self.sandbox: modal.Sandbox = sandbox

    def __init_subclass__(cls, schema: type[T], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.schema = schema
        cls._tools[schema.tool_name] = cls  # type: ignore

    @classmethod
    def from_name(cls, name: str, sandbox: modal.Sandbox) -> SandboxTool[ToolSchema]:
        try:
            return cls._tools[name](sandbox)
        except KeyError as e:
            raise ToolError(f"Invalid tool name: {name}") from e

    @final
    @observe(capture_input=False)
    async def execute(
        self,
        tool_input: T,
        context: SandboxToolContext,
    ) -> str:
        langfuse = get_client()
        langfuse.update_current_span(
            name=f"{self.__class__.__name__}.{self.execute.__name__}",
            input=tool_input,
        )
        try:
            return await self.execute_impl(tool_input, context)
        except modal.exception.NotFoundError:
            logger.error("Unrecoverable error (sandbox timeout?)", exc_info=True)
            raise
        except Exception as exc:
            raise ToolError(f"{exc.__class__.__name__}: {exc}") from exc

    @abstractmethod
    async def execute_impl(
        self,
        tool_input: T,
        context: SandboxToolContext,
    ) -> str: ...

    @final
    @observe
    async def commit(self) -> bool:
        langfuse = get_client()
        langfuse.update_current_span(
            name=f"{self.__class__.__name__}.{self.commit.__name__}"
        )
        process = await self.sandbox.exec.aio("git", "status", "--porcelain")
        if (await process.stdout.read.aio()).strip():
            logger.debug("Committing changes")
            await self.sandbox.exec.aio("git", "add", ".")
            await self.sandbox.exec.aio("git", "commit", "-m", self.schema.tool_name)
            return True
        return False


class SandboxBashTool(SandboxTool[BashToolSchema], schema=BashToolSchema):
    text_limit: ClassVar[int] = 5_000

    @override
    async def execute_impl(
        self,
        tool_input: BashToolSchema,
        context: SandboxToolContext,
    ) -> str:
        def truncate(text: str) -> str:
            if len(text) > self.text_limit:
                return (
                    f"{text[: self.text_limit]}\n"
                    f"...truncated to {self.text_limit} characters ..."
                )
            return text

        process = await self.sandbox.exec.aio(
            "bash",
            "-c",
            tool_input.command,
            timeout=BASH_TIMEOUT,
        )
        exit_code = await process.wait.aio()
        if exit_code != 0:
            raise ToolError(
                f"Command failed with exit code {exit_code}.\n"
                f"STDOUT:\n{truncate(await process.stdout.read.aio())}\n"
                f"STDERR:\n{truncate(await process.stderr.read.aio())}"
            )
        return truncate(await process.stdout.read.aio())


class SandboxDirectoryViewTool(
    SandboxTool[DirectoryViewToolSchema], schema=DirectoryViewToolSchema
):
    limit: ClassVar[int] = 250
    exit_not_found: ClassVar[int] = 101
    exit_not_directory: ClassVar[int] = 102

    @override
    async def execute_impl(
        self,
        tool_input: DirectoryViewToolSchema,
        context: SandboxToolContext,
    ) -> str:
        if tool_input.include_hidden:
            hidden_filter = ""
        else:
            hidden_filter = "-not -path '*/.*' "

        cmd = (
            f"if [ ! -e '{tool_input.path}' ]; then exit {self.exit_not_found}; "
            f"elif [ ! -d '{tool_input.path}' ]; then exit {self.exit_not_directory}; "
            f"else cd '{tool_input.path}' && "
            rf"find . {hidden_filter}-maxdepth 2 \( -type f -printf '%P\n' -o -type d -printf '%P/\n' \) | "
            f"head -{self.limit + 1}; fi"
        )

        # TODO: Run from working directory so that relative paths work
        process = await self.sandbox.exec.aio("bash", "-c", cmd, timeout=BASH_TIMEOUT)
        process_exit = await process.wait.aio()
        if process_exit == -1:
            raise ToolError("Timed out while listing directory contents")
        if process_exit == self.exit_not_found:
            raise ToolInputError(f"{tool_input.path} does not exist")
        elif process_exit == self.exit_not_directory:
            raise ToolInputError(f"{tool_input.path} is not a directory")

        contents = (await process.stdout.read.aio()).splitlines()
        if len(contents) > self.limit:
            contents = contents[: self.limit]
            contents.append("... remaining items omitted ...")

        return "\n".join(contents)


class SandboxFileViewTool(SandboxTool[FileViewToolSchema], schema=FileViewToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: FileViewToolSchema,
        context: SandboxToolContext,
    ) -> str:
        start, end = tool_input.view_range
        view_lines: list[str] = []
        try:
            with await self.sandbox.open.aio(str(tool_input.path), "r") as handle:
                for i, line in enumerate(await handle.readlines.aio(), start=1):
                    if end != -1 and i > end:
                        view_lines.append("... rest of file truncated ...")
                        break
                    if i >= start:
                        view_lines.append(f"{i}\t{line}")
            return "".join(view_lines)
        except FileNotFoundError as e:
            raise ToolInputError(f"{tool_input.path} does not exist") from e
        except IsADirectoryError as e:
            raise ToolInputError(f"{tool_input.path} is a directory") from e


# TODO: Switch to async interfaces
@final
class SandboxPath(PathLike):
    """Compatibility layer to use Sandbox files like Path objects."""

    def __init__(self, path: str, sandbox: modal.Sandbox) -> None:
        self.path = path
        self.sandbox = sandbox

    def exists(self) -> bool:
        try:
            self.sandbox.ls(self.path)
            return True  # Is a directory
        except NotADirectoryError:
            return True  # Is a file
        except FileNotFoundError:
            return False

    def read_text(self) -> str:
        with self.sandbox.open(self.path, "r") as handle:
            return cast(str, handle.read())

    def write_text(self, data: str) -> None:
        with self.sandbox.open(self.path, "w") as handle:
            handle.write(data)

    def __str__(self) -> str:
        return self.path


class SandboxFileWriteTool(
    SandboxTool[FileWriteToolSchema], schema=FileWriteToolSchema
):
    @override
    async def execute_impl(
        self,
        tool_input: FileWriteToolSchema,
        context: SandboxToolContext,
    ) -> str:
        sandbox_path = SandboxPath(str(tool_input.path), self.sandbox)

        # Read old contents to generate diff
        try:
            old_content = sandbox_path.read_text()
        except FileNotFoundError:
            old_content = None
        except IsADirectoryError as e:
            raise ToolInputError(f"{tool_input.path} is a directory") from e

        # Write input content
        sandbox_path.write_text(tool_input.content)

        # Format response
        if old_content is None:
            return f"Created {tool_input.path} ({len(tool_input.content)} bytes)"
        else:
            diff = "".join(
                difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    tool_input.content.splitlines(keepends=True),
                    fromfile="before",
                    tofile="after",
                )
            )
            return f"Updated {tool_input.path}\n\n{diff}"


class SandboxFileEditTool(SandboxTool[FileEditToolSchema], schema=FileEditToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: FileEditToolSchema,
        context: SandboxToolContext,
    ) -> str:
        # Call the merge endpoint
        try:
            diff = await RelaceClient().apply_file(
                path=SandboxPath(str(tool_input.path), self.sandbox),
                edit_snippet=tool_input.code_edit,
                instructions=tool_input.instructions,
            )
        except FileNotFoundError as e:
            raise ToolInputError(f"{tool_input.path} does not exist") from e
        except IsADirectoryError as e:
            raise ToolInputError(f"{tool_input.path} is a directory") from e

        # Return the diff as part of the response
        if diff:
            return f"Applied code changes using Relace API.\n\nChanges made:\n{diff}"
        else:
            return "No changes made"


class SandboxFileDeleteTool(
    SandboxTool[FileDeleteToolSchema], schema=FileDeleteToolSchema
):
    @override
    async def execute_impl(
        self,
        tool_input: FileDeleteToolSchema,
        context: SandboxToolContext,
    ) -> str:
        removed: list[str] = []
        for path in tool_input.target_file_paths:
            try:
                await self.sandbox.rm.aio(str(path), recursive=True)
            except Exception:
                continue
            removed.append(str(path))
        return f"Deleted paths: {removed}"


class SandboxFileSearchTool(
    SandboxTool[FileSearchToolSchema], schema=FileSearchToolSchema
):
    score_threshold: ClassVar[float] = 0.6
    token_limit: ClassVar[int] = 30_000

    @override
    async def execute_impl(
        self,
        tool_input: FileSearchToolSchema,
        context: SandboxToolContext,
    ) -> str:
        logger.info("Streaming candidate files from sandbox")
        pipeline_cmds = [
            # List files from repo (excluding ignored files)
            "git ls-files --others --cached --exclude-standard -z",
            # Filter out non-Unicode files
            'xargs -0 -I {} sh -c \'iconv -f utf-8 -t utf-8 "{}" >/dev/null 2>&1 && printf "%s\\0" "{}"\'',
            # Pack files into tar stream
            "tar --null -czf - --files-from=-",
        ]
        tar_stream = await self.sandbox.exec.aio(
            "bash",
            "-c",
            " | ".join(pipeline_cmds),
            text=False,
            timeout=BASH_TIMEOUT,
        )
        tar_result = await tar_stream.wait.aio()
        if tar_result == -1:
            raise ToolError("Timed out while reading files")
        if tar_result != 0:
            raise ToolError(
                f"Error reading files: {(await tar_stream.stderr.read.aio()).decode()}"
            )

        # Extract tarfile to temporary directory
        tar_bytes = await tar_stream.stdout.read.aio()
        with tempfile.TemporaryDirectory(prefix="repo-") as temp_dir:
            # Extract tarfile asynchronously
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                logger.info("Extracting candidate files to %s", temp_dir)
                await asyncio.to_thread(tar.extractall, path=temp_dir)

            ranked = await RelaceClient().rank_files(
                paths=[path for path in Path(temp_dir).rglob("*") if path.is_file()],
                query=f"<conversation>{context.history.tokenize()}</conversation>\n"
                f"<query>{tool_input.query}</query>",
                token_limit=self.token_limit,
                threshold=self.score_threshold,
            )

            # Read files asynchronously and construct response
            file_contents = []
            for file_path in ranked:
                async with aiofile.async_open(file_path, mode="r") as f:
                    content = await f.read()
                file_contents.append(
                    f"{file_path.relative_to(temp_dir)}\n```\n{content}\n```"
                )
            return "\n\n".join(file_contents)


class SearchAndReplaceToolSchema(ToolSchema, name="search_and_replace"):
    """This is a tool for editing part of an existing file.

    Before using this tool, make sure that you have viewed its contents either with
    `view_file` or `find_relevant_files`.

    The tool will replace ONE occurrence of old_string with new_string in the specified file.
    CRITICAL REQUIREMENTS FOR USING THIS TOOL:
    1. UNIQUENESS: The old_string MUST uniquely identify the specific instance you want to change. This means:
      - Include AT LEAST 3-5 lines of context BEFORE the change point
      - Include AT LEAST 3-5 lines of context AFTER the change point
      - Include all whitespace, indentation, and surrounding code exactly as it appears in the file
    2. SINGLE INSTANCE: This tool can only change ONE instance at a time. If you need to change multiple instances:
      - Make separate calls to this tool for each instance
      - Each call must uniquely identify its specific instance using extensive context
    3. VERIFICATION: Before using this tool:
      - Check how many instances of the target text exist in the file
      - If multiple instances exist, gather enough context to uniquely identify each one
      - Plan separate tool calls for each instance

    WARNING: If you do not follow these requirements:
      - The tool will fail if old_string matches multiple locations
      - The tool will fail if old_string doesn't match exactly (including whitespace)
      - You may change the wrong instance if you don't include enough context
    When making edits:
      - Ensure the edit results in idiomatic, correct code
      - Do not leave the code in a broken state
      - Always use absolute file paths (starting with /)
    """

    path: str = Field(
        ...,
        description="The target file to modify. You must use an absolute path to an "
        "existing file.",
    )
    old_string: str = Field(
        ...,
        description="The text to replace (must be unique within the file, and must "
        "match the file contents exactly, including all whitespace and indentation)",
    )
    new_string: str = Field(
        ..., description="The edited text to replace the old_string"
    )


class SandboxSearchAndReplaceTool(
    SandboxTool[SearchAndReplaceToolSchema], schema=SearchAndReplaceToolSchema
):
    @override
    async def execute_impl(
        self,
        tool_input: SearchAndReplaceToolSchema,
        context: SandboxToolContext,
    ) -> str:
        # Read file content
        try:
            with await self.sandbox.open.aio(tool_input.path, "r") as handle:
                content = cast(str, await handle.read.aio())
        except FileNotFoundError as e:
            raise ToolInputError(f"file not found: {tool_input.path}") from e
        except IsADirectoryError as e:
            raise ToolInputError(f"{tool_input.path} is a directory") from e

        # Check that search term is present exactly once
        count = content.count(tool_input.old_string)
        if count == 0:
            raise ToolInputError(f"old_string not found in {tool_input.path}")
        elif count > 1:
            raise ToolInputError(
                f"old_string found {count} times in {tool_input.path}."
            )

        # Write updated content
        content_replaced = content.replace(
            tool_input.old_string, tool_input.new_string, 1
        )
        with await self.sandbox.open.aio(tool_input.path, "w") as handle:
            await handle.write.aio(content_replaced)

        # Return diff
        diff = "".join(
            difflib.unified_diff(
                content.splitlines(keepends=True),
                content_replaced.splitlines(keepends=True),
                fromfile="before",
                tofile="after",
            )
        )
        return f"Updated {tool_input.path}:\n\n{diff}"


class RipgrepToolSchema(ToolSchema, name="grep_search"):
    """Fast text-based regex search that finds exact pattern matches within files or
    directories, utilizing the ripgrep command for efficient searching. Results will be
    formatted in the style of ripgrep and can be configured to include line numbers and
    content. To avoid overwhelming output, the results are capped at 50 matches. Use the
    include or exclude patterns to filter the search scope by file type or specific
    paths. This is best for finding exact text matches or regex patterns. More precise
    than semantic search for finding specific strings or patterns. This is preferred
    over semantic search when we know the exact symbol/function name/etc. to search in
    some set of directories/file types.
    """

    query: str = Field(..., description="The regex pattern to search for")
    case_sensitive: bool = Field(
        default=True, description="Whether the search should be case sensitive"
    )
    exclude_pattern: str | None = Field(
        default=None, description="Glob pattern for files to exclude"
    )
    include_pattern: str | None = Field(
        default=None,
        description="Glob pattern for files to include (e.g. '.ts' for TypeScript files)",
    )


class SandboxRipgrepTool(SandboxTool[RipgrepToolSchema], schema=RipgrepToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: RipgrepToolSchema,
        context: SandboxToolContext,
    ) -> str:
        args = ["rg", tool_input.query, "./", "--max-count=50", "--color=never"]
        if not tool_input.case_sensitive:
            args.append("--ignore-case")
        if tool_input.exclude_pattern:
            args.append(f"--glob=!{tool_input.exclude_pattern}")
        if tool_input.include_pattern:
            args.append(f"--glob={tool_input.include_pattern}")

        process = await self.sandbox.exec.aio(*args, timeout=BASH_TIMEOUT)
        exit_code = await process.wait.aio()
        if exit_code == 0:
            return await process.stdout.read.aio()
        elif exit_code == 1:
            return "No matches found"
        else:
            raise ToolError(f"Error running ripgrep: {await process.stderr.read.aio()}")


class UndoToolSchema(ToolSchema, name="undo"):
    """Undo previous file changes in the current project.

    This works by calling `git revert` once. It is only able to revert operations that
    resulted in a git commit internally - it may not reverse all of the side effects of
    previous operations. Calling this tool twice in a row will "undo the undo".
    """


class SandboxUndoTool(SandboxTool[UndoToolSchema], schema=UndoToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: UndoToolSchema,
        context: SandboxToolContext,
    ) -> str:
        # Get the commit message of the commit we're about to revert
        commit_msg_process = await self.sandbox.exec.aio(
            "git", "log", "-1", "--pretty=format:%s", "HEAD"
        )
        if await commit_msg_process.wait.aio() != 0:
            raise ToolError("No committed changes to revert")
        commit_message = (await commit_msg_process.stdout.read.aio()).strip()

        # Create a revert commit
        revert_process = await self.sandbox.exec.aio(
            "git", "revert", "--no-edit", "HEAD"
        )
        if await revert_process.wait.aio() != 0:
            raise ToolError(
                f"Error reverting changes: {await revert_process.stderr.read.aio()}"
            )

        return f"Successfully reverted last changes: {commit_message}"
