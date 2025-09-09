from __future__ import annotations

import asyncio
import difflib
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Self, final, override

import aiofile
from langfuse import get_client, observe
from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, Field, ValidationError

from relace_agent.context import Context
from relace_agent.errors import ToolError, ToolInputError
from relace_agent.relace import RelaceClient
from relace_agent.repo import Repo
from relace_agent.server.types import ToolEvent


class ToolSchema(BaseModel):
    """Base class for tool input parameters."""

    tool_name: ClassVar[str]
    tool_description: ClassVar[str]
    _tool_lookup: ClassVar[dict[str, type[ToolSchema]]] = {}

    def __init_subclass__(
        cls,
        name: str,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        description = description or cls.__doc__
        if not description:
            raise ValueError(f"{cls.__name__} is missing a docstring or description")
        cls.tool_name = name
        cls.tool_description = description.strip()
        cls._tool_lookup[name] = cls

    @classmethod
    def from_name(cls, tool_name: str) -> type[ToolSchema]:
        try:
            return cls._tool_lookup[tool_name]
        except KeyError as e:
            raise ToolInputError(f"Unknown tool: {tool_name}") from e

    @classmethod
    def openai(cls) -> ChatCompletionToolParam:
        schema = pydantic_function_tool(
            model=cls,
            name=cls.tool_name,
            description=cls.tool_description,
        )

        # Remove unsupported 'format' from schema for Path fields, recursively
        def remove_path_format(obj: Any) -> None:
            if isinstance(obj, dict):
                if obj.get("format") == "path":
                    obj.pop("format")
                for val in obj.values():
                    remove_path_format(val)
            elif isinstance(obj, list):
                for item in obj:
                    remove_path_format(item)

        schema_params = schema["function"]["parameters"]
        remove_path_format(schema_params)
        return schema

    @classmethod
    def from_json_str(cls, json_str: str) -> Self:
        try:
            return cls.model_validate_json(json_str)
        except ValidationError as e:
            raise ToolInputError(json_str) from e

    @classmethod
    def from_json_dict(cls, json_dict: dict[str, Any]) -> Self:
        try:
            return cls.model_validate(json_dict)
        except ValidationError as e:
            raise ToolInputError(json_dict) from e

    def event(self, context: Context) -> ToolEvent:
        """Convert to public facing SSE event."""
        return ToolEvent(name=self.tool_name)

    def event_with_path(self, context: Context, path: Path) -> ToolEvent:
        if path.is_relative_to(context.repo.root_path):
            path = path.relative_to(context.repo.root_path)
        return ToolEvent(name=self.tool_name, path=str(path))


class Tool[T: ToolSchema](ABC):
    """Base class for tools that an agent can use."""

    schema: ClassVar[type[T]]  # type: ignore

    _tools: ClassVar[dict[str, type[Tool[ToolSchema]]]] = {}

    def __init_subclass__(
        cls,
        schema: type[T],
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.schema = schema
        cls._tools[schema.tool_name] = cls  # type: ignore

    @final
    @observe(capture_input=False)
    async def execute(
        self,
        tool_input: T,
        context: Context,
    ) -> str:
        """Execute the tool with the given parameters.

        Args:
            tool_input: Structured, validated input arguments for the tool.
        """
        langfuse = get_client()
        langfuse.update_current_span(
            name=f"{self.__class__.__name__}.execute",
            input=tool_input,
        )
        return await self.execute_impl(tool_input=tool_input, context=context)

    @abstractmethod
    async def execute_impl(self, tool_input: T, context: Context) -> str: ...

    @classmethod
    def from_name(cls, name: str) -> Tool[ToolSchema]:
        return cls._tools[name]()


class BashToolSchema(ToolSchema, name="bash"):
    """Tool for executing bash commands.

    * Avoid long running commands
    * Avoid dangerous/destructive commands
    * Prefer using other more specialized tools where possible
    """

    command: str = Field(..., description="Bash command to execute")


@final
class BashTool(Tool[BashToolSchema], schema=BashToolSchema):
    timeout: ClassVar[int] = 60 * 2  # 2 minutes
    text_limit: ClassVar[int] = 5_000

    @override
    async def execute_impl(
        self,
        tool_input: BashToolSchema,
        context: Context,
    ) -> str:
        def truncate(text: str) -> str:
            if len(text) > self.text_limit:
                return (
                    f"{text[: self.text_limit]}\n"
                    f"...truncated to {self.text_limit} characters ..."
                )
            return text

        try:
            # Execute the bash command and capture output
            proc = await asyncio.create_subprocess_shell(
                tool_input.command,
                cwd=context.repo.root_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except TimeoutError as e:
            raise ToolError(f"Command timed out after {self.timeout} seconds.") from e

        if proc.returncode != 0:
            raise ToolError(
                f"Command failed with exit code {proc.returncode}.\n"
                f"STDOUT:\n{truncate(stdout.decode())}\n"
                f"STDERR:\n{truncate(stderr.decode())}"
            )
        return truncate(stdout.decode())


class DirectoryViewToolSchema(ToolSchema, name="view_directory"):
    """Tool for viewing the contents of a directory.

    * Lists contents recursively, relative to the input directory
    * Directories are suffixed with a trailing slash '/'
    * Depth might be limited by the tool implementation
    * Output is limited to the first 250 items

    Example output:
    file1.txt
    file2.txt
    subdir1/
    subdir1/file3.txt
    """

    path: Path = Field(
        ...,
        description="Absolute path to a directory, e.g. `/repo/`.",
    )
    include_hidden: bool = Field(
        False,
        description="If true, include hidden files in the output (false by default).",
    )

    @override
    def event(self, context: Context) -> ToolEvent:
        return self.event_with_path(context, path=self.path)


class DirectoryViewTool(Tool[DirectoryViewToolSchema], schema=DirectoryViewToolSchema):
    limit: ClassVar[int] = 250

    @override
    async def execute_impl(
        self,
        tool_input: DirectoryViewToolSchema,
        context: Context,
    ) -> str:
        check_path(tool_input.path, context.repo, must_exist=True, file_ok=False)

        contents = []
        for i, item in enumerate(tool_input.path.rglob("*")):
            if i >= self.limit:
                contents.append("... remaining items omitted ...")
                break
            if not tool_input.include_hidden and item.name.startswith("."):
                continue

            relative_path = item.relative_to(context.repo.root_path)
            if item.is_dir():
                contents.append(f"{relative_path}/")
            else:
                contents.append(f"{relative_path}")

        return "\n".join(contents) if contents else "Directory is empty."


class FileViewToolSchema(ToolSchema, name="view_file"):
    """Tool for viewing/exploring the contents of existing files

    Line numbers are included in the output, indexing at 1. If the output does not
    include the end of the file, it will be noted after the final output line.

    Example (viewing the first 2 lines of a file):
    1\tdef my_function():
    2\t    print("Hello, World!")
    ... rest of file truncated ...
    """

    path: Path = Field(
        ...,
        description="Absolute path to a file, e.g. `/repo/file.py`.",
    )
    # TODO: Split this into two integers for more precise validation
    # NOTE: Tuple validation is not supported by the OpenAI API, so we cannot do that
    view_range: list[int] = Field(
        [1, 100],
        description="Range of file lines to view. If not specified, the first 100 lines "
        "of the file are shown. If provided, the file will be shown in the indicated "
        "line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to "
        "start. Setting `[start_line, -1]` shows all lines from `start_line` to the end "
        "of the file.",
    )

    @override
    def event(self, context: Context) -> ToolEvent:
        return self.event_with_path(context, path=self.path)


@final
class FileViewTool(Tool[FileViewToolSchema], schema=FileViewToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: FileViewToolSchema,
        context: Context,
    ) -> str:
        check_path(tool_input.path, context.repo, must_exist=True, dir_ok=False)

        start, end = tool_input.view_range
        view_lines: list[str] = []
        with tool_input.path.open("r") as handle:
            for i, line in enumerate(handle.readlines(), start=1):
                if end != -1 and i > end:
                    view_lines.append("... rest of file truncated ...")
                    break
                if i >= start:
                    view_lines.append(f"{i}\t{line}")

        return "".join(view_lines)


class FileRenameToolSchema(ToolSchema, name="rename_file"):
    """Tool for renaming/moving a file."""

    old_path: Path = Field(..., description="The old path of the file.")
    new_path: Path = Field(..., description="The new path of the file.")


@final
class FileRenameTool(Tool[FileRenameToolSchema], schema=FileRenameToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: FileRenameToolSchema,
        context: Context,
    ) -> str:
        check_path(tool_input.old_path, context.repo, must_exist=True)
        check_path(tool_input.new_path, context.repo, must_not_exist=True)

        tool_input.old_path.rename(tool_input.new_path)
        return f"Renamed {tool_input.old_path} to {tool_input.new_path}"


class FileDeleteToolSchema(ToolSchema, name="delete_files"):
    """Deletes multiple files or directories at the specified paths.

    Each operation will fail gracefully if:
    * The file doesn't exist
    * The file cannot be deleted
    """

    target_file_paths: list[Path] = Field(
        ..., description="Array of file or directory paths to delete"
    )


@final
class FileDeleteTool(Tool[FileDeleteToolSchema], schema=FileDeleteToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: FileDeleteToolSchema,
        context: Context,
    ) -> str:
        removed = []
        for path in tool_input.target_file_paths:
            try:
                check_path(path, context.repo)
                shutil.rmtree(path)
            except NotADirectoryError:
                path.unlink()
            except Exception:
                continue
            removed.append(str(path))

        return f"Deleted paths: {removed}"


class FileWriteToolSchema(ToolSchema, name="write_file"):
    """Use this tool to create a new file or fully overwrite an existing file.

    For existing files, you should always prefer to use a different tool that is
    optimized for that purpose - use this as a fallback when those tools fail, or your
    change is significant enough that you are effectively replacing the entire file
    contents.
    """

    path: Path = Field(
        ...,
        description="The path/filename of the file. You must use an absolute path.",
    )
    content: str = Field(
        ...,
        description="The content/body to write to the file. Your exact input will be "
        "written to the file without modification.",
    )

    @override
    def event(self, context: Context) -> ToolEvent:
        return self.event_with_path(context, path=self.path)


class FileWriteTool(Tool[FileWriteToolSchema], schema=FileWriteToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: FileWriteToolSchema,
        context: Context,
    ) -> str:
        check_path(tool_input.path, context.repo, dir_ok=False)

        if tool_input.path.is_file():
            async with aiofile.async_open(tool_input.path, "r") as f:
                old_content = await f.read()

            diff = "".join(
                difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    tool_input.content.splitlines(keepends=True),
                    fromfile="before",
                    tofile="after",
                )
            )
            return f"Updated {tool_input.path}\n\n{diff}"
        else:
            tool_input.path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofile.async_open(tool_input.path, "w") as f:
                await f.write(tool_input.content)
            return f"Created {tool_input.path} ({len(tool_input.content)} bytes)"


class FileEditToolSchema(ToolSchema, name="edit_file"):
    """Use this tool to propose an edit to an existing file.

    This will be read by a less intelligent model, which will quickly apply the edit.
    You should make it clear what the edit is, while also minimizing the unchanged code
    you write. When writing the edit, you should specify each edit in sequence, with
    the special comment `// ... existing code ...` to represent unchanged code in
    between edited lines.

    For example:
    ```
    // ... existing code ...
    FIRST_EDIT
    // ... existing code ...
    SECOND_EDIT
    // ... existing code ...
    THIRD_EDIT
    // ... existing code ...
    ```

    You should still bias towards repeating as few lines of the original file as
    possible to convey the change. But, each edit should contain sufficient context of
    unchanged lines around the code you're editing to resolve ambiguity. DO NOT omit
    spans of pre-existing code (or comments) without using the
    `// ... existing code ...` comment to indicate its absence. If you omit the existing
    code comment, the model may inadvertently delete these lines. Make sure it is clear
    what the edit should be, and where it should be applied.
    """

    path: Path = Field(
        ...,
        description="The target file to modify. You must use an absolute path to an "
        "existing file.",
    )
    instructions: str = Field(
        ...,
        description="A single sentence instruction describing what you are going to do "
        "for the sketched edit. This is used to assist the less intelligent model in "
        "applying the edit. Please use the first person to describe what you are going "
        "to do. Dont repeat what you have said previously in normal messages. And use "
        "it to disambiguate uncertainty in the edit.",
    )
    code_edit: str = Field(
        ...,
        description="Specify ONLY the precise lines of code that you wish to edit. "
        "**NEVER specify or write out unchanged code**. Instead, represent all "
        "unchanged code using the comment of the language you're editing in - example: "
        "`// ... existing code ...`",
    )

    @override
    def event(self, context: Context) -> ToolEvent:
        return self.event_with_path(context, path=self.path)


@final
class FileEditTool(Tool[FileEditToolSchema], schema=FileEditToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: FileEditToolSchema,
        context: Context,
    ) -> str:
        check_path(tool_input.path, context.repo, must_exist=True, dir_ok=False)

        # Call the merge endpoint
        try:
            client = RelaceClient()
            diff = await client.apply_file(
                path=tool_input.path,
                edit_snippet=tool_input.code_edit,
                instructions=tool_input.instructions,
            )

            # Return the diff as part of the response
            if diff:
                return f"Applied code changes to {tool_input.path}:\n\n{diff}"
            else:
                return "No changes made"
        except Exception as e:
            raise ToolError(f"Error applying code changes: {str(e)}") from e


class QuickEditToolSchema(ToolSchema, name="quick_edit_file"):
    """Use this tool to propose an edit to an existing file.

    This will be read by a less intelligent model, which will generate the actual code
    change. Always use this tool to make small or simple edits to existing files - it
    is 10 times faster than other methods. Use a different edit method for more complex
    changes that a small model may not understand completely.
    """

    path: Path = Field(
        ...,
        description="The target file to modify. You must use an absolute path to an "
        "existing file. If you pass a file that does not exist, an error will be "
        "returned.",
    )
    instructions: str = Field(
        ...,
        description="A thorough but concise description of the edit you want to make. "
        "These instructions should be complete enough to allow a less intelligent "
        "model to entirely synthesize the code edit.",
    )


class QuickEditTool(Tool[QuickEditToolSchema], schema=QuickEditToolSchema):
    @override
    async def execute_impl(
        self,
        tool_input: QuickEditToolSchema,
        context: Context,
    ) -> str:
        check_path(tool_input.path, context.repo, must_exist=True, dir_ok=False)

        # Call the merge endpoint
        try:
            client = RelaceClient()
            diff = await client.quick_edit_file(
                path=tool_input.path,
                instructions=tool_input.instructions,
            )

            # Return the diff as part of the response
            if diff:
                return f"Edited code using Relace API.\n\nChanges made:\n{diff}"
            else:
                return "No changes made"
        except Exception as e:
            raise ToolError(f"Error editing code: {str(e)}") from e


class FileSearchToolSchema(ToolSchema, name="find_relevant_files"):
    """Use this tool to search for files that are most relevant for a given task or
    question. The conversation history will be passed in with your query. Tool calls in
    the conversation will have the tool name only - inputs and outputs are omitted to
    keep the input size manageable.

    - Call this tool ONE TIME, including all of your tasks/questions.
    - DO NOT call it multiple times before ending your turn.
    - Prefer this as the first step to explore or plan edits within a code repository.
    - Prefer this over using a bash command like `grep` or `find`.
    - Files that are not utf-8 encoded are excluded.
    - Large data files may be excluded.

    The response will be a list of file paths (relative to the working directory) and
    their contents, ordered from most relevant to least relevant. The response will look
    like this:

    path/to/file1
    ```
    file1 content...
    ```

    path/to/file2
    ```
    file2 content...
    ```
    """

    query: str = Field(
        ...,
        description="Natural language description of what you are looking for in the "
        "repository. You can describe a particular change you want to make, a feature "
        "you want to implement, bug you are trying to fix, information you are "
        "searching for, etc.",
    )


@final
class FileSearchTool(Tool[FileSearchToolSchema], schema=FileSearchToolSchema):
    score_threshold: float = 0.6
    token_limit: int = 30_000

    @override
    async def execute_impl(
        self,
        tool_input: FileSearchToolSchema,
        context: Context,
    ) -> str:
        # Call the ranker endpoint
        try:
            client = RelaceClient()
            ranked = await client.rank_files(
                paths=context.repo.list_tracked_files(
                    # TODO: Make this configurable
                    ignore_tracked=(
                        "relace.yaml",
                        ".gitignore",
                        "package-lock.json",
                        "data/*",
                        "*.csv",
                        "*.svg",
                        "*.jsonl",
                    )
                ),
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
                    f"{file_path.relative_to(context.repo.root_path)}\n"
                    f"```\n{content}\n```"
                )

            return "\n\n".join(file_contents)
        except Exception as e:
            raise ToolError(f"Error finding files: {str(e)}") from e


def check_path(
    path: Path,
    repo: Repo,
    must_exist: bool = False,
    must_not_exist: bool = False,
    file_ok: bool = True,
    dir_ok: bool = True,
) -> None:
    if not path.is_relative_to(repo.root_path):
        raise ToolInputError(f"path is not contained in the repository: {path}")
    if must_exist and not path.exists():
        raise ToolInputError(f"path does not exist: {path}")
    if must_not_exist and path.exists():
        raise ToolInputError(f"path already exists: {path}")
    if not file_ok and path.is_file():
        raise ToolInputError(f"path is a file: {path}")
    if not dir_ok and path.is_dir():
        raise ToolInputError(f"path is a directory: {path}")
