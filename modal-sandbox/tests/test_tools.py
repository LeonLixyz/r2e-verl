from pathlib import Path
from uuid import UUID

import pytest

from relace_agent.context import Context
from relace_agent.errors import ToolError, ToolInputError
from relace_agent.repo import Repo
from relace_agent.tools import (
    BashTool,
    BashToolSchema,
    DirectoryViewTool,
    DirectoryViewToolSchema,
    FileDeleteTool,
    FileDeleteToolSchema,
    FileRenameTool,
    FileRenameToolSchema,
    FileViewTool,
    FileViewToolSchema,
)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    return Repo(
        repo_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        root_path=repo_path,
        bundle_path=tmp_path / "repo.bundle",
    )


@pytest.fixture
def context(repo: Repo) -> Context:
    return Context(repo)


class TestBashTool:
    @pytest.fixture
    def tool(self) -> BashTool:
        return BashTool()

    @pytest.mark.asyncio
    async def test_execute_impl(self, tool: BashTool, context: Context) -> None:
        result = await tool.execute_impl(
            BashToolSchema(command="echo 'Hello World!'"), context
        )
        assert result == "Hello World!\n"

    @pytest.mark.asyncio
    async def test_execute_impl_timeout(self, tool: BashTool, context: Context) -> None:
        tool.timeout = 0.1  # type: ignore
        with pytest.raises(ToolError):
            await tool.execute_impl(BashToolSchema(command="sleep 1"), context)

    @pytest.mark.asyncio
    async def test_execute_impl_error(self, tool: BashTool, context: Context) -> None:
        with pytest.raises(ToolError):
            await tool.execute_impl(BashToolSchema(command="exit 1"), context)


class TestDirectoryViewTool:
    @pytest.fixture
    def tool(self) -> DirectoryViewTool:
        return DirectoryViewTool()

    @pytest.mark.asyncio
    async def test_view_directory(
        self, tool: DirectoryViewTool, context: Context
    ) -> None:
        files = [
            context.repo.root_path / "file_1.txt",
            context.repo.root_path / "file_2.txt",
        ]
        for path in files:
            path.touch()

        result = await tool.execute_impl(
            DirectoryViewToolSchema(path=context.repo.root_path, include_hidden=False),
            context,
        )
        assert all(
            str(path.relative_to(context.repo.root_path)) in result for path in files
        )


class TestFileViewTool:
    @pytest.fixture
    def tool(self) -> FileViewTool:
        return FileViewTool()

    @pytest.mark.asyncio
    async def test_view_directory(self, tool: FileViewTool, context: Context) -> None:
        with pytest.raises(ToolInputError):
            await tool.execute_impl(
                FileViewToolSchema(path=context.repo.root_path, view_range=[0, 1]),
                context,
            )

    @pytest.mark.asyncio
    async def test_view_file(self, tool: FileViewTool, context: Context) -> None:
        lines = [5, 6]
        path = context.repo.root_path / "file.txt"
        path.write_text("line\n" * lines[-1] * 2)

        result = await tool.execute_impl(
            FileViewToolSchema(path=path, view_range=lines),
            context,
        )
        assert result == "5\tline\n6\tline\n... rest of file truncated ..."

    @pytest.mark.asyncio
    async def test_view_entire_file(self, tool: FileViewTool, context: Context) -> None:
        lines = [5, -1]
        path = context.repo.root_path / "file.txt"
        path.write_text("line\n" * 7)

        result = await tool.execute_impl(
            FileViewToolSchema(path=path, view_range=lines),
            context,
        )
        assert result == "5\tline\n6\tline\n7\tline\n"


class TestFileRenameTool:
    @pytest.fixture
    def tool(self) -> FileRenameTool:
        return FileRenameTool()

    @pytest.mark.asyncio
    async def test_rename_file(self, tool: FileRenameTool, context: Context) -> None:
        content = "This is a test file."
        original_path = context.repo.root_path / "original.txt"
        original_path.write_text(content)
        new_path = context.repo.root_path / "renamed.txt"

        await tool.execute_impl(
            FileRenameToolSchema(old_path=original_path, new_path=new_path), context
        )
        assert not original_path.exists()
        assert new_path.exists()


class TestFileDeleteTool:
    @pytest.fixture
    def tool(self) -> FileDeleteTool:
        return FileDeleteTool()

    @pytest.mark.asyncio
    async def test_delete_files(self, tool: FileDeleteTool, context: Context) -> None:
        files = [
            context.repo.root_path / "file_1",
            context.repo.root_path / "file_2",
        ]
        for path in files:
            path.touch()

        input_data = FileDeleteToolSchema(target_file_paths=files)
        result = await tool.execute_impl(input_data, context)
        assert result.startswith("Deleted paths:")
        assert not any(path.exists() for path in files)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_files(
        self, tool: FileDeleteTool, context: Context
    ) -> None:
        files = [
            context.repo.root_path / "file_1",
            context.repo.root_path / "file_2",
        ]

        input_data = FileDeleteToolSchema(target_file_paths=files)
        result = await tool.execute_impl(input_data, context)
        assert result == "Deleted paths: []"
        assert not any(path.exists() for path in files)
