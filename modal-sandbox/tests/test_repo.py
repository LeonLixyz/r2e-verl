import shutil
from pathlib import Path
from uuid import UUID

import pygit2
import pytest

from relace_agent.repo import Repo

README_PATH = Path("README.md")
README_COMMITTED = "# Test Repo\n"
README_MODIFIED = "# New Title\n"


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    root = tmp_path / "test-repo"
    root.mkdir()
    (root / README_PATH).write_text(README_COMMITTED)

    repo = pygit2.init_repository(root)
    repo.index.add_all()
    repo.index.write()
    tree = repo.index.write_tree()

    author = pygit2.Signature("Test User", "test@relace.ai")
    repo.create_commit("HEAD", author, author, "Initial commit", tree, [])
    return Repo(
        repo_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
        root_path=root,
        bundle_path=tmp_path / "test-repo.bundle",
    )


def test_get_head(repo: Repo) -> None:
    result = repo.get_head()

    assert result


@pytest.mark.asyncio
async def test_commit_empty(repo: Repo) -> None:
    result = await repo.commit("Nothing changed")

    assert result is None


@pytest.mark.asyncio
async def test_commit_new_file(repo: Repo) -> None:
    (repo.root_path / "new-file").touch()

    result = await repo.commit("Added file")

    assert result is not None


@pytest.mark.asyncio
async def test_commit_edit_file(repo: Repo) -> None:
    (repo.root_path / README_PATH).write_text(README_MODIFIED)

    result = await repo.commit("Changed file")

    assert result is not None
    assert (repo.root_path / README_PATH).read_text() == README_MODIFIED


@pytest.mark.asyncio
async def test_checkout_noop(repo: Repo) -> None:
    head = repo.get_head()

    await repo.checkout()

    assert repo.get_head() == head


@pytest.mark.asyncio
async def test_checkout_edit_file_tracked(repo: Repo) -> None:
    head = repo.get_head()
    (repo.root_path / README_PATH).write_text(README_MODIFIED)

    await repo.checkout()

    assert repo.get_head() == head
    assert (repo.root_path / README_PATH).read_text() == README_COMMITTED


@pytest.mark.asyncio
async def test_checkout_edit_file_untracked(repo: Repo) -> None:
    other_file = repo.root_path / "other-file"
    other_file.touch()

    await repo.checkout()

    assert not other_file.exists()


@pytest.mark.asyncio
async def test_checkout_edit_dir_untracked(repo: Repo) -> None:
    new_dir = repo.root_path / "new-files"
    new_file_1 = new_dir / "file-1"
    new_file_2 = new_dir / "file-2"
    new_dir.mkdir()
    new_file_1.touch()
    new_file_2.touch()

    await repo.checkout()

    assert not new_dir.exists()


@pytest.mark.asyncio
async def test_checkout_old_commit(repo: Repo) -> None:
    head_old = repo.get_head()
    (repo.root_path / "new-file").touch()
    head_new = await repo.commit("Added file")

    await repo.checkout(head_old)

    assert repo.get_head() != head_new
    assert repo.get_head() == head_old


def test_list_tracked_files(repo: Repo) -> None:
    result = repo.list_tracked_files()

    assert result == [repo.root_path / README_PATH]


@pytest.mark.asyncio
async def test_list_tracked_files_ignore(repo: Repo) -> None:
    file_included = repo.root_path / "dir-included" / "nest" / "file"
    file_included.parent.mkdir(parents=True)
    file_included.touch()
    file_excluded = repo.root_path / "dir-excluded" / "nest" / "file"
    file_excluded.parent.mkdir(parents=True)
    file_excluded.touch()
    await repo.commit("Added files")

    result = repo.list_tracked_files(ignore_tracked=("*.md", "dir-excluded/*"))

    assert result == [file_included]


@pytest.mark.skipif(not shutil.which("git"), reason="Requires git CLI")
@pytest.mark.asyncio
async def test_bundle_unbundle(repo: Repo, tmp_path: Path) -> None:
    head = repo.get_head()

    await repo.push_bundle(tmp_path / "bundle")
    shutil.rmtree(repo.root_path)
    repo.root_path.mkdir()
    await repo.clone_bundle(tmp_path / "bundle")

    assert repo.get_head() == head
