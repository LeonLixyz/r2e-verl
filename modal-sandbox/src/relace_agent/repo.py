import asyncio
import fnmatch
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self, cast
from urllib.parse import urlparse

import github
import pygit2
from async_lru import alru_cache
from fastapi import HTTPException, status
from opentelemetry import trace
from pygit2.enums import CheckoutStrategy, FileStatus

from relace_agent.server.database import Database
from relace_agent.server.storage import STORAGE
from relace_agent.server.types import RelaceUserId, RepoId, RepoInfo

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


@alru_cache(maxsize=512)
async def get_github_installation_auth(user_id: str) -> github.Auth.AppInstallationAuth:
    db = Database.from_env()
    github_id = await db.get_github_installation_id(user_id)
    if not github_id:
        raise RuntimeError("User does not have a GitHub installation")

    # Key refresh is handled automatically, so we can cache one instance
    def get_auth_sync() -> github.Auth.AppInstallationAuth:
        app_auth = github.Auth.AppAuth(
            app_id=os.environ["GITHUB_APP_CLIENT_ID"],
            private_key=os.environ["GITHUB_APP_SECRET_KEY"],
        )
        return app_auth.get_installation_auth(installation_id=int(github_id))

    return await asyncio.to_thread(get_auth_sync)


GITHUB_REPO_PATTERN = re.compile(
    r"^(?:https://github\.com/)?(?P<repo_name>[a-zA-Z0-9-]+/[a-zA-Z0-9._-]+)$"
)


async def get_github_url(user_id: RelaceUserId, repo_str: str) -> str:
    try:
        auth = await get_github_installation_auth(user_id)
    except RuntimeError as e:
        logger.info("GitHub auth error: %s", e)
        auth = None

    repo_name_match = GITHUB_REPO_PATTERN.match(repo_str)
    if not repo_name_match:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid GitHub repository format: {repo_str}",
        )

    repo_name = repo_name_match.group("repo_name")
    repo = await asyncio.to_thread(github.Github(auth=auth).get_repo, repo_name)

    # Use access token only if the repo is private
    if auth and repo.private:
        return f"https://x-access-token:{auth.token}@github.com/{repo_name}.git"
    else:
        return f"https://github.com/{repo_name}.git"


def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_github_repo(string: str) -> bool:
    return bool(GITHUB_REPO_PATTERN.match(string))


# TODO: Utilize URL object for cleaner handling
async def resolve_repository_url(user_id: RelaceUserId, template: str) -> str:
    if is_github_repo(template):
        return await get_github_url(user_id, template)
    elif is_url(template):
        return template
    raise HTTPException(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Invalid template repository format: {template}",
    )


@dataclass
class Repo:
    repo_id: RepoId
    root_path: Path
    bundle_path: Path

    @classmethod
    def from_repo_id(cls, repo_id: RepoId) -> Self:
        return cls(
            repo_id=repo_id,
            root_path=STORAGE.repo_cache(repo_id),
            bundle_path=STORAGE.repo_bundle(repo_id),
        )

    @property
    def config_path(self) -> Path:
        return self.root_path / "relace.yaml"

    @tracer.start_as_current_span("Repo.git")
    async def git(
        self,
        *args: str,
        check: bool = True,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        logger.info("Running `git %s` for repo %s", " ".join(args), self.repo_id)
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd or self.root_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()

        if check and proc.returncode != 0:
            logger.error(
                "git %s failed [STDERR]:\n%s", " ".join(args), stderr_str.strip()
            )
            raise subprocess.CalledProcessError(
                proc.returncode or -1, ["git", *args], stdout_str, stderr_str
            )

        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=proc.returncode or -1,
            stdout=stdout_str,
            stderr=stderr_str,
        )

    @tracer.start_as_current_span("Repo.init")
    async def init(self) -> None:
        await asyncio.to_thread(pygit2.init_repository, self.root_path.as_posix())

    @tracer.start_as_current_span("Repo.init_cache")
    async def init_cache(self, head: str | None) -> None:
        cache_head = self.get_head() if self.root_path.exists() else None
        if cache_head == head:
            logger.info("Repo cache is valid (head=%s)", head)
        else:
            logger.info(
                "Repo cache is invalid (cache_head=%s, head=%s); pulling from storage",
                cache_head,
                head,
            )
            await self.clone_bundle()

    @tracer.start_as_current_span("Repo.clone")
    async def clone(
        self,
        user_id: RelaceUserId,
        url: str,
        branch: str | None = None,
    ) -> None:
        # Clone to a temporary directory first so we can atomically populate the cache
        # dir after the entire clone operation completes successfully
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                url = await resolve_repository_url(user_id, url)
                await self.git(
                    "clone",
                    url,
                    "./",
                    *(["--branch", branch] if branch else []),
                    "--single-branch",
                    "--depth=1",
                    cwd=Path(tmpdir),
                )
            except (subprocess.CalledProcessError, github.UnknownObjectException) as e:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    detail=f"Repository is not accessible: {url}",
                ) from e

            Path(tmpdir).rename(self.root_path)

    @tracer.start_as_current_span("Repo.clone_bundle")
    async def clone_bundle(self, path: Path | None = None) -> None:
        if path is None:
            path = self.bundle_path
        with tempfile.TemporaryDirectory() as tmpdir:
            await self.git("clone", path.as_posix(), "./", cwd=Path(tmpdir))

            # NOTE: There might be a more efficient way to do this (e.g. apply the
            # bundle directly to the existing cache). Using this as a simple correct
            # implementation for now.
            if self.root_path.exists():
                logger.info("Clearing existing repo cache: %s", self.root_path)
                await asyncio.to_thread(shutil.rmtree, self.root_path)

            Path(tmpdir).rename(self.root_path)

    @tracer.start_as_current_span("Repo.push_bundle")
    async def push_bundle(self, path: Path | None = None) -> None:
        if path is None:
            path = self.bundle_path
        # If repository is shallow, synthesize a snapshot bundle with a new root commit
        if self._is_shallow():
            logger.info(
                "Shallow repository detected; creating snapshot bundle with orphan root commit"
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                snapshot_path = Path(tmpdir) / "snapshot"
                snapshot_path.mkdir(parents=True, exist_ok=True)

                # Initialize a fresh repository for the snapshot
                await asyncio.to_thread(
                    pygit2.init_repository, snapshot_path.as_posix()
                )
                snapshot_repo = pygit2.Repository(snapshot_path.as_posix())

                # Copy all tracked files from the current index into the snapshot repo
                for abs_src_path in self.list_tracked_files(ignore_tracked=()):
                    # Skip entries that are not regular files (e.g., submodules/directories)
                    if not abs_src_path.is_file():
                        continue
                    rel_path = abs_src_path.relative_to(self.root_path)
                    abs_dst_path = snapshot_path / rel_path
                    abs_dst_path.parent.mkdir(parents=True, exist_ok=True)
                    await asyncio.to_thread(shutil.copy2, abs_src_path, abs_dst_path)

                # Stage and create a single orphan commit on HEAD
                await asyncio.to_thread(snapshot_repo.index.add_all)
                await asyncio.to_thread(snapshot_repo.index.write)
                tree_id = await asyncio.to_thread(snapshot_repo.index.write_tree)
                author = committer = pygit2.Signature(
                    name="Relace Agent", email="noreply@relace.ai"
                )
                await asyncio.to_thread(
                    snapshot_repo.create_commit,
                    "HEAD",
                    author,
                    committer,
                    "Snapshot (shallow) bundle",
                    tree_id,
                    [],
                )

                # Create and verify bundle from the snapshot repo
                with tempfile.NamedTemporaryFile() as tmp:
                    await self.git(
                        "bundle", "create", tmp.name, "--all", cwd=snapshot_path
                    )
                    await self.git("bundle", "verify", tmp.name, cwd=snapshot_path)
                    await asyncio.to_thread(shutil.copy, tmp.name, path)
        else:
            # Full-history bundle
            with tempfile.NamedTemporaryFile() as tmp:
                await self.git("bundle", "create", tmp.name, "--all")
                await self.git("bundle", "verify", tmp.name)
                await asyncio.to_thread(shutil.copy, tmp.name, path)

    def _repo(self) -> pygit2.Repository:
        return pygit2.Repository(self.root_path.as_posix())

    def _is_shallow(self) -> bool:
        git_dir = self.root_path / ".git"
        return (git_dir / "shallow").exists()

    def get_head(self) -> str:
        obj = self._repo().head.peel(None)
        return str(obj.id)

    @tracer.start_as_current_span("Repo.pull_remote")
    async def pull_remote(
        self,
        user_id: RelaceUserId,
        remote: str,
        remote_branch: str | None = None,
        new_branch: str | None = None,
    ) -> None:
        # Use local branch name by default when not provided
        if remote_branch is None:
            try:
                repo_obj = self._repo()
                # Prefer shorthand branch name if HEAD points to a local branch
                branch_name = getattr(repo_obj.head, "shorthand", None)
                if not branch_name:
                    head_name = getattr(repo_obj.head, "name", "")
                    if head_name.startswith("refs/heads/"):
                        branch_name = head_name[len("refs/heads/") :]
                remote_branch = branch_name or "main"
            except Exception:
                remote_branch = "main"
        remote_branch_ref = f"refs/remotes/external/{remote_branch}"

        # If a new target branch is provided, create/checkout it BEFORE fetching/merging
        # so that all subsequent operations occur on the new branch and never touch the
        # caller's current branch state.
        if new_branch:
            try:
                await self.git("checkout", "-b", new_branch)
            except subprocess.CalledProcessError:
                await self.git("checkout", new_branch)

        remote_url = await resolve_repository_url(user_id, remote)
        await self.git("fetch", remote_url, f"{remote_branch}:{remote_branch_ref}")

        # On a new branch, allow a non-FF merge to preserve both histories without
        # assuming fast-forwardability. On the existing branch, still require FF-only.
        if new_branch:
            await self.git("merge", remote_branch_ref)
        else:
            await self.git("merge", remote_branch_ref, "--ff-only")

    @tracer.start_as_current_span("Repo.checkout")
    async def checkout(
        self,
        ref: str = "HEAD",
        clean_ignored: bool = False,
    ) -> None:
        repo = self._repo()
        commit = repo.revparse_single(ref)
        if not isinstance(commit, pygit2.Commit):
            raise ValueError(f"{ref} is not a valid commit")

        # Clean-up files that are not handled by CheckoutStrategy.FORCE
        for filepath, git_status in repo.status(
            "normal", ignored=clean_ignored
        ).items():
            if git_status in (
                FileStatus.WT_NEW,
                FileStatus.INDEX_NEW,
                FileStatus.IGNORED,
            ):
                abs_path = self.root_path / filepath
                if abs_path.is_file():
                    await asyncio.to_thread(abs_path.unlink)
                elif abs_path.is_dir():
                    await asyncio.to_thread(shutil.rmtree, abs_path)

        # Create a branch to ensure commits are never orphaned
        branch_name = f"relace/{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}"
        await asyncio.to_thread(
            repo.create_branch,
            branch_name,
            commit,
        )

        # Force checkout
        await asyncio.to_thread(
            repo.checkout,
            f"refs/heads/{branch_name}",
            strategy=CheckoutStrategy.FORCE,
        )
        commit = cast(pygit2.Commit, repo.head.peel(None))
        logger.info(
            "Repo HEAD -> %s\n\thash: %s\n\ttime: %s",
            repo.head.name,
            commit.id,
            commit.author.time,
        )

    @tracer.start_as_current_span("Repo.commit")
    async def commit(
        self,
        message: str,
        path_specs: list[str | os.PathLike[str]] | None = None,
        allow_empty: bool = False,
    ) -> str | None:
        repo = self._repo()
        await asyncio.to_thread(repo.index.add_all, path_specs)
        await asyncio.to_thread(repo.index.write)

        # Exit if there are no staged changes
        for git_status in repo.status().values():
            if git_status in (
                FileStatus.INDEX_NEW,
                FileStatus.INDEX_MODIFIED,
                FileStatus.INDEX_DELETED,
                FileStatus.INDEX_RENAMED,
                FileStatus.INDEX_TYPECHANGE,
            ):
                break
        else:
            if not allow_empty:
                return None

        tree = await asyncio.to_thread(repo.index.write_tree)
        parents = [] if repo.head_is_unborn else [repo.head.target]
        author = committer = pygit2.Signature(
            name="Relace Agent",
            email="noreply@relace.ai",
        )
        commit = await asyncio.to_thread(
            repo.create_commit,
            "HEAD",
            author,
            committer,
            message,
            tree,
            parents,
        )
        return str(commit)

    # Intentionally not async to allow for synchronous usage in Context.interpolate()
    @tracer.start_as_current_span("Repo.list_tracked_files")
    def list_tracked_files(
        self,
        ignore_tracked: tuple[str, ...] = (
            "relace.yaml",
            ".gitignore",
        ),
    ) -> list[Path]:
        repo = self._repo()
        repo.index.read()
        # TODO: Filter out non-Unicode files
        return [
            abs_path
            for entry in repo.index
            for abs_path in [self.root_path / entry.path]
            if abs_path.is_file()
            and not any(
                fnmatch.fnmatch(entry.path, pattern) for pattern in ignore_tracked
            )
        ]

    def info(self) -> RepoInfo:
        return RepoInfo(
            repo_id=self.repo_id,
            repo_head=self.get_head(),
        )
