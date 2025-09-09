from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import override

import modal

from relace_agent.server.types import RepoId


@dataclass
class StorageMount(ABC):
    storage_root: Path
    cache_root: Path

    @abstractmethod
    def mounts(
        self,
    ) -> dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]: ...

    def setup_cache(self) -> None:
        """Create cache root directory; must be run on app startup"""
        self.cache_root.mkdir(exist_ok=True)

    def repo_cache(self, repo_id: RepoId) -> Path:
        return self.cache_root / str(repo_id)

    def repo_storage(self, repo_id: RepoId) -> Path:
        return self.storage_root / str(repo_id)

    def repo_bundle(self, repo_id: RepoId) -> Path:
        return self.repo_storage(repo_id) / "bundle"

    def repo_history(self, repo_id: RepoId) -> Path:
        return self.repo_storage(repo_id) / "history.json"


@dataclass
class S3Storage(StorageMount):
    bucket_mount: modal.CloudBucketMount

    @override
    def mounts(
        self,
    ) -> dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]:
        return {
            self.storage_root.as_posix(): self.bucket_mount,
        }


STORAGE = S3Storage(
    bucket_mount=modal.CloudBucketMount(
        bucket_name="relace-repos",
        secret=modal.Secret.from_name("aws-secret"),
    ),
    storage_root=Path("/s3/repos"),
    cache_root=Path("/repos"),
)
