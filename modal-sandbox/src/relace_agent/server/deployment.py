import asyncio
import functools
import logging
import os
from pathlib import Path
from typing import Any

import boto3

from relace_agent.server.types import RepoId

logger = logging.getLogger(__name__)

_S3_BUCKET = "relace-static-sites"


@functools.cache
def _s3_client() -> Any:
    """Initialize and return Cloudflare R2 S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["CLOUDFLARE_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CLOUDFLARE_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["CLOUDFLARE_R2_URL"],
    )


def _clear_cloudflare_bucket(prefix: str) -> None:
    logger.info(
        "Deleting existing files with prefix %s from bucket %s", prefix, _S3_BUCKET
    )

    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    delete_list: dict[str, Any] = {"Objects": []}

    # Use the paginator to handle more than 1000 objects
    for page in paginator.paginate(Bucket=_S3_BUCKET, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                delete_list["Objects"].append({"Key": obj["Key"]})

                # Delete in batches of 1000 objects (S3 API limit)
                if len(delete_list["Objects"]) >= 1000:
                    client.delete_objects(Bucket=_S3_BUCKET, Delete=delete_list)
                    logger.info(
                        "Deleted batch of %d objects", len(delete_list["Objects"])
                    )
                    delete_list = {"Objects": []}

    # Delete any remaining objects
    if len(delete_list["Objects"]) > 0:
        client.delete_objects(Bucket=_S3_BUCKET, Delete=delete_list)
        logger.info("Deleted remaining %d objects", len(delete_list["Objects"]))


async def clear_cloudflare_bucket(prefix: str) -> None:
    """Delete all objects with given prefix from Cloudflare R2 bucket."""
    await asyncio.to_thread(_clear_cloudflare_bucket, prefix)


def _upload_to_cloudflare(local_folder: str, prefix: str) -> bool:
    logger.info(
        "Uploading contents of %s to s3://%s/%s", local_folder, _S3_BUCKET, prefix
    )

    if not os.path.exists(local_folder):
        logger.warning("Folder %s does not exist, skipping upload", local_folder)
        return False

    # Recursively upload files
    client = _s3_client()
    for root, _, files in os.walk(local_folder):
        for file in files:
            full_path = os.path.join(root, file)
            key = os.path.relpath(full_path, local_folder).replace("\\", "/")
            if prefix:
                key = f"{prefix}{key}"
            logger.info("Uploading %s to s3://%s/%s", full_path, _S3_BUCKET, key)
            client.upload_file(full_path, _S3_BUCKET, key)

    logger.info(
        "Successfully uploaded all files from %s to s3://%s/%s",
        local_folder,
        _S3_BUCKET,
        prefix,
    )
    return True


async def upload_to_cloudflare(local_folder: str, prefix: str) -> bool:
    """Upload the built Vite app to Cloudflare R2."""
    return await asyncio.to_thread(_upload_to_cloudflare, local_folder, prefix)


def _deploy_to_cloudflare(local_dist_path: Path, prefix: str) -> bool:
    _clear_cloudflare_bucket(prefix)
    return _upload_to_cloudflare(
        local_folder=str(local_dist_path),
        prefix=prefix,
    )


async def deploy_to_cloudflare(local_dist_path: Path, prefix: str) -> bool:
    """Deploy the built Vite app to Cloudflare R2."""
    return await asyncio.to_thread(_deploy_to_cloudflare, local_dist_path, prefix)


def repo_dist(repo_id: RepoId) -> str:
    return f"{repo_id}/dist/"
