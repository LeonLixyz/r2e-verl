import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Literal, TypedDict, cast

import clickhouse_connect
import modal
from langfuse import get_client, observe

from relace_agent.context import Context
from relace_agent.errors import BuildError, TestError
from relace_agent.logging import setup_logging
from relace_agent.modal.agent_runner import get_db
from relace_agent.repo import Repo
from relace_agent.server.storage import STORAGE
from relace_agent.server.types import RepoId

logger = logging.getLogger(__name__)

app = modal.App(name="Ranker-Repos")

image = modal.Image.debian_slim(python_version="3.13").apt_install("git").uv_sync()

test_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "build-essential", "clang", "curl")
    .run_commands(
        # Install Node.js 22 using official Node.js repository
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g depcheck",
    )
    .uv_sync()
    .run_commands("playwright install --with-deps --only-shell")
    .add_local_file("./relace.yaml", "/relace.yaml")
)

REPOS_RAW_VOL = modal.Volume.from_name("ranker-repos", create_if_missing=True)
REPOS_RAW_MNT = Path("/volumes/ranker-repos")

REPOS_POST_VOL = modal.Volume.from_name(
    "ranker-repos-categorized", create_if_missing=True
)
REPOS_POST_MNT = Path("/volumes/ranker-repos-categorized")

UUID_NAMESPACE = uuid.UUID("63fac21e-2e01-4767-81c0-5bdd71c441f2")


def construct_repo_id(request_data: str) -> RepoId:
    return RepoId(uuid.uuid5(UUID_NAMESPACE, request_data).hex)


async def construct_repo(bundle_root: Path, request_data: str) -> None:
    repo_id = construct_repo_id(request_data)
    repo_bundle_path = bundle_root / f"{repo_id}.bundle"
    if repo_bundle_path.exists():
        logger.info("Skipping existing repo %s", repo_id)
        return

    code_base = parse_code_base(request_data)
    if not code_base:
        logger.info("Skipping empty codebase for repo %s", repo_id)
        return

    repo = Repo.from_repo_id(repo_id)
    await repo.init()
    for code_file in code_base:
        code_path = repo.root_path / code_file["filename"]
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code_file["code"])
        logger.debug(
            "+ %s (%d bytes)",
            code_path.relative_to(repo.root_path),
            len(code_file["code"]),
        )

    await repo.commit("Added codebase from ranker")
    await repo.push_bundle(repo_bundle_path)
    logger.info("Saved %s (%d files)", repo_bundle_path, len(code_base))


async def push_repo(
    repo: Repo, category: Literal["build-fail", "test-fail", "test-pass"]
) -> None:
    repo_dest = REPOS_POST_MNT / category
    repo_dest.mkdir(exist_ok=True)
    await repo.push_bundle(repo_dest / f"{repo.repo_id}.bundle")
    logger.info("Pushed repo %s to %s", repo.repo_id, repo_dest)


DEFAULT_GITIGNORE = """
node_modules/
dist/
"""

DEFAULT_TAILWIND_CONFIG = """
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""


async def setup_gitignore(repo: Repo) -> None:
    gitignore_path = repo.root_path / ".gitignore"
    if gitignore_path.exists():
        return

    logger.info("Writing .gitignore")
    gitignore_path.write_text(DEFAULT_GITIGNORE)
    await repo.commit("Added .gitignore")


async def setup_tailwind(repo: Repo) -> None:
    tailwind_path = repo.root_path / "tailwind.config.js"
    if tailwind_path.exists():
        return

    logger.info("Writing tailwind.config.js")
    tailwind_path.write_text(DEFAULT_TAILWIND_CONFIG)
    await repo.commit("Write tailwind.config.js")


async def setup_package(repo: Repo) -> None:
    package_json_path = repo.root_path / "package.json"
    if package_json_path.exists():
        return

    logger.info("Writing package.json")
    package_json_path.write_text(
        json.dumps(
            {
                "name": str(repo.repo_id),
                "private": True,
                "version": "1.0.0",
                "scripts": {
                    "dev": "vite",
                    "build": "vite build",
                    "preview": "vite preview",
                },
                "type": "module",
                "dependencies": {},
                "devDependencies": {
                    "postcss": "^8.4.45",
                    "tailwindcss": "^3.4.10",
                    "typescript": "^5.6.0",
                    "vite": "^5.4.0",
                },
            }
        )
    )
    await repo.commit("Write package.json template")


async def setup_dependencies(repo: Repo) -> None:
    # Check for missing dependencies
    proc = await asyncio.create_subprocess_exec(
        "depcheck",
        "--json",
        cwd=repo.root_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    # Install missing dependencies
    missing_deps = json.loads(stdout.decode())["missing"]
    if missing_deps:
        logger.info("Installing missing dependencies: %s", " ".join(missing_deps))
        proc = await asyncio.create_subprocess_exec(
            "npm",
            "install",
            *missing_deps,
            cwd=repo.root_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("npm install failed")
        await repo.commit("Installed missing dependencies")


class CodeFile(TypedDict):
    filename: str
    code: str


def parse_code_base(request_data: str) -> list[CodeFile]:
    try:
        request_json_str = json.loads(request_data)
        request_json = json.loads(request_json_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON: %s", e)
        return []

    return cast(list[CodeFile], request_json["codebase"])


@app.function(
    image=image,
    volumes={
        REPOS_RAW_MNT.as_posix(): REPOS_RAW_VOL,
    },
    secrets=[
        modal.Secret.from_name("clickhouse-secrets"),
    ],
    timeout=60 * 60 * 4,
)
async def scrape_repos(
    user_id: str,
    http_code: int = 200,
    limit: int | None = None,
) -> None:
    setup_logging(logging.INFO)
    client = clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_URL"],
        port=8443,
        username=os.environ["CLICKHOUSE_USERNAME"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        secure=True,
    )

    # Construct query
    conditions = [
        "request_data != '\"\"'",
        "request_url LIKE '%/code/rank'",
        f"user_id = '{user_id}'",
        f"response_http_code = {http_code}",
    ]
    conditions_str = " AND ".join(conditions)
    query = f"SELECT (request_data) FROM logs WHERE {conditions_str}"
    if limit:
        query += f" LIMIT {limit}"

    logger.info(query)
    bundle_root = REPOS_RAW_MNT / user_id
    bundle_root.mkdir(exist_ok=True)
    with client.query_rows_stream(query) as stream:
        for row in stream:
            await construct_repo(bundle_root, row[0])


@app.function(
    image=test_image,
    volumes={
        REPOS_RAW_MNT.as_posix(): REPOS_RAW_VOL,
        REPOS_POST_MNT.as_posix(): REPOS_POST_VOL,
    },
    secrets=[
        modal.Secret.from_name("ranker-repos-langfuse"),
    ],
    max_containers=256,
)
@observe(capture_input=False, capture_output=False)
async def test_repo_bundle(bundle_path: Path, user_id: str) -> None:
    setup_logging(logging.INFO)
    repo_id = RepoId(bundle_path.stem)

    langfuse = get_client()
    langfuse.update_current_trace(
        user_id=user_id,
        session_id=str(repo_id),
    )

    with langfuse.start_as_current_span(name="setup"):
        repo = Repo.from_repo_id(repo_id)
        await repo.clone_bundle(bundle_path)
        await setup_gitignore(repo)
        await setup_tailwind(repo)
        await setup_package(repo)
        await setup_dependencies(repo)

    logger.info("Running tests")
    context = Context(repo=repo)

    try:
        with langfuse.start_as_current_span(name="build"):
            await context.run_build()

        with langfuse.start_as_current_span(name="test"):
            await context.run_test()
    except BuildError:
        logger.error("Build failed!")
        await push_repo(repo, "build-fail")
    except TestError:
        logger.error("Tests failed!")
        await push_repo(repo, "test-fail")
    else:
        logger.info("Tests passed!")
        await push_repo(repo, "test-pass")


@app.function(
    image=test_image,
    volumes={
        REPOS_RAW_MNT.as_posix(): REPOS_RAW_VOL,
    },
    timeout=60 * 60 * 4,
)
def test_repos(
    user_id: str,
    limit: int | None = None,
) -> None:
    setup_logging(logging.INFO)

    repo_bundle_root = REPOS_RAW_MNT / user_id
    logger.info("Listing %s", repo_bundle_root)
    bundles = list(repo_bundle_root.iterdir())
    if limit is not None:
        bundles = bundles[:limit]

    logger.info("Processing %d bundles", len(bundles))
    test_repo_bundle.for_each(
        bundles,
        kwargs={"user_id": user_id},
        ignore_exceptions=True,
    )
    logger.info("Done")


@app.function(
    image=test_image,
    volumes={
        REPOS_POST_MNT.as_posix(): REPOS_POST_VOL,
        **STORAGE.mounts(),
    },
    secrets=[
        modal.Secret.from_name("relace-agent-supabase"),
        modal.Secret.from_name("cloudflare-secret"),
    ],
    timeout=60 * 60,
)
async def export_repos(
    agent_user_id: str,
    category: str = "test-pass",
    limit: int = 10,
    deploy: bool = False,
) -> None:
    setup_logging(logging.INFO)

    db = get_db()
    ranker_repos = REPOS_POST_MNT / category
    logger.info("Exporting repos from %s", ranker_repos)
    for i, ranker_repo_bundle in enumerate(ranker_repos.iterdir()):
        if i >= limit:
            break

        logger.info("Exporting %s", ranker_repo_bundle)
        ranker_repo_id = RepoId(ranker_repo_bundle.stem)
        agent_repo_id: RepoId = uuid.uuid4()
        agent_repo = Repo.from_repo_id(agent_repo_id)
        await agent_repo.clone_bundle(ranker_repo_bundle)
        await db.insert_repo(
            agent_repo_id,
            agent_user_id,
            metadata={
                "slug": f"lovable-{ranker_repo_id}",
                "ranker_repo_id": str(ranker_repo_id),
                "ranker_category": category,
            },
            remote=str(ranker_repo_id),
            remote_branch=None,
            head=agent_repo.get_head(),
        )
        STORAGE.repo_storage(agent_repo_id).mkdir()
        await agent_repo.push_bundle()
        logger.info("Exported to %s", agent_repo_id)

        if deploy:
            context = Context(repo=agent_repo)
            await context.run_build()
            await context.run_deploy()
            logger.info("Deployed %s", agent_repo_id)
