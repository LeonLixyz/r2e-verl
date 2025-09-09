import asyncio
import itertools
import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import clickhouse_connect
import httpx
import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from relace_agent.server.types import (
    File,
    RepoCreateFilesSource,
    RepoCreateRequest,
    RepoId,
    RepoInfo,
    RepoUpdateFiles,
    RepoUpdateRequest,
)

logger = logging.getLogger(__name__)

app = typer.Typer()


@dataclass
class Result:
    status_code: int
    latency: float
    response: httpx.Response | None = None
    error: Exception | None = None


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    request: httpx.Request,
) -> Result:
    # Build request to merge base_url and headers
    request = client.build_request(
        method=request.method,
        url=request.url,
        content=request.content,
        headers=request.headers,
    )
    async with semaphore:
        time_start = time.monotonic()
        try:
            result = await client.send(request)
            result.raise_for_status()
            return Result(
                status_code=result.status_code,
                latency=time.monotonic() - time_start,
                response=result,
            )
        except Exception as e:
            logger.error("Error during request", exc_info=True)
            return Result(
                status_code=500,
                latency=time.monotonic() - time_start,
                error=e,
            )


async def send_requests_parallel(
    base_url: str,
    requests: list[httpx.Request],
    *,
    name: str = "Requests",
    api_key: str,
    timeout: float = 30.0,
    parallel: int = 1,
) -> list[Result]:
    semaphore = asyncio.Semaphore(parallel)
    async with httpx.AsyncClient(
        timeout=timeout,
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    ) as client:
        results: list[Result] = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task(
                f"[cyan bold]{name}",
                total=len(requests),
            )
            tasks = [
                asyncio.create_task(
                    send_request(client=client, semaphore=semaphore, request=request)
                )
                for request in requests
            ]
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                progress.update(task_id, advance=1)
        return results


def print_results(results: list[Result]) -> None:
    # Count response codes
    status_counts: dict[int, int] = Counter()
    latencies = []

    for r in results:
        status_counts[r.status_code] += 1
        if r.status_code == 200:
            latencies.append(r.latency)

    total = len(results)
    success = status_counts.get(200, 0)
    success_rate = (success / total) * 100 if total else 0

    # Prepare table
    table = Table(title="Load Test Results")
    table.add_column("Status Code", justify="right")
    table.add_column("Count", justify="right")
    for code, count in sorted(status_counts.items()):
        table.add_row(str(code), str(count))

    console = Console()
    console.print(table)
    console.print(f"[bold]Success Rate:[/bold] {success_rate:.2f}%")

    if latencies:
        console.print("[bold]Latency (s) for successful requests:[/bold]")
        console.print(f"  Avg: {np.mean(latencies):.3f}")
        console.print(f"  p50: {np.percentile(latencies, 50):.3f}")
        console.print(f"  p90: {np.percentile(latencies, 90):.3f}")
        console.print(f"  p99: {np.percentile(latencies, 99):.3f}")
    else:
        console.print(
            "[bold red]No successful requests to calculate latency percentiles.[/bold red]"
        )


def user_request_dir(user_id: str) -> Path:
    return Path(__file__).parent / "data" / "ranker" / user_id


def iter_test_repos(request_dir: Path) -> Iterator[list[File]]:
    for file_path in request_dir.iterdir():
        file_data = json.loads(file_path.read_text(encoding="utf-8"))
        yield [
            File(
                filename=f["filename"],
                content=f.get("content") or f["code"],
            )
            for f in file_data["codebase"]
        ]


@app.command("scrape")
def scrape_test_repos(
    user_id: str,
    limit: int = 10,
    http_code: int = 200,
    last_n_days: int = 7,
    clickhouse_url: str = typer.Option(..., envvar="CLICKHOUSE_URL"),
    clickhouse_username: str = typer.Option(..., envvar="CLICKHOUSE_USERNAME"),
    clickhouse_password: str = typer.Option(..., envvar="CLICKHOUSE_PASSWORD"),
) -> None:
    out_path = user_request_dir(user_id)
    out_path.mkdir(exist_ok=True, parents=True)
    client = clickhouse_connect.get_client(
        host=clickhouse_url,
        port=8443,
        username=clickhouse_username,
        password=clickhouse_password,
        secure=True,
    )

    # Construct query
    conditions = [
        "request_data != '\"\"'",
        "request_url LIKE '%/code/rank'",
        f"user_id = '{user_id}'",
        f"response_http_code = {http_code}",
        f"timestamp >= now() - INTERVAL {last_n_days} DAY",
    ]
    query = f"""
        SELECT (request_data)
        FROM logs
        WHERE {" AND ".join(conditions)}
        ORDER BY RAND()
        LIMIT {limit}
    """

    with client.query_rows_stream(query) as stream:
        for i, row in enumerate(stream, start=1):
            data = json.loads(json.loads(row[0]))
            file_path = out_path / f"request_{i}.json"
            with file_path.open("w") as f:
                json.dump(data, f, indent=2)


def repo_create_request(body: RepoCreateRequest) -> httpx.Request:
    return httpx.Request(
        method="POST",
        url="/repo",
        json=body.model_dump(mode="json"),
    )


@app.command("create")
def repo_create(
    user_id: str,
    parallel_requests: int = 1,
    total_requests: int | None = None,
    base_url: str = "https://api.relace.run/v1",
    timeout: int = 30,
    api_key: str = typer.Option(..., envvar="RELACE_API_KEY"),
) -> None:
    run_id = str(uuid.uuid4())
    request_dir = user_request_dir(user_id)
    requests: list[httpx.Request] = []
    for files in iter_test_repos(request_dir):
        request = RepoCreateRequest(
            source=RepoCreateFilesSource(type="files", files=files),
            metadata={
                "tag": "load_test",
                "user_id": user_id,
                "run_id": run_id,
            },
        )
        requests.append(repo_create_request(request))
    if not requests:
        raise ValueError(f"No requests found in {request_dir}")

    # Expand requests to match total_requests
    if total_requests:
        source_repeats = total_requests // len(requests)
        source_remainder = total_requests % len(requests)
        requests = requests * source_repeats + requests[:source_remainder]

    results: list[Result] = asyncio.run(
        send_requests_parallel(
            base_url=base_url,
            requests=requests,
            api_key=api_key,
            timeout=timeout,
            parallel=parallel_requests,
            name="Create repos",
        )
    )
    print_results(results)


def repo_update_request(repo_id: RepoId, body: RepoUpdateRequest) -> httpx.Request:
    return httpx.Request(
        method="POST",
        url=f"/repo/{repo_id}/update",
        json=body.model_dump(mode="json"),
    )


@app.command("update")
def repo_update(
    user_id: str,
    parallel_updates: int = 1,
    sequential_updates: int = 1,
    base_url: str = "https://api.relace.run/v1",
    timeout: int = 30,
    api_key: str = typer.Option(..., envvar="RELACE_API_KEY"),
) -> None:
    run_id = str(uuid.uuid4())

    # Create repo for each parallel update sequence
    repo_create_requests: list[httpx.Request] = []
    for i in range(parallel_updates):
        request = RepoCreateRequest(
            metadata={
                "tag": "load_test",
                "user_id": user_id,
                "run_id": run_id,
                "repo": f"repo_{i}",
            }
        )
        repo_create_requests.append(repo_create_request(request))
    create_results = asyncio.run(
        send_requests_parallel(
            base_url=base_url,
            requests=repo_create_requests,
            api_key=api_key,
            timeout=timeout,
            parallel=parallel_updates,
            name="Create repos",
        )
    )
    repo_ids = []
    for result in create_results:
        if result.status_code != 200 or not result.response:
            raise ValueError(f"Failed to create repo: {result}")
        repo_ids.append(RepoInfo.model_validate(result.response.json()).repo_id)

    # Run sequential updates for each repo
    update_results: list[Result] = []
    request_dir = user_request_dir(user_id)
    for files in itertools.islice(iter_test_repos(request_dir), sequential_updates):
        update_request = RepoUpdateRequest(
            source=RepoUpdateFiles(type="files", files=files),
        )
        update_requests = [
            repo_update_request(repo_id=repo_id, body=update_request)
            for repo_id in repo_ids
        ]
        update_results.extend(
            asyncio.run(
                send_requests_parallel(
                    base_url=base_url,
                    requests=update_requests,
                    api_key=api_key,
                    timeout=timeout,
                    parallel=parallel_updates,
                    name="Update repos",
                )
            )
        )
    print_results(update_results)


if __name__ == "__main__":
    app()
