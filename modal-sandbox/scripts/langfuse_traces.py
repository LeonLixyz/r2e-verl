"""Script for interacting with project traces on Langfuse.

Set the following environment variables:
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-yyy
LANGFUSE_HOST=https://us.cloud.langfuse.com
LANGFUSE_TIMEOUT=30
"""

import csv
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Self

import typer
from langfuse import get_client
from langfuse.api import TraceWithDetails
from rich import print
from rich.console import Console, RenderableType
from rich.json import JSON
from rich.status import Status
from rich.style import Style
from rich.table import Table

from relace_agent.modal.agent_toolbox_eval import UiTestCase

app = typer.Typer(name="langfuse")


def iter_traces(
    user_id: str | None = None,
    page: int | None = None,
    page_size: int = 50,
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
    order_by: str | None = None,
) -> Iterator[TraceWithDetails]:
    langfuse = get_client()
    status_text = "Fetching traces"
    if user_id:
        status_text += f" ({user_id=})"

    with Status(status_text, spinner="dots") as status:
        while True:
            status.update(f"{status_text} (page {page or 1})")
            traces = langfuse.api.trace.list(
                user_id=user_id,
                limit=page_size,
                page=page,
                from_timestamp=from_timestamp,
                to_timestamp=to_timestamp,
                order_by=order_by,
            )
            yield from traces.data

            page = traces.meta.page + 1
            if page > traces.meta.total_pages:
                break


_JSON_PATH = re.compile(r"[^.\[]+|\[[^\]]*\]")


def resolve_json_path(data: Any, path: str) -> Any:
    current = data
    for token in _JSON_PATH.findall(path):
        if token.startswith("["):
            current = current[int(token[1:-1])]
        else:
            current = current[token]
    return current


@app.command("summary")
def summarize_traces(
    user_id: str = "qwen-apply",
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
) -> None:
    counts: Counter[str] = Counter()
    for trace in iter_traces(
        user_id=user_id,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    ):
        if trace.output:
            counts[trace.output] += 1

    console = Console()
    table = Table(title=f"Trace Outputs ({user_id})")
    table.add_column("Output", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("Percentage", style="green", justify="right")

    total = counts.total()
    for key, val in counts.items():
        percentage = f"{(val / total) * 100:.2f}%"
        table.add_row(str(key), str(val), percentage)

    table.add_section()
    table.add_row("total", str(total))
    console.print(table)


@app.command("list")
def list_traces(
    user_id: str = "qwen-apply",
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
    input_json_path: str | None = None,
    filter_output: str | None = None,
    include_input: bool = True,
    include_output: bool = True,
    include_latency: bool = True,
    show_lines: bool = True,
) -> None:
    table = Table(title=f"Traces ({user_id})", show_lines=show_lines)
    table.add_column("ID", style="cyan", no_wrap=True)
    if include_input:
        table.add_column("Input", style="magenta")
    if include_output:
        table.add_column("Output", style="green", justify="right")
    if include_latency:
        table.add_column("Latency (sec)", style="yellow", justify="right")
    for trace in iter_traces(
        user_id=user_id,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    ):
        if filter_output and filter_output != trace.output:
            continue

        row: list[RenderableType] = [trace.id]
        if include_input:
            data = trace.input
            if input_json_path:
                data = resolve_json_path(data, input_json_path)
            row.append(JSON.from_data(data))
        if include_output:
            row.append(f"{trace.output!s}" if trace.output else "")
        if include_latency:
            row.append(f"{trace.latency:.3f}")

        table.add_row(*row)

    console = Console()
    console.print(table)


@app.command("dupes")
def dupes(
    user_id: str = "qwen-apply",
    delete_old: bool = False,
) -> None:
    # Use dict for groups to avoid ingesting duplicate traces
    trace_groups: defaultdict[str, dict[str, TraceWithDetails]] = defaultdict(dict)
    for trace in iter_traces(user_id=user_id, order_by="timestamp.desc"):
        image_tag = resolve_json_path(trace.input, "args[0].image_tag")
        trace_groups[image_tag][trace.id] = trace

    table = Table(title=f"Traces ({user_id})")
    table.add_column("Image Tag", style="cyan")
    table.add_column("Trace ID", style="cyan")
    table.add_column("Output", style="green", justify="right")
    table.add_column("Latency (sec)", style="yellow", justify="right")
    table.add_column("Time", style="yellow")
    stale: list[TraceWithDetails] = []
    for image_tag, traces in trace_groups.items():
        if len(traces) < 2:
            continue

        table.add_section()
        for i, trace in enumerate(traces.values()):
            style: Style | None = None
            if i > 0:
                style = Style(dim=True)
                stale.append(trace)
            table.add_row(
                image_tag,
                trace.id,
                trace.output,
                f"{trace.latency:.3f}",
                trace.timestamp.strftime("%d/%m/%Y, %H:%M:%S"),
                style=style,
            )

    console = Console()
    console.print(table)
    if delete_old:
        with console.status(f"[red]Deleting {len(stale)} stale traces"):
            langfuse = get_client()
            langfuse.api.trace.delete_multiple(trace_ids=[trace.id for trace in stale])
        console.print(f"[red]Deleted {len(stale)} stale traces")
    else:
        console.print(f"[yellow]Found {len(stale)} stale traces")


@app.command("compare")
def compare(
    user_ids: list[str],
    diff_only: bool = False,
) -> None:
    # Group traces by image_tag across all users
    image_traces: defaultdict[str, list[TraceWithDetails]] = defaultdict(list)
    for user in user_ids:
        for trace in iter_traces(user_id=user):
            image_tag = resolve_json_path(trace.input, "args[0].image_tag")
            image_traces[image_tag].append(trace)

    table = Table(title="Results By Test Case")
    table.add_column("Image Tag", style="cyan")
    table.add_column("Trace ID", style="cyan")
    table.add_column("User", justify="left")
    table.add_column("Latency (sec)", style="yellow", justify="right")

    user_scores: defaultdict[str, Counter[str]] = defaultdict(Counter)
    user_events: defaultdict[str, list[int]] = defaultdict(list)
    user_latencies: defaultdict[str, list[float]] = defaultdict(list)
    for image_tag, traces in image_traces.items():
        # Skip traces where outcomes are all the same
        if diff_only and len({trace.output for trace in traces}) == 1:
            continue

        table.add_section()
        for trace in traces:
            if trace.user_id is None or trace.output is None:
                print(f"[yellow]Skipping invalid trace:[/yellow] {trace.id}")
                continue

            match trace.output:
                case "success":
                    style = "[green]"
                case "partial_success":
                    style = "[yellow]"
                case "failure":
                    style = "[red]"
                case other:
                    raise ValueError(f"Unexpected output: {other!r}")
            table.add_row(
                image_tag,
                trace.id,
                f"{style}{trace.user_id}",
                f"{trace.latency:.3f}",
            )
            user_scores[trace.user_id][trace.output] += 1
            user_events[trace.user_id].append(len(trace.observations))
            user_latencies[trace.user_id].append(trace.latency)

    score_table = Table(title="Results By User")
    score_table.add_column("User", style="cyan", justify="left")
    score_table.add_column("success rate", style="green", justify="right")
    score_table.add_column("improvement rate", style="yellow", justify="right")
    score_table.add_column("events (avg)", style="blue", justify="right")
    score_table.add_column("events (med)", style="blue", justify="right")
    score_table.add_column("latency (avg)", style="yellow", justify="right")
    score_table.add_column("latency (med)", style="yellow", justify="right")
    for user, scores in user_scores.items():
        success_rate = scores["success"] / scores.total()
        improvement_rate = (
            scores["success"] + scores["partial_success"]
        ) / scores.total()
        events = user_events[user]
        events_avg = sum(events) / len(events)
        events_med = events[len(events) // 2]
        latencies = user_latencies[user]
        latency_avg = sum(latencies) / len(latencies)
        latency_med = latencies[len(latencies) // 2]
        score_table.add_row(
            user,
            f"{success_rate * 100:.1f}%",
            f"{improvement_rate * 100:.1f}%",
            f"{round(events_avg)}",
            f"{events_med}",
            f"{latency_avg:.3f}",
            f"{latency_med:.3f}",
        )

    console = Console()
    console.print(table)
    console.print(score_table)


@dataclass
class UiTrace:
    trace_id: str
    user_id: str
    prompt: str
    slug: str | None

    @classmethod
    def from_langfuse(cls, trace: TraceWithDetails) -> Self:
        if trace.input is None:
            raise ValueError("Trace is missing input")
        if trace.session_id is None:
            raise ValueError("Trace is missing session_id")

        test_case_json = resolve_json_path(trace.input, "args[0]")
        test_case = UiTestCase.model_validate(test_case_json)
        return cls(
            trace_id=trace.id,
            user_id=test_case.matrix_key,
            prompt=test_case.user_prompt,
            slug=f"eval-{trace.session_id}" if trace.output == "success" else None,
        )

    def as_row(self, model_num: int) -> dict[str, str | None]:
        return {
            f"trace_{model_num}": self.trace_id,
            f"model_{model_num}": self.user_id,
            f"slug_{model_num}": self.slug,
        }


@app.command("ui-arena")
def ui_arena(
    csv_in: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    csv_out: Annotated[Path, typer.Argument(writable=True, dir_okay=False)],
    user_groups: Annotated[
        list[str],
        typer.Option(
            "--user-group",
            help="Comma separated list of user IDs to be grouped (user1,user2,...)",
        ),
    ],
) -> None:
    groups = [group.split(",") for group in user_groups]
    groups_sizes = {len(group) for group in groups}
    if len(groups_sizes) != 1:
        raise ValueError("All user groups must have the same number of users")
    group_size = groups_sizes.pop()

    # Group traces by prompt for each user group
    traces_by_prompt: defaultdict[tuple[int, str], list[UiTrace]] = defaultdict(list)
    for group_idx, group in enumerate(groups):
        for user_id in group:
            for trace in iter_traces(user_id=user_id):
                ui_trace = UiTrace.from_langfuse(trace)
                traces_by_prompt[(group_idx, ui_trace.prompt)].append(ui_trace)

    with csv_in.open("r") as f_in, csv_out.open("w") as f_out:
        csv_reader = csv.DictReader(f_in)
        # Insert headers if there are additional user groups
        csv_fields = set(csv_reader.fieldnames or [])
        for i in range(group_size):
            csv_fields.add(f"trace_{i + 1}")
            csv_fields.add(f"model_{i + 1}")
            csv_fields.add(f"slug_{i + 1}")

        csv_writer = csv.DictWriter(f_out, fieldnames=sorted(csv_fields))
        csv_writer.writeheader()
        for row in csv_reader:
            # Create a new row for each user group with results
            for group_idx in range(len(groups)):
                ui_traces = traces_by_prompt.get((group_idx, row["prompt"]), [])
                if ui_traces:
                    # Remove duplicates from same user
                    ui_traces = list(
                        {trace.user_id: trace for trace in ui_traces}.values()
                    )

                    # Randomize order to avoid bias
                    random.shuffle(ui_traces)
                    group_row = row.copy()
                    for model_num, model_trace in enumerate(ui_traces, start=1):
                        group_row.update(model_trace.as_row(model_num=model_num))
                    csv_writer.writerow(group_row)


if __name__ == "__main__":
    app()
