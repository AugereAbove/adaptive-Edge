from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import platform
import sqlite3
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, ParamSpec, TypeVar

import typer
from pydantic import ValidationError

from adaptiveedge.api import AdaptiveEdge, default_registry
from adaptiveedge.benchmarks.evaluation import evaluate_policies
from adaptiveedge.benchmarks.runner import BenchmarkRunner
from adaptiveedge.config.schema import load_config
from adaptiveedge.core.models import Category, Constraints, TaskDefinition
from adaptiveedge.profiling.hardware import hardware_snapshot
from adaptiveedge.reporting.export import export_csv, export_json
from adaptiveedge.reporting.reports import generate_report
from adaptiveedge.selection.engine import Selector, Weights
from adaptiveedge.storage.repository import Repository

app = typer.Typer(
    no_args_is_help=True, help="Local model benchmarking and evidence-based selection."
)
benchmarks = typer.Typer(no_args_is_help=True, help="Run, inspect and reproduce experiments.")
app.add_typer(benchmarks, name="benchmark")
debug_enabled = False
P = ParamSpec("P")
R = TypeVar("R")
DatabaseOption = Annotated[Path, typer.Option("--db", help="Local SQLite database")]
JsonOption = Annotated[bool, typer.Option("--json", help="Machine-readable output")]


def guarded(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as exc:
            if debug_enabled:
                raise
            message = {"status": "error", "error": type(exc).__name__, "detail": str(exc)}
            typer.echo(
                json.dumps(message)
                if kwargs.get("json_output")
                else f"Error ({type(exc).__name__}): {exc}",
                err=True,
            )
            raise typer.Exit(code=2) from exc

    return wrapper


def emit(value: object, as_json: bool = True) -> None:
    typer.echo(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) if as_json else str(value)
    )


@app.callback()
def settings(
    debug: Annotated[bool, typer.Option(help="Show exception tracebacks")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    global debug_enabled
    debug_enabled = debug
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR, format="%(levelname)s %(message)s"
    )


@app.command()
@guarded
def hardware(json_output: JsonOption = False) -> None:
    snapshot = hardware_snapshot()
    emit(snapshot.model_dump(mode="json"), json_output)


@app.command()
@guarded
def doctor(db: DatabaseOption = Path("adaptiveedge.db"), json_output: JsonOption = False) -> None:
    registry = default_registry()

    async def availability() -> dict[str, object]:
        return {
            name: {
                **(value := await registry.get(name).availability()).model_dump(mode="json"),
                "status": "ok" if value.available else "unavailable",
            }
            for name in registry.names
        }

    try:
        with Repository(db) as repository:
            integrity = repository.connection.execute("PRAGMA quick_check").fetchone()[0]
        database = {"integrity": integrity, "status": "ok" if integrity == "ok" else "error"}
    except (OSError, sqlite3.Error, ValueError) as exc:
        database = {"status": "error", "detail": str(exc)}
    try:
        snapshot = hardware_snapshot().model_dump(mode="json")
        hardware_status = "ok"
    except Exception as exc:
        snapshot = {"error": str(exc)}
        hardware_status = "error"
    from adaptiveedge import __version__

    emit(
        {
            "version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "database": database,
            "hardware_status": hardware_status,
            "plotting_status": "ok" if importlib.util.find_spec("matplotlib") else "unavailable",
            "hardware": snapshot,
            "backends": asyncio.run(availability()),
            "optional_dependencies": {
                name: importlib.util.find_spec(name) is not None
                for name in ("httpx", "transformers", "matplotlib")
            },
        },
        json_output,
    )


@app.command()
@guarded
def models(backend: str = "mock", json_output: JsonOption = False) -> None:
    runtime = default_registry().get(backend)
    available = asyncio.run(runtime.availability())
    if not available.available:
        raise ValueError(available.detail)
    emit({"backend": backend, "models": asyncio.run(runtime.discover())}, json_output)


@benchmarks.command("run")
@guarded
def benchmark_run(
    config: Path, db: DatabaseOption = Path("adaptiveedge.db"), json_output: JsonOption = False
) -> None:
    configuration = load_config(config)
    with Repository(db) as repository:
        identifier = asyncio.run(BenchmarkRunner(repository, default_registry()).run(configuration))
        rows = repository.results(identifier)
        failures = sum(not row.success for row in rows)
    emit(
        {
            "experiment_id": identifier,
            "runs": len(rows),
            "failures": failures,
            "simulated": any(row.inference.simulated for row in rows),
        },
        json_output,
    )
    if failures:
        raise typer.Exit(1)


@benchmarks.command("list")
@guarded
def benchmark_list(
    db: DatabaseOption = Path("adaptiveedge.db"), json_output: JsonOption = False
) -> None:
    with Repository(db) as repository:
        emit(repository.experiments(), json_output)


@benchmarks.command("show")
@guarded
def benchmark_show(
    experiment_id: str,
    db: DatabaseOption = Path("adaptiveedge.db"),
    json_output: JsonOption = False,
) -> None:
    with Repository(db) as repository:
        emit(
            {
                "manifest": repository.manifest(experiment_id),
                "runs": len(repository.results(experiment_id)),
            },
            json_output,
        )


@benchmarks.command("reproduce")
@guarded
def reproduce(
    experiment_id: str,
    db: DatabaseOption = Path("adaptiveedge.db"),
    json_output: JsonOption = False,
) -> None:
    with Repository(db) as repository:
        configuration = repository.configuration(experiment_id)
        manifest = repository.manifest(experiment_id)
        extra = manifest.get("extra")
        if isinstance(extra, dict) and extra.get("kind") in {
            "policy_evaluation",
            "adaptive_inference",
        }:
            raise ValueError(
                "reproduce accepts benchmark experiments; "
                "rerun evaluation with its frozen training manifest"
            )
        resolved_tasks = extra.get("resolved_tasks") if isinstance(extra, dict) else None
        if isinstance(resolved_tasks, list):
            tasks = tuple(TaskDefinition.model_validate(t) for t in resolved_tasks)
            configuration = configuration.model_copy(update={"suite": "custom", "tasks": tasks})
        registry = default_registry()

        async def validate_models() -> None:
            for model in configuration.models:
                async with asyncio.timeout(configuration.experiment.timeout_seconds):
                    await registry.get(model.backend).validate(model)

        asyncio.run(validate_models())
        identifier = asyncio.run(BenchmarkRunner(repository, registry).run(configuration))
        repository.event(identifier, "reproduction_of", {"experiment_id": experiment_id})
        failures = sum(not r.success for r in repository.results(identifier))
    emit(
        {
            "experiment_id": identifier,
            "reproduced_from": experiment_id,
            "failures": failures,
            "notice": (
                "Rerun with archived tasks; model tags, runtime and hardware may have changed"
            ),
        },
        json_output,
    )
    if failures:
        raise typer.Exit(1)


def request_constraints(
    max_latency_ms: float | None,
    max_memory_gb: float | None,
    min_quality: float | None,
    constraint_file: Path | None,
) -> Constraints:
    values = (
        {} if constraint_file is None else json.loads(constraint_file.read_text(encoding="utf-8"))
    )
    for key, value in (
        ("max_latency_ms", max_latency_ms),
        ("max_memory_gb", max_memory_gb),
        ("min_quality", min_quality),
    ):
        if value is not None:
            values[key] = value
    return Constraints.model_validate(values)


def make_selector(weights_file: Path | None) -> Selector:
    weights = (
        None
        if weights_file is None
        else Weights.model_validate_json(weights_file.read_text(encoding="utf-8"))
    )
    return Selector(weights=weights)


@app.command("select")
@guarded
def select_model(
    task: Annotated[Category, typer.Option()],
    db: DatabaseOption = Path("adaptiveedge.db"),
    policy: str = "adaptive",
    max_latency_ms: float | None = None,
    max_memory_gb: float | None = None,
    min_quality: float | None = None,
    constraints_file: Path | None = None,
    weights_file: Path | None = None,
    json_output: JsonOption = False,
) -> None:
    constraints = request_constraints(max_latency_ms, max_memory_gb, min_quality, constraints_file)
    edge = AdaptiveEdge(db, selector=make_selector(weights_file))
    try:
        result = edge.select(task, constraints, policy)
    finally:
        edge.close()
    emit(result.model_dump(mode="json") if json_output else result.reason, json_output)
    if result.selected is None:
        raise typer.Exit(3)


@app.command("run")
@guarded
def run_inference(
    task: Annotated[Category, typer.Option()],
    input_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    db: DatabaseOption = Path("adaptiveedge.db"),
    policy: str = "adaptive",
    max_latency_ms: float | None = None,
    max_memory_gb: float | None = None,
    min_quality: float | None = None,
    constraints_file: Path | None = None,
    weights_file: Path | None = None,
    json_output: JsonOption = False,
) -> None:
    constraints = request_constraints(max_latency_ms, max_memory_gb, min_quality, constraints_file)
    edge = AdaptiveEdge(db, selector=make_selector(weights_file))
    try:
        result = asyncio.run(
            edge.run(task, input_file.read_text(encoding="utf-8"), constraints, policy)
        )
    finally:
        edge.close()
    if json_output:
        emit(result.model_dump(mode="json"))
    else:
        emit(f"{result.selection.reason}\n\n{result.output or ''}", False)
    if result.inference is None:
        raise typer.Exit(3)
    if result.inference.error:
        raise typer.Exit(1)


@app.command("evaluate")
@guarded
def evaluate(
    training_id: str,
    config: Path,
    db: DatabaseOption = Path("adaptiveedge.db"),
    constraints_file: Path | None = None,
    weights_file: Path | None = None,
    allow_overlap: bool = False,
    json_output: JsonOption = False,
) -> None:
    configuration = load_config(config)
    constraints = request_constraints(None, None, None, constraints_file)
    with Repository(db) as repository:
        identifier = asyncio.run(
            evaluate_policies(
                repository,
                default_registry(),
                hardware_snapshot(),
                training_id,
                configuration,
                constraints,
                make_selector(weights_file),
                allow_overlap=allow_overlap,
            )
        )
        failures = sum(not r.success for r in repository.results(identifier))
    emit({"experiment_id": identifier, "failures": failures}, json_output)
    if failures:
        raise typer.Exit(1)


@app.command()
@guarded
def report(
    experiment_id: str,
    output: Path = Path("outputs/report"),
    db: DatabaseOption = Path("adaptiveedge.db"),
    plots: bool = False,
    json_output: JsonOption = False,
) -> None:
    with Repository(db) as repository:
        path = generate_report(repository, experiment_id, output)
        if plots:
            from adaptiveedge.reporting.plots import plot_quality_latency, publication_plots

            try:
                plot_quality_latency(
                    repository.results(experiment_id), output / "quality-latency.png"
                )
                publication_plots(repository.results(experiment_id), output / "figures")
            except ImportError as exc:
                raise ValueError("Install adaptiveedge[plotting] for plots") from exc
    emit({"status": "ok", "path": str(path)}, True) if json_output else typer.echo(str(path))


@app.command("export")
@guarded
def export_results(
    experiment_id: str,
    output: Path,
    db: DatabaseOption = Path("adaptiveedge.db"),
    json_output: JsonOption = False,
) -> None:
    with Repository(db) as repository:
        repository.configuration(experiment_id)
        rows = repository.results(experiment_id)
    if output.suffix == ".csv":
        export_csv(rows, output)
    elif output.suffix == ".json":
        export_json(rows, output)
    else:
        raise ValueError("export filename must end in .json or .csv")
    emit({"status": "ok", "path": str(output)}, True) if json_output else typer.echo(str(output))


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("Interrupted; completed observations were preserved.", err=True)
        raise SystemExit(130) from None
    except ValidationError as exc:
        typer.echo(str(exc), err=True)
        raise SystemExit(2) from exc
