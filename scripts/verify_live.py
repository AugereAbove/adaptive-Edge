import asyncio
import json
import urllib.request
from contextlib import suppress
from pathlib import Path

import psutil
from typer.testing import CliRunner

from adaptiveedge.api import AdaptiveEdge
from adaptiveedge.cli.app import app
from adaptiveedge.config.schema import BenchmarkConfiguration
from adaptiveedge.datasets.baseline import baseline_tasks


def verify(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    original = BenchmarkConfiguration.model_validate_json(
        Path("outputs/baseline-real/calibration.json").read_text(encoding="utf-8")
    )
    config = original.model_copy(
        update={
            "models": original.models[:1],
            "tasks": baseline_tasks("calibration")[6:7],
            "experiment": original.experiment.model_copy(
                update={"name": "live-cli-smoke", "repetitions": 1}
            ),
        }
    )
    edge = AdaptiveEdge(destination / "live.db")
    try:
        identifier = asyncio.run(edge.benchmark(config))
        assert all(
            row.success and not row.inference.simulated
            for row in edge.repository.results(identifier)
        )
        response = asyncio.run(
            edge.run("reasoning", "What is 5 + 5? Give only the numeric answer.")
        )
        assert response.inference is not None and response.inference.error is None
        assert response.experiment_id != identifier
        assert edge.repository.results(response.experiment_id)[0].split == "inference"
    finally:
        edge.close()
    common = ["--db", str(destination / "live.db")]
    commands = [
        ["benchmark", "show", identifier, *common, "--json"],
        ["benchmark", "reproduce", identifier, *common, "--json"],
        ["select", "--task", "reasoning", *common, "--json"],
        ["export", identifier, str(destination / "raw.csv"), *common],
        ["export", identifier, str(destination / "raw.json"), *common],
        ["report", identifier, "--output", str(destination / "report"), *common],
    ]
    outcomes = []
    runner = CliRunner()
    for command in commands:
        outcome = runner.invoke(app, command)
        assert outcome.exit_code == 0, outcome.output
        outcomes.append(
            {"command": command[0:2], "exit_code": outcome.exit_code, "output": outcome.output}
        )
    snapshots = []
    for process in psutil.process_iter(["pid", "ppid", "name"]):
        if "ollama" in (process.info["name"] or "").casefold():
            with suppress(psutil.Error):
                snapshots.append({**process.info, "rss_bytes": process.memory_info().rss})
    loaded = json.load(urllib.request.urlopen("http://localhost:11434/api/ps", timeout=10))
    (destination / "verification.json").write_text(
        json.dumps(
            {
                "benchmark": identifier,
                "adaptive_execution": response.experiment_id,
                "commands": outcomes,
                "process_rss_snapshot": snapshots,
                "resident_models_snapshot": loaded,
                "memory_scope": (
                    "Postrun process RSS and server-reported resident allocation. "
                    "Neither is per-request model peak RAM/VRAM; "
                    "do not use for memory constraints."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    verify(Path("outputs/release/live-smoke"))
