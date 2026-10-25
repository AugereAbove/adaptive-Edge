import argparse
import asyncio
import json
import urllib.request
from pathlib import Path

from adaptiveedge.api import default_registry
from adaptiveedge.benchmarks.evaluation import evaluate_policies
from adaptiveedge.benchmarks.runner import BenchmarkRunner
from adaptiveedge.config.schema import BenchmarkConfiguration, ExperimentSettings
from adaptiveedge.core.models import Constraints, Generation, ModelConfiguration, RuntimeOption
from adaptiveedge.datasets.baseline import baseline_tasks
from adaptiveedge.profiling.hardware import hardware_snapshot
from adaptiveedge.reporting.reports import generate_report
from adaptiveedge.selection.engine import Selector, Weights
from adaptiveedge.storage.repository import Repository


async def run(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    endpoint = "http://localhost:11434"
    inventory = json.load(urllib.request.urlopen(endpoint + "/api/tags", timeout=10))["models"]
    selected = [
        next(row for row in inventory if row["name"] == name)
        for name in ("qwen2.5:0.5b", "smollm2:1.7b", "qwen3:4b")
    ]
    (destination / "models.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    models = tuple(
        ModelConfiguration(
            backend="ollama",
            model=row["name"],
            size_gb=row["size"] / 1e9,
            family=row["details"]["family"],
            quantization=row["details"]["quantization_level"],
            context_size=2048,
            generation=Generation(max_output_tokens=96),
            options=(RuntimeOption(name="think", value=False),)
            if row["name"].startswith("qwen3:")
            else (),
        )
        for row in selected
    )
    hardware = hardware_snapshot()
    registry = default_registry()
    config = BenchmarkConfiguration(
        experiment=ExperimentSettings(name="real-calibration", repetitions=3, warmup_runs=1),
        models=models,
        suite="custom",
        tasks=baseline_tasks("calibration"),
    )
    heldout = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(
                update={"name": "real-heldout", "split": "evaluation"}
            ),
            "tasks": baseline_tasks("evaluation"),
        }
    )
    for name, value in (("calibration", config), ("evaluation", heldout)):
        (destination / f"{name}.json").write_text(
            value.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    with Repository(destination / "baseline.db") as repository:
        training = await BenchmarkRunner(repository, registry, hardware).run(config)
        print("Calibration", training, flush=True)
        generate_report(repository, training, destination / "calibration")
        constraints = Constraints(max_latency_ms=15000)
        evaluation = await evaluate_policies(
            repository, registry, hardware, training, heldout, constraints
        )
        print("Evaluation", evaluation, flush=True)
        generate_report(repository, evaluation, destination / "evaluation")
        ablation = await evaluate_policies(
            repository,
            registry,
            hardware,
            training,
            heldout.model_copy(
                update={
                    "experiment": heldout.experiment.model_copy(
                        update={"name": "equal-weights-ablation"}
                    )
                }
            ),
            constraints,
            Selector(weights=Weights(quality=1, latency_ms=1, ram_gb=0)),
            policies=("adaptive",),
        )
        print("Ablation", ablation, flush=True)
        generate_report(repository, ablation, destination / "ablation")
        ids = {"calibration": training, "evaluation": evaluation, "ablation": ablation}
        (destination / "experiments.json").write_text(
            json.dumps(ids, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline-real"))
    asyncio.run(run(parser.parse_args().output))
