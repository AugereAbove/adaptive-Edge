import argparse
import asyncio
from pathlib import Path

from adaptiveedge.api import default_registry
from adaptiveedge.benchmarks.evaluation import evaluate_policies
from adaptiveedge.benchmarks.runner import BenchmarkRunner
from adaptiveedge.config.schema import load_config
from adaptiveedge.profiling.hardware import hardware_snapshot
from adaptiveedge.reporting.reports import generate_report
from adaptiveedge.storage.repository import Repository


async def experiment(
    training: Path, evaluation: Path, database: Path, output: Path, smoke: bool
) -> None:
    registry = default_registry()
    hardware = hardware_snapshot()
    with Repository(database) as repository:
        training_id = await BenchmarkRunner(repository, registry, hardware).run(
            load_config(training)
        )
        evaluation_id = await evaluate_policies(
            repository,
            registry,
            hardware,
            training_id,
            load_config(evaluation),
            allow_overlap=smoke,
        )
        generate_report(repository, training_id, output / "training")
        generate_report(repository, evaluation_id, output / "evaluation")
        print(f"Training: {training_id}\nEvaluation: {evaluation_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-evidence policy experiment")
    parser.add_argument("training", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--db", type=Path, default=Path("adaptiveedge.db"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(
        experiment(
            arguments.training,
            arguments.evaluation,
            arguments.db,
            arguments.output,
            arguments.smoke,
        )
    )


if __name__ == "__main__":
    main()
