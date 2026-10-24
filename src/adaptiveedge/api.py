from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from adaptiveedge.backends.base import BackendRegistry
from adaptiveedge.backends.mock import MockBackend
from adaptiveedge.backends.ollama import OllamaBackend
from adaptiveedge.benchmarks.runner import BenchmarkRunner, software_versions
from adaptiveedge.config.schema import BenchmarkConfiguration, ExperimentSettings
from adaptiveedge.core.models import (
    BenchmarkResult,
    Category,
    Constraints,
    HardwareSnapshot,
    InferenceResult,
    ModelConfiguration,
    Record,
    SelectionResult,
    TaskDefinition,
)
from adaptiveedge.profiling.hardware import hardware_snapshot
from adaptiveedge.selection.engine import Selector
from adaptiveedge.storage.repository import Repository


def default_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register("mock", MockBackend())
    registry.register("ollama", OllamaBackend())
    return registry


class AdaptiveResponse(Record):
    selection: SelectionResult
    inference: InferenceResult | None
    experiment_id: str | None = None

    @property
    def output(self) -> str | None:
        return self.inference.output if self.inference is not None else None


class AdaptiveEdge:
    def __init__(
        self,
        database: Path | str = "adaptiveedge.db",
        registry: BackendRegistry | None = None,
        selector: Selector | None = None,
        hardware: HardwareSnapshot | None = None,
    ) -> None:
        self.repository = Repository(database)
        self.registry = registry or default_registry()
        self.selector = selector or Selector()
        self.hardware = hardware or hardware_snapshot()

    def close(self) -> None:
        self.repository.close()

    def _task(self, task: TaskDefinition | str, input_text: str | None) -> TaskDefinition:
        if isinstance(task, TaskDefinition):
            return task
        category = Category(task)
        examples = [
            r.task
            for r in self.repository.results()
            if r.task.category == category and r.policy is None and r.split == "calibration"
        ]
        if not examples:
            raise ValueError(
                f"No benchmark task context for {category}; benchmark this category first"
            )
        template = examples[-1]
        return template.model_copy(
            update={
                "prompt": input_text or template.prompt,
                "reference": None,
                "expected_labels": (),
            }
        )

    def candidates(self) -> tuple[ModelConfiguration, ...]:
        models = {
            r.configuration.identity: r.configuration
            for r in self.repository.results()
            if r.policy is None and r.split == "calibration"
        }
        return tuple(models[key] for key in sorted(models))

    def select(
        self,
        task: TaskDefinition | str,
        constraints: Constraints | None = None,
        policy: str = "adaptive",
        models: Sequence[ModelConfiguration] | None = None,
        input_text: str | None = None,
    ) -> SelectionResult:
        return self.selector.select(
            self._task(task, input_text),
            self.hardware,
            self.candidates() if models is None else models,
            self.repository.results(),
            constraints,
            policy,
        )

    async def benchmark(self, configuration: BenchmarkConfiguration) -> str:
        return await BenchmarkRunner(self.repository, self.registry, self.hardware).run(
            configuration
        )

    async def run(
        self,
        task: TaskDefinition | str,
        input_text: str | None = None,
        constraints: Constraints | None = None,
        policy: str = "adaptive",
        models: Sequence[ModelConfiguration] | None = None,
        seed: int = 42,
        timeout_seconds: float = 120,
    ) -> AdaptiveResponse:
        resolved = self._task(task, input_text)
        selection = self.select(resolved, constraints, policy, models)
        if selection.selected is None:
            return AdaptiveResponse(selection=selection, inference=None)
        configuration = BenchmarkConfiguration(
            experiment=ExperimentSettings(
                split="inference",
                name=f"inference-{policy}",
                repetitions=1,
                warmup_runs=0,
                seed=seed,
                timeout_seconds=timeout_seconds,
            ),
            models=(selection.selected,),
            suite="custom",
            tasks=(resolved,),
        )
        identifier = f"{configuration.identity[:12]}-{uuid4().hex}"
        software = software_versions()
        self.repository.create_experiment(
            identifier,
            configuration,
            self.hardware,
            software,
            {
                "constraints": (constraints or Constraints()).model_dump(mode="json"),
                "selection": selection.model_dump(mode="json"),
                "kind": "adaptive_inference",
            },
        )
        runner = BenchmarkRunner(self.repository, self.registry, self.hardware)
        measured = await runner.measure(selection.selected, resolved, seed, timeout_seconds, False)
        measured, quality = runner.score(resolved, measured)
        self.repository.save_result(
            BenchmarkResult(
                run_id=uuid4().hex,
                split="inference",
                experiment_id=identifier,
                configuration=selection.selected,
                hardware=self.hardware,
                task=resolved,
                inference=measured,
                quality=quality,
                seed=seed,
                repetition=0,
                software=software,
                policy=policy,
                selection=selection.model_dump(mode="json"),
            )
        )
        self.repository.event(identifier, "completed", {})
        return AdaptiveResponse(selection=selection, inference=measured, experiment_id=identifier)
