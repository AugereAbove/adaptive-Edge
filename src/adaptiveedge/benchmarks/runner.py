import asyncio
import logging
import platform
import sys
from time import perf_counter
from uuid import uuid4

from pydantic import JsonValue

from adaptiveedge._version import __version__
from adaptiveedge.backends.base import BackendRegistry
from adaptiveedge.config.schema import BenchmarkConfiguration
from adaptiveedge.core.models import (
    BenchmarkResult,
    Failure,
    HardwareSnapshot,
    InferenceResult,
    ModelConfiguration,
    TaskDefinition,
    utcnow,
)
from adaptiveedge.datasets.basic import basic_suite
from adaptiveedge.metrics.quality import MetricRegistry
from adaptiveedge.profiling.hardware import ProcessSampler, hardware_snapshot
from adaptiveedge.storage.repository import Repository

logger = logging.getLogger("adaptiveedge")


def software_versions() -> dict[str, str]:
    return {"adaptiveedge": __version__, "python": sys.version, "platform": platform.platform()}


class BenchmarkRunner:
    def __init__(
        self,
        repository: Repository,
        registry: BackendRegistry,
        hardware: HardwareSnapshot | None = None,
        metrics: MetricRegistry | None = None,
    ) -> None:
        self.repository = repository
        self.registry = registry
        self.hardware = hardware or hardware_snapshot()
        self.metrics = metrics or MetricRegistry()

    def score(
        self, task: TaskDefinition, result: InferenceResult
    ) -> tuple[InferenceResult, dict[str, float]]:
        if result.error is not None:
            return result, {}
        try:
            return result, self.metrics.evaluate(task, result.output)
        except Exception as exc:
            failed = result.model_copy(
                update={
                    "error": Failure(kind=type(exc).__name__, message=str(exc), phase="metrics")
                }
            )
            return failed, {}

    async def measure(
        self,
        model: ModelConfiguration,
        task: TaskDefinition,
        seed: int,
        timeout: float,
        cold: bool,
    ) -> InferenceResult:
        started = utcnow()
        clock = perf_counter()
        async with ProcessSampler() as sampler:
            try:
                async with asyncio.timeout(timeout):
                    result = await self.registry.get(model.backend).infer(model, task, seed, cold)
            except Exception as exc:
                logger.warning(
                    "inference_failed", extra={"backend": model.backend, "kind": type(exc).__name__}
                )
                result = InferenceResult(
                    configuration=model,
                    latency_ms=(perf_counter() - clock) * 1000,
                    started_at=started,
                    finished_at=utcnow(),
                    simulated=model.backend == "mock",
                    error=Failure(kind=type(exc).__name__, message=str(exc)),
                )
        return result.model_copy(
            update={
                "client_peak_ram_gb": sampler.measurement(),
                "metadata": {
                    **result.metadata,
                    "mode": "cold" if cold else "warm",
                    "input_characters": len(task.prompt),
                },
            }
        )

    async def run(self, configuration: BenchmarkConfiguration) -> str:
        tasks = basic_suite() if configuration.suite == "basic" else configuration.tasks
        self.metrics.validate(tasks)
        identifier = f"{configuration.identity[:12]}-{uuid4().hex}"
        software = software_versions()
        runtime: dict[str, JsonValue] = {}
        for backend_name in sorted({m.backend for m in configuration.models}):
            try:
                async with asyncio.timeout(configuration.experiment.timeout_seconds):
                    available = await self.registry.get(backend_name).availability()
                runtime[backend_name] = available.model_dump(mode="json")
            except Exception as exc:
                runtime[backend_name] = {"available": False, "error": type(exc).__name__}
        self.repository.create_experiment(
            identifier,
            configuration,
            self.hardware,
            software,
            {"runtimes": runtime, "resolved_tasks": [t.model_dump(mode="json") for t in tasks]},
        )
        settings = configuration.experiment
        interrupted = True
        try:
            for model in configuration.models:
                phase = "availability"
                try:
                    async with asyncio.timeout(settings.timeout_seconds):
                        backend = self.registry.get(model.backend)
                        available = await backend.availability()
                        if not available.available:
                            raise RuntimeError(available.detail)
                        await backend.validate(model)
                    phase = "warmup"
                    if settings.mode == "warm":
                        for _ in range(settings.warmup_runs):
                            warm = await self.measure(
                                model, tasks[0], settings.seed, settings.timeout_seconds, False
                            )
                            if warm.error:
                                raise RuntimeError(f"warm-up failed: {warm.error.kind}")
                except Exception as exc:
                    error = Failure(kind=type(exc).__name__, message=str(exc), phase=phase)
                    for task in tasks:
                        for repetition in range(settings.repetitions):
                            self.repository.save_result(
                                BenchmarkResult(
                                    run_id=uuid4().hex,
                                    split=settings.split,
                                    experiment_id=identifier,
                                    configuration=model,
                                    hardware=self.hardware,
                                    task=task,
                                    inference=InferenceResult(
                                        configuration=model,
                                        latency_ms=None,
                                        error=error,
                                        simulated=model.backend == "mock",
                                        metadata={
                                            "latency_notice": (
                                                "No inference attempted; exclude from statistics"
                                            )
                                        },
                                    ),
                                    seed=(settings.seed + repetition) % 2147483648,
                                    repetition=repetition,
                                    software=software,
                                )
                            )
                    continue
                for task in tasks:
                    for repetition in range(settings.repetitions):
                        seed = (settings.seed + repetition) % 2147483648
                        result = await self.measure(
                            model, task, seed, settings.timeout_seconds, settings.mode == "cold"
                        )
                        result, quality = self.score(task, result)
                        self.repository.save_result(
                            BenchmarkResult(
                                run_id=uuid4().hex,
                                split=settings.split,
                                experiment_id=identifier,
                                configuration=model,
                                hardware=self.hardware,
                                task=task,
                                inference=result,
                                quality=quality,
                                seed=seed,
                                repetition=repetition,
                                software=software,
                            )
                        )
            interrupted = False
        finally:
            self.repository.event(identifier, "interrupted" if interrupted else "completed", {})
        return identifier
