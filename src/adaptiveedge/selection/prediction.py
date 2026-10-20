from collections.abc import Sequence
from statistics import median
from typing import Literal, Protocol

from adaptiveedge.core.models import (
    BenchmarkResult,
    Estimates,
    HardwareSnapshot,
    ModelConfiguration,
    TaskDefinition,
)


class Predictor(Protocol):
    def predict(
        self,
        configuration: ModelConfiguration,
        task: TaskDefinition,
        hardware: HardwareSnapshot,
        history: Sequence[BenchmarkResult],
    ) -> Estimates: ...


class HistoricalMedianPredictor:
    def __init__(
        self, mode: Literal["warm", "cold"] = "warm", input_length_ratio: float | None = None
    ) -> None:
        if input_length_ratio is not None and input_length_ratio < 1:
            raise ValueError("input_length_ratio must be >= 1")
        self.mode = mode
        self.input_length_ratio = input_length_ratio

    def predict(
        self,
        configuration: ModelConfiguration,
        task: TaskDefinition,
        hardware: HardwareSnapshot,
        history: Sequence[BenchmarkResult],
    ) -> Estimates:
        primary_metric = task.metrics[0] if task.metrics else None
        rows = [
            row
            for row in history
            if row.configuration.identity == configuration.identity
            and row.split == "calibration"
            and row.hardware.fingerprint == hardware.fingerprint
            and row.task.category == task.category
            and row.task.dataset_version == task.dataset_version
            and row.task.dataset == task.dataset
            and (row.task.generation or configuration.generation)
            == (task.generation or configuration.generation)
            and row.policy is None
            and row.inference.metadata.get("mode") == self.mode
            and row.success
            and row.inference.simulated == (configuration.backend == "mock")
            and (primary_metric is None or primary_metric in row.quality)
            and self._length_matches(len(row.task.prompt), len(task.prompt))
        ]

        def middle(values: list[float | None]) -> float | None:
            present = [value for value in values if value is not None]
            return median(present) if present else None

        return Estimates(
            quality=middle([row.quality.get(primary_metric) for row in rows])
            if primary_metric
            else None,
            latency_ms=middle([row.inference.latency_ms for row in rows]),
            ram_gb=middle([row.inference.peak_ram_gb.value for row in rows]),
            vram_gb=middle([row.inference.peak_vram_gb.value for row in rows]),
            throughput_tps=middle([row.inference.throughput_tps for row in rows]),
            size_gb=configuration.size_gb,
            samples=len(rows),
        )

    def _length_matches(self, historical: int, current: int) -> bool:
        if self.input_length_ratio is None:
            return True
        return (
            max(historical, current) / max(1, min(historical, current)) <= self.input_length_ratio
        )
