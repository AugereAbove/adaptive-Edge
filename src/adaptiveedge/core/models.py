from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @property
    def identity(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


def canonical_hash(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class Category(StrEnum):
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CODE = "code"


class Generation(Record):
    temperature: Annotated[float, Field(ge=0, le=2)] = 0
    top_p: Annotated[float, Field(gt=0, le=1)] = 1
    max_output_tokens: Annotated[int, Field(gt=0)] = 256


class RuntimeOption(Record):
    name: Annotated[str, Field(min_length=1)]
    value: str | int | float | bool


class ModelConfiguration(Record):
    backend: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    label: str = ""
    family: str | None = None
    parameter_count: Annotated[int, Field(gt=0)] | None = None
    size_gb: Nonnegative | None = None
    quantization: str | None = None
    context_size: Annotated[int, Field(gt=0)] = 4096
    generation: Generation = Generation()
    device: str = "auto"
    options: tuple[RuntimeOption, ...] = ()

    @model_validator(mode="after")
    def unique_options(self) -> Self:
        if len({o.name for o in self.options}) != len(self.options):
            raise ValueError("runtime option names must be unique")
        return self

    @property
    def display_name(self) -> str:
        return self.label or self.model


class Measurement(Record):
    value: Nonnegative | None = None
    status: Literal["measured", "simulated", "unsupported", "unavailable", "failed"]
    unit: str
    scope: str
    detail: str = ""

    @model_validator(mode="after")
    def consistent_value(self) -> Self:
        if (self.status in {"measured", "simulated"}) != (self.value is not None):
            raise ValueError("only measured or simulated measurements must have a value")
        return self


def unknown_memory(scope: str = "model_runtime") -> Measurement:
    return Measurement(status="unsupported", unit="GB", scope=scope)


class HardwareSnapshot(Record):
    os: str
    cpu: str
    physical_cores: int | None = None
    logical_cores: int | None = None
    total_ram_gb: Positive | None = None
    available_ram_gb: Nonnegative | None = None
    gpu_models: tuple[str, ...] = ()
    gpu_count: Annotated[int, Field(ge=0)] | None = None
    total_vram_gb: Measurement = Field(default_factory=unknown_memory)
    available_vram_gb: Measurement = Field(default_factory=unknown_memory)
    accelerator: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)

    @property
    def fingerprint(self) -> str:
        return canonical_hash(
            {
                "os": self.os,
                "cpu": self.cpu,
                "cores": self.logical_cores,
                "ram": self.total_ram_gb,
                "gpu": list(self.gpu_models),
                "accelerator": self.accelerator,
            }
        )


MetricName = Annotated[str, Field(min_length=1)]
DataSplit = Literal["calibration", "validation", "evaluation", "functional", "inference"]


class TaskDefinition(Record):
    id: Annotated[str, Field(min_length=1)]
    category: Category
    prompt: Annotated[str, Field(min_length=1)]
    reference: str | None = None
    expected_labels: tuple[str, ...] = ()
    metrics: tuple[MetricName, ...] = ("normalized_exact_match",)
    generation: Generation | None = None
    dataset: str = "custom"
    dataset_version: str = "1"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def metric_reference(self) -> Self:
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metric names must be unique")
        if (
            self.expected_labels
            and self.reference is not None
            and self.reference not in self.expected_labels
        ):
            raise ValueError("reference must be an expected label")
        return self


class Failure(Record):
    kind: str
    message: str
    phase: Literal["availability", "warmup", "inference", "metrics"] = "inference"


class InferenceResult(Record):
    configuration: ModelConfiguration
    output: str = ""
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    latency_ms: Nonnegative | None
    ttft_ms: Nonnegative | None = None
    generation_ms: Nonnegative | None = None
    load_ms: Nonnegative | None = None
    throughput_tps: Nonnegative | None = None
    peak_ram_gb: Measurement = Field(default_factory=unknown_memory)
    peak_vram_gb: Measurement = Field(default_factory=unknown_memory)
    client_peak_ram_gb: Measurement = Field(
        default_factory=lambda: unknown_memory("client_process")
    )
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime = Field(default_factory=utcnow)
    simulated: bool = False
    error: Failure | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class BenchmarkResult(Record):
    run_id: str
    split: DataSplit = "calibration"
    experiment_id: str
    configuration: ModelConfiguration
    hardware: HardwareSnapshot
    task: TaskDefinition
    inference: InferenceResult
    quality: dict[str, Score] = Field(default_factory=dict)
    seed: int
    repetition: Annotated[int, Field(ge=0)]
    software: dict[str, str]
    policy: str | None = None
    selection: dict[str, JsonValue] | None = None
    timestamp: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def consistent_observation(self) -> Self:
        if self.inference.configuration != self.configuration:
            raise ValueError("inference configuration must match benchmark configuration")
        if not set(self.quality).issubset(self.task.metrics):
            raise ValueError("quality contains metrics not requested by the task")
        if self.inference.error is not None and self.quality:
            raise ValueError("failed observations cannot contain quality scores")
        return self

    @property
    def success(self) -> bool:
        return self.inference.error is None


class Constraints(Record):
    max_latency_ms: Positive | None = None
    max_memory_gb: Positive | None = None
    max_vram_gb: Positive | None = None
    min_quality: Score | None = None
    min_throughput_tps: Nonnegative | None = None
    max_model_size_gb: Positive | None = None
    allowed_backends: tuple[str, ...] | None = None
    allowed_models: tuple[str, ...] | None = None
    preferred_device: str | None = None


class Estimates(Record):
    quality: Score | None = None
    latency_ms: Nonnegative | None = None
    ram_gb: Nonnegative | None = None
    vram_gb: Nonnegative | None = None
    throughput_tps: Nonnegative | None = None
    size_gb: Nonnegative | None = None
    samples: Annotated[int, Field(ge=0)] = 0


class Violation(Record):
    field: str
    predicted: float | str | None
    limit: float | str
    reason: str


class CandidateEvaluation(Record):
    configuration: ModelConfiguration
    estimates: Estimates
    violations: tuple[Violation, ...] = ()
    utility: float | None = None
    pareto: bool = False


class SelectionResult(Record):
    selected: ModelConfiguration | None
    policy: str
    candidates: tuple[CandidateEvaluation, ...]
    reason: str
