import asyncio
from typing import Annotated

from pydantic import Field

from adaptiveedge.backends.base import Availability
from adaptiveedge.core.models import (
    InferenceResult,
    Measurement,
    ModelConfiguration,
    Nonnegative,
    Record,
    TaskDefinition,
    utcnow,
)


class MockBehavior(Record):
    outputs: dict[str, str] = Field(default_factory=dict)
    default_output: str = "mock output"
    latency_ms: Nonnegative = 10
    ram_gb: Nonnegative = 0.25
    vram_gb: Nonnegative | None = None
    prompt_tokens: Annotated[int, Field(ge=0)] = 12
    completion_tokens: Annotated[int, Field(ge=0)] = 4
    delay_seconds: Nonnegative = 0
    fail_tasks: tuple[str, ...] = ()


class MockBackend:
    def __init__(self, behaviors: dict[str, MockBehavior] | None = None) -> None:
        self.behaviors = behaviors if behaviors is not None else demo_behaviors()

    async def availability(self) -> Availability:
        return Availability(
            available=True, detail="Deterministic simulated test runtime", version="1"
        )

    async def discover(self) -> tuple[str, ...]:
        return tuple(sorted(self.behaviors))

    async def validate(self, configuration: ModelConfiguration) -> None:
        if configuration.model not in self.behaviors:
            raise ValueError(f"unknown mock model: {configuration.model}")

    async def infer(
        self, configuration: ModelConfiguration, task: TaskDefinition, seed: int, cold: bool = False
    ) -> InferenceResult:
        await self.validate(configuration)
        behavior = self.behaviors[configuration.model]
        started = utcnow()
        await asyncio.sleep(behavior.delay_seconds)
        if task.id in behavior.fail_tasks:
            raise RuntimeError("configured mock failure")
        vram = (
            Measurement(
                value=behavior.vram_gb, status="simulated", unit="GB", scope="model_runtime"
            )
            if behavior.vram_gb is not None
            else Measurement(status="unsupported", unit="GB", scope="model_runtime")
        )
        return InferenceResult(
            configuration=configuration,
            output=behavior.outputs.get(task.id, behavior.default_output),
            prompt_tokens=behavior.prompt_tokens,
            completion_tokens=behavior.completion_tokens,
            latency_ms=behavior.latency_ms,
            generation_ms=behavior.latency_ms,
            throughput_tps=(
                behavior.completion_tokens * 1000 / behavior.latency_ms
                if behavior.latency_ms > 0
                else None
            ),
            peak_ram_gb=Measurement(
                value=behavior.ram_gb, status="simulated", unit="GB", scope="model_runtime"
            ),
            peak_vram_gb=vram,
            simulated=True,
            started_at=started,
            finished_at=utcnow(),
            metadata={"notice": "SIMULATED fixture data; not hardware benchmark evidence"},
        )


def demo_behaviors() -> dict[str, MockBehavior]:
    answers = {
        "sentiment": "positive",
        "extract": "Ada",
        "arithmetic": "42",
        "summary": "Rain delayed the match.",
        "code": "def add(a, b):\n    return a + b",
    }
    return {
        "mock-small": MockBehavior(
            outputs={**answers, "arithmetic": "41", "code": "pass"}, latency_ms=20, ram_gb=0.5
        ),
        "mock-large": MockBehavior(outputs=answers, latency_ms=100, ram_gb=2),
    }
