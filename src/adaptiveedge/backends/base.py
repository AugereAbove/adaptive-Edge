from typing import Protocol

from pydantic import Field, JsonValue

from adaptiveedge.core.models import InferenceResult, ModelConfiguration, Record, TaskDefinition


class Availability(Record):
    available: bool
    detail: str
    version: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Backend(Protocol):
    async def availability(self) -> Availability: ...

    async def discover(self) -> tuple[str, ...]: ...

    async def validate(self, configuration: ModelConfiguration) -> None: ...

    async def infer(
        self, configuration: ModelConfiguration, task: TaskDefinition, seed: int, cold: bool = False
    ) -> InferenceResult: ...


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}

    def register(self, name: str, backend: Backend) -> None:
        if name in self._backends:
            raise ValueError(f"backend already registered: {name}")
        self._backends[name] = backend

    def get(self, name: str) -> Backend:
        if name not in self._backends:
            raise ValueError(f"backend is not registered: {name}")
        return self._backends[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))
