from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, model_validator

from adaptiveedge.core.models import DataSplit, ModelConfiguration, Positive, Record, TaskDefinition


class ExperimentSettings(Record):
    name: Annotated[str, Field(min_length=1)]
    split: DataSplit = "calibration"
    repetitions: Annotated[int, Field(ge=1)] = 3
    warmup_runs: Annotated[int, Field(ge=0)] = 1
    seed: Annotated[int, Field(ge=0, le=2147483647)] = 42
    timeout_seconds: Positive = 120
    mode: Literal["warm", "cold"] = "warm"


class BenchmarkConfiguration(Record):
    experiment: ExperimentSettings
    models: Annotated[tuple[ModelConfiguration, ...], Field(min_length=1)]
    suite: Literal["basic", "custom"] = "basic"
    tasks: tuple[TaskDefinition, ...] = ()

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.suite == "custom" and not self.tasks:
            raise ValueError("custom suite requires tasks")
        if self.suite == "basic" and self.tasks:
            raise ValueError("basic suite cannot override tasks; use custom")
        if len({m.identity for m in self.models}) != len(self.models):
            raise ValueError("duplicate model configurations")
        if len({t.id for t in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task identifiers")
        return self


def load_config(path: Path) -> BenchmarkConfiguration:
    if path.suffix == ".toml":
        import tomllib

        with path.open("rb") as stream:
            data = tomllib.load(stream)
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BenchmarkConfiguration.model_validate(data)
