from collections.abc import Sequence
from typing import Protocol, Self

from pydantic import model_validator

from adaptiveedge.core.models import (
    BenchmarkResult,
    CandidateEvaluation,
    Constraints,
    Estimates,
    HardwareSnapshot,
    ModelConfiguration,
    Nonnegative,
    Record,
    SelectionResult,
    TaskDefinition,
    Violation,
)
from adaptiveedge.selection.pareto import MAXIMIZE, Objective, frontier, normalize_values
from adaptiveedge.selection.prediction import HistoricalMedianPredictor, Predictor


class Weights(Record):
    quality: Nonnegative = 0.6
    latency_ms: Nonnegative = 0.3
    ram_gb: Nonnegative = 0.1
    vram_gb: Nonnegative = 0
    throughput_tps: Nonnegative = 0
    size_gb: Nonnegative = 0

    @model_validator(mode="after")
    def nonzero(self) -> Self:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one policy weight must be positive")
        return self


def constraint_violations(
    model: ModelConfiguration, estimates: Estimates, constraints: Constraints
) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    for field, value, limit, minimum in (
        ("latency_ms", estimates.latency_ms, constraints.max_latency_ms, False),
        ("ram_gb", estimates.ram_gb, constraints.max_memory_gb, False),
        ("vram_gb", estimates.vram_gb, constraints.max_vram_gb, False),
        ("quality", estimates.quality, constraints.min_quality, True),
        ("throughput_tps", estimates.throughput_tps, constraints.min_throughput_tps, True),
        ("size_gb", estimates.size_gb, constraints.max_model_size_gb, False),
    ):
        if limit is not None and (value is None or (value < limit if minimum else value > limit)):
            violations.append(
                Violation(
                    field=field,
                    predicted=value,
                    limit=limit,
                    reason="measurement unavailable"
                    if value is None
                    else "predicted constraint violation",
                )
            )
    for field, identifier, allowed in (
        ("backend", model.backend, constraints.allowed_backends),
        ("model", model.model, constraints.allowed_models),
    ):
        if allowed is not None and identifier not in allowed:
            violations.append(
                Violation(
                    field=field,
                    predicted=identifier,
                    limit=", ".join(allowed),
                    reason="not allowed",
                )
            )
    if constraints.preferred_device is not None and model.device != constraints.preferred_device:
        violations.append(
            Violation(
                field="device",
                predicted=model.device,
                limit=constraints.preferred_device,
                reason="device preference requires an explicitly matching configuration",
            )
        )
    if estimates.samples == 0:
        violations.append(
            Violation(
                field="evidence", predicted=0, limit=1, reason="no compatible historical evidence"
            )
        )
    return tuple(violations)


class Policy(Protocol):
    def score(self, candidates: Sequence[CandidateEvaluation]) -> tuple[float | None, ...]: ...


class MetricPolicy:
    def __init__(self, metric: Objective, maximize: bool) -> None:
        self.metric = metric
        self.maximize = maximize

    def score(self, candidates: Sequence[CandidateEvaluation]) -> tuple[float | None, ...]:
        scores: list[float | None] = []
        for candidate in candidates:
            value = getattr(candidate.estimates, self.metric)
            scores.append(None if value is None else value if self.maximize else -value)
        return tuple(scores)


class AdaptivePolicy:
    def __init__(self, weights: Weights | None = None) -> None:
        self.weights = weights or Weights()

    def objectives(self, candidates: Sequence[CandidateEvaluation]) -> tuple[Objective, ...]:
        names: tuple[Objective, ...] = (
            "quality",
            "latency_ms",
            "ram_gb",
            "vram_gb",
            "throughput_tps",
            "size_gb",
        )
        return tuple(
            name
            for name in names
            if getattr(self.weights, name) > 0
            and all(getattr(candidate.estimates, name) is not None for candidate in candidates)
        )

    def score(self, candidates: Sequence[CandidateEvaluation]) -> tuple[float | None, ...]:
        objectives = self.objectives(candidates)
        if not objectives:
            return tuple(None for _ in candidates)
        scores = [0.0] * len(candidates)
        denominator = sum(getattr(self.weights, name) for name in objectives)
        for name in objectives:
            values = [float(getattr(c.estimates, name)) for c in candidates]
            scaled = normalize_values(values, maximize=name in MAXIMIZE)
            weight = getattr(self.weights, name) / denominator
            for index, value in enumerate(scaled):
                scores[index] += weight * value
        keep = frontier([c.estimates for c in candidates], objectives)
        return tuple(value if i in keep else None for i, value in enumerate(scores))


class Selector:
    def __init__(self, predictor: Predictor | None = None, weights: Weights | None = None) -> None:
        self.predictor = predictor or HistoricalMedianPredictor()
        self.weights = weights or Weights()
        self.policies: dict[str, Policy] = {
            "smallest": MetricPolicy("size_gb", False),
            "largest": MetricPolicy("size_gb", True),
            "fastest": MetricPolicy("latency_ms", False),
            "highest_quality": MetricPolicy("quality", True),
            "lowest_memory": MetricPolicy("ram_gb", False),
            "adaptive": AdaptivePolicy(self.weights),
        }

    def register_policy(self, name: str, policy: Policy) -> None:
        if name in self.policies:
            raise ValueError(f"policy already registered: {name}")
        self.policies[name] = policy

    def select(
        self,
        task: TaskDefinition,
        hardware: HardwareSnapshot,
        models: Sequence[ModelConfiguration],
        history: Sequence[BenchmarkResult],
        constraints: Constraints | None = None,
        policy: str = "adaptive",
    ) -> SelectionResult:
        constraints = constraints or Constraints()
        if policy not in self.policies:
            raise ValueError(f"unknown policy: {policy}")
        if len({m.identity for m in models}) != len(models):
            raise ValueError("candidate configurations must be unique")
        ordered = sorted(models, key=lambda m: (m.display_name, m.identity))
        candidates: list[CandidateEvaluation] = []
        for model in ordered:
            estimates = self.predictor.predict(model, task, hardware, history)
            violations = list(constraint_violations(model, estimates, constraints))
            if (
                hardware.available_ram_gb is not None
                and estimates.ram_gb is not None
                and estimates.ram_gb > hardware.available_ram_gb
            ):
                violations.append(
                    Violation(
                        field="available_ram_gb",
                        predicted=estimates.ram_gb,
                        limit=hardware.available_ram_gb,
                        reason="exceeds currently available host RAM",
                    )
                )
            candidates.append(
                CandidateEvaluation(
                    configuration=model, estimates=estimates, violations=tuple(violations)
                )
            )
        feasible_indices = [i for i, c in enumerate(candidates) if not c.violations]
        feasible = [candidates[i] for i in feasible_indices]
        chosen_policy = self.policies[policy]
        scores = chosen_policy.score(feasible)
        for index, score in zip(feasible_indices, scores, strict=True):
            candidates[index] = candidates[index].model_copy(
                update={
                    "utility": score,
                    "pareto": isinstance(chosen_policy, AdaptivePolicy) and score is not None,
                }
            )
        ranked = sorted(
            (c for c in candidates if not c.violations and c.utility is not None),
            key=lambda c: (
                -float(c.utility or 0),
                c.configuration.display_name,
                c.configuration.identity,
            ),
        )
        if not ranked:
            return SelectionResult(
                selected=None,
                policy=policy,
                candidates=tuple(candidates),
                reason="No model satisfies all constraints with sufficient comparable evidence.",
            )
        winner = ranked[0]
        rejected = "; ".join(
            f"{c.configuration.display_name}: "
            + (
                ", ".join(f"{v.field} ({v.reason})" for v in c.violations)
                if c.violations
                else "lower utility, dominated, unavailable policy metric, or tie-break"
            )
            for c in candidates
            if c.configuration != winner.configuration
        )
        active = (
            ", ".join(chosen_policy.objectives(feasible))
            if isinstance(chosen_policy, AdaptivePolicy)
            else policy
        )
        return SelectionResult(
            selected=winner.configuration,
            policy=policy,
            candidates=tuple(candidates),
            reason=(
                f"Selected {winner.configuration.display_name} using {policy}; "
                f"predicted quality={winner.estimates.quality}, "
                f"latency={winner.estimates.latency_ms} ms, "
                f"RAM={winner.estimates.ram_gb} GB; {winner.estimates.samples} historical samples; "
                f"satisfies all predicted hard constraints. Active objectives: {active}. "
                f"Alternatives: {rejected or 'none'}. "
                "Predictions do not guarantee runtime compliance."
            ),
        )
