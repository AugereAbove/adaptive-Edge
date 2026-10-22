from collections.abc import Sequence
from math import isfinite
from random import Random
from statistics import mean, median, stdev

from adaptiveedge.core.models import Record


class Statistics(Record):
    count: int
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    ci95_lower: float | None = None
    ci95_upper: float | None = None
    uncertainty: str = "insufficient"
    ci_method: str = "percentile bootstrap of the mean; 2000 resamples; seed 42"


def describe(values: Sequence[float], confidence: bool = True) -> Statistics:
    if any(not isfinite(v) for v in values):
        raise ValueError("statistics require finite values")
    if not values:
        return Statistics(count=0)
    bootstrap = []
    if confidence and len(values) >= 3:
        random = Random(42)
        bootstrap = sorted(mean(random.choices(values, k=len(values))) for _ in range(2000))
    return Statistics(
        count=len(values),
        ci95_lower=bootstrap[49] if bootstrap else None,
        ci95_upper=bootstrap[1949] if bootstrap else None,
        uncertainty="repetitions_not_independent"
        if not confidence
        else "limited"
        if 3 <= len(values) < 10
        else "estimated"
        if len(values) >= 10
        else "insufficient",
        mean=mean(values),
        median=median(values),
        standard_deviation=stdev(values) if len(values) > 1 else None,
        minimum=min(values),
        maximum=max(values),
    )
