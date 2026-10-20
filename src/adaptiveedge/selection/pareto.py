from collections.abc import Sequence
from typing import Literal

from adaptiveedge.core.models import Estimates

Objective = Literal["quality", "latency_ms", "ram_gb", "vram_gb", "throughput_tps", "size_gb"]
MAXIMIZE = {"quality", "throughput_tps"}


def dominates(left: Estimates, right: Estimates, objectives: Sequence[Objective]) -> bool:
    if not objectives:
        return False
    strict = False
    for name in objectives:
        a = getattr(left, name)
        b = getattr(right, name)
        if a is None or b is None:
            return False
        if name in MAXIMIZE:
            a, b = -a, -b
        if a > b:
            return False
        strict = strict or a < b
    return strict


def frontier(points: Sequence[Estimates], objectives: Sequence[Objective]) -> tuple[int, ...]:
    return tuple(
        index
        for index, point in enumerate(points)
        if not any(dominates(other, point, objectives) for other in points)
    )


def normalize_values(values: Sequence[float], maximize: bool = True) -> tuple[float, ...]:
    if not values:
        return ()
    lower, upper = min(values), max(values)
    if lower == upper:
        return tuple(1.0 for _ in values)
    scaled = tuple((value - lower) / (upper - lower) for value in values)
    return scaled if maximize else tuple(1 - value for value in scaled)
