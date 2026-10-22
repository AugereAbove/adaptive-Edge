from collections import defaultdict
from collections.abc import Sequence
from functools import partial

from pydantic import JsonValue

from adaptiveedge.core.models import BenchmarkResult
from adaptiveedge.reporting.statistics import describe as describe_statistics

describe = partial(describe_statistics, confidence=False)


def policy_comparison(results: Sequence[BenchmarkResult]) -> list[dict[str, JsonValue]]:
    groups: dict[tuple[str, ...], list[BenchmarkResult]] = defaultdict(list)
    for row in results:
        if row.policy is None:
            continue
        metric = row.task.metrics[0] if row.task.metrics else "unscored"
        key: tuple[str, ...] = (
            row.policy,
            row.task.category,
            row.task.dataset,
            row.task.dataset_version,
            metric,
            row.hardware.fingerprint,
            str(row.inference.simulated),
            str(row.inference.metadata.get("mode")),
            row.split,
        )
        groups[key].append(row)
    comparison: list[dict[str, JsonValue]] = []
    for key, rows in sorted(groups.items()):
        successful = [r for r in rows if r.success]
        known = []
        unknown = 0
        for row in successful:
            violations = row.inference.metadata.get("actual_constraint_violations")
            if not isinstance(violations, list) or any(
                isinstance(v, dict) and v.get("reason") == "measurement unavailable"
                for v in violations
            ):
                unknown += 1
            else:
                known.append(bool(violations))
        comparison.append(
            {
                "policy": key[0],
                "category": key[1],
                "dataset": key[2],
                "dataset_version": key[3],
                "metric": key[4],
                "hardware": key[5],
                "simulated": rows[0].inference.simulated,
                "mode": key[7],
                "split": key[8],
                "failure_rate": (len(rows) - len(successful)) / len(rows),
                "attempted": len(rows),
                "failures": len(rows) - len(successful),
                "quality": describe(
                    [r.quality[key[4]] for r in successful if key[4] in r.quality]
                ).model_dump(mode="json"),
                "latency_ms": describe(
                    [
                        r.inference.latency_ms
                        for r in successful
                        if r.inference.latency_ms is not None
                    ]
                ).model_dump(mode="json"),
                "ram_gb": describe(
                    [
                        r.inference.peak_ram_gb.value
                        for r in successful
                        if r.inference.peak_ram_gb.value is not None
                    ]
                ).model_dump(mode="json"),
                "constraint_violation_rate": sum(known) / len(known) if known else None,
                "constraint_compliance_observations": len(known),
                "constraint_compliance_unknown": unknown,
            }
        )
    return comparison
