from collections import defaultdict
from collections.abc import Sequence
from statistics import mean, median
from typing import Any, cast

from adaptiveedge.core.models import BenchmarkResult
from adaptiveedge.reporting.statistics import describe


def measurements(row: BenchmarkResult) -> dict[str, float | None]:
    return {
        "quality": row.quality.get(row.task.metrics[0]) if row.task.metrics else None,
        "latency_ms": row.inference.latency_ms,
        "ram_gb": row.inference.peak_ram_gb.value,
        "vram_gb": row.inference.peak_vram_gb.value,
        "throughput_tps": row.inference.throughput_tps,
    }


def research_analysis(results: Sequence[BenchmarkResult]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[BenchmarkResult]] = defaultdict(list)
    for row in results:
        groups[
            (
                row.policy or row.configuration.display_name,
                row.task.category,
                row.task.dataset,
                row.task.dataset_version,
                row.task.metrics[0] if row.task.metrics else "unscored",
                row.hardware.fingerprint,
                str(row.inference.metadata.get("mode")),
                str(row.inference.simulated),
                row.split,
                row.experiment_id,
            )
        ].append(row)
    aggregates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        tasks: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for row in rows:
            tasks[row.task.identity].append(row)
        statistics = {}
        for metric in measurements(rows[0]):
            task_means = []
            for observations in tasks.values():
                values = [measurements(r)[metric] for r in observations if r.success]
                available = [value for value in values if value is not None]
                if available:
                    task_means.append(mean(available))
            statistics[metric] = describe(task_means).model_dump(mode="json")
        compliance = [
            not r.inference.metadata["actual_constraint_violations"]
            for r in rows
            if r.success
            and isinstance(r.inference.metadata.get("actual_constraint_violations"), list)
            and not any(
                isinstance(v, dict) and v.get("reason") == "measurement unavailable"
                for v in cast(list[Any], r.inference.metadata["actual_constraint_violations"])
            )
        ]
        aggregate = {
            "policy_or_model": key[0],
            "category": key[1],
            "dataset": key[2],
            "dataset_version": key[3],
            "metric": key[4],
            "hardware": key[5],
            "mode": key[6],
            "simulated": rows[0].inference.simulated,
            "split": key[8],
            "experiment_id": key[9],
            "task_ids": sorted(tasks),
            "attempted": len(rows),
            "tasks": len(tasks),
            "failure_rate": sum(not r.success for r in rows) / len(rows),
            "constraint_satisfaction_rate": mean(compliance) if compliance else None,
            "compliance_known": len(compliance),
            "statistics": statistics,
        }
        aggregates.append(aggregate)
        for metric in measurements(rows[0]):
            by_task: dict[str, list[tuple[float, float, float | None]]] = defaultdict(list)
            for row in rows:
                if not row.success or row.selection is None:
                    continue
                candidates = row.selection.get("candidates", [])
                if not isinstance(candidates, list):
                    continue
                selected = next(
                    (
                        c
                        for c in candidates
                        if isinstance(c, dict)
                        and c.get("configuration") == row.configuration.model_dump(mode="json")
                    ),
                    None,
                )
                if not selected or not isinstance(selected.get("estimates"), dict):
                    continue
                estimate = selected["estimates"]
                if not isinstance(estimate, dict):
                    continue
                predicted, actual = estimate.get(metric), measurements(row)[metric]
                if isinstance(predicted, (int, float)) and actual is not None:
                    absolute = abs(predicted - actual)
                    by_task[row.task.identity].append(
                        (
                            absolute,
                            predicted - actual,
                            absolute / abs(actual) * 100 if actual else None,
                        )
                    )
            if by_task:
                absolute_errors = [mean(v[0] for v in values) for values in by_task.values()]
                percentages = [
                    mean(v[2] for v in values if v[2] is not None)
                    for values in by_task.values()
                    if any(v[2] is not None for v in values)
                ]
                errors.append(
                    {
                        "policy": key[0],
                        "category": key[1],
                        "metric": metric,
                        "quality_metric": key[4],
                        "hardware": key[5],
                        "mode": key[6],
                        "simulated": rows[0].inference.simulated,
                        "split": key[8],
                        "experiment_id": key[9],
                        "task_count": len(by_task),
                        "observations": sum(map(len, by_task.values())),
                        "mae": mean(absolute_errors),
                        "median_absolute_error": median(absolute_errors),
                        "mape_percent": mean(percentages)
                        if percentages and metric != "quality"
                        else None,
                        "percentage_task_count": len(percentages) if metric != "quality" else 0,
                        "absolute_error_statistics": describe(absolute_errors).model_dump(
                            mode="json"
                        ),
                    }
                )
    effects = []
    for item in aggregates:
        baseline = next(
            (
                a
                for a in aggregates
                if a["policy_or_model"] == "highest_quality"
                and all(
                    a[k] == item[k]
                    for k in (
                        "category",
                        "dataset",
                        "dataset_version",
                        "metric",
                        "hardware",
                        "mode",
                        "simulated",
                        "split",
                        "task_ids",
                        "experiment_id",
                    )
                )
            ),
            None,
        )
        if baseline is None:
            continue
        effect = {
            k: item[k]
            for k in (
                "policy_or_model",
                "category",
                "metric",
                "hardware",
                "mode",
                "simulated",
                "split",
            )
        }
        effect["reference"] = "highest_quality"
        for metric in measurements(results[0]):
            value, reference = (
                item["statistics"][metric]["mean"],
                baseline["statistics"][metric]["mean"],
            )
            effect[f"{metric}_difference"] = (
                value - reference
                if value is not None
                and reference is not None
                and item["statistics"][metric]["count"] == item["tasks"]
                and baseline["statistics"][metric]["count"] == baseline["tasks"]
                else None
            )
            effect[f"{metric}_ratio"] = (
                value / reference
                if value is not None
                and reference is not None
                and reference > 0
                and item["statistics"][metric]["count"] == item["tasks"]
                and baseline["statistics"][metric]["count"] == baseline["tasks"]
                else None
            )
        for metric in ("latency_ms", "ram_gb", "vram_gb"):
            ratio = effect[f"{metric}_ratio"]
            effect[f"{metric}_reduction_percent"] = (1 - ratio) * 100 if ratio is not None else None
        ratio = effect["throughput_tps_ratio"]
        effect["throughput_increase_percent"] = (ratio - 1) * 100 if ratio is not None else None
        actual, reference = (
            item["constraint_satisfaction_rate"],
            baseline["constraint_satisfaction_rate"],
        )
        effect["constraint_violation_difference"] = (
            reference - actual if actual is not None and reference is not None else None
        )
        effects.append(effect)
    return {
        "unit": "task mean over successful repetitions",
        "aggregates": aggregates,
        "prediction_errors": errors,
        "effects": effects,
        "uncertainty": (
            "Percentile bootstrap of independent task means; no hypothesis tests; "
            "fewer than 3 tasks suppresses CI; fewer than 10 is limited. "
            "Failures and abstentions are not quality scores."
        ),
    }
