import json
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from adaptiveedge.core.models import BenchmarkResult, Estimates
from adaptiveedge.reporting.export import export_csv, export_json
from adaptiveedge.reporting.policies import policy_comparison
from adaptiveedge.reporting.research import research_analysis
from adaptiveedge.reporting.statistics import describe as describe_statistics
from adaptiveedge.reporting.tables import export_tables
from adaptiveedge.selection.pareto import frontier
from adaptiveedge.storage.repository import Repository

describe = partial(describe_statistics, confidence=False)


def summarize(results: Sequence[BenchmarkResult]) -> list[dict[str, JsonValue]]:
    groups: dict[tuple[str, ...], list[BenchmarkResult]] = defaultdict(list)
    for row in results:
        key: tuple[str, ...] = (
            row.configuration.identity,
            row.task.category,
            row.task.dataset,
            row.task.dataset_version,
            row.hardware.fingerprint,
            str(row.inference.simulated),
            str(row.inference.metadata.get("mode", "unknown")),
            row.policy or "benchmark",
            json.dumps(row.task.metrics),
            row.split,
        )
        groups[key].append(row)
    summaries: list[dict[str, JsonValue]] = []
    for key in sorted(groups):
        rows = groups[key]
        valid = [r for r in rows if r.success]
        measurements: dict[str, list[float | None]] = {
            "latency_ms": [r.inference.latency_ms for r in valid],
            "ram_gb": [r.inference.peak_ram_gb.value for r in valid],
            "vram_gb": [r.inference.peak_vram_gb.value for r in valid],
            "client_ram_gb": [r.inference.client_peak_ram_gb.value for r in valid],
            "throughput_tps": [r.inference.throughput_tps for r in valid],
            "ttft_ms": [r.inference.ttft_ms for r in valid],
            "generation_ms": [r.inference.generation_ms for r in valid],
            "load_ms": [r.inference.load_ms for r in valid],
        }
        for metric in sorted({name for row in valid for name in row.quality}):
            measurements[f"quality:{metric}"] = [r.quality.get(metric) for r in valid]
        summaries.append(
            {
                "model": rows[0].configuration.display_name,
                "configuration_id": key[0],
                "category": key[1],
                "dataset": key[2],
                "dataset_version": key[3],
                "hardware": key[4],
                "simulated": rows[0].inference.simulated,
                "mode": key[6],
                "policy": key[7],
                "split": rows[0].split,
                "observations": len(rows),
                "failures": len(rows) - len(valid),
                "statistics": {
                    name: describe([v for v in values if v is not None]).model_dump(mode="json")
                    for name, values in measurements.items()
                },
            }
        )
    return summaries


def report_frontiers(
    results: Sequence[BenchmarkResult], objectives: tuple[str, ...] = ("quality", "latency_ms")
) -> list[dict[str, JsonValue]]:
    from typing import cast

    from adaptiveedge.selection.pareto import Objective

    valid_objectives = {"quality", "latency_ms", "ram_gb", "vram_gb", "throughput_tps", "size_gb"}
    if not objectives or not set(objectives).issubset(valid_objectives):
        raise ValueError("invalid Pareto objectives")
    groups: dict[tuple[str, ...], list[BenchmarkResult]] = defaultdict(list)
    for row in results:
        if row.success and row.task.metrics:
            groups[
                (
                    row.task.identity,
                    row.hardware.fingerprint,
                    str(row.inference.simulated),
                    str(row.inference.metadata.get("mode")),
                    row.split,
                    row.policy or "benchmark",
                )
            ].append(row)
    outputs: list[dict[str, JsonValue]] = []
    for group, rows in sorted(groups.items()):
        by_model: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for row in rows:
            by_model[row.configuration.identity].append(row)
        models = sorted(by_model)
        points = []
        for model in models:
            observations = by_model[model]
            metric = observations[0].task.metrics[0]
            quality = describe(
                [r.quality[metric] for r in observations if metric in r.quality]
            ).median
            latency = describe(
                [r.inference.latency_ms for r in observations if r.inference.latency_ms is not None]
            ).median
            points.append(
                Estimates(
                    quality=quality,
                    latency_ms=latency,
                    ram_gb=describe(
                        [
                            r.inference.peak_ram_gb.value
                            for r in observations
                            if r.inference.peak_ram_gb.value is not None
                        ]
                    ).median,
                    vram_gb=describe(
                        [
                            r.inference.peak_vram_gb.value
                            for r in observations
                            if r.inference.peak_vram_gb.value is not None
                        ]
                    ).median,
                    throughput_tps=describe(
                        [
                            r.inference.throughput_tps
                            for r in observations
                            if r.inference.throughput_tps is not None
                        ]
                    ).median,
                    size_gb=observations[0].configuration.size_gb,
                )
            )
        indices = frontier(points, cast(tuple[Objective, ...], objectives))
        outputs.append(
            {
                "task_hash": group[0],
                "hardware": group[1],
                "simulated": group[2],
                "mode": group[3],
                "objectives": list(objectives),
                "configurations": [models[i] for i in indices],
            }
        )
    return outputs


def generate_report(repository: Repository, experiment_id: str, destination: Path) -> Path:
    configuration = repository.configuration(experiment_id)
    results = repository.results(experiment_id)
    manifest = repository.manifest(experiment_id)
    destination.mkdir(parents=True, exist_ok=True)
    groups = summarize(results)
    summary = {
        "experiment_id": experiment_id,
        "configuration_hash": configuration.identity,
        "manifest": manifest,
        "groups": groups,
        "pareto_frontiers": report_frontiers(results),
        "policy_comparison": policy_comparison(results),
        "research": research_analysis(results),
    }
    analysis = research_analysis(results)
    events = manifest.get("events", [])
    requested_policies = manifest.get("extra", {})
    expected = (
        requested_policies.get("policies", []) if isinstance(requested_policies, dict) else []
    )
    request_summary = []
    for policy in expected if isinstance(expected, list) else []:
        served = [r for r in results if r.policy == policy]
        abstained = (
            sum(
                isinstance(e, dict)
                and e.get("kind") == "abstention"
                and isinstance(e.get("payload"), dict)
                and cast(dict[str, Any], e["payload"]).get("policy") == policy
                for e in events
            )
            if isinstance(events, list)
            else 0
        )
        total = len(served) + abstained
        request_summary.append(
            {
                "policy": policy,
                "requested": total,
                "served": len(served),
                "abstentions": abstained,
                "abstention_rate": abstained / total if total else None,
                "failures": sum(not r.success for r in served),
            }
        )
    analysis["request_summary"] = request_summary
    summary["research"] = analysis
    export_tables(analysis, destination)
    (destination / "research.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    export_json(results, destination / "raw.json")
    export_csv(results, destination / "raw.csv")
    lines = [
        f"# AdaptiveEdge experiment {experiment_id}",
        "",
        "SIMULATED observations are functional fixtures, not scientific evidence.",
        "",
        "Raw observations: raw.json / raw.csv. Units: ms, decimal GB, tokens/s.",
        "",
        "Statistics use successful observations; failures and missing-value counts remain visible.",
        "Quality metrics are separate; lexical overlap does not measure human quality.",
        "",
        "| Model | Category | Policy | Simulated | Runs | Failures |",
        "|---|---|---|---|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group['model']} | {group['category']} | {group['policy']} | "
            f"{group['simulated']} | {group['observations']} | {group['failures']} |"
        )
    lines.extend(
        [
            "",
            "## Measurements",
            "",
            "Compatible model/category/hardware/mode groups. Unavailable values are —.",
            "",
            "| Model / category / policy | Metric | Count | Mean | Median | SD | Min | Max |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in groups:
        statistics = group["statistics"]
        if not isinstance(statistics, dict):
            continue
        for name, statistic in statistics.items():
            if not isinstance(statistic, dict):
                continue
            cells = []
            for field in ("count", "mean", "median", "standard_deviation", "minimum", "maximum"):
                value = statistic[field]
                cells.append(f"{value:.3g}" if isinstance(value, (int, float)) else "—")
            lines.append(
                f"| {group['model']} / {group['category']} / {group['policy']} | {name} | "
                + " | ".join(cells)
                + " |"
            )
    policies = policy_comparison(results)
    if policies:
        lines.extend(
            [
                "",
                "## Policy comparison",
                "",
                "Task metrics stay separate. Constraint compliance uses observed values.",
                "",
                "| Policy | Category | Metric | Attempted | Failures | "
                "Violation rate | Unknown compliance |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for policy in policies:
            lines.append(
                f"| {policy['policy']} | {policy['category']} | {policy['metric']} | "
                f"{policy['attempted']} | {policy['failures']} | "
                f"{policy['constraint_violation_rate']} | "
                f"{policy['constraint_compliance_unknown']} |"
            )
    events = manifest.get("events", [])
    if isinstance(events, list):
        abstentions = sum(isinstance(e, dict) and e.get("kind") == "abstention" for e in events)
        lines.extend(
            [
                "",
                f"Abstentions (no inference attempted): {abstentions}. Details in manifest events.",
            ]
        )
    lines.extend(
        [
            "",
            "Detailed count, mean, median, sample standard deviation, minimum and maximum",
            "for each metric and compatible group are available in summary.json.",
            "",
            "Hardware, software, configuration and lifecycle events are in manifest.json.",
            "",
        ]
    )
    path = destination / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
