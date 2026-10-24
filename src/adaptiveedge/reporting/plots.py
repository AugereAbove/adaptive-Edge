import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adaptiveedge.core.models import BenchmarkResult, Estimates
from adaptiveedge.reporting.research import measurements, research_analysis
from adaptiveedge.selection.pareto import frontier


def plot_quality_latency(results: Sequence[BenchmarkResult], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(9, 6))
    for row in sorted(results, key=lambda r: (r.task.id, r.configuration.identity, r.repetition)):
        if not row.success or not row.task.metrics:
            continue
        metric = row.task.metrics[0]
        quality = row.quality.get(metric)
        if quality is None or row.inference.latency_ms is None:
            continue
        label = f"{row.configuration.display_name} / {row.task.category} / {metric}"
        axes.scatter(
            row.inference.latency_ms,
            quality,
            label=label,
            marker="x" if row.inference.simulated else "o",
        )
    axes.set_xlabel("Request latency (ms)")
    axes.set_ylabel("Task metric score (metric-specific; not a shared quality scale)")
    identifiers = sorted({r.experiment_id for r in results})
    hardware = sorted({r.hardware.fingerprint[:12] for r in results})
    provenance = (
        "Includes SIMULATED fixture values (x markers)"
        if any(r.inference.simulated for r in results)
        else "Measured backend observations"
    )
    axes.set_title(
        f"{provenance}\nExperiments: {', '.join(identifiers)}\nHardware: {', '.join(hardware)}",
        fontsize=8,
    )
    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    if unique:
        axes.legend(unique.values(), unique.keys(), fontsize=7)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, metadata={"Software": "AdaptiveEdge"})
    plt.close(figure)


def publication_plots(results: Sequence[BenchmarkResult], destination: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "adaptiveedge-0.1"
    destination.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, ...], list[BenchmarkResult]] = defaultdict(list)
    for row in results:
        groups[
            (
                row.experiment_id,
                row.task.category,
                row.task.metrics[0] if row.task.metrics else "unscored",
                row.hardware.fingerprint,
                str(row.inference.metadata.get("mode")),
                str(row.inference.simulated),
                row.split,
            )
        ].append(row)
    outputs: list[str] = []
    unavailable: list[dict[str, str]] = []
    labels = {
        "quality": "Task score (0–1)",
        "latency_ms": "Request latency (ms)",
        "ram_gb": "Model peak RAM (GB)",
        "vram_gb": "Model peak VRAM (GB)",
        "throughput_tps": "Server generation throughput (tokens/s)",
    }

    def save(figure: Any, axes: Any, name: str, title: str, xlabel: str, ylabel: str) -> None:
        axes.set_title(title, fontsize=9)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xlim(left=0)
        axes.set_ylim(bottom=0)
        axes.grid(alpha=0.2)
        handles, legend_labels = axes.get_legend_handles_labels()
        if handles:
            axes.legend(handles, legend_labels, fontsize=7)
        figure.tight_layout()
        for extension in ("svg", "png"):
            path = destination / f"{name}.{extension}"
            metadata = {"Date": None} if extension == "svg" else {"Software": "AdaptiveEdge"}
            figure.savefig(path, dpi=300, metadata=metadata)
            outputs.append(path.name)
        plt.close(figure)

    for index, (key, rows) in enumerate(sorted(groups.items())):
        aggregates = research_analysis(rows)["aggregates"]
        prefix = f"{index:02d}-{key[1]}"
        title = (
            f"{'SIMULATED ' if rows[0].inference.simulated else ''}"
            f"{key[1]} / {key[2]} / {key[4]}\nExperiment {key[0]}"
        )
        for x, y, name in (
            ("latency_ms", "quality", "quality-latency"),
            ("ram_gb", "quality", "quality-memory"),
            ("latency_ms", "throughput_tps", "throughput-latency"),
            ("latency_ms", "quality", "pareto"),
        ):
            available = [
                a
                for a in aggregates
                if a["statistics"][x]["mean"] is not None and a["statistics"][y]["mean"] is not None
            ]
            if not available:
                unavailable.append(
                    {
                        "context": prefix,
                        "figure": name,
                        "reason": "required model measurement unavailable",
                    }
                )
                continue
            figure, axes = plt.subplots(figsize=(8, 5))
            for item in available:
                axes.scatter(
                    item["statistics"][x]["mean"],
                    item["statistics"][y]["mean"],
                    label=item["policy_or_model"],
                )
            if name == "pareto":
                points = [
                    Estimates(
                        quality=a["statistics"]["quality"]["mean"],
                        latency_ms=a["statistics"]["latency_ms"]["mean"],
                    )
                    for a in available
                ]
                kept = sorted(
                    (points[i] for i in frontier(points, ("quality", "latency_ms"))),
                    key=lambda p: p.latency_ms or 0,
                )
                axes.plot(
                    [p.latency_ms or 0.0 for p in kept],
                    [p.quality or 0.0 for p in kept],
                    linestyle="--",
                    color="black",
                    label="Descriptive frontier",
                )
            if y == "quality":
                axes.set_ylim(0, 1.05)
            save(figure, axes, f"{prefix}-{name}", title, labels[x], labels[y])
        if any(r.policy for r in rows):
            for metric in (
                "quality",
                "latency_ms",
                "ram_gb",
                "vram_gb",
                "throughput_tps",
                "constraint_satisfaction_rate",
            ):
                available = [
                    (
                        a,
                        a.get(metric)
                        if metric == "constraint_satisfaction_rate"
                        else a["statistics"][metric]["mean"],
                    )
                    for a in aggregates
                ]
                available = [(a, v) for a, v in available if v is not None]
                if not available:
                    unavailable.append(
                        {
                            "context": prefix,
                            "figure": f"policy-{metric}",
                            "reason": "measurement unavailable",
                        }
                    )
                    continue
                figure, axes = plt.subplots(figsize=(8, 5))
                axes.bar([a["policy_or_model"] for a, _ in available], [v for _, v in available])
                axes.tick_params(axis="x", rotation=25)
                if metric in {"quality", "constraint_satisfaction_rate"}:
                    axes.set_ylim(0, 1.05)
                save(
                    figure,
                    axes,
                    f"{prefix}-policy-{metric}",
                    title,
                    "Policy",
                    labels.get(metric, "Observed constraint satisfaction (0–1)"),
                )
            for metric in ("latency_ms", "quality", "ram_gb", "throughput_tps"):
                pairs = []
                for row in rows:
                    if not row.success or not row.selection:
                        continue
                    candidates = row.selection.get("candidates")
                    if not isinstance(candidates, list):
                        continue
                    for candidate in candidates:
                        if not isinstance(candidate, dict) or candidate.get(
                            "configuration"
                        ) != row.configuration.model_dump(mode="json"):
                            continue
                        estimate = candidate.get("estimates")
                        predicted = estimate.get(metric) if isinstance(estimate, dict) else None
                        actual = measurements(row)[metric]
                        if isinstance(predicted, (int, float)) and actual is not None:
                            pairs.append((predicted, actual, row.policy))
                if not pairs:
                    continue
                figure, axes = plt.subplots(figsize=(8, 5))
                for policy in sorted({str(p[2]) for p in pairs}):
                    subset = [p for p in pairs if p[2] == policy]
                    axes.scatter(
                        [p[0] for p in subset], [p[1] for p in subset], label=policy, alpha=0.6
                    )
                maximum = max(max(p[0], p[1]) for p in pairs)
                axes.plot(
                    [0, maximum], [0, maximum], color="black", linestyle="--", label="Identity"
                )
                save(
                    figure,
                    axes,
                    f"{prefix}-prediction-{metric}",
                    title,
                    f"Predicted {labels[metric]}",
                    f"Observed {labels[metric]}",
                )
    (destination / "availability.json").write_text(
        json.dumps(unavailable, indent=2) + "\n", encoding="utf-8"
    )
    return outputs
