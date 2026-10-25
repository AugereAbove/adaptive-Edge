import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from adaptiveedge.profiling.hardware import hardware_snapshot
from adaptiveedge.reporting.plots import publication_plots
from adaptiveedge.reporting.reports import generate_report
from adaptiveedge.reporting.research import research_analysis
from adaptiveedge.storage.repository import Repository


def curate(source: Path, destination: Path) -> None:
    identifiers = json.loads((source / "experiments.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    machine_details = source / "machine-details.json"
    manifest = {
        "version": "0.1.0",
        "machine_details": json.loads(machine_details.read_text(encoding="utf-8-sig"))
        if machine_details.exists()
        else None,
        "experiment_ids": identifiers,
        "models": json.loads((source / "models.json").read_text(encoding="utf-8")),
        "supplemental_postrun_hardware": hardware_snapshot().model_dump(mode="json"),
        ("hardware_note"): (
            "GPU support was installed after calibration began. The "
            "original fingerprint is frozen; the postrun NVML snapshot is "
            "supplemental, not an inference-peak measurement."
        ),
        ("execution_note"): (
            "Sequential model requests on a personal Windows desktop with "
            "other applications and development/verification processes "
            "active. Fixed order; no thermal or background-load control. "
            "Preliminary in-template evidence only."
        ),
        "software_packages": {
            name: importlib.metadata.version(name)
            for name in (
                "adaptiveedge",
                "pydantic",
                "PyYAML",
                "psutil",
                "typer",
                "httpx",
                "matplotlib",
                "numpy",
            )
        },
        ("warmup"): (
            "One first-task call per calibration model; one actual-task "
            "call before each policy/ablation measurement. Excluded from "
            "observations."
        ),
        "calibration": None,
    }
    with Repository(source / "baseline.db") as repository:
        for name, identifier in identifiers.items():
            generate_report(repository, identifier, source / name)
            publication_plots(repository.results(identifier), source / name / "figures")
        calibration = repository.results(identifiers["calibration"])
        evaluation = repository.results(identifiers["evaluation"])
        ablation = repository.results(identifiers["ablation"])
        assert calibration and evaluation and ablation
        assert all(
            r.success and not r.inference.simulated for r in (*calibration, *evaluation, *ablation)
        )
        assert {r.task.prompt for r in calibration}.isdisjoint(r.task.prompt for r in evaluation)
        manifest["calibration"] = {
            "hardware": calibration[0].hardware.model_dump(mode="json"),
            "software": calibration[0].software,
            "observations": len(calibration),
            "task_count": len({r.task.id for r in calibration}),
            "config_hash": repository.configuration(identifiers["calibration"]).identity,
        }
        for name in ("evaluation", "ablation"):
            current = repository.manifest(identifiers[name])
            manifest[name] = {
                "configuration": repository.configuration(identifiers[name]).model_dump(
                    mode="json"
                ),
                "provenance": current["extra"],
                "events": current["events"],
            }
        manifest["ollama"] = repository.manifest(identifiers["calibration"])["extra"]["runtimes"][
            "ollama"
        ]["version"]
        analysis = json.loads((source / "evaluation" / "research.json").read_text(encoding="utf-8"))
        ablated = research_analysis(ablation)
        for filename in (
            "research.json",
            "aggregate-results.csv",
            "policy-comparison.csv",
            "prediction-errors.csv",
            "effects.csv",
        ):
            shutil.copyfile(source / "evaluation" / filename, destination / filename)
        for name in ("calibration", "evaluation", "ablation"):
            shutil.copyfile(source / name / "raw.csv", destination / f"{name}-raw.csv")
        (destination / "ablation-research.json").write_text(
            json.dumps(ablated, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("calibration", "evaluation"):
            shutil.copyfile(source / f"{name}.json", destination / f"{name}-config.json")
        figures = destination / "figures"
        figures.mkdir(exist_ok=True)
        for path in sorted((source / "evaluation" / "figures").glob("*-prediction-latency_ms.svg")):
            shutil.copyfile(path, figures / path.name)
        for path in sorted((source / "evaluation" / "figures").glob("*-policy-quality.svg")):
            shutil.copyfile(path, figures / path.name)
        lines = [
            "# Preliminary real baseline",
            "",
            (
                "Generated from immutable SQLite observations by "
                "scripts/curate_results.py. No simulated observations are "
                "included."
            ),
            "",
            (
                f"Calibration: {len(calibration)} observations across "
                f"{len({r.task.id for r in calibration})} tasks and "
                f"{len({r.configuration.identity for r in calibration})} "
                f"models. Held-out: {len({r.task.id for r in evaluation})} "
                f"tasks, "
                f"{repository.configuration(identifiers['evaluation']).experiment.repetitions} "
                f"repetitions, {len(evaluation)} served requests. Ablation: "
                f"{len(ablation)} observations."
            ),
            "",
            (
                f"Ollama {manifest['ollama']}; Windows; RAM "
                f"{calibration[0].hardware.total_ram_gb:.3f} GB. Hardware "
                f"snapshots, model digests, settings, timestamps and frozen "
                f"training IDs are in manifest.json. GPU identity is a "
                f"supplemental postrun snapshot."
            ),
            "",
            "Models: " + ", ".join(sorted({r.configuration.model for r in calibration})) + ".",
            "",
            "## Adaptive versus highest-quality policy",
            "",
            (
                "Highest-quality means the policy selected from calibration; "
                "it is not a held-out oracle. Each row uses its own task "
                "metric. Negative latency reduction means slower. No pooled "
                "quality score is calculated."
            ),
            "",
            (
                "| Category | Adaptive quality | Reference quality | Quality "
                "retention | Latency reduction | Latency MAE (ms) | Quality "
                "MAE |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for effect in analysis["effects"]:
            if effect["policy_or_model"] != "adaptive":
                continue
            category = effect["category"]
            item = next(
                a
                for a in analysis["aggregates"]
                if a["policy_or_model"] == "adaptive" and a["category"] == category
            )
            reference = next(
                a
                for a in analysis["aggregates"]
                if a["policy_or_model"] == "highest_quality" and a["category"] == category
            )
            error = {
                e["metric"]: e["mae"]
                for e in analysis["prediction_errors"]
                if e["policy"] == "adaptive" and e["category"] == category
            }
            retention = (
                f"{effect['quality_ratio'] * 100:.2f}%"
                if effect["quality_ratio"] is not None
                else "undefined"
            )
            lines.append(
                f"| {category} | {item['statistics']['quality']['mean']:.3f} | "
                f"{reference['statistics']['quality']['mean']:.3f} | "
                f"{retention} | {effect['latency_ms_reduction_percent']:.2f}% | "
                f"{error['latency_ms']:.2f} | {error['quality']:.3f} |"
            )
        lines += [
            "",
            "## Request coverage",
            "",
            "| Policy | Requested | Served | Abstentions | Failures |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in analysis["request_summary"]:
            lines.append(
                f"| {item['policy']} | {item['requested']} | {item['served']} | "
                f"{item['abstentions']} | {item['failures']} |"
            )
        lines += [
            "",
            (
                "Model peak RAM/VRAM is unavailable; client RSS and resident "
                "allocation are not substituted. No model-memory reduction is "
                "claimed. Served-request constraint satisfaction is in "
                "research.json and is conditional on service; abstentions are "
                "separate."
            ),
            "",
            "## Equal-weight ablation",
            "",
            (
                "Same frozen calibration, held-out tasks, seeds and "
                "repetitions. Declared adaptive quality/latency weights are "
                "compared with equal weights; unavailable memory objectives "
                "are inactive. The ablation runs later, so latency changes are "
                "descriptive, not causal."
            ),
            "",
            (
                "| Category | Original quality | Equal-weight quality | "
                "Original latency (ms) | Equal-weight latency (ms) |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
        for item in ablated["aggregates"]:
            original = next(
                a
                for a in analysis["aggregates"]
                if a["policy_or_model"] == "adaptive" and a["category"] == item["category"]
            )
            lines.append(
                f"| {item['category']} | "
                f"{original['statistics']['quality']['mean']:.3f} | "
                f"{item['statistics']['quality']['mean']:.3f} | "
                f"{original['statistics']['latency_ms']['mean']:.2f} | "
                f"{item['statistics']['latency_ms']['mean']:.2f} |"
            )
        lines += [
            "",
            "## Uncertainty and interpretation",
            "",
            (
                "Tables use task means over successful repetitions. Percentile "
                "bootstrap: 2000 resamples, seed 42, empirical order "
                "statistics 50 and 1950 (one-based), 95% interval for the "
                "mean. Fewer than three task means suppresses intervals; fewer "
                "than ten is marked limited. Each category here has only three "
                "independent task instances. Intervals describe this tiny "
                "sample and are not population guarantees. No significance "
                "tests were performed."
            ),
            "",
            (
                "Calibration and evaluation have disjoint prompts/IDs but "
                "share task templates. This is a preliminary in-template "
                "baseline, not evidence of general routing superiority. The "
                "reasoning result illustrates a quality loss; calibration "
                "medians can hide individual task failures. Background work, "
                "fixed order and prompt caching limit latency interpretation. "
                "Code quality is restricted-expression test success, not full "
                "Python pass@1."
            ),
            "",
            (
                "See ../../../docs/methodology.md and "
                "../../../research/threats_to_validity.md. Full raw text, "
                "SQLite and high-resolution SVG/PNG figures remain in ignored "
                "outputs/baseline-real/. Curated CSVs omit generated text; "
                "configurations preserve authored fixtures. Checksums detect "
                "accidental artifact changes, not scientific validity."
            ),
            "",
        ]
        report = "\n".join(lines)
        (destination / "report.md").write_text(report, encoding="utf-8")
        block = (
            report[
                report.index("## Adaptive versus") : report.index("## Request coverage")
            ].rstrip()
            + "\n"
        )
        (destination / "readme-results.md").write_text(block, encoding="utf-8")
        for path in figures.glob("*.svg"):
            content = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in content.splitlines()) + "\n", encoding="utf-8"
            )
        manifest["checksums"] = {
            str(path.relative_to(destination)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/baseline-real"))
    parser.add_argument("--output", type=Path, default=Path("examples/results/baseline-real"))
    arguments = parser.parse_args()
    curate(arguments.source, arguments.output)
