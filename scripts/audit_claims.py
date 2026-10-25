import csv
import hashlib
import json
from pathlib import Path

from adaptiveedge.config.schema import BenchmarkConfiguration


def audit(root: Path) -> None:
    artifact = root / "examples/results/baseline-real"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["checksums"].items():
        assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == expected, name
    configs = {
        name: BenchmarkConfiguration.model_validate_json(
            (artifact / f"{name}-config.json").read_text(encoding="utf-8")
        )
        for name in ("calibration", "evaluation")
    }
    assert configs["calibration"].experiment.split == "calibration"
    assert configs["evaluation"].experiment.split == "evaluation"
    assert {t.id for t in configs["calibration"].tasks}.isdisjoint(
        t.id for t in configs["evaluation"].tasks
    )
    assert {t.prompt for t in configs["calibration"].tasks}.isdisjoint(
        t.prompt for t in configs["evaluation"].tasks
    )
    all_ids: set[str] = set()
    for name in ("calibration", "evaluation", "ablation"):
        with (artifact / f"{name}-raw.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert rows and all(r["simulated"] == "False" and r["success"] == "True" for r in rows)
        assert all(
            r["split"] == ("calibration" if name == "calibration" else "evaluation") for r in rows
        )
        assert all_ids.isdisjoint(r["run_id"] for r in rows)
        all_ids.update(r["run_id"] for r in rows)
        if name == "calibration":
            assert (
                len(rows)
                == len(configs[name].models)
                * len(configs[name].tasks)
                * configs[name].experiment.repetitions
            )
    research = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
    requested = sum(item["requested"] for item in research["request_summary"])
    assert requested == len(configs["evaluation"].tasks) * configs[
        "evaluation"
    ].experiment.repetitions * len(research["request_summary"])
    effects = {(e["policy_or_model"], e["category"]): e for e in research["effects"]}
    for item in research["aggregates"]:
        reference = next(
            a
            for a in research["aggregates"]
            if a["policy_or_model"] == "highest_quality" and a["category"] == item["category"]
        )
        effect = effects[(item["policy_or_model"], item["category"])]
        numerator, denominator = (
            item["statistics"]["quality"]["mean"],
            reference["statistics"]["quality"]["mean"],
        )
        assert effect["quality_ratio"] == (numerator / denominator if denominator else None)
        assert item["statistics"]["quality"]["count"] <= len(configs["evaluation"].tasks)
    readme = (root / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- BEGIN GENERATED BASELINE -->\n", 1)[1].split(
        "<!-- END GENERATED BASELINE -->", 1
    )[0]
    assert block == (artifact / "readme-results.md").read_text(encoding="utf-8")
    print(
        "Scientific-claim audit passed: measured provenance, splits, checksums, "
        "ratios and generated README table"
    )


if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[1])
