import csv
import json
from pathlib import Path
from typing import Any


def flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = json.dumps(item, sort_keys=True) if isinstance(item, list) else item
    return result


def export_tables(analysis: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("aggregates", "aggregate-results"),
        ("aggregates", "policy-comparison"),
        ("prediction_errors", "prediction-errors"),
        ("effects", "effects"),
    ):
        rows = [
            flatten(row)
            for row in analysis[key]
            if key != "aggregates" or name != "policy-comparison" or "policy_or_model" in row
        ]
        fields = sorted({field for row in rows for field in row})
        with (destination / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
