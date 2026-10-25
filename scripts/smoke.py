import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def smoke(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    database = output / "smoke.db"
    executed = []

    def command(arguments: list[str], expected: int = 0) -> str:
        completed = subprocess.run(
            [sys.executable, "-m", "adaptiveedge", *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
        )
        executed.append({"arguments": arguments, "exit_code": completed.returncode})
        if completed.returncode != expected:
            raise RuntimeError(
                f"Command failed: {arguments}\n{completed.stdout}\n{completed.stderr}"
            )
        return completed.stdout

    db_options = ["--db", str(database)]
    command(["--help"])
    command(["benchmark", "--help"])
    doctor = json.loads(command(["doctor", *db_options, "--json"]))
    benchmark = json.loads(
        command(["benchmark", "run", "configs/benchmarks/basic.yaml", *db_options, "--json"])
    )
    identifier = benchmark["experiment_id"]
    assert benchmark["runs"] == 30 and benchmark["failures"] == 0 and benchmark["simulated"]
    command(["benchmark", "list", *db_options, "--json"])
    shown = json.loads(command(["benchmark", "show", identifier, *db_options, "--json"]))
    assert shown["runs"] == 30
    unconstrained = json.loads(command(["select", "--task", "reasoning", *db_options, "--json"]))
    constrained = json.loads(
        command(["select", "--task", "reasoning", "--max-latency-ms", "30", *db_options, "--json"])
    )
    assert unconstrained["selected"]["model"] == "mock-large"
    assert constrained["selected"]["model"] == "mock-small"
    impossible = json.loads(
        command(
            ["select", "--task", "reasoning", "--max-latency-ms", "1", *db_options, "--json"],
            expected=3,
        )
    )
    assert impossible["selected"] is None
    inference = json.loads(
        command(
            [
                "run",
                "--task",
                "reasoning",
                "--input-file",
                "configs/examples/input.txt",
                "--max-latency-ms",
                "30",
                *db_options,
                "--json",
            ]
        )
    )
    assert inference["inference"]["simulated"]
    report_args = ["report", identifier, "--output", str(output / "report"), *db_options]
    if importlib.util.find_spec("matplotlib"):
        report_args.append("--plots")
    command(report_args)
    for suffix in ("json", "csv"):
        command(["export", identifier, str(output / f"export.{suffix}"), *db_options])
    assert len(json.loads((output / "export.json").read_text(encoding="utf-8"))) == 30
    with (output / "export.csv").open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 30
    evaluation = json.loads(
        command(
            [
                "evaluate",
                identifier,
                "configs/benchmarks/evaluation-smoke.yaml",
                "--allow-overlap",
                *db_options,
                "--json",
            ]
        )
    )
    command(
        [
            "report",
            evaluation["experiment_id"],
            "--output",
            str(output / "policy-report"),
            *db_options,
        ]
    )
    reproduction = json.loads(
        command(["benchmark", "reproduce", identifier, *db_options, "--json"])
    )
    summary = {
        "status": "passed",
        "benchmark": benchmark,
        "constraint_changed_selection": True,
        "no_feasible_exit_code": 3,
        "inference_experiment": inference["experiment_id"],
        "evaluation": evaluation,
        "reproduction": reproduction,
        "doctor": doctor,
        "commands": executed,
        "notice": "All model-performance observations in this smoke run are simulated fixtures",
    }
    (output / "verification.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Smoke verification passed; evidence: {output / 'verification.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    smoke(parser.parse_args().output)
