import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def verify() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "outputs/release"
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .strip("\0")
        .split("\0")
    )
    checks = []

    def check(name: str, arguments: list[str], live: bool = False) -> None:
        current = dict(environment)
        current.pop("ADAPTIVEEDGE_REAL_MODEL", None)
        if live:
            current["ADAPTIVEEDGE_REAL_MODEL"] = "qwen2.5:0.5b"
        outcome = subprocess.run(
            arguments,
            cwd=root,
            env=current,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        log = outcome.stdout + outcome.stderr
        path = destination / f"{name}.txt"
        path.write_text(log, encoding="utf-8")
        checks.append(
            {
                "check": name,
                "arguments": arguments,
                "exit_code": outcome.returncode,
                "status": "passed" if outcome.returncode == 0 else "failed",
                "log": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(name, checks[-1]["status"], flush=True)

    python = sys.executable
    commands = [
        ("ruff", [python, "-m", "ruff", "check", "."]),
        ("format", [python, "-m", "ruff", "format", "--check", "."]),
        ("mypy", [python, "-m", "mypy", "src"]),
        (
            "tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "--cov-fail-under=94.28",
                "--cov-report=json:outputs/release/coverage.json",
                "--cov-report=xml",
            ],
        ),
        ("source-audit", [python, "scripts/audit_source.py"]),
        ("repository-audit", [python, "scripts/audit_repository.py"]),
        ("claim-audit", [python, "scripts/audit_claims.py"]),
        (
            "dependency-audit",
            [
                python,
                "-m",
                "pip_audit",
                "--format=json",
                "--output=outputs/release/dependency-audit.json",
            ],
        ),
        ("pip-check", [python, "-m", "pip", "check"]),
        ("pre-commit", [python, "-m", "pre_commit", "run", "--files", *files]),
        ("build", [python, "-m", "build"]),
        ("mock-smoke", [python, "scripts/smoke.py", "--output", "outputs/release/mock-smoke"]),
        ("live-smoke", [python, "scripts/verify_live.py"]),
    ]
    for name, arguments in commands:
        check(name, arguments)
    check(
        "live-tests",
        [
            python,
            "-m",
            "pytest",
            "tests/integration/test_publication.py",
            "-m",
            "real_backend",
            "-q",
            "--no-cov",
        ],
        live=True,
    )
    coverage = json.loads((destination / "coverage.json").read_text(encoding="utf-8"))
    clean = json.loads((destination / "clean-sdist-coverage.json").read_text(encoding="utf-8"))
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
        "coverage": coverage["totals"],
        "clean_sdist_coverage": clean["totals"],
        "core_coverage": {
            name: data["summary"]["percent_covered"]
            for name, data in coverage["files"].items()
            if any(part in name for part in ("selection", "storage"))
        },
        "real_experiments": json.loads(
            (root / "outputs/baseline-real/experiments.json").read_text()
        ),
        "project_file_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
            if (root / name).is_file()
        },
        "published": False,
        "committed": False,
    }
    (destination / "verification.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    if any(c["exit_code"] for c in checks):
        raise SystemExit("Release verification has failing checks; inspect logs")


if __name__ == "__main__":
    verify()
