import re
import subprocess
from pathlib import Path


def audit_repository(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    names = sorted(set(result.stdout.decode().split("\0")) - {""})
    patterns = (
        re.compile(r"[A-Za-z]:[\\/](?:Users|home)[\\/]"),
        re.compile(r"/(?:Users|home)/[^/\s]+/"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    )
    violations = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        if path.stat().st_size > 500_000:
            violations.append(f"{name}: exceeds 500 KB")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"{name}: unexpected binary file")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns):
                violations.append(f"{name}:{number}: possible secret or absolute personal path")
    return violations


if __name__ == "__main__":
    failures = audit_repository(Path(__file__).resolve().parents[1])
    if failures:
        raise SystemExit("\n".join(failures))
    print("Repository audit passed: no matched secrets, personal paths, large files or binaries")
