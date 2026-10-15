import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def code_test_success(output: str, reference: str) -> float:
    if len(output) > 16384 or len(reference) > 8192:
        return 0.0
    worker = Path(__file__).with_name("code_worker.py")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"SYSTEMROOT", "TEMP", "TMP"}
    }
    with TemporaryDirectory(prefix="adaptiveedge-code-") as directory:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(worker)],
                input=json.dumps({"source": output, "fixture": json.loads(reference)}),
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
                cwd=directory,
                env=environment,
            )
            return float(json.loads(result.stdout)["score"])
        except (subprocess.SubprocessError, ValueError, KeyError, OSError):
            return 0.0
