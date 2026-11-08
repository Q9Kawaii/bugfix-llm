"""Run untrusted, model-generated Python in a subprocess with a timeout."""

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def run_program(program: str, timeout: float = 10.0) -> dict:
    """Execute `program` in a fresh Python process. Passing = exit code 0."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = proc.returncode == 0
            err = "" if passed else (proc.stderr or proc.stdout)[-1500:]
            return {"passed": passed, "error": err}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"timeout after {timeout}s"}


def run_many(programs: list[str], timeout: float = 10.0, workers: int = 8) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: run_program(p, timeout), programs))
