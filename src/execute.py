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


# ---------------------------------------------------- pytest-style test files
# Minimal stand-in so tests using pytest.raises / pytest.approx run without pytest.
PYTEST_SHIM = '''
import sys as _sys, types as _types, re as _re
_pt = _types.ModuleType("pytest")
class _Raises:
    def __init__(self, exc, match=None): self.exc, self.match, self.value = exc, match, None
    def __enter__(self): return self
    def __exit__(self, et, ev, tb):
        if et is None: raise AssertionError("DID NOT RAISE")
        if not issubclass(et, self.exc): return False
        if self.match and not _re.search(self.match, str(ev)): raise AssertionError("match failed")
        self.value = ev
        return True
def _approx(expected, rel=1e-6, abs_tol=1e-12, **kw):
    abs_tol = kw.get("abs", abs_tol)
    class _A:
        def __eq__(self, got):
            if isinstance(expected, (list, tuple)):
                return len(got) == len(expected) and all(_approx(e, rel, abs_tol) == g for e, g in zip(expected, got))
            return abs(got - expected) <= max(rel * abs(expected), abs_tol)
    return _A()
class _Mark:
    def __getattr__(self, name): return lambda *a, **k: (lambda f: f)
_pt.raises, _pt.approx, _pt.mark = _Raises, _approx, _Mark()
_sys.modules["pytest"] = _pt
'''

RUNNER = '''
import inspect as _inspect
_tests = [f for n, f in list(globals().items())
          if n.startswith("test") and _inspect.isfunction(f) and not _inspect.signature(f).parameters]
assert _tests, "no runnable tests"
for _t in _tests:
    _t()
'''


def program(code: str, tests: str) -> str:
    """Code + pytest-style test functions -> a script that runs every test."""
    return f"{PYTEST_SHIM}\n{code}\n\n{tests}\n{RUNNER}"
