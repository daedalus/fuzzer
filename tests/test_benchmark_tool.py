"""tools/benchmark.py: one entry point for every benchmark harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
ENTRY = TOOLS / "benchmark.py"

sys.path.insert(0, str(TOOLS))

import benchmark  # noqa: E402

# The dispatcher itself and library modules, not runnable benchmarks.
_NOT_BENCHMARKS = {"benchmark.py", "bench_lock.py"}


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRY), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_every_bench_script_is_registered():
    """Drift guard: a new tools/bench_* script must be reachable."""
    on_disk = {p.name for p in TOOLS.glob("bench*.py")} | {p.name for p in TOOLS.glob("bench*.sh")}
    registered = {Path(s).name for s in benchmark.BENCHMARKS.values()}

    assert on_disk - _NOT_BENCHMARKS <= registered


def test_registered_scripts_exist():
    for name, script in benchmark.BENCHMARKS.items():
        assert (TOOLS / script).is_file(), name


def test_list_names_every_benchmark():
    out = _run("list")

    assert out.returncode == 0
    for name in benchmark.BENCHMARKS:
        assert name in out.stdout


def test_falsify_unknown_benchmark_fails():
    """An unregistered name must fail loudly, not run something else."""
    out = _run("no-such-bench")

    assert out.returncode != 0
    assert "no-such-bench" in out.stderr


def test_adversarial_flags_forwarded_verbatim():
    """`--help` after the name belongs to the benchmark, not the dispatcher."""
    out = _run("diff", "--help")

    assert out.returncode == 0
    assert "--baseline" in out.stdout


def test_exit_code_propagates():
    """bench_diff's argparse error (missing required args) exits 2."""
    out = _run("diff")

    assert out.returncode == 2
    assert "--baseline" in out.stderr
