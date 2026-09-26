"""tools/benchmark.py: one entry point for every benchmark harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
LIB = TOOLS / "lib"
ENTRY = TOOLS / "benchmark.py"

sys.path.insert(0, str(TOOLS))

import benchmark  # noqa: E402

# Shared helpers, not runnable benchmarks.
_NOT_BENCHMARKS = {"bench_lock.py", "bench_common.sh"}


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRY), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_every_bench_script_is_registered():
    """Drift guard: a new tools/lib/bench_* script must be reachable."""
    on_disk = {p.name for p in LIB.glob("bench*.py")} | {p.name for p in LIB.glob("bench*.sh")}
    registered = {Path(s).name for s in benchmark.BENCHMARKS.values()}

    assert on_disk - _NOT_BENCHMARKS <= registered


def test_registered_scripts_exist():
    for name, script in benchmark.BENCHMARKS.items():
        assert (LIB / script).is_file(), name


def test_no_benchmark_left_in_tools_root():
    """Falsification: benchmark.py is the only benchmark file under tools/."""
    stray = {p.name for p in TOOLS.glob("bench*")} - {"benchmark.py"}
    registered = {Path(s).name for s in benchmark.BENCHMARKS.values()}

    assert stray == set()
    assert not {p.name for p in TOOLS.iterdir()} & registered


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


# Harnesses whose --help is side-effect free (argparse). The rest run at import.
_HELP_SAFE = ("paired", "replicated", "noise", "diff", "lineage")


@pytest.mark.parametrize("name", _HELP_SAFE)
def test_adversarial_harness_imports_resolve(name):
    """Fresh interpreter: a sibling import left behind in tools/ fails here,
    even when another test module already put tools/ on sys.path."""
    out = _run(name, "--help")

    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout
