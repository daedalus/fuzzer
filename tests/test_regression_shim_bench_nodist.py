"""Regression: shim micro-benchmark must build with distance mode off.

The warm loop called ``__afl_probe_distance`` unguarded, so
``tools/build-shim-bench.sh --nodist`` (``-D__AFL_DISTANCE_MODE=0``) failed
with an implicit-declaration error.
"""

import os
import shutil
import subprocess

import pytest

_TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ADAPTERS = os.path.join(_TOP, "src", "fuzzer_tool", "adapters")
_SHIM = os.path.join(_ADAPTERS, "afl_shim.c")
_BENCH = os.path.join(_TOP, "tools", "shim-micro-benchmark.c")


def _build(tmp_path, *defines: str) -> subprocess.CompletedProcess:
    if not shutil.which("clang"):
        pytest.skip("clang not installed")

    cmd = ["clang", "-O0", f"-I{_ADAPTERS}", "-include", _SHIM, *defines, _BENCH]
    return subprocess.run(
        [*cmd, "-o", str(tmp_path / "bench")],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_regression_shim_bench_nodist(tmp_path):
    r = _build(tmp_path, "-D__AFL_DISTANCE_MODE=0")
    assert r.returncode == 0, r.stderr


def test_regression_shim_bench_nodist_dist_mode_noop(tmp_path):
    """Adversarial: ``dist`` on a nodist build runs and reports, not crashes."""
    r = _build(tmp_path, "-D__AFL_DISTANCE_MODE=0")
    assert r.returncode == 0, r.stderr

    out = subprocess.run(
        [str(tmp_path / "bench"), "dist", "1024", "64", "1000", "1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("dist")


def test_shim_bench_default_builds(tmp_path):
    """Control: the default build compiles."""
    r = _build(tmp_path)
    assert r.returncode == 0, r.stderr
