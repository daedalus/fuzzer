"""Regression: perf_shim.c opened counters without FD_CLOEXEC.

Every target the fuzzer spawned inherited the counter fds.
"""

import fcntl
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.perf_event import PERF_COUNT_HW_INSTRUCTIONS, PerfCounters, PerfShim

SHIM = "src/fuzzer_tool/adapters/perf_shim.c"


@pytest.fixture
def shim(tmp_path):
    if shutil.which("clang") is None:
        pytest.skip("no clang")
    out = tmp_path / "perf_shim.so"
    r = subprocess.run(["clang", "-O2", "-shared", "-fPIC", "-o", str(out), SHIM])
    assert r.returncode == 0
    return PerfShim(str(out))


def test_regression_shim_fd_is_cloexec(shim):
    if not PerfCounters(counter_names=["instructions"]).has_pmu:
        pytest.skip("no PMU")
    fd = shim.perf_open(0, PERF_COUNT_HW_INSTRUCTIONS)
    assert fd >= 0
    try:
        assert fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    finally:
        shim.perf_close(fd)


def test_shim_open_failure_is_negative(shim):
    """Adversarial: a bad pid reports failure, not a stray fd."""
    assert shim.perf_open(-5, PERF_COUNT_HW_INSTRUCTIONS) < 0
