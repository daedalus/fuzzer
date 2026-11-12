"""--hw-perf on every execution backend.

Counts must be the target's: a busy input reads far more CPU time than an
idle one. Software ``task_clock`` keeps this runnable without a PMU.
"""

import os
import platform
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters.perf_event import PerfCounters

SPIN_RATIO = 5

_SPIN = r"""
#include <unistd.h>
int main(void) {
    char c = 0;
    if (read(0, &c, 1) == 1 && c == '1') {
        volatile unsigned long x = 0;
        for (unsigned long i = 0; i < 60000000UL; i++) x += i;
    }
    return 0;
}
"""


def _sw_ok() -> bool:
    return PerfCounters(counter_names=["task_clock"]).available


needs_perf = pytest.mark.skipif(not _sw_ok(), reason="perf_event_open unavailable")
needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")


@pytest.fixture
def spin(tmp_path):
    src, exe = tmp_path / "spin.c", tmp_path / "spin"
    src.write_text(_SPIN)
    r = subprocess.run(["clang", "-O0", "-o", str(exe), str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:300]
    return str(exe)


class _SpyPerf:
    """Records attach calls; reads nothing."""

    def __init__(self):
        self.opened: list[int] = []
        self.ensured: list[int] = []

    def open_for_pid(self, pid):
        self.opened.append(pid)
        return True

    def ensure_open(self, pid):
        self.ensured.append(pid)
        return True

    def read_and_reset(self):
        return {"task_clock": 1}


@needs_perf
@needs_clang
def test_forkserver_counts_forked_child(spin):
    from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled

    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader not buildable")
    perf = PerfCounters(counter_names=["task_clock"])
    fs = ForkserverRunner(spin, timeout=5, perf_counters=perf)
    try:
        assert fs.start()
        perf.read_and_reset()
        fs.run_one(b"0")
        idle = perf.read_and_reset()["task_clock"]
        fs.run_one(b"1")
        busy = perf.read_and_reset()["task_clock"]
    finally:
        fs.stop()
        perf.close()

    # Inherited counters fold the exited child's time into the loader's.
    assert busy > SPIN_RATIO * max(1, idle)


@needs_clang
def test_forkserver_attaches_before_init(spin):
    """Falsification: the attach is on the loader pid, once per start."""
    from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled

    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader not buildable")
    spy = _SpyPerf()
    fs = ForkserverRunner(spin, timeout=5, perf_counters=spy)
    try:
        assert fs.start()
        pid = fs._proc.pid
        fs.run_one(b"0")
        fs.run_one(b"0")
    finally:
        fs.stop()
    assert spy.opened == [pid]


def _sh_stop(tmp_path):
    script = tmp_path / "t.sh"
    script.write_text("#!/bin/sh\nkill -STOP $$\nsleep 5\n")
    script.chmod(0o755)
    return str(script)


def test_persistent_attaches_on_start(tmp_path):
    """The attach follows the fork, before start() waits for the stop:
    asserted on the fork's pid, not on start()'s timing-bound verdict."""
    from fuzzer_tool.adapters.persistent_signal import PersistentRunner

    spy = _SpyPerf()
    pids = []
    pr = PersistentRunner(_sh_stop(tmp_path), timeout=1, perf_counters=spy)
    fork = os.fork

    def record_fork():
        pid = fork()
        if pid:
            pids.append(pid)
        return pid

    os.fork = record_fork
    try:
        pr.start()
    finally:
        os.fork = fork
        pr.stop()
    assert spy.opened == pids
    assert len(pids) == 1


def test_persistent_without_perf_unchanged(tmp_path):
    """Adversarial: no counters, no attach attempt, no error."""
    from fuzzer_tool.adapters.persistent_signal import PersistentRunner

    pr = PersistentRunner(_sh_stop(tmp_path), timeout=1)
    try:
        pr.start()
    finally:
        pr.stop()
    assert pr._perf_counters is None


@needs_perf
@pytest.mark.skipif(platform.machine() != "x86_64", reason="ptrace runner is x86-64")
def test_ptrace_counts_tracee(tmp_path):
    from fuzzer_tool.services.ptrace_coverage import PtraceCoverage
    from fuzzer_tool.services.runner import TargetRunner

    def run(body):
        script = tmp_path / "t.sh"
        script.write_text(body)
        script.chmod(0o755)
        perf = PerfCounters(counter_names=["task_clock"])
        f = SimpleNamespace(
            ptrace_cov=PtraceCoverage(str(script)),
            target=str(script),
            timeout=10,
            _last_child_pid=None,
            _last_fault_addr=None,
            _last_regs={},
            _perf_counters=perf,
            _last_perf_deltas={},
        )
        TargetRunner(f)._run_target_ptrace(b"")
        perf.close()
        return f._last_perf_deltas.get("task_clock", 0)

    idle = run("#!/bin/sh\nexit 0\n")
    busy = run("#!/bin/sh\ni=0\nwhile [ $i -lt 200000 ]; do i=$((i+1)); done\n")
    assert busy > SPIN_RATIO * max(1, idle)


def _inproc_fuzzer(spy, perf_pid):
    ip = SimpleNamespace(
        run_one=lambda data: (0, ""),
        perf_pid=lambda: perf_pid,
        _last_fault_addr=None,
        _last_regs={},
        direct_lite=True,
    )
    return SimpleNamespace(
        multi_targets=False,
        shm_cov=None,
        _cmplog=None,
        _inprocess_runner=ip,
        pt_cov=None,
        branch_cov=None,
        _perf_counters=spy,
        _last_perf_deltas={},
        _last_fault_addr=None,
        _last_regs={},
    )


def test_inprocess_opens_once():
    """Direct modes: counters on this thread, kept for the run."""
    from fuzzer_tool.services.runner import TargetRunner

    spy = _SpyPerf()
    f = _inproc_fuzzer(spy, 0)
    runner = TargetRunner(f)
    for _ in range(3):
        runner.run_target(b"x")

    assert spy.opened == []
    assert spy.ensured == [0, 0, 0]
    assert f._last_perf_deltas == {"task_clock": 1}


def test_inprocess_loader_pid_counted():
    """Falsification: a loader subprocess is counted, not this process."""
    from fuzzer_tool.services.runner import TargetRunner

    spy = _SpyPerf()
    TargetRunner(_inproc_fuzzer(spy, 4321)).run_target(b"x")
    assert spy.ensured == [4321]


def test_inprocess_per_call_loader_not_attached():
    """Adversarial: no long-lived process, nothing to attach to."""
    from fuzzer_tool.services.runner import TargetRunner

    spy = _SpyPerf()
    TargetRunner(_inproc_fuzzer(spy, None)).run_target(b"x")
    assert spy.ensured == []


_SPIN_SO = r"""
#include <stddef.h>
#include <stdint.h>
int LLVMFuzzerTestOneInput(const uint8_t *d, size_t n) {
    if (n && d[0] == '1') {
        volatile unsigned long x = 0;
        for (unsigned long i = 0; i < 60000000UL; i++) x += i;
    }
    return 0;
}
"""


@needs_perf
@needs_clang
def test_inprocess_persistent_loader_counts_child(tmp_path):
    from fuzzer_tool.adapters.inprocess import InProcessRunner

    src, so = tmp_path / "spin.c", tmp_path / "spin.so"
    src.write_text(_SPIN_SO)
    r = subprocess.run(
        ["clang", "-O0", "-shared", "-fPIC", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[:300]
    perf = PerfCounters(counter_names=["task_clock"])
    ip = InProcessRunner(target=str(so), timeout=10, perf_counters=perf)
    try:
        assert ip._persistent is not None
        assert ip.perf_pid() == ip._persistent._proc.pid
        perf.read_and_reset()
        ip.run_one(b"0")
        idle = perf.read_and_reset()["task_clock"]
        ip.run_one(b"1")
        busy = perf.read_and_reset()["task_clock"]
    finally:
        ip.stop()
        perf.close()

    assert busy > SPIN_RATIO * max(1, idle)


@needs_perf
@needs_clang
def test_hw_perf_keeps_forkserver(tmp_path, spin):
    """--hw-perf used to force spawn-per-exec; the forkserver now counts."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a").write_bytes(b"0")
    f = Fuzzer(
        target=spin,
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "k"),
        timeout=5,
        hw_perf_counters=["task_clock"],
        quiet_stats=True,
    )
    try:
        assert f._forkserver is not None
        f._run_target(b"1")
        assert f._last_perf_deltas.get("task_clock", 0) > 0
    finally:
        if f._forkserver is not None:
            f._forkserver.stop()
        f._perf_counters.close()
