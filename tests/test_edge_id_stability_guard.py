"""Tests for the two F1 guards: per-process edge ids under ASLR.

F1 (docs/handover/handover_edge_id_axis_2026-09-18.md): afl_shim.c's
``__afl_get_caller_ctx`` hashes the return address of the frame above the
edge, so with ``__AFL_CTX_SENSITIVE=1`` and ASLR on, the same input reports
a different edge set in every process. ``Fuzzer.__init__`` calls
``disable_aslr()``, which normally makes the question moot; the shim now
also resolves that address relative to the load base when
``FUZZER_KEEP_ASLR=1`` is set in the *target's* environment.

Those two conditions are not the same one, which is what the guards under
test are about:

* :meth:`Fuzzer._ensure_ctx_ids_are_exec_stable` -- ``disable_aslr()`` also
  returns False when ``personality()`` is refused (seccomp, containers,
  non-Linux), and there the variable is unset, so the shim stays in raw
  mode while ASLR is on. The fuzzer sets the variable itself in that case
  rather than asking the user to set something named for keeping ASLR on a
  host that never turned it off.
* :meth:`Fuzzer._report_edge_id_stability` -- a three-execution probe at the
  end of seed calibration. A target built before the shim grew relative
  mode ignores the variable and exports no symbol saying so, so no static
  check can find it. This one measures.

The probe tests are end to end on real binaries, not mocks: F1 is a return
address moving between two processes, and nothing but two real processes
can show it. gcc is enough -- the driver calls the shim's callback
directly, so no ``-fsanitize-coverage`` support is needed (the same trick
tests/test_ctx_and_map_size.py uses; the driver is a copy, kept independent
on purpose so a change to either file's build matrix cannot silently alter
the other's subject).

Measured while writing these (gcc 13, PIE, 40 guards, six executions of the
identical input), with the current shim:

===========================  =====  =====  ============  =======
regime                       sizes  union  intersection  Jaccard
===========================  =====  =====  ============  =======
ctx=8, ASLR off              22 x6     22            22    1.000
ctx=8, ASLR on, raw mode     22 x6     76             0    0.000
ctx=8, ASLR on, relative     22 x6     22            22    1.000
ctx=0, ASLR on               22 x6     22            22    1.000
===========================  =====  =====  ============  =======

Row 2 is the gap the first guard closes; row 3 is what it closes it to.
Row 4 is the control that pins the context term as the cause rather than
ASLR by itself.
"""

import os
import shutil
import subprocess
import sys

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.services.fuzzer import Fuzzer

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
SHIM = os.path.join(_SRC, "fuzzer_tool", "adapters", "afl_shim.c")

_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    uint32_t n = (uint32_t)atoi(argv[1]);
    for (uint32_t g = 1; g <= n; g++) { uint32_t guard = g;
        __sanitizer_cov_trace_pc_guard(&guard); }
    return 0;
}
"""

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="personality() is Linux-only"
)


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    """Build the driver context-sensitive and context-free. {bits: path}."""
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("f1")
    src = d / "drv.c"
    src.write_text(_DRIVER)
    out = {}
    for bits, flags in {
        8: ["-D__AFL_CTX_SENSITIVE=1", "-fno-omit-frame-pointer"],
        0: ["-D__AFL_CTX_SENSITIVE=0"],
    }.items():
        exe = d / f"drv_{bits}"
        r = subprocess.run(
            ["gcc", "-O1", "-g", *flags, "-include", SHIM, "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            pytest.skip(f"shim failed to build at ctx_bits={bits}: {r.stderr[:300]}")
        out[bits] = str(exe)
    return out


class _Stub:
    """Just enough of a Fuzzer for the two guards, with no __init__.

    Binding the unbound methods keeps the subject the real implementation
    while avoiding a full Fuzzer construction (which needs a corpus, a
    runner, a state store and a target profile). Every attribute the two
    methods touch is set explicitly below, so a new dependency in either
    method shows up as an AttributeError here rather than passing silently.
    """

    _ensure_ctx_ids_are_exec_stable = Fuzzer._ensure_ctx_ids_are_exec_stable
    _repeat_edge_sets = Fuzzer._repeat_edge_sets
    _report_edge_id_stability = Fuzzer._report_edge_id_stability

    def __init__(self, target="/bin/true", aslr_disabled=True, shm=None, coverage=True):
        self.target = target
        self._aslr_disabled = aslr_disabled
        self.use_coverage = coverage
        self.shm_cov = shm
        self._target_shm_covs = {}


class TestRelativeModeFix:
    """_ensure_ctx_ids_are_exec_stable: the static half of the guard.

    Every test explicitly controls FUZZER_KEEP_ASLR, because the method
    both reads and writes it and the tests share a process.
    """

    @needs_cc
    def test_sets_the_variable_when_aslr_survived(self, drivers, capsys, monkeypatch):
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        f = _Stub(target=drivers[8], aslr_disabled=False, shm=object())
        f._ensure_ctx_ids_are_exec_stable([drivers[8]])
        assert os.environ.get("FUZZER_KEEP_ASLR") == "1"
        out = capsys.readouterr().out
        assert "context-sensitive" in out
        assert "load-base-relative" in out

    @needs_cc
    def test_silent_when_aslr_is_disabled(self, drivers, capsys, monkeypatch):
        """The production path. Same target; only the flag differs."""
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        f = _Stub(target=drivers[8], aslr_disabled=True, shm=object())
        f._ensure_ctx_ids_are_exec_stable([drivers[8]])
        assert capsys.readouterr().out == ""
        assert "FUZZER_KEEP_ASLR" not in os.environ

    @needs_cc
    def test_does_not_touch_a_context_free_build(self, drivers, capsys, monkeypatch):
        """ctx=0 is exec-stable under ASLR already; relative mode would only
        buy it a dladdr() call per distinct return address."""
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        f = _Stub(target=drivers[0], aslr_disabled=False, shm=object())
        f._ensure_ctx_ids_are_exec_stable([drivers[0]])
        assert capsys.readouterr().out == ""
        assert "FUZZER_KEEP_ASLR" not in os.environ

    def test_silent_when_the_marker_is_absent(self, capsys, monkeypatch):
        """detect_ctx_bits returns None for an unknown shim -- not evidence."""
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        f = _Stub(target="/bin/true", aslr_disabled=False, shm=object())
        f._ensure_ctx_ids_are_exec_stable(["/bin/true"])
        assert capsys.readouterr().out == ""
        assert "FUZZER_KEEP_ASLR" not in os.environ

    @needs_cc
    def test_says_nothing_when_the_user_already_set_it(self, drivers, capsys, monkeypatch):
        """Announcing a change that was not made is worse than silence."""
        monkeypatch.setenv("FUZZER_KEEP_ASLR", "1")
        f = _Stub(target=drivers[8], aslr_disabled=False, shm=object())
        f._ensure_ctx_ids_are_exec_stable([drivers[8]])
        assert capsys.readouterr().out == ""

    @needs_cc
    def test_leaves_blind_and_ptrace_runs_alone(self, drivers, capsys, monkeypatch):
        """--no-coverage reads no edge ids; --no-shm does not go through the
        shim's context hash. Neither can be damaged by F1."""
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        blind = _Stub(target=drivers[8], aslr_disabled=False, shm=object(), coverage=False)
        blind._ensure_ctx_ids_are_exec_stable([drivers[8]])
        ptrace = _Stub(target=drivers[8], aslr_disabled=False, shm=None)
        ptrace._ensure_ctx_ids_are_exec_stable([drivers[8]])
        assert capsys.readouterr().out == ""
        assert "FUZZER_KEEP_ASLR" not in os.environ


# The probe, run in a child so the parent's persona (one-shot, cached) is
# not what decides the result. Two levels for the same reason
# tests/test_aslr.py gives: personality() governs the NEXT execve, so the
# launcher stands in for the fuzzer parent and the driver for the target.
_PROBE_LAUNCHER = """
import ctypes, ctypes.util, os, subprocess, sys
sys.path.insert(0, {src!r})
from fuzzer_tool.adapters.process import ADDR_NO_RANDOMIZE
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.services.fuzzer import Fuzzer

if os.environ.get("DISABLE_ASLR") == "1":
    from fuzzer_tool.adapters.process import disable_aslr
    assert disable_aslr(), "could not disable ASLR; arm is not meaningful"
else:
    # Clear the flag rather than assume it is clear. personality() is
    # inherited, and any earlier test in the same pytest process that
    # constructed a Fuzzer has already set it for this process tree -- which
    # would hand this arm a stable load base and a green test for the wrong
    # reason. (tests/test_aslr.py::test_aslr_still_randomizes_without_the_call
    # has the same exposure and does not do this.)
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.personality.argtypes = [ctypes.c_ulong]
    libc.personality.restype = ctypes.c_int
    cur = libc.personality(0xFFFFFFFF)
    libc.personality(ctypes.c_ulong(cur & ~ADDR_NO_RANDOMIZE))
    # The libc mapping, not the first line of maps: the interpreter itself
    # may be non-PIE and load at a fixed 0x400000 on some distributions,
    # which would read as "no ASLR" on a host that has it.
    probe = "print([l.split('-')[0] for l in open('/proc/self/maps') if 'libc.so' in l][0])"
    bases = {{subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True).stdout.strip() for _ in range(4)}}
    if len(bases) == 1:
        print("NO_ASLR_AVAILABLE")
        raise SystemExit(0)

target = {target!r}
shm = ShmCoverage(size=8192)

class Probe:
    _repeat_edge_sets = Fuzzer._repeat_edge_sets
    _report_edge_id_stability = Fuzzer._report_edge_id_stability
    target = target
    shm_cov = shm
    _aslr_disabled = os.environ.get("DISABLE_ASLR") == "1"
    def _run_target(self, data):
        # Mirrors services/runner.run_target: reset the table, then execute
        # with the parent's environment copied through -- which is how
        # FUZZER_KEEP_ASLR reaches the shim in a real run. The driver takes
        # a guard count rather than a file, so the "input" is fixed by
        # construction, which is what the probe assumes.
        self.shm_cov.reset_edge_map()
        env = dict(os.environ, __AFL_SHM_ID=str(shm.shm_id),
                   AFL_MAP_SIZE=str(shm.num_entries))
        subprocess.run([target, "40"], env=env, capture_output=True)
        return 0, ""

try:
    Probe()._report_edge_id_stability(b"seed")
finally:
    shm.cleanup()
"""


def _run_probe(target: str, disable_aslr: bool, relative: bool = False) -> str:
    """Run the probe against *target*.

    ``relative`` is what the fuzzer itself now sets when ASLR survives
    startup, and what an ASAN user sets by hand; the shim reads it in the
    target process and hashes load-base-relative return addresses.
    """
    env = {**os.environ, "DISABLE_ASLR": "1" if disable_aslr else "0"}
    if relative:
        env["FUZZER_KEEP_ASLR"] = "1"
    else:
        env.pop("FUZZER_KEEP_ASLR", None)
    r = subprocess.run(
        [sys.executable, "-c", _PROBE_LAUNCHER.format(src=_SRC, target=target)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    if "NO_ASLR_AVAILABLE" in r.stdout:
        pytest.skip("ASLR is off system-wide (randomize_va_space=0); F1 cannot be shown")
    return r.stdout


class TestStabilityProbe:
    """_report_edge_id_stability: the half that measures rather than infers."""

    @needs_cc
    @linux_only
    def test_fires_on_real_id_drift(self, drivers):
        """ASLR on, shim in raw mode: the regime a target built before
        base-relative mode is stuck in no matter what the fuzzer sets."""
        out = _run_probe(drivers[8], disable_aslr=False)
        assert "WARNING" in out
        assert "Jaccard" in out
        assert "predates" in out, "the cause must be named, not just the symptom"

    @needs_cc
    @linux_only
    def test_relative_mode_is_exec_stable_under_aslr(self, drivers):
        """The fix, measured through the fuzzer's own probe rather than a
        standalone harness: same binary, same ASLR, variable set."""
        out = _run_probe(drivers[8], disable_aslr=False, relative=True)
        assert "WARNING" not in out
        assert "reproduced exactly" in out

    @needs_cc
    @linux_only
    def test_silent_when_ids_reproduce(self, drivers):
        """The default production path: ASLR disabled, raw addresses already
        stable. The probe must not cry wolf there."""
        out = _run_probe(drivers[8], disable_aslr=True)
        assert "WARNING" not in out
        assert "reproduced exactly" in out

    @needs_cc
    @linux_only
    def test_context_free_build_is_stable_under_aslr(self, drivers):
        """The control that makes the first test mean what it says: with the
        context term gone, ASLR alone moves nothing."""
        out = _run_probe(drivers[0], disable_aslr=False)
        assert "WARNING" not in out


class TestProbeAbstains:
    """Cases where the measurement cannot be taken. Silence, not a guess."""

    def test_no_seed(self, capsys):
        _Stub(shm=object())._report_edge_id_stability(None)
        assert capsys.readouterr().out == ""

    def test_no_shm_coverage(self, capsys):
        _Stub(shm=None)._report_edge_id_stability(b"seed")
        assert capsys.readouterr().out == ""

    def test_unrunnable_seed(self, capsys):
        f = _Stub(shm=ShmCoverage(size=1024))
        try:
            f._run_target = lambda data: (_ for _ in ()).throw(OSError("boom"))
            f._report_edge_id_stability(b"seed")
        finally:
            f.shm_cov.cleanup()
        assert capsys.readouterr().out == ""

    def test_drops_invalidate_the_measurement(self, capsys, monkeypatch):
        """A full map discards edges by arrival order, so set divergence is
        not evidence of drift -- the same reason _calibrate_seed_stability
        abstains rather than masking a subset."""
        shm = ShmCoverage(size=1024)
        try:
            f = _Stub(shm=shm)
            f._run_target = lambda data: (0, "")
            monkeypatch.setattr(type(shm), "dropped_edges_delta", lambda self: 3)
            monkeypatch.setattr(type(shm), "get_edge_ids", lambda self: {1, 2, 3})
            monkeypatch.setattr(type(shm), "read_path_hash", lambda self: 7)
            f._report_edge_id_stability(b"seed")
        finally:
            shm.cleanup()
        out = capsys.readouterr().out
        assert "not measured" in out
        assert "dropped" in out


class TestSharedMeasurement:
    """_repeat_edge_sets is shared with _calibrate_seed_stability."""

    def test_drains_the_backlog_before_measuring(self, monkeypatch):
        """The drop counter is cumulative for the segment: without the
        discarding first read, one drop early in a campaign would veto every
        later measurement for the rest of the session."""
        shm = ShmCoverage(size=1024)
        try:
            f = _Stub(shm=shm)
            f._run_target = lambda data: (0, "")
            deltas = iter([999, 0, 0, 0])  # backlog, then three clean runs
            monkeypatch.setattr(type(shm), "dropped_edges_delta", lambda self: next(deltas))
            monkeypatch.setattr(type(shm), "get_edge_ids", lambda self: {1, 2})
            monkeypatch.setattr(type(shm), "read_path_hash", lambda self: 7)
            _sets, _hashes, dropped = f._repeat_edge_sets(b"seed", 3)
            assert dropped == 0
        finally:
            shm.cleanup()

    def test_too_few_runs_is_not_a_measurement(self):
        f = _Stub(shm=ShmCoverage(size=1024))
        try:
            assert f._repeat_edge_sets(b"seed", 1) is None
        finally:
            f.shm_cov.cleanup()
