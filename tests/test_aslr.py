"""Tests for adapters/process.disable_aslr.

The property that matters is not "personality() returned 0" — it is that a
freshly spawned child lands at the same address on every exec.  That is the
invariant afl_shim.c's caller-context edge hashing depends on, since
ShmCoverage._seen_edge_ids compares edge IDs across target processes.

These tests spawn subprocesses rather than mutating the test runner's own
persona, so the outcome does not depend on test ordering.
"""

import os
import subprocess
import sys

import pytest

from fuzzer_tool.adapters import process as proc_mod
from fuzzer_tool.adapters.process import ADDR_NO_RANDOMIZE, disable_aslr

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# Two-level, because personality() governs the NEXT execve, not the current
# mapping: a process that disables ASLR on itself is already loaded at a
# random base.  The launcher stands in for the fuzzer parent (which calls
# disable_aslr in Fuzzer.__init__, before anything spawns) and the probes
# stand in for target processes.  A single-level test would report the
# launcher's own base and pass for the wrong reason.
#
# The launcher first clears ADDR_NO_RANDOMIZE it may have inherited from the
# pytest process (any test that built a Fuzzer set it there); otherwise the
# FUZZER_KEEP_ASLR=1 control's children stop moving and the control fails.
_LAUNCHER = """
import ctypes, ctypes.util, os, subprocess, sys
sys.path.insert(0, {src!r})
from fuzzer_tool.adapters.process import ADDR_NO_RANDOMIZE, disable_aslr
libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
libc.personality.argtypes = [ctypes.c_ulong]
libc.personality(libc.personality(0xFFFFFFFF) & ~ADDR_NO_RANDOMIZE)
ok = disable_aslr()
probe = "print([l.split('-')[0] for l in open('/proc/self/maps') if 'libc.so' in l][0])"
bases = [
    subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.strip()
    for _ in range(5)
]
print(ok, " ".join(bases))
"""

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="personality() is Linux-only"
)

_QUERY_PERSONA = 0xFFFFFFFF


def _personality(persona: int) -> int:
    """Raw personality(2); 0xFFFFFFFF queries without changing it."""
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.personality.argtypes = [ctypes.c_ulong]
    libc.personality.restype = ctypes.c_int
    return libc.personality(persona)


@pytest.fixture
def _restore_persona():
    """This process's persona is inherited by every later child; put it back."""
    saved = _personality(_QUERY_PERSONA)
    yield
    _personality(saved)


def _launch(env: dict | None = None) -> tuple[str, list[str]]:
    """Run the launcher; return (disable_aslr result, child load bases)."""
    r = subprocess.run(
        [sys.executable, "-c", _LAUNCHER.format(src=_SRC)],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    assert r.returncode == 0, r.stderr
    ok, *bases = r.stdout.split()
    return ok, bases


class TestDisableAslr:
    @linux_only
    def test_child_load_base_is_stable_across_execs(self):
        ok, bases = _launch(env={"FUZZER_KEEP_ASLR": "0"})
        assert ok == "True"
        assert len(set(bases)) == 1, f"load base varied across execs: {set(bases)}"

    @linux_only
    def test_aslr_still_randomizes_without_the_call(self):
        """Guards against a false pass: confirm the baseline really does move."""
        ok, bases = _launch(env={"FUZZER_KEEP_ASLR": "1"})
        assert ok == "False"
        assert len(set(bases)) > 1, "ASLR appears disabled system-wide; test is not meaningful"

    @linux_only
    def test_regression_control_ignores_inherited_persona(self, _restore_persona):
        """Any earlier test that built a Fuzzer left ADDR_NO_RANDOMIZE on this
        process; the control's children inherited it and stopped moving."""
        _personality(_personality(_QUERY_PERSONA) | ADDR_NO_RANDOMIZE)

        ok, bases = _launch(env={"FUZZER_KEEP_ASLR": "1"})

        assert ok == "False"
        assert len(set(bases)) > 1, f"control inherited the parent persona: {set(bases)}"

    @linux_only
    def test_persona_flag_is_set_on_this_process(self, monkeypatch, _restore_persona):
        monkeypatch.setattr(proc_mod, "_aslr_disabled", None)
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        assert disable_aslr() is True
        assert _personality(_QUERY_PERSONA) & ADDR_NO_RANDOMIZE

    def test_opt_out_env_var(self, monkeypatch):
        monkeypatch.setattr(proc_mod, "_aslr_disabled", None)
        monkeypatch.setenv("FUZZER_KEEP_ASLR", "1")
        assert disable_aslr() is False

    def test_result_is_cached(self, monkeypatch):
        """Second call must not re-issue the syscall — it returns the cache."""
        monkeypatch.setattr(proc_mod, "_aslr_disabled", False)
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        assert disable_aslr() is False  # cached False wins over a would-be True

    def test_never_raises_when_syscall_unavailable(self, monkeypatch):
        monkeypatch.setattr(proc_mod, "_aslr_disabled", None)
        monkeypatch.delenv("FUZZER_KEEP_ASLR", raising=False)
        monkeypatch.setattr(proc_mod.sys, "platform", "darwin")
        assert disable_aslr() is False
