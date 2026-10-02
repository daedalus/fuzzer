"""Coverage segment write guard: only the shim may write the edge table.

The target runs attacker-chosen input in the same address space as the
shim's mapping of the coverage segment, so a wild store from a memory bug
could plant or erase edges. With the guard armed the mapping is write-locked
(x86 protection key, WD bit) and the shim unlocks it around its own writes.

Opt-in: `--shm-write-guard {off,pkey,mprotect}` (default off, not part of
--hail-mary) exports `__AFL_WGUARD`; unset, the shim arms nothing.

The pkey path needs PKU hardware. `mprotect` arms the same open/close
brackets with mprotect instead, so every write site is checked on any host:
an unbracketed shim write faults exactly like a stray one.
"""

from __future__ import annotations

import inspect
import os
import signal
import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import _ENTRY_DTYPE, ShmCoverage
from fuzzer_tool.cli import commands
from fuzzer_tool.core.shim_health import WGUARD_ENV, ShimField, WriteGuard, export_wguard
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

MAP_ENTRIES = 1024
STRAY_ID = 0xDEADBEEF

# argv[1]: "guards" fires the uninstrumented callbacks only; "all" adds the
# self-instrumented entry points (map_edge, sfuzz, map_reset), whose inlined
# guard branches add mode-dependent edges; "stray" also writes the table
# directly, the way a wild store in the target would.
_DRIVER = """
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    uint32_t g1 = 7, g2 = 9;
    __sanitizer_cov_trace_pc_guard(&g1);
    __sanitizer_cov_trace_pc_guard(&g2);
    __sanitizer_cov_trace_pc_indir((uintptr_t)main);
    if (strcmp(argv[1], "guards") != 0) {
        __afl_map_reset();
        __afl_map_edge(0x1101);
        __sfuzz_state(1, 2);
    }

    uint64_t h[16] = {0};
    __afl_shim_health(h, 16);
    printf("%llu\\n", (unsigned long long)h[FIELD]);
    fflush(stdout);

    if (strcmp(argv[1], "stray") == 0)
        __afl_area[3].edge_id = STRAY;
    return 0;
}
""".replace("FIELD", str(int(ShimField.WGUARD))).replace("STRAY", hex(STRAY_ID))


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("wguard"),
        _DRIVER,
        "-fsanitize-coverage=trace-pc-guard",
        "-D__AFL_CTX_SENSITIVE=0",
    )


def _run(exe, cov: ShmCoverage, mode: str | None, arg: str, entries: int = MAP_ENTRIES):
    env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(entries))
    if mode is not None:
        env["__AFL_WGUARD"] = mode
    cov.reset_edge_map()
    return subprocess.run([str(exe), arg], env=env, capture_output=True, text=True, timeout=30)


def _edges(exe, mode: str | None, arg: str = "guards") -> set[int]:
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(exe, cov, mode, arg)
        assert r.returncode == 0, r.stderr
        return cov.get_edge_ids()
    finally:
        cov.cleanup()


def _raw_ids(cov: ShmCoverage) -> set[int]:
    """Every non-zero id in the table, ignoring generation tags."""
    table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
    return {int(e) for e in table["edge_id"] if e}


@requires_clang
def test_control_unguarded_stray_write_lands(target):
    """Control: with the guard off the stray store reaches the table, so the
    adversarial test below can tell a blocked write from a missed one."""
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(target, cov, "0", "stray")
        assert r.returncode == 0, r.stderr
        assert int(r.stdout.split()[0]) == WriteGuard.OFF
        assert STRAY_ID in _raw_ids(cov)
    finally:
        cov.cleanup()


@requires_clang
def test_stray_write_faults_under_guard(target):
    """Adversarial: a store into the table from target code dies on SIGSEGV
    and never reaches the segment."""
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(target, cov, "mprotect", "stray")
        assert int(r.stdout.split()[0]) == WriteGuard.MPROTECT
        assert r.returncode == -signal.SIGSEGV, (r.returncode, r.stderr)
        assert STRAY_ID not in _raw_ids(cov)
    finally:
        cov.cleanup()


@requires_clang
def test_guard_records_same_edges(target):
    """Falsification: the guard neither drops nor alters recorded edges."""
    control = _edges(target, "0")
    assert control == _edges(target, "0")
    assert control and _edges(target, "mprotect") == control


@requires_clang
def test_every_write_site_bracketed(target):
    """map_edge, sfuzz, map_reset and the exit-time distance tail all write
    the segment; an unbracketed one faults under the guard."""
    assert _edges(target, "mprotect", "all")


@requires_clang
def test_drop_counter_written_under_guard(target):
    """Edge case: a 1-entry table overflows, so __afl_note_drop writes the
    header from inside the probe loop."""
    cov = ShmCoverage(size=1)
    try:
        r = _run(target, cov, "mprotect", "all", entries=1)
        assert r.returncode == 0, r.stderr
        assert cov.read_dropped_edges() > 0
    finally:
        cov.cleanup()


@requires_clang
def test_unset_env_is_off(target):
    """Falsification of the default: no knob, no guard."""
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(target, cov, None, "stray")
        assert r.returncode == 0, r.stderr
        assert int(r.stdout.split()[0]) == WriteGuard.OFF
        assert STRAY_ID in _raw_ids(cov)
    finally:
        cov.cleanup()


@requires_clang
def test_unknown_mode_is_off(target):
    """Adversarial: a garbage knob must not arm, disable coverage or crash."""
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(target, cov, "banana", "guards")
        assert int(r.stdout.split()[0]) == WriteGuard.OFF
    finally:
        cov.cleanup()
    assert _edges(target, "banana") == _edges(target, "0")


@requires_clang
def test_pkey_guard_blocks_stray_write(target):
    """pkey mode on PKU hardware; skipped where pkey_alloc is refused."""
    cov = ShmCoverage(size=MAP_ENTRIES)
    try:
        probe = _run(target, cov, "pkey", "guards")
        if int(probe.stdout.split()[0]) != WriteGuard.PKEY:
            pytest.skip("no protection keys on this host")
        r = _run(target, cov, "pkey", "stray")
        assert r.returncode == -signal.SIGSEGV, (r.returncode, r.stderr)
        assert STRAY_ID not in _raw_ids(cov)
    finally:
        cov.cleanup()


# ── CLI / wiring ───────────────────────────────────────────────────────


def _fuzz_kwargs(monkeypatch, tmp_path, extra: list[str]) -> dict:
    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    argv = ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    commands.main()
    return captured


def test_cli_default_off(monkeypatch, tmp_path):
    assert _fuzz_kwargs(monkeypatch, tmp_path, [])["shm_write_guard"] is WriteGuard.OFF


def test_hail_mary_leaves_guard_off(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])
    assert kw["shm_write_guard"] is WriteGuard.OFF


@pytest.mark.parametrize("mode", [WriteGuard.PKEY, WriteGuard.MPROTECT])
def test_cli_mode_reaches_fuzzer(monkeypatch, tmp_path, mode):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--shm-write-guard", mode.name.lower()])
    assert kw["shm_write_guard"] is mode


def test_cli_rejects_unknown_mode(monkeypatch, tmp_path):
    """Adversarial: a typo must fail loudly, not silently run unguarded."""
    with pytest.raises(SystemExit):
        _fuzz_kwargs(monkeypatch, tmp_path, ["--shm-write-guard", "banana"])


def test_export_sets_env_only_when_on(monkeypatch):
    monkeypatch.delenv(WGUARD_ENV, raising=False)
    export_wguard(WriteGuard.OFF)
    assert WGUARD_ENV not in os.environ

    export_wguard(WriteGuard.PKEY)
    assert os.environ[WGUARD_ENV] == "pkey"
    monkeypatch.delenv(WGUARD_ENV)


def test_fuzzer_exports_guard():
    # Fuzzer() needs a live target; the call is checked in source.
    assert "export_wguard(shm_write_guard)" in inspect.getsource(Fuzzer.__init__)
