"""Cmplog record stream paused on rounds whose records are never parsed.

The collector drains the record stream every 20th round once the pair pool
saturates, but the shim formatted and wrote records on every execution. On
ffmpeg that is ~400k records per run: the target ran 3.1x slower (78 -> 25
execs/s, ASAN build) to emit text that the FIFO drain then dropped.

``__cmplog_pause(1)`` stops record emission; the per-execution counts and
site channels keep running. ``FuzzRound`` pauses records on every round that
will not collect, so the one round that does sees only its own records.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.cmplog import CMPLOG_MAX_LINES_PER_READ, CmplogCollector, CmplogRecords
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Phase 1 runs paused, phase 2 live. Operands come from argv so nothing folds.
_TARGET_C = r"""
#include <stdlib.h>
#include <string.h>
void __cmplog_pause(int paused);
void __tracecmp_flush(void);
static int phase(const char *s, const char *magic, unsigned long v, unsigned long k) {
    volatile int r = memcmp(s, magic, 8);
    volatile int c = (v == k);
    return r + c;
}
int main(int argc, char **argv) {
    if (argc < 3) return 1;
    unsigned long v = strtoul(argv[2], NULL, 0);
    __cmplog_pause(1);
    phase(argv[1], "PAUSEDAA", v, 0x51525354UL);
    __tracecmp_flush();
    __cmplog_pause(0);
    phase(argv[1], "LIVEBBBB", v, 0x61626364UL);
    __tracecmp_flush();
    return 0;
}
"""


@pytest.fixture(scope="module")
def pause_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmplog_pause")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "clang",
            "-O0",
            "-fsanitize-coverage=trace-pc-guard,trace-cmp",
            "-D__AFL_CMPLOG=1",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(exe),
            str(src),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"pause target did not build: {proc.stderr[-400:]}")
    return exe


def _run(exe: Path, tmp_path: Path) -> tuple[str, str]:
    records = tmp_path / "records.cmplog"
    counts = tmp_path / "counts.txt"
    env = dict(os.environ, _CMPLOG_OUT=str(records), _CMPLOG_COUNTS=str(counts))
    run = subprocess.run(
        [str(exe), "XXXXXXXX", "0x1234"], capture_output=True, text=True, env=env, timeout=60
    )
    assert run.returncode == 0, run.stderr
    return records.read_text(), counts.read_text()


@requires_clang
class TestShimPause:
    def test_paused_records_are_not_written(self, pause_target, tmp_path):
        records, _ = _run(pause_target, tmp_path)
        assert b"PAUSEDAA".hex() not in records
        assert "54535251" not in records  # 0x51525354, little-endian hex

    def test_live_records_are_written(self, pause_target, tmp_path):
        """Falsification: an always-silent shim would pass the test above."""
        records, _ = _run(pause_target, tmp_path)
        assert b"LIVEBBBB".hex() in records
        assert "64636261" in records

    def test_pause_keeps_counts_channel(self, pause_target, tmp_path):
        """Adversarial: per-exec counters must not go quiet with the records."""
        _, counts = _run(pause_target, tmp_path)
        memcmp_fired = sum(
            int(p[2])
            for p in (ln.split() for ln in counts.splitlines())
            if p[:2] == ["CNT", "memcmp"]
        )
        assert memcmp_fired >= 2


# ── FIFO path read cap ───────────────────────────────────────────────


class _FakeFifo:
    def __init__(self, data: bytes):
        self._data = data

    def drain(self) -> bytes:
        data, self._data = self._data, b""
        return data


def _fifo_collector(n_lines: int) -> CmplogCollector:
    c = CmplogCollector()
    c.fifo_sink = True
    hexes = [(b"B%07d" % i).hex().encode() for i in range(n_lines)]
    body = b"".join(b"CMP %s %s 1 8\n" % (h, h) for h in hexes)
    c._fifo = _FakeFifo(body)
    return c


class TestFifoReadCap:
    def test_fifo_drain_capped_like_file_path(self):
        c = _fifo_collector(CMPLOG_MAX_LINES_PER_READ * 2)
        assert len(c.collect_tokens()) == CMPLOG_MAX_LINES_PER_READ

    def test_fifo_keeps_first_records(self):
        c = _fifo_collector(CMPLOG_MAX_LINES_PER_READ + 5)
        tokens = set(c.collect_tokens())
        assert b"B0000000" in tokens
        assert b"B%07d" % CMPLOG_MAX_LINES_PER_READ not in tokens

    def test_fifo_under_cap_reads_everything(self):
        """Falsification: the cap must not trim a small drain."""
        c = _fifo_collector(37)
        assert len(c.collect_tokens()) == 37


# ── FuzzRound gating ─────────────────────────────────────────────────


class _Cmplog:
    def __init__(self, n_pairs: int):
        self.pairs = [(b"a", b"b")] * n_pairs
        self.collects = 0

    def collect_tokens(self):
        self.collects += 1
        return []


def _fake_fuzzer(n_pairs: int):
    modes: list = []
    f = SimpleNamespace(
        _cmplog=_Cmplog(n_pairs),
        _cmplog_skip_counter=0,
        _cmplog_records=modes.append,
        _rewind_cmplog_shim=lambda: None,
        _add_dict_tokens=lambda tokens: None,
    )
    return f, modes


def _drive(n_pairs: int, rounds: int):
    """Per round: the record mode set before exec, and whether it collected."""
    f, modes = _fake_fuzzer(n_pairs)
    out = []
    for _ in range(rounds):
        rnd = FuzzRound(f, b"S")
        rnd._gate_records()
        before = f._cmplog.collects
        rnd._collect_tokens()
        out.append((modes[-1], f._cmplog.collects > before))
    return out


class TestRoundGating:
    @pytest.mark.parametrize("n_pairs", [0, 499, 500, 1999, 2000, 5000])
    def test_records_on_exactly_when_collected(self, n_pairs):
        for mode, collected in _drive(n_pairs, 60):
            assert (mode is CmplogRecords.ON) == collected

    def test_saturated_pool_pauses_most_rounds(self):
        rounds = _drive(2000, 40)
        on = [i for i, (mode, _) in enumerate(rounds) if mode is CmplogRecords.ON]
        assert on == [19, 39]

    def test_unsaturated_pool_never_pauses(self):
        """Adversarial: the pool-building phase must see every record."""
        assert all(mode is CmplogRecords.ON for mode, _ in _drive(10, 10))

    def test_no_cmplog_no_gate_call(self):
        f, modes = _fake_fuzzer(0)
        f._cmplog = None
        FuzzRound(f, b"S")._gate_records()
        assert modes == []


# ── Fuzzer → shim ────────────────────────────────────────────────────


class _Lib:
    def __init__(self):
        self.calls: list[int] = []

    def __getattr__(self, name):
        if name != "__cmplog_pause":
            raise AttributeError(name)
        return self.calls.append


def _bare_fuzzer(lib) -> Fuzzer:
    f = Fuzzer.__new__(Fuzzer)
    f._cmplog = object()
    f._inprocess_runner = SimpleNamespace(direct_lite=True, _lib=lib)
    f._cmplog_records_mode = CmplogRecords.ON
    return f


class TestFuzzerRecordsSwitch:
    def test_off_then_on_reaches_shim(self):
        lib = _Lib()
        f = _bare_fuzzer(lib)
        f._cmplog_records(CmplogRecords.OFF)
        f._cmplog_records(CmplogRecords.ON)
        assert lib.calls == [1, 0]

    def test_repeated_mode_skips_ctypes_call(self):
        lib = _Lib()
        f = _bare_fuzzer(lib)
        for _ in range(5):
            f._cmplog_records(CmplogRecords.OFF)
        assert lib.calls == [1]

    def test_old_build_without_symbol_is_noop(self):
        """Adversarial: a target built before the pause export must still run."""
        f = _bare_fuzzer(SimpleNamespace())
        f._cmplog_records(CmplogRecords.OFF)
        assert f._cmplog_records_mode is CmplogRecords.ON

    def test_i2s_probe_turns_records_on(self):
        lib = _Lib()
        f = _bare_fuzzer(lib)
        f._cmplog_records(CmplogRecords.OFF)
        f._cmplog = SimpleNamespace(collect_tokens=lambda: [], last_pairs=[])
        f._runner = SimpleNamespace(run_target=lambda data: None)
        f.exec_count = 0
        f._add_dict_tokens = lambda tokens: None
        f._rewind_cmplog_shim = lambda: None
        f._reset_cmplog = lambda: None
        f._i2s_probe(b"x")
        assert lib.calls[-1] == 0
