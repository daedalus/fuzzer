"""``--cmplog-records shm``: collector, CLI, Fuzzer and bench wiring.

Shim side: tests/test_cmplog_record_shm.py. Here a real CmplogCollector
drives the compiled target through both channels; the pairs and conds it
ingests must match.
"""

from __future__ import annotations

import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.adapters.cmplog_rec_shm import (
    CMPLOG_REC_ENV,
    INT_SITES,
    RecDrops,
    site_slots_for,
)
from fuzzer_tool.cli import commands
from fuzzer_tool.core.cmplog import CmplogCollector, Preload, RecordChannel
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.stats import StatsReporter
from tests.conftest import requires_clang
from tests.test_cmplog_record_shm import _TARGET_C, AFL_SHIM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS  # noqa: E402

_NO_ASLR = ["setarch", platform.machine(), "-R"]


@pytest.fixture(scope="module")
def target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmplog_channel")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "clang",
            "-O2",
            "-fno-builtin",
            "-fsanitize-coverage=trace-pc-guard,trace-cmp,trace-div",
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
        pytest.skip(f"target did not build: {proc.stderr[-400:]}")
    return exe


def _collect(exe: Path, tmp_path: Path, channel: RecordChannel) -> CmplogCollector:
    """One subprocess exec through a collector on *channel*, then a drain."""
    c = CmplogCollector(
        workdir=str(tmp_path / channel.value), preload=Preload.NONE, record_channel=channel
    )
    (tmp_path / channel.value).mkdir()
    # setup_env needs a shim path (start() sets it); Preload.NONE keeps it off LD_PRELOAD.
    c._shim_path = str(tmp_path / "unused_shim.so")
    try:
        env = c.setup_env(dict(os.environ))
        run = subprocess.run(
            [*_NO_ASLR, str(exe), "0x1234", "MAGICVAX"], env=env, capture_output=True, timeout=60
        )
        assert run.returncode == 0, run.stderr
        c.collect_tokens()
    except BaseException:
        c.stop()
        raise
    return c


@requires_clang
class TestCollector:
    def test_text_channel_sets_no_segment(self, target, tmp_path):
        c = CmplogCollector(workdir=str(tmp_path), preload=Preload.NONE)
        c._shim_path = str(tmp_path / "unused_shim.so")
        try:
            assert CMPLOG_REC_ENV not in c.setup_env({})
        finally:
            c.stop()

    def test_shm_matches_text(self, target, tmp_path):
        text = _collect(target, tmp_path, RecordChannel.TEXT)
        shm = _collect(target, tmp_path, RecordChannel.SHM)
        try:
            assert shm.records_shm is not None and shm.records_shm.attached
            assert set(shm.pairs) == set(text.pairs) and shm.pairs
            assert set(shm.tokens) == set(text.tokens)
            key = lambda cs: {(x.base.op_a, x.base.op_b, x.base.pc) for x in cs}  # noqa: E731
            assert key(shm.last_conds) == key(text.last_conds)
        finally:
            text.stop()
            shm.stop()

    def test_div_pairs_still_arrive(self, target, tmp_path):
        """DIV records stay on text and must still reach the pool."""
        shm = _collect(target, tmp_path, RecordChannel.SHM)
        text = _collect(target, tmp_path, RecordChannel.TEXT)
        try:
            div = set(text.pairs) - {(c.base.op_a, c.base.op_b) for c in text.last_conds}
            assert div and div <= set(shm.pairs)
        finally:
            text.stop()
            shm.stop()

    def test_stop_releases_segment(self, target, tmp_path):
        shm = _collect(target, tmp_path, RecordChannel.SHM)
        shm.stop()
        assert shm.records_shm is None

    def test_env_restored_after_run(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CMPLOG_REC_ENV, raising=False)
        c = CmplogCollector(
            workdir=str(tmp_path), preload=Preload.NONE, record_channel=RecordChannel.SHM
        )
        c._shim_path = str(tmp_path / "unused_shim.so")
        try:
            c.setup_env_for_run()
            assert os.environ[CMPLOG_REC_ENV] == c.records_shm.env_id
            c.restore_env()
            assert CMPLOG_REC_ENV not in os.environ
        finally:
            c.stop()


# ── CLI -> Fuzzer -> collector ──────────────────────────────────────────


def _fuzz_kwargs(monkeypatch, tmp_path, extra: list[str]) -> dict:
    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    monkeypatch.setattr(
        sys, "argv", ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    )
    commands.main()
    return captured


def test_flag_reaches_fuzzer(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--cmplog-records", "shm"])
    assert kw["cmplog_records"] is RecordChannel.SHM


def test_falsify_default_is_text(monkeypatch, tmp_path):
    assert _fuzz_kwargs(monkeypatch, tmp_path, [])["cmplog_records"] is RecordChannel.TEXT


def test_adversarial_unknown_channel_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _fuzz_kwargs(monkeypatch, tmp_path, ["--cmplog-records", "pipe"])


def test_fuzzer_hands_channel_to_collector():
    src = inspect.getsource(Fuzzer.__init__)
    assert "record_channel=cmplog_records" in src
    assert "record_call_sites=" in src


# ── Site table sized per target ─────────────────────────────────────────


def _shm_collector(tmp_path, **kw) -> CmplogCollector:
    c = CmplogCollector(
        workdir=str(tmp_path), preload=Preload.NONE, record_channel=RecordChannel.SHM, **kw
    )
    c._shim_path = str(tmp_path / "unused_shim.so")
    return c


def test_site_table_sized_from_call_sites(tmp_path):
    c = _shm_collector(tmp_path, record_call_sites=132_959)
    try:
        c.setup_env({})
        assert c.records_shm.int_sites == site_slots_for(132_959)
    finally:
        c.stop()


def test_falsify_unknown_call_sites_take_default(tmp_path):
    c = _shm_collector(tmp_path)
    try:
        c.setup_env({})
        assert c.records_shm.int_sites == INT_SITES
    finally:
        c.stop()


# ── Stats + bench ───────────────────────────────────────────────────────


def _stats(dropped: int, drops: RecDrops | None = None) -> str:
    cmplog = SimpleNamespace(
        tokens=[],
        pairs=[],
        records_dropped=dropped,
        records_drops=drops or RecDrops(0, 0, dropped, 0),
        get_eviction_stats=lambda: (0, 0),
        total_comparisons=lambda: (0, 0),
    )
    return StatsReporter._print_stats_cmplog_str(None, SimpleNamespace(_cmplog=cmplog))


def test_stats_show_table_drops():
    assert "rdrop:7" in _stats(7)


def test_stats_break_drops_down_by_cause():
    assert "rdrop:10(s1/h2/r3/b4)" in _stats(10, RecDrops(1, 2, 3, 4))


def test_stats_hide_zero_drops():
    assert "rdrop" not in _stats(0)


def test_arm_and_control():
    assert ARMS["cmp-rec-shm"] == ["--cmplog-records", "shm"]
    assert ARM_BASELINES["cmp-rec-shm"] == "baseline"
    assert ARMS["cmp-rec-shm-ctl"] == ARMS["baseline"]
    assert ARM_BASELINES["cmp-rec-shm-ctl"] == "baseline"
