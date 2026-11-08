"""pc-table and control-flow tables dumped by the shim ($__AFL_SANCOV_OUT).

``-fsanitize-coverage=pc-table`` hands ``__sanitizer_cov_pcs_init`` one
(PC, flags) pair per instrumented block, index-matched to the guards;
``control-flow`` hands ``__sanitizer_cov_cfs_init`` the CFG. The shim
writes both, base-relative via the load bias, to one sidecar file:

    [u32 magic][u32 kind][u32 first guard][u32 pad][u64 bias][u64 count][count x u64]

The first process to create the file owns it (O_EXCL), so a run of
one-shot executions dumps once.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from fuzzer_tool.adapters.sancov_tables import (
    NO_FIRST,
    PC_FLAG_ENTRY,
    SANCOV_ENV,
    SANCOV_MAGIC,
    SancovKind,
    SancovTables,
    parse_sancov,
)
from fuzzer_tool.adapters.shm import BlockBitmapShm, ShmCoverage
from fuzzer_tool.core.sancov_cfg import CFS_INDIRECT, cfs_functions
from fuzzer_tool.services.stats import StatsReporter
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

MAP_ENTRIES = 8192
BITMAP_ENV = "__AFL_BLOCK_BITMAP_ID"
MODES = "-fsanitize-coverage=trace-pc-guard,pc-table,control-flow"
TARGET_FUNCS = ("main", "deep", "high", "low")

_SRC = """
static volatile int sink;
__attribute__((noinline)) void high(int a) { sink = a * 2; }
__attribute__((noinline)) void low(int a) { sink = a; }
__attribute__((noinline)) void deep(int a) {
    if (a > '3') high(a);
    else low(a);
}
int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] == 'T') deep(argv[1][1]);
    return 0;
}
"""


def _record(kind: int, words: list[int], first: int = NO_FIRST, bias: int = 0) -> bytes:
    """One sidecar record, packed independently of the parser."""
    head = struct.pack("<IIIIQQ", SANCOV_MAGIC, kind, first, 0, bias, len(words))
    return head + struct.pack(f"<{len(words)}Q", *words)


def _symbols(path: Path) -> dict[str, int]:
    """name -> ELF vaddr of defined text symbols (nm, not our ELF parser)."""
    out = subprocess.run(["nm", "--defined-only", str(path)], capture_output=True, text=True)
    syms = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in "tT":
            syms[parts[2]] = int(parts[0], 16)
    return syms


def _run(exe: Path, out: Path, *argv: str, bitmap: BlockBitmapShm | None = None) -> int:
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES))
        env[SANCOV_ENV] = str(out)
        if bitmap is not None:
            env[BITMAP_ENV] = bitmap.env_id
        return subprocess.run([str(exe), *argv], env=env, timeout=30).returncode
    finally:
        shm.cleanup()


def _of_kind(records, kind):
    return [r for r in records if r.kind == kind]


# ── parser ───────────────────────────────────────────────────────────


class TestParse:
    def test_round_trip(self):
        pcs = [0x1010, PC_FLAG_ENTRY, 0x1020, 0]
        cfs = [0x1010, 0x1020, 0, 0, 0x1020, 0, CFS_INDIRECT, 0]
        buf = _record(SancovKind.PCS, pcs, first=5, bias=0x1000) + _record(SancovKind.CFS, cfs)
        recs = parse_sancov(buf)
        assert [r.kind for r in recs] == [SancovKind.PCS, SancovKind.CFS]
        assert recs[0].first == 5
        assert recs[0].bias == 0x1000
        assert recs[0].words.tolist() == pcs
        assert recs[1].words.tolist() == cfs

    def test_entry_vaddrs_subtract_bias(self):
        rec = parse_sancov(_record(SancovKind.PCS, [0x5010, 1, 0x5020, 0], bias=0x5000))[0]
        assert rec.pcs_vaddrs().tolist() == [0x10, 0x20]
        assert rec.pcs_flags().tolist() == [1, 0]

    @pytest.mark.parametrize(
        "tail",
        [
            b"\x00" * 7,  # short header
            struct.pack("<IIIIQQ", 0xDEADBEEF, 1, 0, 0, 0, 0),  # bad magic
            struct.pack("<IIIIQQ", SANCOV_MAGIC, 1, 0, 0, 0, 1 << 60),  # count past EOF
        ],
    )
    def test_adversarial_garbage_tail_is_dropped(self, tail):
        """A torn or hostile tail stops the parse; earlier records survive."""
        good = _record(SancovKind.PCS, [0x10, 1])
        recs = parse_sancov(good + tail)
        assert len(recs) == 1
        assert recs[0].words.tolist() == [0x10, 1]

    def test_adversarial_unknown_kind_is_skipped(self):
        buf = _record(99, [1, 2, 3]) + _record(SancovKind.CFS, [0x10, 0, 0])
        assert [r.kind for r in parse_sancov(buf)] == [SancovKind.CFS]


class TestTables:
    def test_entry_indices_offset_by_first(self, tmp_path):
        t = SancovTables()
        try:
            words = [0x10, PC_FLAG_ENTRY, 0x20, 0, 0x30, PC_FLAG_ENTRY]
            Path(t.path).write_bytes(_record(SancovKind.PCS, words, first=7))
            assert t.entry_indices().tolist() == [7, 9]
        finally:
            t.cleanup()

    def test_falsify_unpaired_region_has_no_indices(self):
        """A pc-table the shim could not pair with a guard region is not
        counted against the block bitmap: its indices would be wrong."""
        t = SancovTables()
        try:
            Path(t.path).write_bytes(_record(SancovKind.PCS, [0x10, PC_FLAG_ENTRY]))
            assert t.entry_indices().size == 0
        finally:
            t.cleanup()

    def test_rereads_after_growth(self):
        t = SancovTables()
        try:
            assert t.entry_indices().size == 0  # not written yet
            Path(t.path).write_bytes(_record(SancovKind.PCS, [0x10, 1], first=0))
            assert t.entry_indices().tolist() == [0]
        finally:
            t.cleanup()

    def test_cleanup_removes_the_dump(self):
        t = SancovTables()
        Path(t.path).write_bytes(b"x")
        t.cleanup()
        assert not Path(t.path).parent.exists()


# ── CFG converter ────────────────────────────────────────────────────


class TestCfsFunctions:
    def test_splits_functions_at_unreached_blocks(self):
        # f: 0x10 -> 0x20 (calls g); g: 0x40 (indirect call). bias 0.
        words = np.array(
            [0x10, 0x20, 0, 0, 0x20, 0, 0x40, 0, 0x40, 0, CFS_INDIRECT, 0], dtype=np.uint64
        )
        names = {0x10: "f", 0x40: "g"}
        fns = cfs_functions(words, 0, names.get)
        assert set(fns) == {"f", "g"}
        f, g = fns["f"], fns["g"]
        assert f.blocks[0x10].is_entry and not f.blocks[0x20].is_entry
        assert f.blocks[0x10].successors == [0x20]
        assert f.blocks[0x20].callees == {"g"}
        assert g.blocks[0x40].indirect_call

    def test_bias_is_subtracted(self):
        words = np.array([0x7010, 0, 0x7020, 0], dtype=np.uint64)
        fns = cfs_functions(words, 0x7000, {0x10: "f"}.get)
        assert set(fns["f"].blocks) == {0x10}
        assert fns["f"].blocks[0x10].callees == {"sub_20"}

    def test_adversarial_truncated_table(self):
        """No terminator: the partial block is dropped, earlier ones kept."""
        words = np.array([0x10, 0, 0, 0x20, 0x30], dtype=np.uint64)
        fns = cfs_functions(words, 0, {}.get)
        assert set(fns) == {"sub_10"}
        assert set(fns["sub_10"].blocks) == {0x10}

    def test_adversarial_empty(self):
        assert cfs_functions(np.zeros(0, dtype=np.uint64), 0, {}.get) == {}


# ── live shim ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def table_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("sancov_tables"), _SRC, MODES)


@requires_clang
class TestLiveShim:
    def test_entries_are_the_target_functions(self, table_exe, tmp_path):
        out = tmp_path / "sancov.bin"
        assert _run(table_exe, out) == 0
        (pcs,) = _of_kind(parse_sancov(out.read_bytes()), SancovKind.PCS)
        syms = _symbols(table_exe)
        entries = pcs.pcs_vaddrs()[(pcs.pcs_flags() & PC_FLAG_ENTRY) != 0]
        assert set(entries.tolist()) == {syms[n] for n in TARGET_FUNCS}

    def test_falsify_no_env_no_dump(self, table_exe, tmp_path):
        shm = ShmCoverage(size=MAP_ENTRIES)
        try:
            env = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES))
            assert subprocess.run([str(table_exe)], env=env, cwd=tmp_path).returncode == 0
        finally:
            shm.cleanup()
        assert list(tmp_path.iterdir()) == []

    def test_adversarial_second_process_does_not_append(self, table_exe, tmp_path):
        out = tmp_path / "sancov.bin"
        _run(table_exe, out)
        once = out.read_bytes()
        _run(table_exe, out, "T9")
        assert out.read_bytes() == once

    def test_entry_bits_follow_the_call_graph(self, table_exe, tmp_path):
        """Function entries hit = functions the input reaches."""
        out = tmp_path / "sancov.bin"
        reached = {"X": {"main"}, "T9": {"main", "deep", "high"}}
        for arg, funcs in reached.items():
            b = BlockBitmapShm(num_bits=1 << 12)
            try:
                out.unlink(missing_ok=True)
                _run(table_exe, out, arg, bitmap=b)
                (pcs,) = _of_kind(parse_sancov(out.read_bytes()), SancovKind.PCS)
                assert pcs.first != NO_FIRST
                idx = pcs.entry_indices()
                assert b.covered_at(idx) == len(funcs)
            finally:
                b.cleanup()

    def test_cfg_matches_the_source(self, table_exe, tmp_path):
        out = tmp_path / "sancov.bin"
        _run(table_exe, out)
        (cfs,) = _of_kind(parse_sancov(out.read_bytes()), SancovKind.CFS)
        syms = _symbols(table_exe)
        by_addr = {a: n for n, a in syms.items()}
        fns = cfs_functions(cfs.words, cfs.bias, by_addr.get)
        assert set(TARGET_FUNCS) <= set(fns)
        for name in TARGET_FUNCS:
            entry = [b for b in fns[name].blocks.values() if b.is_entry]
            assert [b.start for b in entry] == [syms[name]]
        assert {"high", "low"} <= set().union(*(b.callees for b in fns["deep"].blocks.values()))
        assert "deep" in set().union(*(b.callees for b in fns["main"].blocks.values()))

    def test_adversarial_dlopen_bias(self, tmp_path):
        """A randomly-loaded .so still dumps ELF vaddrs (load bias removed)."""
        so = _build(tmp_path, _SRC.replace("int main", "int run"), MODES, shared=True)
        out = tmp_path / "sancov.bin"
        script = textwrap.dedent(
            """
            import ctypes, sys
            ctypes.CDLL(sys.argv[1])
            """
        )
        env = _clean_env()
        env[SANCOV_ENV] = str(out)
        r = subprocess.run([sys.executable, "-c", script, str(so)], env=env, timeout=60)
        assert r.returncode == 0
        (pcs,) = _of_kind(parse_sancov(out.read_bytes()), SancovKind.PCS)
        syms = _symbols(so)
        entries = pcs.pcs_vaddrs()[(pcs.pcs_flags() & PC_FLAG_ENTRY) != 0]
        assert {syms[n] for n in ("run", "deep", "high", "low")} == set(entries.tolist())


# ── wiring ───────────────────────────────────────────────────────────


def _cov_fuzzer(block_bitmap, tables) -> MagicMock:
    f = MagicMock()
    f._block_bitmap = block_bitmap
    f._sancov_tables = tables
    return f


def _bitmap(covered: int, total: int, entries_hit: int) -> MagicMock:
    bb = MagicMock()
    bb.covered.return_value = covered
    bb.total.return_value = total
    bb.covered_at.return_value = entries_hit
    return bb


def test_stats_shows_funcs():
    tables = MagicMock()
    tables.entry_indices.return_value = np.array([0, 3, 9, 12], dtype=np.uint32)
    line = StatsReporter._blocks_str(_cov_fuzzer(_bitmap(5, 40, 2), tables))
    assert line == " blocks: 5/40 funcs: 2/4"


@pytest.mark.parametrize("tables", [None, "empty"])
def test_stats_omits_funcs_without_pc_table(tables):
    if tables == "empty":
        tables = MagicMock()
        tables.entry_indices.return_value = np.zeros(0, dtype=np.uint32)
    line = StatsReporter._blocks_str(_cov_fuzzer(_bitmap(5, 40, 0), tables))
    assert line == " blocks: 5/40"


def test_covered_at_counts_only_requested_bits():
    b = BlockBitmapShm(num_bits=64)
    try:
        ctypes.c_uint8.from_address(b._ptr + 8).value = 0b1010_0101
        assert b.covered_at(np.array([0, 1, 2, 5], dtype=np.uint32)) == 3
        # Adversarial: indices past the payload are ignored, not read.
        assert b.covered_at(np.array([0, 1 << 20], dtype=np.uint32)) == 1
    finally:
        b.cleanup()


def test_fuzzer_exports_the_dump_path(monkeypatch):
    from fuzzer_tool.services.fuzzer import Fuzzer

    monkeypatch.delenv(SANCOV_ENV, raising=False)
    tables = Fuzzer._open_sancov_tables(MagicMock())
    try:
        assert os.environ[SANCOV_ENV] == tables.path
        assert not os.path.exists(tables.path), "the shim creates it (O_EXCL)"
    finally:
        tables.cleanup()
        monkeypatch.delenv(SANCOV_ENV, raising=False)
