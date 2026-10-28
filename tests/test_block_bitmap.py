"""Block coverage bitmap: guard sites hit over guard sites instrumented.

The shim sets bit ``i`` of ``BlockBitmapShm`` the first time global guard
index ``i`` fires, and writes the number of guard sites it initialised into
the header. The bitmap is cumulative: Python never clears it, only counts.

    [u32 head = 4 + payload][u32 total guards][payload bits ...]
"""

from __future__ import annotations

import ctypes
import subprocess
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.adapters.shm import BlockBitmapShm, ShmCoverage
from fuzzer_tool.core.elf import parse_sancov_guard_count
from fuzzer_tool.services.stats import StatsReporter
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

MAP_ENTRIES = 8192
BITMAP_ENV = "__AFL_BLOCK_BITMAP_ID"

_SRC = """
static volatile int sink;
/* noinline arms: an if/else of plain stores folds into one select block. */
__attribute__((noinline)) static void high(int a) { sink = a * 2; }
__attribute__((noinline)) static void low(int a) { sink = a; }
__attribute__((noinline)) static void deep(int a) {
    if (a > '3') high(a);
    else low(a);
}
int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] == 'T') deep(argv[1][1]);
    return 0;
}
"""


# ── segment layout ───────────────────────────────────────────────────


def test_layout_head_counts_total_slot_and_payload():
    b = BlockBitmapShm(num_bits=100)
    try:
        payload = (100 + 7) // 8
        assert ctypes.c_uint32.from_address(b._ptr).value == 4 + payload
        assert b.total() == 0
        assert b.covered() == 0
    finally:
        b.cleanup()


def test_covered_counts_set_bits():
    b = BlockBitmapShm(num_bits=64)
    try:
        ctypes.c_uint8.from_address(b._ptr + 8).value = 0b1011
        ctypes.c_uint8.from_address(b._ptr + 15).value = 0x80
        assert b.covered() == bin(0b1011).count("1") + 1
        assert b.covered() == 4, "reading must not clear the bitmap"
    finally:
        b.cleanup()


# ── live shim ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def guard_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("blocks"), _SRC, "-fsanitize-coverage=trace-pc-guard")


def _run(exe, bmp: BlockBitmapShm, *argv: str, env_id: str | None = None) -> int:
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES))
        env[BITMAP_ENV] = env_id or bmp.env_id
        return subprocess.run([str(exe), *argv], env=env, timeout=30).returncode
    finally:
        shm.cleanup()


@requires_clang
def test_total_is_elf_guard_count(guard_exe):
    b = BlockBitmapShm(num_bits=1 << 12)
    try:
        assert _run(guard_exe, b, "X") == 0
        assert b.total() == parse_sancov_guard_count(str(guard_exe))
    finally:
        b.cleanup()


@requires_clang
def test_branch_input_covers_more_blocks(guard_exe):
    flat, deep = BlockBitmapShm(num_bits=1 << 12), BlockBitmapShm(num_bits=1 << 12)
    try:
        _run(guard_exe, flat, "X")
        _run(guard_exe, deep, "T9")
        assert 0 < flat.covered() < deep.covered()
    finally:
        flat.cleanup()
        deep.cleanup()


@requires_clang
def test_bitmap_is_cumulative_and_idempotent(guard_exe):
    """Union across runs; re-running an input adds nothing."""
    b = BlockBitmapShm(num_bits=1 << 12)
    try:
        _run(guard_exe, b, "T9")
        once = b.covered()
        _run(guard_exe, b, "T9")
        assert b.covered() == once
        _run(guard_exe, b, "T1")
        assert b.covered() > once
    finally:
        b.cleanup()


@requires_clang
def test_falsify_not_every_site_is_marked(guard_exe):
    """A shim marking every initialised guard would pass the tests above."""
    b = BlockBitmapShm(num_bits=1 << 12)
    try:
        _run(guard_exe, b, "X")
        assert b.covered() < b.total()
    finally:
        b.cleanup()


@requires_clang
def test_adversarial_undersized_bitmap_stays_in_bounds(guard_exe):
    """Guard indices past the payload are dropped, never written."""
    b = BlockBitmapShm(num_bits=8)
    try:
        assert _run(guard_exe, b, "T9") == 0
        assert b.covered() <= 8
        assert b.total() == parse_sancov_guard_count(str(guard_exe))
    finally:
        b.cleanup()


@requires_clang
def test_adversarial_lying_header_is_refused(guard_exe):
    """A head larger than the segment must not become a write bound."""
    b = BlockBitmapShm(num_bits=64)
    try:
        ctypes.c_uint32.from_address(b._ptr).value = 1 << 20
        assert _run(guard_exe, b, "T9") == 0
        assert b.total() == 0
        assert b.covered() == 0
    finally:
        b.cleanup()


# ── stats line ───────────────────────────────────────────────────────


def _cov_fuzzer(block_bitmap) -> MagicMock:
    f = MagicMock()
    f.multi_targets = None
    f.shm_cov.read_dropped_edges.return_value = 0
    f.shm_cov.cumulative_edges = 7
    f._edge_tracker.good_turing_estimate.return_value = {
        "n": 7,
        "estimated_undiscovered": 3,
        "saturation": 0.7,
    }
    f._block_bitmap = block_bitmap
    return f


def _bare_reporter() -> StatsReporter:
    return StatsReporter.__new__(StatsReporter)


def test_stats_line_shows_blocks():
    bb = MagicMock()
    bb.covered.return_value = 5
    bb.total.return_value = 40
    line = _bare_reporter()._print_stats_cov_str(_cov_fuzzer(bb))
    assert line.endswith(" blocks: 5/40")


@pytest.mark.parametrize("total", [0, None])
def test_stats_line_omits_blocks_without_guards(total):
    """No bitmap, or a shim that saw no guard sites: no blocks field."""
    bb = None
    if total is not None:
        bb = MagicMock()
        bb.total.return_value = total
    line = _bare_reporter()._print_stats_cov_str(_cov_fuzzer(bb))
    assert "blocks:" not in line
