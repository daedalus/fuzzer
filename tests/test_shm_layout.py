"""The segment layout is a contract between afl_shim.c and adapters/shm.py.

A mismatch is not a degradation. The target writes entries at ITS table
offset and the fuzzer reads them at ITS table offset, so a difference of d
bytes means every edge id read back is a splice of two adjacent entries, and
the fuzzer's own header words are read as edges. The run looks healthy --
plausible ids, a growing corpus -- while every coverage decision is made on
garbage. Nothing downstream can detect that, so it is pinned here instead.

These tests parse the C source rather than compiling it, so they run without
a compiler and cannot be satisfied by a stale object file.
"""

import ctypes
import os
import re
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import (
    SHM_DROP_OFFSET,
    SHM_METADATA_SIZE,
    SIZEOF_ENTRY,
    ShmCoverage,
)
from fuzzer_tool.core.elf import SHM_LAYOUT_CURRENT, detect_shm_layout

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

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


def _shim_define(name: str) -> int:
    """Read a `#define <name> <integer>` out of the shim source."""
    with open(SHIM) as f:
        src = f.read()
    m = re.search(rf"^#define\s+{re.escape(name)}\s+(\d+)", src, re.M)
    assert m, f"{name} not defined in afl_shim.c"
    return int(m.group(1))


class TestLayoutConstantsAgree:
    def test_table_offset_matches_the_shim(self):
        assert _shim_define("SHM_TABLE_OFFSET") == SHM_METADATA_SIZE

    def test_drop_offset_matches_the_shim(self):
        assert _shim_define("SHM_DROP_OFFSET") == SHM_DROP_OFFSET

    def test_layout_generation_matches_the_shim(self):
        assert _shim_define("__AFL_SHM_LAYOUT") == SHM_LAYOUT_CURRENT

    def test_drop_word_sits_between_header_and_table(self):
        """The counter is inside the front region, not overlapping either side."""
        header_end = _shim_define("SHM_HEADER_SIZE")
        assert header_end <= SHM_DROP_OFFSET
        assert SHM_DROP_OFFSET + 4 <= SHM_METADATA_SIZE

    def test_table_stays_eight_byte_aligned(self):
        """An 8-byte entry at a 4-mod-8 offset straddles a boundary on every
        access, and one in eight straddles a cache line — paid on the single
        hottest store in the system. That is what the pad at offset 28 buys.
        """
        assert SHM_METADATA_SIZE % SIZEOF_ENTRY == 0


class TestLayoutMarkerSymbol:
    @needs_cc
    def test_built_target_advertises_the_current_layout(self, tmp_path):
        src = tmp_path / "d.c"
        src.write_text(_DRIVER)
        exe = tmp_path / "d"
        r = subprocess.run(
            ["gcc", "-O1", "-include", SHIM, "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        assert detect_shm_layout(str(exe)) == SHM_LAYOUT_CURRENT

    def test_absent_marker_reads_as_layout_one(self):
        """Every shim built before the dedicated drop counter produced layout
        1 and exported nothing to say so, so absence must not read as
        'current' — that is precisely the binary the check exists to catch.
        """
        assert detect_shm_layout("/bin/true") == 1

    def test_unreadable_path_does_not_claim_a_stale_layout(self):
        """A path that cannot be parsed is unknown, not stale. Reporting 1
        here would abort the run for every non-ELF target.
        """
        assert detect_shm_layout("/nonexistent/binary") == SHM_LAYOUT_CURRENT


class TestSegmentSizing:
    def test_allocation_covers_the_whole_front_region(self):
        cov = ShmCoverage(size=256)
        try:
            entry0 = ctypes.addressof(cov._entries)
            assert entry0 - cov._ptr == SHM_METADATA_SIZE
            # The drop word must be inside the segment and below the table.
            assert cov._ptr + SHM_DROP_OFFSET + 4 <= entry0
        finally:
            cov.cleanup()


class TestStaleLayoutIsRefused:
    """A layout mismatch must abort, not degrade.

    There is no usable fallback: a layout-1 target attached to a layout-2
    segment writes every entry eight bytes below where the fuzzer reads it,
    so each id read back splices the tail of one entry onto the head of the
    next, and the fuzzer's own edge_count header reads as an edge. The result
    is a plausible-looking stream of never-before-seen ids -- a corpus that
    grows on garbage, which is strictly worse than a run reporting nothing.
    """

    def _fuzzer(self):
        import tempfile
        from unittest.mock import patch

        from fuzzer_tool.services.fuzzer import Fuzzer

        d = tempfile.mkdtemp(prefix="layout_")
        with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=f"{d}/corpus",
                crashes_dir=f"{d}/crashes",
                max_len=64,
                timeout=1,
                mutations_per_input=1,
            )
        # The check is gated on there being a segment to get the offsets
        # wrong about; both of these are what the coverage-guided default
        # produces at runtime.
        f.use_coverage = True
        f.shm_cov = object()  # only its presence is checked
        return f

    def test_old_layout_raises(self):
        from unittest.mock import patch

        f = self._fuzzer()
        with patch("fuzzer_tool.core.elf.detect_shm_layout", return_value=1):
            with pytest.raises(RuntimeError, match="SHM layout 1"):
                f._check_shm_layout("/some/stale/target")

    def test_current_layout_passes(self):
        from unittest.mock import patch

        f = self._fuzzer()
        with patch("fuzzer_tool.core.elf.detect_shm_layout", return_value=SHM_LAYOUT_CURRENT):
            f._check_shm_layout("/some/fresh/target")

    def test_no_coverage_means_no_layout_contract(self):
        """--no-coverage never attaches a segment, so the offsets are moot
        and a stale target is a legitimate thing to fuzz blind."""
        from unittest.mock import patch

        f = self._fuzzer()
        f.use_coverage = False
        with patch("fuzzer_tool.core.elf.detect_shm_layout", return_value=1):
            f._check_shm_layout("/some/stale/target")
