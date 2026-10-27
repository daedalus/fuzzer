"""``minimize -c --count-aware``: set cover over (edge, hit-count) items.

Binary cover keeps one seed per edge set, so seeds that differ only in how
often an edge ran are dropped, although the fuzzer's virgin map (AFL count
buckets) treats a new bucket as new coverage.

    loop edge 10->10 runs 0, 1, 4, 8 times   (n fire(10) calls = n-1 repeats)
    binary: one edge set {->1, 1->10, 10->10} is shared by 3 seeds -> 1 seed
    bucket items (edge, AFL bucket): repeats 1, 4-7, 8-15 differ -> 3 seeds
                                     (the 1-call seed is dominated by the 2-call one)
    thermo items (edge, count >= 2^k): the 8-repeat seed holds all levels -> 1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.count_items import CountItems, count_items
from fuzzer_tool.services.minimize import ExecMode, _EdgeBook, minimize_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# fire(1) once, fire(10) buf[0] times. The shim edge is (prev, cur), so the loop
# adds a 10->10 edge counted buf[0]-1 times.
TARGET_SRC = r"""
#include <stddef.h>
#include <stdint.h>
void __sanitizer_cov_trace_pc_guard(uint32_t *guard);
static void fire(uint32_t g) { __sanitizer_cov_trace_pc_guard(&g); }
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    fire(1);
    for (unsigned i = 0; len && i < buf[0]; i++) fire(10);
    return 0;
}
"""


def _u32(*v):
    return np.asarray(v, dtype=np.uint32)


class TestCountItems:
    def test_off_returns_ids(self):
        assert count_items(_u32(3, 5), _u32(1, 9), CountItems.OFF).tolist() == [3, 5]

    def test_bucket_one_item_per_edge(self):
        got = count_items(_u32(3, 5), _u32(1, 9), CountItems.BUCKET)
        assert got.size == 2
        same_bucket = count_items(_u32(3), _u32(4), CountItems.BUCKET)
        other = count_items(_u32(3), _u32(7), CountItems.BUCKET)
        assert same_bucket.tolist() == other.tolist()
        assert count_items(_u32(3), _u32(8), CountItems.BUCKET).tolist() != other.tolist()

    def test_thermo_levels(self):
        # counts 1, 2-3, 4-7, 8-15 -> 1, 2, 3, 4 items
        for c, n in ((1, 1), (2, 2), (3, 2), (4, 3), (7, 3), (8, 4), (15, 4), (128, 8), (10**6, 8)):
            assert count_items(_u32(9), _u32(c), CountItems.THERMO).size == n, c

    def test_thermo_nested(self):
        small = set(count_items(_u32(9), _u32(3), CountItems.THERMO).tolist())
        big = set(count_items(_u32(9), _u32(9), CountItems.THERMO).tolist())
        assert small < big

    def test_sorted_unique_uint64(self):
        got = count_items(_u32(7, 2, 9), _u32(5, 40, 1), CountItems.THERMO)
        assert got.dtype == np.uint64
        assert got.tolist() == sorted(set(got.tolist()))

    def test_wide_edge_ids_do_not_collide(self):
        big = 2**32 - 1
        a = count_items(_u32(big), _u32(1), CountItems.BUCKET)
        b = count_items(_u32(big - 1), _u32(1), CountItems.BUCKET)
        assert a.tolist() != b.tolist()

    def test_zero_count_dropped(self):
        assert count_items(_u32(3), _u32(0), CountItems.THERMO).size == 0

    def test_empty(self):
        assert count_items(_u32(), _u32(), CountItems.BUCKET).size == 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            count_items(_u32(1, 2), _u32(1), CountItems.BUCKET)


def test_edge_book_collapses_uint64_twins():
    from fuzzer_tool.services.minimize import Twins

    book = _EdgeBook(Twins.COLLAPSE, dtype=np.uint64)
    a = np.asarray([2**40 + 1, 2**40 + 2], dtype=np.uint64)
    book.add("big", a, 10)
    book.add("small", a.copy(), 4)
    assert list(book.edges) == ["small"]
    assert book.edges["small"].dtype == np.uint64
    assert book.edges["small"].tolist() == a.tolist()


@pytest.fixture
def build_so(tmp_path):
    cc = next((c for c in ("clang", "gcc", "cc") if shutil.which(c)), None)
    if cc is None:
        pytest.skip("no C compiler available")
    src = tmp_path / "t.c"
    src.write_text(TARGET_SRC)
    so = tmp_path / "t.so"
    proc = subprocess.run(
        [cc, "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"target failed to build: {proc.stderr[-400:]}")
    return str(so)


SEEDS = {
    "h1.bin": b"\x01",
    "h2.bin": b"\x02",
    "h5.bin": b"\x05",
    "h9.bin": b"\x09",
    "z.bin": b"\x00",
}


def _kept(tmp_path, so, mode):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, data in SEEDS.items():
        (corpus / name).write_bytes(data)
    minimize_corpus(
        so,
        str(corpus),
        use_coverage=True,
        exec_mode=ExecMode.DIRECT_LITE,
        inprocess_func="fuzz_test",
        output_dir=str(tmp_path / "out"),
        count_items=mode,
    )
    return sorted(p.name for p in (tmp_path / "out").rglob("*") if p.is_file())


class TestMinimize:
    def test_binary_keeps_one_hit_seed(self, tmp_path, build_so):
        assert len(_kept(tmp_path, build_so, CountItems.OFF)) == 1

    def test_bucket_keeps_distinct_buckets(self, tmp_path, build_so):
        assert _kept(tmp_path, build_so, CountItems.BUCKET) == ["h2.bin", "h5.bin", "h9.bin"]

    def test_thermo_keeps_the_dominating_seed(self, tmp_path, build_so):
        assert _kept(tmp_path, build_so, CountItems.THERMO) == ["h9.bin"]


class TestFlag:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        self.seen: list[argparse.Namespace] = []
        monkeypatch.setattr(commands, "cmd_minimize", lambda a: self.seen.append(a) or 0)
        self.monkeypatch = monkeypatch

    def _args(self, *argv):
        self.monkeypatch.setattr(sys, "argv", ["fuzzer-tool", *argv])
        assert commands.main() == 0
        return self.seen[-1]

    def test_default_off_and_values(self):
        assert self._args("minimize", "t", "-d", "c").count_aware is None
        assert (
            self._args("minimize", "t", "-d", "c", "--count-aware", "bucket").count_aware
            == "bucket"
        )
        assert (
            self._args("minimize", "t", "-d", "c", "--count-aware", "thermo").count_aware
            == "thermo"
        )
