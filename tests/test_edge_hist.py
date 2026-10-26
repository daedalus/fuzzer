"""``--edge-hist``: seeds-per-edge-count histogram and edges/seed ratio.

Both ``fuzz`` and ``minimize -c`` print it behind the flag. Bins are powers
of two (0, 1, 2-3, 4-7, ...): per-seed edge counts span orders of magnitude.

    counts [0, 1, 3, 4]  ->  bin 0: 1 seed, bin 1: 1, bin 2-3: 1, bin 4-7: 1
    ratio = unique edges / seeds
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.edge_hist import (
    BAR_WIDTH,
    bin_counts,
    bin_label,
    fuzz_hist,
    hist_report,
)
from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.services.minimize import ExecMode, minimize_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# a={1,10,11}, b={1,20,21}, c={1,10,11,30}; empty input reaches edge 1 only.
TARGET_SRC = r"""
#include <stddef.h>
#include <stdint.h>
void __sanitizer_cov_trace_pc_guard(uint32_t *guard);
static void fire(uint32_t g) { __sanitizer_cov_trace_pc_guard(&g); }
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    fire(1);
    if (!len) return 0;
    if (buf[0] == 'a')      { fire(10); fire(11); }
    else if (buf[0] == 'b') { fire(20); fire(21); }
    else if (buf[0] == 'c') { fire(10); fire(11); fire(30); }
    return 0;
}
"""


def _rows(report: str) -> dict[str, int]:
    """Histogram rows as {bin label: seeds}."""
    rows = {}
    for line in report.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit() and (parts[0] == "0" or parts[0][0].isdigit()):
            rows[parts[0]] = int(parts[1])
    return rows


class TestBins:
    def test_boundaries(self):
        got = bin_counts([0, 1, 2, 3, 4, 7, 8, 15, 16])
        assert got.tolist() == [1, 1, 2, 2, 2, 1]

    def test_matches_bit_length_oracle(self):
        """Independent oracle: bin index of n is n.bit_length()."""
        vals = [0, 1, 2, 3, 5, 8, 100, 1023, 1024, 65535, 65536]
        want = np.zeros(max(vals).bit_length() + 1, dtype=np.int64)
        for v in vals:
            want[v.bit_length()] += 1
        assert bin_counts(vals).tolist() == want.tolist()

    def test_control_oracle_against_itself(self):
        """Same input twice -> same bins; a comparison that can fail."""
        vals = [3, 9, 9, 200]
        assert bin_counts(vals).tolist() == bin_counts(list(vals)).tolist()

    def test_labels(self):
        assert [bin_label(i) for i in range(5)] == ["0", "1", "2-3", "4-7", "8-15"]

    def test_empty(self):
        assert bin_counts([]).size == 0

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            bin_counts([3, -1])

    def test_huge_count(self):
        got = bin_counts([2**40])
        assert got.size == 42
        assert got[41] == 1


class TestReport:
    def test_ratio_and_rows(self):
        out = hist_report([3, 3, 4, 1], unique_edges=6, title="t")
        assert "Edges/seed:" in out
        assert "1.50" in out  # 6 edges / 4 seeds
        assert _rows(out) == {"1": 1, "2-3": 2, "4-7": 1}

    def test_empty_bins_between_are_listed(self):
        out = hist_report([1, 100], unique_edges=101, title="t")
        rows = _rows(out)
        assert rows["2-3"] == 0
        assert rows["32-63"] == 0
        assert rows["64-127"] == 1

    def test_no_seeds(self):
        out = hist_report([], unique_edges=0, title="t")
        assert "no seeds" in out.lower()

    def test_all_zero_edges_no_div_error(self):
        """Coverage blackout: every seed has 0 edges; ratio is 0, not a crash."""
        out = hist_report([0, 0, 0], unique_edges=0, title="t")
        assert _rows(out) == {"0": 3}
        assert "0.00" in out

    def test_bar_scales_to_width(self):
        out = hist_report([1] * 50 + [2] * 5, unique_edges=2, title="t")
        bars = {ln.split()[0]: ln.count("#") for ln in out.splitlines() if "#" in ln}
        assert bars["1"] == BAR_WIDTH
        assert bars["2-3"] == round(5 / 50 * BAR_WIDTH)

    def test_tiny_bin_still_visible(self):
        """1 seed vs 10_000 must not round to a zero-width bar."""
        out = hist_report([1] * 10_000 + [2], unique_edges=2, title="t")
        bars = {ln.split()[0]: ln.count("#") for ln in out.splitlines() if "#" in ln}
        assert bars["2-3"] == 1

    def test_title_and_population_note(self):
        out = hist_report([1, 2], unique_edges=2, title="fuzz", population=40)
        assert "fuzz" in out.splitlines()[1]
        assert "of 40" in out

    def test_summary_stats(self):
        out = hist_report([2, 4, 6], unique_edges=7, title="t")
        assert "mean 4.0" in out
        assert "median 4" in out
        assert "max 6" in out


class TestFuzzHist:
    def _fuzzer(self, seed_edges, corpus=()):
        return SimpleNamespace(
            _edge_tracker=SimpleNamespace(seed_edges=seed_edges), corpus=list(corpus)
        )

    def test_counts_from_tracker(self):
        f = self._fuzzer({"s1": {1, 2, 3}, "s2": {3, 4}, "s3": set()})
        out = fuzz_hist(f)
        assert _rows(out) == {"0": 1, "1": 0, "2-3": 2}
        assert "Unique edges:" in out
        assert "4" in out.split("Unique edges:")[1].splitlines()[0]  # union {1,2,3,4}

    def test_shared_edges_not_double_counted(self):
        f = self._fuzzer({"a": {1, 2}, "b": {1, 2}})
        line = next(ln for ln in fuzz_hist(f).splitlines() if "Edges/seed" in ln)
        assert "1.00" in line  # 2 unique / 2 seeds

    def test_untracked_corpus_is_noted(self):
        f = self._fuzzer({"a": {1}}, corpus=["a", "b", "c"])
        assert "of 3" in fuzz_hist(f)

    def test_no_tracker_data(self):
        assert "no seeds" in fuzz_hist(self._fuzzer({})).lower()


class TestFlag:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        """Run the real parser in main(); capture the Namespace each cmd gets."""
        self.seen: list[argparse.Namespace] = []
        for name in ("cmd_minimize", "cmd_fuzz"):
            monkeypatch.setattr(commands, name, lambda a: self.seen.append(a) or 0)
        self.monkeypatch = monkeypatch

    def _args(self, *argv: str) -> argparse.Namespace:
        self.monkeypatch.setattr(sys, "argv", ["fuzzer-tool", *argv])
        assert commands.main() == 0
        return self.seen[-1]

    @pytest.mark.parametrize("cmd", [("minimize", "t", "-d", "c"), ("fuzz", "t")])
    def test_gate_default_off_and_on(self, cmd):
        assert self._args(*cmd).edge_hist is False
        assert self._args(*cmd, "--edge-hist").edge_hist is True


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


def _corpus(tmp_path: Path, seeds: dict[str, bytes]) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    for name, data in seeds.items():
        (d / name).write_bytes(data)
    return d


class TestMinimize:
    SEEDS = {"a.bin": b"a", "b.bin": b"b", "c.bin": b"c", "empty.bin": b""}

    def _run(self, tmp_path, so, flag):
        corpus = _corpus(tmp_path, self.SEEDS)
        minimize_corpus(
            so,
            str(corpus),
            use_coverage=True,
            exec_mode=ExecMode.DIRECT_LITE,
            inprocess_func="fuzz_test",
            output_dir=str(tmp_path / "out"),
            edge_hist=flag,
        )

    def test_off_prints_nothing(self, tmp_path, build_so, capsys):
        self._run(tmp_path, build_so, Switch.OFF)
        assert "Edge Histogram" not in capsys.readouterr().out

    def test_on_reports_replayed_and_kept(self, tmp_path, build_so, capsys):
        self._run(tmp_path, build_so, Switch.ON)
        out = capsys.readouterr().out

        replayed, kept = out.split("Edge Histogram")[1:3]
        # Replayed: empty=1 edge, a/b=3, c=4; 6 unique over 4 seeds.
        assert _rows(replayed) == {"1": 1, "2-3": 2, "4-7": 1}
        assert "1.50" in replayed
        # Kept: set-cover keeps b and c; same 6 edges over 2 seeds.
        assert _rows(kept) == {"2-3": 1, "4-7": 1}
        assert "3.00" in kept
