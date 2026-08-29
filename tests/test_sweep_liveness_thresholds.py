"""Tests for the item-B liveness calibration mode of the sweep tool.

`tools/sweep_liveness_thresholds.py --synthetic-target` is the run the
handover called "one run left" for four rounds. It was not runnable: the
tool's `--target` argument was accepted and never read, there was no
`--unstable` flag, and both paths drove `FormatLearner` over fabricated
transitions rather than executing anything. The mode under test here drives
the real `LiveBitMaskEstimator` against `gen_synthetic_target.py`'s
known-dead region.

Two layers, deliberately:

- The replay/diff helpers are tested against hand-computable inputs with no
  compiler involved, so the *logic* stays covered on a machine without gcc.
  These would pass even if the estimator were never fed real coverage, which
  is exactly why the second layer exists.
- One gcc-gated end-to-end test actually builds and runs the target and
  asserts the verdict matches ground truth. This is the one that fails if the
  measurement silently stops measuring — the failure mode the synthetic
  target exists to make visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.conftest import requires_gcc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from sweep_liveness_thresholds import (  # noqa: E402
    SYNTH_DEAD_REGION,
    SYNTH_LIVE_REGION,
    SweepConfig,
    _diff_bits,
    _flip_in_region,
    _replay_estimator,
    _synthetic_report,
)


class TestDiffBits:
    """`_diff_bits` must reproduce exactly what `record_coverage_diff` hands
    to `observe()`; if it drifts, the calibration measures a different
    transform than production uses."""

    def test_symmetric_difference_not_union(self):
        # Edge 3 is in both, so it must not appear. An implementation using
        # union instead of XOR would set bit 3 too.
        bits = _diff_bits(frozenset({1, 3}), frozenset({3, 5}), 65536)
        assert bits == (1 << 1) | (1 << 5)

    def test_identical_sets_give_zero(self):
        assert _diff_bits(frozenset({7, 9}), frozenset({7, 9}), 65536) == 0

    def test_edge_ids_fold_modulo_map_bits(self):
        # 8 and 0 alias under an 8-bit map; the estimator tolerates aliasing
        # (it only ever costs a too-eager "not dead yet"), but the folding
        # has to actually happen or observe() raises on out-of-range bits.
        assert _diff_bits(frozenset({8}), frozenset(), 8) == 1


class TestReplayEstimator:
    """Verdict semantics must match `_region_liveness_factor`: DEAD requires
    convergence *and* an empty mask, not merely an empty mask so far."""

    def test_all_zero_sequence_converges_dead_at_threshold(self):
        r = _replay_estimator([0] * 300, switch_after=200, map_bits=65536)
        assert r["verdict"] == "DEAD"
        assert r["converged_at"] == 200
        assert r["mask_bits"] == 0

    def test_short_all_zero_sequence_is_unresolved_not_dead(self):
        """The fail-closed property: a not-yet-converged empty mask is not a
        dead verdict. Without this, any region would read dead on sample 1."""
        r = _replay_estimator([0] * 50, switch_after=200, map_bits=65536)
        assert r["verdict"] == "UNRESOLVED"
        assert r["converged_at"] is None

    def test_any_revealed_edge_makes_it_live(self):
        r = _replay_estimator([0] * 500 + [1 << 4] + [0] * 500, 200, 65536)
        assert r["verdict"] == "LIVE"
        assert r["mask_bits"] == 1

    def test_leading_zero_run_measures_the_cold_live_window(self):
        """The number that says how long a live region can *look* dead. It is
        the whole reason the synthetic target cannot set a lower bound on
        switch_after -- its live region reveals an edge on sample 1."""
        r = _replay_estimator([0, 0, 0, 1 << 2, 0], switch_after=200, map_bits=65536)
        assert r["leading_zero_run"] == 3


class TestFlipInRegion:
    def test_flip_stays_inside_the_region(self):
        import random

        base = bytes(64) * 4
        rng = random.Random(1)
        for _ in range(50):
            out = _flip_in_region(rng, base, SYNTH_DEAD_REGION)
            changed = [i for i, (a, b) in enumerate(zip(base, out, strict=True)) if a != b]
            assert len(changed) == 1
            assert SYNTH_DEAD_REGION[0] <= changed[0] < SYNTH_DEAD_REGION[1]

    def test_regions_do_not_overlap(self):
        assert SYNTH_LIVE_REGION[1] <= SYNTH_DEAD_REGION[0]


@requires_gcc
class TestSyntheticCalibrationEndToEnd:
    """Builds and runs the real target. Slow, and the only layer that can
    catch the calibration silently measuring nothing."""

    @staticmethod
    @pytest.fixture(scope="class")
    def report():
        # Small budget: switch_after grid capped so the run stays short while
        # still exercising convergence at the top of the grid.
        cfg = SweepConfig(
            synthetic_target=True,
            blocks=200,
            fanout=16,
            unstable=0,
            calib_samples=60,
            switch_grid=(10, 25, 50),
        )
        return _synthetic_report(cfg)

    def test_dead_region_never_moves_coverage(self, report):
        assert "dead-region mutations moving coverage: 0/60" in report

    def test_live_region_always_moves_coverage(self, report):
        assert "live-region mutations moving coverage: 60/60" in report

    def test_verdict_is_correct_across_the_grid(self, report):
        assert "CORRECT across the grid" in report
        assert "MISCLASSIFICATION" not in report

    def test_no_dead_verdict_is_given_to_the_live_region(self, report):
        """The false-positive half. Ground truth says the live region is live
        at every threshold; a DEAD column entry for it would mean the
        down-weight suppresses bytes that do drive coverage."""
        rows = [ln for ln in report.splitlines() if ln.startswith("| ") and "DEAD" in ln]
        assert rows, "no verdict rows in report"
        for row in rows:
            # columns: | switch | dead | conv@ | live | mask bits | zero run |
            cols = [c.strip() for c in row.strip("|").split("|")]
            assert cols[1] == "DEAD", f"dead region misclassified: {row}"
            assert cols[3] == "LIVE", f"live region given a dead verdict: {row}"

    def test_calibration_does_not_disable_aslr_process_wide(self, report):
        """personality() is process-global and irreversible, so disabling
        ASLR in *this* process would leave every later test in the same
        pytest run without it -- which makes
        test_synthetic_target.py's unstable-variant checks silently skip
        rather than fail. A suite quietly losing coverage is worse than a
        red test, so the sweep sets it per-child instead."""
        import ctypes
        import ctypes.util

        from fuzzer_tool.adapters.process import ADDR_NO_RANDOMIZE

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.personality.argtypes = [ctypes.c_ulong]
        libc.personality.restype = ctypes.c_int
        assert not (libc.personality(0xFFFFFFFF) & ADDR_NO_RANDOMIZE), (
            "the calibration leaked ADDR_NO_RANDOMIZE into the test process"
        )

    def test_calibration_runs_with_aslr_disabled_like_production(self, report):
        """services/fuzzer.py disables ASLR for every target it runs, so a
        calibration under ASLR *on* characterises a target the fuzzer never
        executes. The first version of this tool did exactly that."""
        assert "ASLR disabled (production condition): True" in report

    def test_unstable_variant_is_deterministic_once_aslr_is_off(self):
        """The --unstable blocks are gated on a heap address, so they are
        only nondeterministic while ASLR is on. With it off they still fire,
        as a fixed predicate -- so the variant becomes a valid calibration
        target rather than a ruined one."""
        cfg = SweepConfig(
            synthetic_target=True,
            blocks=200,
            fanout=16,
            unstable=4,
            calib_samples=40,
            switch_grid=(10,),
        )
        out = _synthetic_report(cfg)
        assert "identical-input reruns stable: True" in out
        assert "dead-region mutations moving coverage: 0/40" in out
        assert "WARNING" not in out

    def test_nondeterminism_is_flagged_not_silently_calibrated(self):
        """With --keep-aslr the ASLR-gated blocks make the target
        nondeterministic, which makes a dead verdict unfalsifiable. The tool
        must say so rather than emit numbers that look like a calibration.

        Keyed on observed instability, not on the --unstable flag: the flag
        was never what mattered, and a target can lose determinism for
        reasons that have nothing to do with it."""
        cfg = SweepConfig(
            synthetic_target=True,
            blocks=200,
            fanout=16,
            unstable=4,
            calib_samples=30,
            switch_grid=(10,),
            keep_aslr=True,
        )
        out = _synthetic_report(cfg)
        assert "ASLR disabled (production condition): False" in out
        # ASLR variance is probabilistic; only assert the flagging logic when
        # the run actually lost determinism, or the test is flaky by
        # construction (the lesson from the ground-truth doc's postscript).
        if "identical-input reruns stable: False" in out:
            assert "WARNING: the target is not deterministic" in out
            assert "INCONCLUSIVE" in out
