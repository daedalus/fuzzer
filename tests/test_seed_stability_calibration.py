"""Seed stability calibration: mask edges that don't reproduce.

There was no AFL-style per-seed calibration. `_run_calibration` is a
bootstrap warm-up loop that measures throughput, not stability. Without
one, a nondeterministic edge -- ASLR-, time-, thread-, or
uninitialized-memory-dependent -- is rediscovered on every execution and
reads as new coverage forever, permanently absorbing mutation energy that
should be going somewhere productive.

`_calibrate_seed_stability` runs an accepted seed n times and masks any
edge that appears in some runs but not all.

Two design points the tests pin down, because both are easy to get wrong:

1. Path-hash divergence is a *screen*, not the verdict. The hash is order-
   and multiplicity-sensitive, so it moves whenever the same edges fire in a
   different order or a different number of times. Masking on hash
   divergence alone would mask edges that are perfectly deterministic about
   *which* code runs. The per-edge set-diff decides.
2. Calibration never rejects a seed. It runs after the seed is committed to
   the corpus and only ever masks edges.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest


class _FakeShm:
    """Replays a scripted sequence of edge sets, one per run."""

    def __init__(
        self,
        runs: list[set[int]],
        hashes: list[int] | None = None,
        drops: list[int] | None = None,
    ):
        self._runs = list(runs)
        self._hashes = list(hashes) if hashes is not None else None
        # Per-run dropped-edge counts, returned by dropped_edges_delta().
        # Default is a clean map: no drops, so the set-diff verdict stands.
        self._drops = list(drops) if drops is not None else None
        self._i = -1
        # Drops already on the counter when calibration starts, i.e. the
        # rest of the campaign's history.
        self._drop_backlog = 0
        self._last_drop_raw = 0
        self._seen_edge_ids: set[int] = set()
        self._masked_edge_ids: set[int] = set()

    def advance(self):
        self._i += 1

    def dropped_edges_delta(self) -> int:
        """Differences a monotone cumulative counter, as ShmCoverage does.

        Modelled rather than stubbed so the pre-loop read in
        `_calibrate_seed_stability` is actually exercised: that read exists
        to discard drops accumulated *before* this calibration, and a stub
        returning a per-index value would pass whether it was there or not.
        """
        raw = self._drop_backlog
        if self._drops is not None and self._i >= 0:
            raw += sum(self._drops[: min(self._i + 1, len(self._drops))])
        delta = raw - self._last_drop_raw
        self._last_drop_raw = raw
        return delta

    def get_edge_ids(self) -> set[int]:
        return set(self._runs[min(self._i, len(self._runs) - 1)])

    def read_path_hash(self) -> int:
        if self._hashes is not None:
            return self._hashes[min(self._i, len(self._hashes) - 1)]
        # Order-insensitive stand-in: differs iff the edge set differs.
        return hash(frozenset(self.get_edge_ids())) & 0xFFFFFFFF

    def mask_edges(self, edge_ids) -> int:
        ids = {int(e) for e in edge_ids}
        newly = ids - self._seen_edge_ids
        self._seen_edge_ids.update(ids)
        self._masked_edge_ids.update(ids)
        return len(newly)

    @property
    def masked_edges(self) -> set[int]:
        return set(self._masked_edge_ids)


def _make_fuzzer(**kwargs):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="stability_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            **kwargs,
        )


def _wire(f, shm):
    """Point the fuzzer at a fake SHM and make _run_target advance it."""
    f.shm_cov = shm
    f._run_target = lambda data: (shm.advance(), (0, ""))[1]
    return f


class TestUnstableEdgeDetection:
    def test_stable_seed_masks_nothing(self):
        shm = _FakeShm([{1, 2, 3}] * 3)
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()
        assert shm.masked_edges == set()

    def test_edge_present_in_only_some_runs_is_masked(self):
        # Edge 9 fires on run 2 only: the textbook ASLR-dependent edge.
        shm = _FakeShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}])
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {9}
        assert shm.masked_edges == {9}

    def test_stable_edges_are_never_masked(self):
        shm = _FakeShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}])
        f = _wire(_make_fuzzer(), shm)
        f._calibrate_seed_stability(b"x", n_runs=3)
        assert {1, 2, 3}.isdisjoint(shm.masked_edges)

    def test_multiple_unstable_edges_all_masked(self):
        shm = _FakeShm([{1, 7}, {1, 8}, {1, 9}])
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {7, 8, 9}

    def test_unstable_edges_accumulate_across_seeds(self):
        f = _make_fuzzer()
        _wire(f, _FakeShm([{1}, {1, 7}, {1}]))
        f._calibrate_seed_stability(b"a", n_runs=3)
        _wire(f, _FakeShm([{1}, {1, 8}, {1}]))
        f._calibrate_seed_stability(b"b", n_runs=3)
        assert f._unstable_edges == {7, 8}


class TestPathHashIsAScreenNotTheVerdict:
    def test_diverging_hash_with_identical_edge_sets_masks_nothing(self):
        """Same edges, different order or trip count. The hash moves; the
        set does not. Masking here would suppress deterministic edges."""
        shm = _FakeShm([{1, 2, 3}] * 3, hashes=[111, 222, 333])
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()
        assert shm.masked_edges == set()

    def test_identical_hash_short_circuits_before_the_set_diff(self):
        """An identical hash across every run is proof of stability, so the
        set comparison is skipped entirely. Cheap screen, as intended."""
        shm = _FakeShm([{1, 2}, {1, 2, 99}, {1, 2}], hashes=[42, 42, 42])
        f = _wire(_make_fuzzer(), shm)
        # Edge 99 differs, but the hash says stable, so nothing is examined.
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()


class TestGuards:
    def test_no_shm_means_no_calibration(self):
        f = _make_fuzzer()
        f.shm_cov = None
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()

    def test_single_run_cannot_establish_stability(self):
        """One run has nothing to compare against; must not claim a verdict."""
        shm = _FakeShm([{1, 2}])
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=1) == set()
        assert shm.masked_edges == set()

    def test_a_seed_that_fails_to_rerun_is_left_alone(self):
        f = _make_fuzzer()
        f.shm_cov = _FakeShm([{1}])

        def boom(data):
            raise OSError("target vanished")

        f._run_target = boom
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()

    def test_disabled_by_default(self):
        """Opt-in: an unmeasured throughput change does not become the
        default, and this costs n_runs executions per accepted seed."""
        assert _make_fuzzer()._calibrate_stability == 0

    def test_flag_value_is_carried_through(self):
        assert _make_fuzzer(calibrate_stability=3)._calibrate_stability == 3


class TestCorpusWiring:
    def test_save_to_corpus_calibrates_only_when_enabled(self):
        import inspect

        from fuzzer_tool.services import corpus_manager

        src = inspect.getsource(corpus_manager.CorpusManager.save_to_corpus)
        assert "_calibrate_seed_stability" in src
        assert "_calibrate_stability" in src

    def test_cli_exposes_the_flag(self):
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--calibrate-stability"' in src
        assert "calibrate_stability=" in src


class TestMaskEdges:
    """`mask_edges` on the real ShmCoverage.

    Note these write table entries directly rather than going through
    `record_edge()`. That helper deliberately stands in for both the writer
    and the reader -- it pre-marks its own hit-count buckets so a recorded
    edge does not report as new on the next scan -- which makes it useless
    for testing novelty either way: an edge written through it never reads
    as new, masked or not. A test built on it would pass without the feature
    existing at all.
    """

    @staticmethod
    def _place(cov, slot: int, edge_id: int, count: int = 1):
        """Write an entry the way the C shim does: table slot *and* the two
        headers the fast path screens on. Writing only the slot leaves
        edge_count/path_hash unchanged, so `_check_new_coverage` short-
        circuits before it ever scans the table."""
        import ctypes

        cov._entries[slot].edge_id = edge_id
        cov._entries[slot].count = (cov.read_generation() << 24) | count
        ec = ctypes.c_uint64.from_address(cov._ptr + 16)
        ec.value = ec.value + 1
        ph = ctypes.c_uint64.from_address(cov._ptr + 8)
        ph.value = (ph.value * 31) ^ edge_id

    def test_masked_edge_does_not_register_as_new_coverage(self):
        from fuzzer_tool.adapters.shm import ShmCoverage

        cov = ShmCoverage()
        try:
            cov.mask_edges({4242})
            self._place(cov, 0, 4242)
            has_new, edges = cov.is_new_coverage_with_edges()
            assert 4242 in edges, "edge should still be visible, just not novel"
            assert not has_new, "a masked edge registered as new coverage"
        finally:
            cov.cleanup()

    def test_the_same_edge_unmasked_does_register(self):
        """Control: without the mask, the identical sequence must report new
        coverage. This is what makes the test above non-vacuous."""
        from fuzzer_tool.adapters.shm import ShmCoverage

        cov = ShmCoverage()
        try:
            self._place(cov, 0, 4242)
            has_new, edges = cov.is_new_coverage_with_edges()
            assert 4242 in edges
            assert has_new, "control failed: unmasked edge was not new either"
        finally:
            cov.cleanup()

    def test_masking_is_targeted_not_wholesale(self):
        """Masking one edge must not suppress novelty for a different one."""
        from fuzzer_tool.adapters.shm import ShmCoverage

        cov = ShmCoverage()
        try:
            cov.mask_edges({4242})
            self._place(cov, 0, 4242)
            cov.is_new_coverage_with_edges()
            self._place(cov, 1, 777)
            has_new, _ = cov.is_new_coverage_with_edges()
            assert has_new, "masking suppressed an unrelated edge's novelty"
        finally:
            cov.cleanup()

    def test_returns_count_of_newly_masked_only(self):
        from fuzzer_tool.adapters.shm import ShmCoverage

        cov = ShmCoverage()
        try:
            assert cov.mask_edges({1, 2, 3}) == 3
            assert cov.mask_edges({2, 3, 4}) == 1
            assert cov.masked_edges == {1, 2, 3, 4}
        finally:
            cov.cleanup()

    def test_masked_set_is_a_copy(self):
        from fuzzer_tool.adapters.shm import ShmCoverage

        cov = ShmCoverage()
        try:
            cov.mask_edges({5})
            cov.masked_edges.add(99)
            assert cov.masked_edges == {5}
        finally:
            cov.cleanup()


class TestDroppedEdgesVetoTheVerdict:
    """A truncated edge set is not evidence of nondeterminism.

    When the table saturates, an edge whose probe window is full is
    discarded, and WHICH edge loses the slot depends on arrival order
    against whatever the previous execution left behind — reset_edge_map()
    bumps a generation tag, it does not clear the table. So the same input,
    firing the same edges in the same order, can report different edge sets
    across runs with no nondeterminism in the target at all.

    Measured on an 8192-entry table with a fixed 4000-guard sequence and
    only the prior occupancy differing between runs: three runs shared 819
    edges out of a 2,507-edge union. Calibration would have masked the other
    1,688 — every one reproducible — and masking is permanent.

    Declining to decide is the correct answer rather than masking a subset:
    from the edge sets alone there is no way to tell which divergences were
    drops and which were real.
    """

    def test_divergence_is_not_masked_when_edges_were_dropped(self):
        shm = _FakeShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}], drops=[0, 40, 0])
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()
        assert shm.masked_edges == set(), "masked edges on evidence a full map invalidates"

    def test_drops_in_any_run_veto_the_whole_calibration(self):
        """One truncated run is enough: the intersection it participates in
        is missing edges that every other run recorded."""
        for drops in ([7, 0, 0], [0, 0, 7]):
            shm = _FakeShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}], drops=drops)
            f = _wire(_make_fuzzer(), shm)
            assert f._calibrate_seed_stability(b"x", n_runs=3) == set(), drops

    def test_clean_runs_still_reach_the_verdict(self):
        """The veto must not disable calibration on a healthy map."""
        shm = _FakeShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}], drops=[0, 0, 0])
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {9}
        assert shm.masked_edges == {9}

    def test_drops_from_before_the_calibration_do_not_veto(self):
        """The pre-loop read exists to discard the backlog. Without it, a
        single drop anywhere earlier in the run would veto calibration for
        the rest of the campaign."""
        shm = _FakeShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}], drops=[0, 0, 0])
        shm._drop_backlog = 5000
        f = _wire(_make_fuzzer(), shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {9}

    def test_vetoed_calibration_still_counts_as_performed(self):
        """The calibration counter drives reporting and the opt-in's cost
        accounting; a vetoed run consumed the same n_runs executions."""
        shm = _FakeShm([{1, 2}, {1, 2, 9}, {1, 2}], drops=[0, 3, 0])
        f = _wire(_make_fuzzer(), shm)
        before = f._stability_calibrations
        f._calibrate_seed_stability(b"x", n_runs=3)
        assert f._stability_calibrations == before + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
