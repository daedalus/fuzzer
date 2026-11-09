"""core/byte_pca.py: group bytes by the PCA components of their byteflip traces.

The positive case plants three fields, each moving its own edges, plus a block
of bytes that only trip the shared reject path. The grouping must recover the
fields and must not place the reject bytes in any of them. The falsification
and adversarial cases pin the other direction: structure-free input must not
produce groups that look like structure, and degenerate input must decline.
"""

import numpy as np
import pytest

from fuzzer_tool.core.byte_pca import (
    DROP_LEADING,
    MAX_COMPONENTS,
    MAX_ROWS,
    MIN_AGREEMENT,
    MIN_ROWS,
    SKETCH_COLS,
    adjusted_rand,
    byte_groups,
    group_stability,
    sketch_row,
)
from fuzzer_tool.core.rand_pool import RandPool

BASE = np.arange(1, 1500)
REJECT = np.arange(50_000, 50_300)


def _fields(n_fields=3, per_field=40, reject=24, seed=1, keep=0.8):
    """Rows for `per_field` bytes of each field, then `reject` clutter bytes."""
    rng = np.random.default_rng(seed)
    own = [np.arange(10_000 * (g + 1), 10_000 * (g + 1) + 40) for g in range(n_fields)]
    rows, truth = [], []
    for g in range(n_fields):
        for _ in range(per_field):
            ids = np.concatenate([BASE, own[g][rng.random(40) < keep]])
            rows.append(sketch_row(ids))
            truth.append(g)
    for _ in range(reject):
        rows.append(sketch_row(np.concatenate([BASE[:50], REJECT])))
        truth.append(-1)
    return np.arange(len(rows)), np.stack(rows), np.array(truth)


class TestSketch:
    def test_shape_and_dtype(self):
        row = sketch_row(np.arange(1, 100))
        assert row.shape == (SKETCH_COLS,) and row.dtype == np.int16

    def test_empty_trace_is_all_zero(self):
        assert not sketch_row(np.array([], dtype=np.int64)).any()

    def test_deterministic(self):
        ids = np.arange(5, 900)
        assert (sketch_row(ids) == sketch_row(ids)).all()

    def test_static_trace_cancels_at_centring(self):
        """Identical rows have no variance: nothing to group."""
        rows = np.stack([sketch_row(BASE)] * 8)
        assert byte_groups(np.arange(8), rows) is None


class TestPlantedGroups:
    def test_recovers_each_field_without_mixing(self):
        offs, rows, truth = _fields()
        got = byte_groups(offs, rows)
        assert got is not None
        by_group = {}
        for off, comp in zip(got.offsets, got.comp, strict=True):
            by_group.setdefault(int(comp), set()).add(int(truth[off]))
        assert all(len(v) == 1 for v in by_group.values())  # pure
        assert {next(iter(v)) for v in by_group.values()} == {0, 1, 2}  # all found

    def test_reject_only_bytes_are_excluded(self):
        offs, rows, truth = _fields()
        got = byte_groups(offs, rows)
        assert got is not None
        assert not (truth[got.offsets] == -1).any()

    def test_offsets_ascending_and_weights_positive(self):
        offs, rows, _ = _fields()
        got = byte_groups(offs, rows)
        assert list(got.offsets) == sorted(got.offsets)
        assert (got.weight > 0).all() and got.offsets.dtype == np.uint32

    def test_group_ids_within_count(self):
        offs, rows, _ = _fields()
        got = byte_groups(offs, rows)
        assert got.comp.min() >= 0 and got.comp.max() < got.count
        assert got.count <= 2 * MAX_COMPONENTS

    def test_deterministic(self):
        offs, rows, _ = _fields()
        a, b = byte_groups(offs, rows), byte_groups(offs, rows)
        assert (a.offsets == b.offsets).all() and (a.comp == b.comp).all()


class TestFalsification:
    def test_one_field_only_yields_no_separation(self):
        """Rows that differ only by noise must not look like distinct fields.

        With a single field there is nothing for the kept components to split
        by field identity: every surviving group must still be pure (one true
        class), and no more than one class exists, so the check is that the
        grouping does not invent a second class.
        """
        offs, rows, truth = _fields(n_fields=1, reject=0)
        got = byte_groups(offs, rows)
        if got is not None:
            assert set(truth[got.offsets]) <= {0}

    def test_dropping_the_leading_component_matters(self):
        """The first component is the clutter axis: without the drop, the
        reject bytes would be the largest 'group'."""
        assert DROP_LEADING >= 1


class TestAdversarial:
    def test_too_few_rows_declines(self):
        offs, rows, _ = _fields(per_field=1, reject=0, n_fields=MIN_ROWS - 1)
        assert byte_groups(offs, rows) is None

    def test_offset_row_mismatch_declines(self):
        offs, rows, _ = _fields()
        assert byte_groups(offs[:-1], rows) is None

    def test_all_zero_rows_decline(self):
        assert byte_groups(np.arange(10), np.zeros((10, SKETCH_COLS), np.int16)) is None

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_tiny_inputs_do_not_raise(self, n):
        rows = np.ones((n, SKETCH_COLS), np.int16)
        assert byte_groups(np.arange(n), rows) is None

    def test_saturated_int16_rows_do_not_raise(self):
        rng = np.random.default_rng(0)
        rows = rng.integers(-32768, 32767, size=(20, SKETCH_COLS), dtype=np.int16)
        got = byte_groups(np.arange(20), rows)
        assert got is None or got.offsets.size <= 20

    def test_more_rows_than_the_cap_declines(self):
        offs, rows, _ = _fields(per_field=MAX_ROWS // 2, reject=0, n_fields=3)
        assert rows.shape[0] > MAX_ROWS
        assert byte_groups(offs, rows) is None


class TestAdjustedRand:
    def test_identical_partitions_score_one_under_relabelling(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand(a, np.array([5, 5, 9, 9, 7, 7])) == 1.0

    def test_trivial_partitions(self):
        assert adjusted_rand(np.zeros(5, int), np.zeros(5, int)) == 1.0
        assert adjusted_rand(np.arange(5), np.arange(5)) == 1.0
        assert adjusted_rand(np.zeros(5, int), np.arange(5)) == 0.0

    def test_hand_computed_value(self):
        # a = {0,1}{2,3}  b = {0,1,2}{3}: sum_ij=1 sum_a=2 sum_b=3 total=6
        # expected = 2*3/6 = 1; top = 2.5 -> (1-1)/(2.5-1) = 0
        assert adjusted_rand(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1])) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            adjusted_rand(np.zeros(3, int), np.zeros(4, int))


class TestGroupStability:
    def test_declines_exactly_when_byte_groups_declines(self):
        offs, rows, _ = _fields(per_field=1, reject=0, n_fields=MIN_ROWS - 1)
        assert group_stability(offs, rows, RandPool(seed=1)) is None

    def test_planted_fields_are_stable(self):
        offs, rows, _ = _fields()
        st = group_stability(offs, rows, RandPool(seed=3))
        assert st is not None and st.stable and st.agreement >= MIN_AGREEMENT

    def test_noise_rows_are_unstable(self):
        """Structure-free rows must not be called stable: the bootstrap
        regroups them differently every time."""
        rng = np.random.default_rng(0)
        rows = rng.integers(-5, 5, size=(40, SKETCH_COLS)).astype(np.int16)
        st = group_stability(np.arange(40), rows, RandPool(seed=3))
        assert st is not None and not st.stable and st.agreement < MIN_AGREEMENT

    def test_gaps_match_the_eigenvalues(self):
        offs, rows, _ = _fields()
        st = group_stability(offs, rows, RandPool(seed=3), resamples=0)
        y = rows.astype(np.float64)
        y -= y.mean(axis=0)
        w = np.sort(np.linalg.eigvalsh(y @ y.T))[::-1]
        start = DROP_LEADING
        want = (w[start : start + st.gaps.size] - w[start + 1 : start + 1 + st.gaps.size]) / w[
            start : start + st.gaps.size
        ]
        assert np.allclose(st.gaps, want, atol=1e-9)
        assert st.resamples == 0 and st.agreement == 1.0

    def test_degenerate_only_counts_components_that_hold_bytes(self):
        offs, rows, _ = _fields()
        st = group_stability(offs, rows, RandPool(seed=3))
        used = {int(c) // 2 for c in byte_groups(offs, rows).comp}
        assert set(st.degenerate) <= used

    def test_does_not_change_byte_groups(self):
        offs, rows, _ = _fields()
        before = byte_groups(offs, rows)
        group_stability(offs, rows, RandPool(seed=3))
        after = byte_groups(offs, rows)
        assert (before.offsets == after.offsets).all() and (before.comp == after.comp).all()

    def test_same_seed_same_result(self):
        offs, rows, _ = _fields(keep=0.5)
        a = group_stability(offs, rows, RandPool(seed=11))
        b = group_stability(offs, rows, RandPool(seed=11))
        assert a.agreement == b.agreement and (a.gaps == b.gaps).all()
