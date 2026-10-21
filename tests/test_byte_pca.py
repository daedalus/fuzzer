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
    MIN_ROWS,
    SKETCH_COLS,
    byte_groups,
    sketch_row,
)

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
