"""Tests for tools/sim_byte_ica.py: PCA argmax vs ICA rotation of byteflip groups.

FastICA is checked against a rotation the test builds itself, never against the
tool's own bookkeeping. The grouping arms are scored against planted truth.
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.core.byte_pca import (
    MIN_GAP_RATIO,
    adjusted_rand,
    gap_rank,
    signed_labels,
    sketch_row,
    varimax,
)
from fuzzer_tool.core.rand_pool import RandPool

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import sim_byte_ica as sim  # noqa: E402

N = 4000
RECOVERY = 0.98  # |corr| a matched source must reach


def _whitened_sources(rand: RandPool) -> np.ndarray:
    """Three unit-variance non-Gaussian sources: Laplace, uniform, +-1."""
    u = rand.random_array((N, 3))
    lap = np.where(u[:, 0] < 0.5, np.log(2 * u[:, 0]), -np.log(2 - 2 * u[:, 0]))
    uni = u[:, 1] - 0.5
    pm = np.where(u[:, 2] < 0.5, -1.0, 1.0)
    s = np.column_stack([lap, uni, pm])
    s -= s.mean(axis=0)
    return s / s.std(axis=0)


def _rotation(rand: RandPool, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(rand.normal_array((k, k)))
    return q


def _best_match(c: np.ndarray) -> np.ndarray:
    """Matched |corr| per source over all permutations (k is tiny)."""
    k = c.shape[0]
    best = max(itertools.permutations(range(k)), key=lambda p: c[range(k), p].sum())
    return c[range(k), best]


class TestFastIca:
    def test_recovers_known_rotation(self):
        rand = RandPool(seed=11)
        s = _whitened_sources(rand)
        z = s @ _rotation(rand, 3).T  # still white: orthogonal mix
        w, _ = sim.fast_ica(z, rand)
        got = z @ w.T
        c = np.abs(np.corrcoef(s.T, got.T)[:3, 3:])
        assert _best_match(c).min() >= RECOVERY

    def test_rotation_is_orthogonal(self):
        rand = RandPool(seed=12)
        z = _whitened_sources(rand)
        w, _ = sim.fast_ica(z, rand)
        assert np.allclose(w @ w.T, np.eye(3), atol=1e-8)

    def test_reports_non_convergence(self):
        rand = RandPool(seed=13)
        z = _whitened_sources(rand) @ _rotation(rand, 3).T
        _, conv = sim.fast_ica(z, rand, max_iter=1)
        assert conv is sim.Converge.NO

    def test_single_component_is_identity(self):
        rand = RandPool(seed=14)
        z = _whitened_sources(rand)[:, :1]
        w, conv = sim.fast_ica(z, rand)
        assert w.shape == (1, 1) and abs(abs(w[0, 0]) - 1.0) < 1e-12
        assert conv is sim.Converge.YES


class TestPlanted:
    def test_shapes_and_truth(self):
        spec = sim.Spec(n_fields=3, per_field=30, overlap=0.5, reject=20)
        offs, rows, truth = sim.planted(RandPool(seed=1), spec)
        assert rows.shape == (3 * 30 + 20, rows.shape[1])
        assert offs.size == truth.size == rows.shape[0]
        assert set(truth.tolist()) == {-1, 0, 1, 2}

    def test_overlap_shares_edges_between_fields(self):
        """overlap > 0 makes field signatures non-orthogonal; 0 keeps them apart."""

        def mean_cos(overlap):
            spec = sim.Spec(n_fields=2, per_field=40, overlap=overlap, reject=0, keep=1.0)
            _, rows, truth = sim.planted(RandPool(seed=2), spec)
            y = rows.astype(np.float64) - sketch_row(sim.BASE)  # drop the static trace
            a, b = y[truth == 0].mean(axis=0), y[truth == 1].mean(axis=0)
            return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

        assert mean_cos(0.75) > mean_cos(0.0)


class TestArms:
    def test_same_membership_both_arms(self):
        """Arms differ only in which group a member joins, never in who joins."""
        spec = sim.Spec(n_fields=3, per_field=30, overlap=0.5, reject=20)
        offs, rows, _ = sim.planted(RandPool(seed=3), spec)
        p = sim.pca_labels(offs, rows)
        i = sim.ica_labels(offs, rows, RandPool(seed=4))
        assert p is not None and i is not None
        assert ((p >= 0) == (i >= 0)).all()

    def test_ica_recovers_disjoint_fields(self):
        spec = sim.Spec(n_fields=3, per_field=40, overlap=0.0, reject=24)
        offs, rows, truth = sim.planted(RandPool(seed=5), spec)
        got = sim.ica_labels(offs, rows, RandPool(seed=6))
        assert got is not None
        assert adjusted_rand(truth, got) >= 0.9

    def test_pca_labels_match_byte_groups(self):
        """pca_labels is byte_groups re-indexed per row, nothing more."""
        from fuzzer_tool.core.byte_pca import byte_groups

        spec = sim.Spec(n_fields=3, per_field=30, overlap=0.25, reject=10)
        offs, rows, _ = sim.planted(RandPool(seed=7), spec)
        g = byte_groups(offs, rows)
        labels = sim.pca_labels(offs, rows)
        assert (labels[g.offsets] == g.comp).all()
        assert (np.delete(labels, g.offsets) == -1).all()

    def test_falsification_no_structure_no_agreement(self):
        """Fieldless rows: neither arm may agree with an arbitrary planted split."""
        rand = RandPool(seed=8)
        rows = np.stack(
            [sketch_row(np.array(rand.randint_list(1, 1 << 30, 60))) for _ in range(120)]
        )
        offs = np.arange(120)
        fake = np.repeat(np.arange(3), 40)
        for labels in (sim.pca_labels(offs, rows), sim.ica_labels(offs, rows, rand)):
            assert labels is not None
            assert abs(adjusted_rand(fake, labels)) < 0.1

    @pytest.mark.parametrize("n_rows", [1, 3])
    def test_adversarial_too_few_rows_declines(self, n_rows):
        spec = sim.Spec(n_fields=1, per_field=n_rows, overlap=0.0, reject=0)
        offs, rows, _ = sim.planted(RandPool(seed=9), spec)
        assert sim.pca_labels(offs, rows) is None
        assert sim.ica_labels(offs, rows, RandPool(seed=9)) is None

    def test_adversarial_identical_rows_declines(self):
        rows = np.stack([sketch_row(np.arange(1, 500))] * 16)
        offs = np.arange(16)
        assert sim.pca_labels(offs, rows) is None
        assert sim.ica_labels(offs, rows, RandPool(seed=10)) is None


class TestSignP:
    def test_no_pairs_is_one(self):
        assert sim.sign_p(0, 0) == 1.0

    def test_all_wins_exact(self):
        assert sim.sign_p(12, 0) == pytest.approx(2 * 0.5**12)

    def test_symmetric(self):
        assert sim.sign_p(9, 3) == sim.sign_p(3, 9)

    def test_matches_binomial_tail(self):
        """Derived independently: two-sided tail of Binomial(n, 1/2)."""
        wins, losses = 14, 6
        n = wins + losses
        tail = sum(math.comb(n, j) for j in range(0, min(wins, losses) + 1)) / 2**n
        assert sim.sign_p(wins, losses) == pytest.approx(min(1.0, 2 * tail))


def _skewed_sources(rand: RandPool) -> np.ndarray:
    """Three unit-variance, zero-mean, right-skewed sources (exponential)."""
    s = -np.log(1.0 - rand.random_array((N, 3)))
    s -= s.mean(axis=0)
    return s / s.std(axis=0)


class TestSkewContrast:
    def test_recovers_rotated_skewed_sources(self):
        rand = RandPool(seed=21)
        s = _skewed_sources(rand)
        z = s @ _rotation(rand, 3).T
        w, conv = sim.fast_ica(z, rand, contrast=sim.Contrast.SKEW)
        c = np.abs(np.corrcoef(s.T, (z @ w.T).T)[:3, 3:])
        assert conv is sim.Converge.YES
        assert _best_match(c).min() >= RECOVERY


class TestVarimax:
    def test_recovers_rotated_sparse_loadings(self):
        """One nonzero per row (a byte in one field); varimax must undo Q."""
        rand = RandPool(seed=22)
        k, per = 3, 20
        l0 = np.zeros((k * per, k))
        l0[np.arange(k * per), np.repeat(np.arange(k), per)] = 1.0 + rand.random_array(k * per)
        q = _rotation(rand, k)
        got = l0 @ q @ varimax(l0 @ q)
        c = np.abs(got.T @ l0) / np.outer(np.linalg.norm(got, axis=0), np.linalg.norm(l0, axis=0))
        assert _best_match(c).min() >= 0.999

    def test_rotation_is_orthogonal(self):
        r = varimax(RandPool(seed=23).normal_array((40, 4)))
        assert np.allclose(r @ r.T, np.eye(4), atol=1e-10)

    def test_single_column_is_identity(self):
        assert np.allclose(varimax(np.ones((5, 1))), np.eye(1))


class TestGapRank:
    def test_largest_relative_gap(self):
        assert gap_rank(np.array([100.0, 50, 40, 1, 0.9, 0.8]), 1) == 3

    def test_min_rank_skips_leading_gap(self):
        """A dominant first eigenvalue (reject split) must not cap the rank at 1."""
        eig = np.array([2000.0, 50, 40, 30, 1, 0.9])
        assert gap_rank(eig, 1) == 1
        assert gap_rank(eig, 2) == 4

    @pytest.mark.parametrize("eig", [np.array([5.0]), np.array([3.0, 3.0])])
    def test_adversarial_short_spectrum_bounded(self, eig):
        assert 1 <= gap_rank(eig, 2) <= eig.size


class TestSignedAssign:
    def test_vertex_and_rest(self):
        """Positive tail names the field; no positive score -> rest group (k)."""
        src = np.array([[3.0, -1, -1], [-1, 2.5, -1], [-1, -1, 4], [-1, -1, -0.5], [-2, -2, -2]])
        src = np.vstack([src, -0.1 * np.ones((20, 3))])  # bulk keeps each tail positive
        member = np.ones(len(src), dtype=bool)
        member[-1] = False
        got = signed_labels(src, member)
        assert got[:5].tolist() == [0, 1, 2, 3, 3]
        assert got[-1] == -1


FOUR = sim.Spec(n_fields=4, per_field=30, overlap=0.5, reject=24)


class TestImprovedArms:
    def test_ica_fix_keeps_pca_membership(self):
        offs, rows, _ = sim.planted(RandPool(seed=31), FOUR)
        p = sim.pca_labels(offs, rows)
        i = sim.ica_fix_labels(offs, rows, RandPool(seed=32))
        assert ((p >= 0) == (i >= 0)).all()

    @pytest.mark.parametrize("arm", ["ica_fix", "vmax"])
    def test_recovers_four_overlapping_fields(self, arm):
        """The tetrahedron case where logcosh ICA scored ~0.69."""
        offs, rows, truth = sim.planted(RandPool(seed=33), FOUR)
        got = (
            sim.ica_fix_labels(offs, rows, RandPool(seed=34))
            if arm == "ica_fix"
            else sim.vmax_labels(offs, rows)
        )
        assert adjusted_rand(truth, got) >= 0.95

    def test_vmax_keeps_pc1_field_without_reject(self):
        spec = sim.Spec(n_fields=4, per_field=30, overlap=0.5, reject=0)
        offs, rows, truth = sim.planted(RandPool(seed=35), spec)
        assert adjusted_rand(truth, sim.vmax_labels(offs, rows)) >= 0.95

    def test_vmax_rank_one_without_reject(self):
        """Two fields, no reject: one real axis; a forced 2nd axis splits noise."""
        spec = sim.Spec(n_fields=2, per_field=30, overlap=0.5, reject=0)
        ari = []
        for seed in range(40, 50):  # fixed draws, not a retry loop: all are scored
            offs, rows, truth = sim.planted(RandPool(seed=seed), spec)
            ari.append(adjusted_rand(truth, sim.vmax_labels(offs, rows)))
        assert np.mean(ari) >= 0.95

    def test_gap_rank_falls_back_without_later_gap(self):
        """No ratio past min_rank reaches min_gap: the leading gap decides."""
        eig = np.array([500.0, 10, 9.5, 9, 8.6, 8.2])
        assert gap_rank(eig, 2) == 3  # 9.5 / 9: largest noise ratio
        assert gap_rank(eig, 2, min_gap=MIN_GAP_RATIO) == 1

    def test_taper_shrinks_later_fields(self):
        spec = sim.Spec(n_fields=3, per_field=40, overlap=0.0, reject=0, taper=0.5)
        _, _, truth = sim.planted(RandPool(seed=36), spec)
        assert np.bincount(truth).tolist() == [40, 20, 10]

    def test_falsification_no_structure(self):
        rand = RandPool(seed=37)
        rows = np.stack(
            [sketch_row(np.array(rand.randint_list(1, 1 << 30, 60))) for _ in range(120)]
        )
        offs = np.arange(120)
        fake = np.repeat(np.arange(3), 40)
        for labels in (sim.ica_fix_labels(offs, rows, rand), sim.vmax_labels(offs, rows)):
            assert labels is not None
            assert abs(adjusted_rand(fake, labels)) < 0.1

    def test_adversarial_degenerate_declines(self):
        rows = np.stack([sketch_row(np.arange(1, 500))] * 16)
        offs = np.arange(16)
        assert sim.ica_fix_labels(offs, rows, RandPool(seed=38)) is None
        assert sim.vmax_labels(offs, rows) is None
        assert sim.vmax_labels(offs[:3], rows[:3]) is None
