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

from fuzzer_tool.core.byte_pca import adjusted_rand, sketch_row
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
