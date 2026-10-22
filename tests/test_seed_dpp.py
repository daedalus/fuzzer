"""SeedDPPScheduler: sequential DPP over hashed edge-set features.

Every pick is one scripted ``random()`` draw (Hard Rule 39). Residuals are
checked against a pure-Python determinant-ratio oracle, never against the
scheduler's own arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.core.chi_squared import chi_squared_test
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_dpp import HASH_DIM, SeedDPPScheduler
from tests.support.scripted_rng import ScriptedRng

P_FLOOR = 1e-3  # distribution tests: reject only on strong evidence
TRIALS = 4000


def _sched(window: int = 8, **rng_kw) -> SeedDPPScheduler:
    return SeedDPPScheduler(rng=ScriptedRng(**rng_kw), window=window)


def _tracker(seed_edges: dict[str, set[int]]) -> SimpleNamespace:
    return SimpleNamespace(seed_edges=seed_edges)


def _features(n: int, dim: int, seed: int) -> np.ndarray:
    return np.asarray(RandPool(seed=seed).normal_array((n, dim)), dtype=np.float32)


# --- pure-Python oracle --------------------------------------------------------


def _det(m: list[list[float]]) -> float:
    """Determinant by Gaussian elimination with partial pivoting."""
    a = [row[:] for row in m]
    n = len(a)
    det = 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-300:
            return 0.0
        if p != c:
            a[c], a[p] = a[p], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            k = a[r][c] / a[c][c]
            for j in range(c, n):
                a[r][j] -= k * a[c][j]
    return det


def _gram(phi: np.ndarray) -> list[list[float]]:
    rows = [[float(x) for x in r] for r in phi]
    return [[sum(x * y for x, y in zip(a, b, strict=True)) for b in rows] for a in rows]


def _ratio(gram: list[list[float]], window: list[int], i: int) -> float:
    """det(L_{W∪i}) / det(L_W): the DPP conditional weight of i."""
    if i in window:
        return 0.0
    sub = window + [i]
    num = _det([[gram[a][b] for b in sub] for a in sub])
    den = _det([[gram[a][b] for b in window] for a in window]) if window else 1.0
    return num / den


def _oracle_pick(gram, window, u):
    w = [_ratio(gram, window, i) for i in range(len(gram))]
    acc, target = 0.0, u * sum(w)
    for i, x in enumerate(w):
        acc += x
        if acc > target:
            return i
    return len(w) - 1


# --- policy ------------------------------------------------------------------


class TestPolicy:
    def test_first_pick_proportional_to_squared_norm(self):
        """Row norms 1 and 2 -> weights 1 and 4; u=0.21 lands past 1/5."""
        s = _sched(randoms=[0.21])
        s._load(["a", "b"], np.array([[1.0, 0.0], [0.0, 2.0]], np.float32))

        assert s.select_seed(["a", "b"]) == "b"

    def test_residuals_match_det_ratio_oracle(self):
        phi = _features(6, 4, seed=1)
        keys = [f"k{i}" for i in range(6)]
        gram = _gram(phi)
        draws = [0.1, 0.7, 0.4, 0.9]
        s = _sched(window=len(draws), randoms=draws)
        s._load(keys, phi)

        window: list[int] = []
        for _u in draws:
            expected = [_ratio(gram, window, i) for i in range(6)]
            assert s._d2 == pytest.approx(expected, rel=1e-4, abs=1e-6)
            window.append(keys.index(s.select_seed(keys)))

    def test_pick_sequence_matches_oracle(self):
        phi = _features(5, 3, seed=2)
        keys = [f"k{i}" for i in range(5)]
        gram = _gram(phi)
        draws = [0.05, 0.5, 0.95]
        s = _sched(window=3, randoms=draws)
        s._load(keys, phi)

        window: list[int] = []
        for u in draws:
            expected = _oracle_pick(gram, window, u)
            assert s.select_seed(keys) == keys[expected]
            window.append(expected)

    def test_window_resets_after_k_picks(self):
        s = _sched(window=2, randoms=[0.0, 0.0, 0.0])
        s._load(["a", "b", "c"], np.eye(3, dtype=np.float32))

        picks = [s.select_seed(["a", "b", "c"]) for _ in range(3)]

        # Third pick still sees a, b conditioned out; then the window restarts at c.
        assert picks == ["a", "b", "c"]
        assert len(s._chol) == 1

    def test_unfitted_is_uniform(self):
        s = _sched(randoms=[0.5])

        assert s.select_seed(["a", "b", "c", "d"]) == "c"

    def test_rare_edges_raise_quality(self):
        """Seed with a unique edge outweighs one whose edges are shared."""
        s = _sched()
        s.refit(_tracker({"common1": {1}, "common2": {1}, "rare": {2}}))

        d2 = dict(zip(sorted(s._row), s._d2[[s._row[k] for k in sorted(s._row)]], strict=True))
        assert d2["rare"] > d2["common1"] == pytest.approx(d2["common2"])


class TestFalsification:
    def test_duplicate_of_last_pick_is_skipped(self):
        """A, B identical; C distinct. u at B's original CDF slot must yield C.

        A uniform or quality-only picker returns B for the second draw.
        """
        s = _sched(randoms=[0.0, 0.3])
        s.refit(_tracker({"A": {1, 2}, "B": {1, 2}, "C": {3, 4}}))
        keys = ["A", "B", "C"]
        before = s._d2[[s._row[k] for k in keys]]
        lo = before[0] / before.sum()
        hi = (before[0] + before[1]) / before.sum()
        assert lo < 0.3 < hi  # 0.3 is inside B's original slot

        assert s.select_seed(keys) == "A"
        assert s.select_seed(keys) == "C"


class TestAdversarial:
    def test_identical_corpus_never_nan(self):
        n = 1000
        s = SeedDPPScheduler(rng=RandPool(seed=5), window=8)
        s.refit(_tracker({f"s{i}": {7, 9} for i in range(n)}))
        keys = [f"s{i}" for i in range(n)]

        picks = [s.select_seed(keys) for _ in range(50)]

        assert set(picks) <= set(keys)
        assert np.isfinite(s._d2).all()
        assert (s._d2 >= 0).all()

    def test_zero_edge_seed_never_picked_while_others_have_mass(self):
        s = _sched(randoms=[0.0, 0.999, 0.5])
        s.refit(_tracker({"z": set(), "a": {1}}))

        assert [s.select_seed(["z", "a"]) for _ in range(3)] == ["a", "a", "a"]

    def test_all_zero_edge_falls_back_to_uniform(self):
        s = _sched(randoms=[0.75])
        s.refit(_tracker({"x": set(), "y": set()}))

        assert s.select_seed(["x", "y"]) == "y"

    def test_novel_key_gets_max_live_residual(self):
        """'n' joined after refit: weight = max(d2 of a) -> equal odds."""
        s = _sched(randoms=[0.6])
        s.refit(_tracker({"a": {1}}))

        assert s.select_seed(["a", "n"]) == "n"

    def test_dropped_key_never_returned(self):
        s = _sched(randoms=[0.0, 0.0, 0.999])
        s._load(["a", "b", "c"], np.eye(3, dtype=np.float32))

        picks = [s.select_seed(["a", "c"]) for _ in range(3)]

        assert "b" not in picks

    def test_key_list_grown_in_place(self):
        """The row cache keys on list identity; an in-place append must miss it."""
        s = _sched(randoms=[0.0, 0.99])
        s._load(["a", "b"], np.eye(2, dtype=np.float32))
        keys = ["a", "b"]
        s.select_seed(keys)
        keys.append("n")

        assert s.select_seed(keys) == "n"

    def test_single_key_needs_no_draw(self):
        assert _sched().select_seed(["only"]) == "only"

    def test_empty_list(self):
        assert _sched().select_seed([]) == ""

    def test_refit_mid_window_resets(self):
        s = _sched(randoms=[0.0])
        s.refit(_tracker({"a": {1}, "b": {2}}))
        s.select_seed(["a", "b"])
        assert s._chol

        s.refit(_tracker({"a": {1}, "b": {2}}))

        assert s._chol == []

    def test_negative_and_huge_edge_ids(self):
        s = SeedDPPScheduler(rng=RandPool(seed=6))

        assert s.refit(_tracker({"a": {-1, 2**63}, "b": {0}}))
        assert s.select_seed(["a", "b"]) in {"a", "b"}

    def test_requires_rng(self):
        with pytest.raises(ValueError):
            SeedDPPScheduler(rng=None)

    def test_empty_tracker_does_not_refit(self):
        s = _sched()

        assert s.refit(_tracker({})) is False
        assert s.refit(SimpleNamespace()) is False


class TestRefitCadence:
    def test_first_call_refits_then_waits_interval(self):
        s = SeedDPPScheduler(rng=RandPool(seed=1), refit_interval=100)
        t = _tracker({"a": {1}, "b": {2}})

        assert s.maybe_refit(t, 0)
        assert not s.maybe_refit(t, 50)
        assert s.refits == 1


class TestFuzzRoundHook:
    def _round(self, dpp):
        from fuzzer_tool.services.fuzz_round import FuzzRound

        rnd = FuzzRound.__new__(FuzzRound)
        rnd._f = SimpleNamespace(
            _seed_dpp=dpp, _edge_tracker=_tracker({"a": {1}, "b": {2}}), exec_count=0
        )
        return rnd

    def test_refits_enabled_arm(self):
        dpp = SeedDPPScheduler(rng=RandPool(seed=1))
        self._round(dpp)._refit_dpp()

        assert dpp.refits == 1

    def test_disabled_arm_is_skipped(self):
        self._round(None)._refit_dpp()  # no AttributeError on None

    def test_called_from_feed_models(self):
        import inspect

        from fuzzer_tool.services.fuzz_round import FuzzRound

        assert "self._refit_dpp()" in inspect.getsource(FuzzRound._feed_models)


class TestBounds:
    def test_memory_bounded(self):
        n, k = 64, 4
        s = SeedDPPScheduler(rng=RandPool(seed=2), window=k)
        s.refit(_tracker({f"s{i}": {i, i + 1} for i in range(n)}))
        keys = [f"s{i}" for i in range(n)]

        for _ in range(3 * k):
            s.select_seed(keys)

        assert s._phi.nbytes == n * HASH_DIM * 4
        assert len(s._chol) <= k


# --- distribution (Hard Rule 46: control first) --------------------------------


def _pairs_sched(phi, keys, rng) -> list[tuple[str, str]]:
    s = SeedDPPScheduler(rng=rng, window=2)
    out = []
    for _ in range(TRIALS):
        s._load(keys, phi)
        out.append((s.select_seed(keys), s.select_seed(keys)))
    return out


def _pairs_oracle(gram, keys, rng) -> list[tuple[str, str]]:
    out = []
    for _ in range(TRIALS):
        a = _oracle_pick(gram, [], rng.random())
        b = _oracle_pick(gram, [a], rng.random())
        out.append((keys[a], keys[b]))
    return out


def _table(x, y) -> list[list[float]]:
    cells = sorted(set(x) | set(y))
    return [[float(x.count(c)) for c in cells], [float(y.count(c)) for c in cells]]


class TestDistribution:
    PHI = np.array([[1.0, 0.0], [0.9, 0.3], [0.0, 1.2], [0.5, 0.5]], np.float32)
    KEYS = ["a", "b", "c", "d"]

    def test_control_oracle_against_itself(self):
        gram = _gram(self.PHI)
        x = _pairs_oracle(gram, self.KEYS, RandPool(seed=10))
        y = _pairs_oracle(gram, self.KEYS, RandPool(seed=11))

        assert chi_squared_test(_table(x, y))[1] > P_FLOOR

    def test_matches_oracle_in_distribution(self):
        gram = _gram(self.PHI)
        x = _pairs_oracle(gram, self.KEYS, RandPool(seed=10))
        y = _pairs_sched(self.PHI, self.KEYS, RandPool(seed=12))

        assert chi_squared_test(_table(x, y))[1] > P_FLOOR
