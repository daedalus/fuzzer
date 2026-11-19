"""Naive Bayes exec gate (core/nb_gate.py)."""

from __future__ import annotations

import io
import pickle
from fractions import Fraction

import numpy as np
import pytest

from fuzzer_tool.core.nb_gate import (
    CARDS,
    POS_BUCKETS,
    SIZE_BUCKETS,
    NBGate,
    OpIds,
    Outcome,
    pos_bucket,
    size_bucket,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import safe_load
from tests.support.scripted_rng import ScriptedRng

_NO_DRAWS = ScriptedRng()  # any draw raises StopIteration


def _gate(rng=_NO_DRAWS, **kw):
    kw.setdefault("warmup", 0)
    return NBGate((2, 2), rng, **kw)


def _laplace(c_feat: int, n_cls: int, card: int) -> Fraction:
    return Fraction(c_feat + 1, n_cls + card)


# --- Posterior ----------------------------------------------------------------


def test_posterior_matches_hand_laplace():
    g = _gate()
    g.record((0, 0), Outcome.NOVEL)
    g.record((0, 1), Outcome.BORING)
    g.record((1, 1), Outcome.BORING)

    # N1 = 1, N0 = 2; class prior (N_y + 1) / (N + 2).
    p1 = Fraction(2, 5) * _laplace(1, 1, 2) * _laplace(1, 1, 2)
    p0 = Fraction(3, 5) * _laplace(1, 2, 2) * _laplace(0, 2, 2)
    assert g.posterior((0, 0)) == pytest.approx(float(p1 / (p1 + p0)), rel=1e-12)


def test_untrained_is_half():
    assert _gate().posterior((1, 0)) == pytest.approx(0.5)


def test_batch_matches_scalar():
    g = NBGate(CARDS, _NO_DRAWS, warmup=0)
    rng = RandPool(seed=5)
    rows = np.array([rng.randint_list(0, c - 1, 200) for c in CARDS]).T
    for row, y in zip(rows, rng.randint_list(0, 1, 200), strict=True):
        g.record(tuple(row), Outcome(y))

    batch = g.posterior_batch(rows)
    scalar = [g.posterior(tuple(r)) for r in rows]
    # Control (Hard Rule 46): the scalar path against itself first.
    assert scalar == [g.posterior(tuple(r)) for r in rows]
    np.testing.assert_allclose(batch, scalar, rtol=1e-12)


# --- Gate ---------------------------------------------------------------------


def _dead_op_gate(randoms=(), eps=0.1):
    # Op 1 never yields, op 0 always does: op 1's posterior sits far below base.
    g = _gate(ScriptedRng(randoms=randoms), eps=eps, lift=0.5)
    for _ in range(50):
        g.record((0, 0), Outcome.NOVEL)
        g.record((1, 0), Outcome.BORING)
    return g


def test_skips_dead_op_without_exploration_hit():
    g = _dead_op_gate(randoms=[0.1])  # 0.1 >= eps: no exploration
    assert g.admit((1, 0)) is False
    assert g.skipped == 1


def test_eps_floor_explores_dead_op():
    g = _dead_op_gate(randoms=[0.0999])
    assert g.admit((1, 0)) is True
    assert g.explored == 1


def test_live_op_admitted_without_draw():
    # High posterior: no RNG draw at all (ScriptedRng would raise).
    assert _dead_op_gate().admit((0, 0)) is True


def test_warmup_admits_without_draw():
    g = NBGate((2, 2), _NO_DRAWS, warmup=10)
    for _ in range(9):
        g.record((1, 1), Outcome.BORING)
    assert g.admit((1, 1)) is True


def test_falsify_gate_is_not_always_admit():
    # A gate that never skips would admit all of these; the scripted RNG
    # never explores, so every dead-op mutant must be skipped.
    g = _dead_op_gate(randoms=[0.5] * 20)
    assert [g.admit((1, 0)) for _ in range(20)] == [False] * 20


def test_eps_one_never_starves():
    g = _dead_op_gate(randoms=[0.0] * 5, eps=1.0)
    assert all(g.admit((1, 0)) for _ in range(5))


# --- Bounds / adversarial -----------------------------------------------------


def test_counts_halve_at_cap():
    g = _gate(cap=8)
    for _ in range(8):
        g.record((0, 1), Outcome.BORING)
    assert g.total == 4
    assert g.posterior((0, 1)) == g.posterior((0, 1))  # still finite
    assert g.nbytes == len(g.to_dict()["counts"]) + 2 * 8


def test_adversarial_all_novel_never_skips():
    g = _gate(lift=1.0)
    for _ in range(1000):
        g.record((1, 1), Outcome.NOVEL)
    assert g.admit((1, 1)) is True


@pytest.mark.parametrize(
    "kw", [dict(eps=-0.1), dict(eps=1.5), dict(lift=-1.0), dict(cap=1), dict(warmup=-1)]
)
def test_rejects_bad_args(kw):
    with pytest.raises(ValueError):
        _gate(**kw)


def test_rejects_empty_cards():
    with pytest.raises(ValueError):
        NBGate((), _NO_DRAWS)


# --- Feature encoders ---------------------------------------------------------


def test_pos_bucket_edges():
    assert pos_bucket(0, 100) == 0
    assert pos_bucket(99, 100) == POS_BUCKETS - 1
    assert pos_bucket(None, 100) == POS_BUCKETS  # delocalised op
    assert pos_bucket(10**9, 3) == POS_BUCKETS - 1  # offset past a shrunk buffer
    assert pos_bucket(0, 0) == 0


def test_size_bucket_is_log2_and_clamped():
    assert [size_bucket(n) for n in (0, 1, 2, 3, 4)] == [0, 1, 2, 2, 3]
    assert size_bucket(1 << 200) == SIZE_BUCKETS - 1


def test_op_ids_bounded():
    ids = OpIds(slots=3)
    assert [ids.of(n) for n in ("a", "b", "a")] == [0, 1, 0]
    # Past the cap every new name shares the overflow slot.
    assert ids.of("c") == ids.of("d") == 2
    assert len(ids) == 2


# --- Persistence --------------------------------------------------------------


def test_state_roundtrip_through_safe_loader():
    g = _dead_op_gate()
    raw = pickle.dumps(g.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    back = NBGate.from_dict(safe_load(io.BytesIO(raw)), _NO_DRAWS)
    assert back.posterior((1, 0)) == g.posterior((1, 0))
    assert back.total == g.total


def test_from_dict_rejects_shape_mismatch():
    state = _gate().to_dict()
    state["counts"] = bytes(7 * 8)
    with pytest.raises(ValueError):
        NBGate.from_dict(state, _NO_DRAWS)


def test_op_ids_restore_same_slots():
    ids = OpIds(slots=4)
    for n in ("havoc", "splice", "flip"):
        ids.of(n)
    back = OpIds.from_names(pickle.loads(pickle.dumps(ids.names())), slots=4)
    assert [back.of(n) for n in ("flip", "havoc", "splice")] == [2, 0, 1]


def test_regression_state_section_survives_store(tmp_path):
    # Protocol-5 ndarrays were refused by the safe loader and the section
    # silently dropped from state.pkl.gz.
    from fuzzer_tool.core.state_store import StateStore

    g = _dead_op_gate()
    store = StateStore(tmp_path)
    store.set("nb_gate", {"gate": g.to_dict(), "ops": ["x"]})
    assert store.save()

    back = StateStore(tmp_path)
    back.load()
    assert NBGate.from_dict(back.get("nb_gate")["gate"], _NO_DRAWS).total == g.total
