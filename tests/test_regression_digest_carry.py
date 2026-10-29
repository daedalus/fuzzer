"""Op-effect digest carry: an op's after-digest is the next op's before-digest.

A k-op stack used to hash 2k times; carrying the digest pays k+1.
The carry is valid only while the returned buffer is the hashed one, so a
max_len clamp must drop it.
"""

from __future__ import annotations

import itertools

import pytest

from fuzzer_tool.services import operators as operators_mod
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer

_K = 8  # ops per round


def _engine(ops, max_len=0):
    """Engine whose round applies *ops* (name -> handler) in turn."""
    f = make_minimal_fuzzer(seed=3)
    f.mutations_per_input = _K
    f._last_perf_score = 100.0
    f._track_op_effect = True
    f.max_len = max_len
    engine = OperatorEngine(f)

    names = list(ops)
    turn = itertools.count()
    engine.build_ops = lambda data: names
    engine.select_op = lambda _ops: names[next(turn) % len(names)]
    f._op_dispatch = dict(ops)
    return f, engine


def _count_hash64(monkeypatch):
    calls = []
    real = operators_mod.hash64

    def counted(data):
        calls.append(1)
        return real(data)

    monkeypatch.setattr(operators_mod, "hash64", counted)
    return calls


def _flip(buf, _idx, _data):
    buf[0] ^= 1


def _noop(buf, _idx, _data):
    return None


def test_regression_digest_carry_k_plus_one(monkeypatch):
    """A k-op round hashes k + 1 times, not 2k."""
    calls = _count_hash64(monkeypatch)
    f, engine = _engine({"flip": _flip})

    engine.mutate(b"A" * 64)

    assert len(f._last_ops_used) == _K
    assert len(calls) == _K + 1


def test_regression_digest_carry_noop_after_change_declines(monkeypatch):
    """Falsification: a stale carry (op N's *before* digest) would mark the
    no-op after a flip effective."""
    f, engine = _engine({"flip": _flip, "noop": _noop})
    order = iter(["flip"] + ["noop"] * (_K - 1))
    engine.select_op = lambda _ops: next(order)

    engine.mutate(b"A" * 64)

    assert f._last_ops_effective == {"flip"}
    assert f._op_declines.get("noop") == _K - 1


@pytest.mark.parametrize("grow", ["replace", "in_place"])
def test_regression_digest_carry_dropped_on_clamp(grow):
    """Adversarial: the after-digest covers the pre-clamp buffer. Carrying
    it past the clamp makes the following no-op look effective."""
    cap = 16

    def big(buf, _idx, _data):
        if grow == "replace":
            return bytearray(buf) + b"Z" * cap
        buf.extend(b"Z" * cap)
        return None

    f, engine = _engine({"big": big, "noop": _noop}, max_len=cap)
    order = iter(["big"] + ["noop"] * (_K - 1))
    engine.select_op = lambda _ops: next(order)

    engine.mutate(b"A" * cap)

    assert "noop" not in f._last_ops_effective
    assert f._op_declines.get("noop") == _K - 1


def test_regression_digest_carry_reset_per_round():
    """A carry left by the previous round must not seed this round's first op."""
    f, engine = _engine({"noop": _noop})
    engine.mutate(b"A" * 32)
    f._op_declines.clear()

    engine.mutate(b"B" * 32)

    assert "noop" not in f._last_ops_effective
    assert f._op_declines.get("noop") == _K
