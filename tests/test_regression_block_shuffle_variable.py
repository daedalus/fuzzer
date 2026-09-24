"""Regression: block_shuffle_variable dropped colliding cut points, so short
inputs got fewer blocks than the drawn k (order_theory P4-1)."""

from __future__ import annotations

from fuzzer_tool.core.mutations.generic import block_shuffle_variable
from tests.support.scripted_rng import ScriptedRng


class _ExpRng(ScriptedRng):
    """ScriptedRng plus scripted expovariate; records shuffled blocks."""

    def __init__(self, exps, **kw):
        super().__init__(**kw)
        self._exps = iter(exps)
        self.blocks: list[bytes] = []

    def expovariate(self, _lambd):
        return next(self._exps)

    def shuffle(self, seq):
        self.blocks = list(seq)
        super().shuffle(seq)


def _run(data: bytes, k: int, exps) -> _ExpRng:
    rng = _ExpRng(exps, randints=[k])
    out = block_shuffle_variable(data, rng)
    assert sorted(out) == sorted(data)
    return rng


def test_regression_block_shuffle_variable_keeps_k_blocks():
    # All four cuts collide at len/2 before the fix -> 2 blocks.
    tiny = 1e-9
    rng = _run(bytes(range(8)), 5, [1.0, tiny, tiny, tiny, 1.0])

    assert len(rng.blocks) == 5
    assert all(rng.blocks)
    assert b"".join(rng.blocks) == bytes(range(8))


def test_block_shuffle_variable_edge_cuts_pushed_inward():
    """Adversarial: all mass on the last gap pins every raw cut at 1."""
    tiny = 1e-9
    data = bytes(range(8))
    rng = _run(data, 5, [tiny, tiny, tiny, tiny, 1.0])

    assert [len(b) for b in rng.blocks] == [1, 1, 1, 1, 4]


def test_block_shuffle_variable_distinct_cuts_unchanged():
    """Falsification: well-spread cuts keep their exact positions."""
    data = bytes(range(100))
    rng = _run(data, 3, [1.0, 1.0, 2.0])

    # cum/s = 1/4, 2/4 -> cuts 25, 50
    assert [len(b) for b in rng.blocks] == [25, 25, 50]
