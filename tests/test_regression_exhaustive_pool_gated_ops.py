"""Regression: ``ExhaustivePool.sample`` takes a range or bytearray, and the
option-gated operators decline instead of raising when their option is unset.

The enumerator walks every registered operator through ``ExhaustivePool``.
``arith_mid`` calls ``rng.sample(range(n), k)``, which ``RandPool`` accepts and
the pool did not; ``poly1305_fix`` and ``rs_fix`` read config the minimal
fuzzer lacked and crashed on ``None`` when called unconfigured.
"""

from __future__ import annotations

import itertools

import pytest

from fuzzer_tool.core.exhaustive_pool import ExhaustivePool
from fuzzer_tool.services.operators import OperatorEngine

from .support.operator_env import make_minimal_fuzzer


def _selections(population, k):
    pool = ExhaustivePool(max_depth=16, max_runs=1000)
    return [tuple(pool.sample(population, k)) for _ in pool.runs()]


def test_regression_sample_range_enumerates_ordered_selections():
    assert sorted(_selections(range(3), 2)) == sorted(itertools.permutations(range(3), 2))


def test_regression_sample_bytearray_enumerates_ordered_selections():
    assert sorted(_selections(bytearray(b"abc"), 2)) == sorted(itertools.permutations(b"abc", 2))


def test_regression_sample_range_clamps_k_and_handles_empty():
    assert sorted(_selections(range(2), 5)) == [(0, 1), (1, 0)]
    assert _selections(range(0), 2) == [()]


_BUF = b"abcdefghijklmnopqrstuvwxyz012345"


@pytest.mark.parametrize("name", ["poly1305_fix", "rs_fix"])
def test_regression_gated_op_declines_without_its_option(name):
    eng = OperatorEngine(make_minimal_fuzzer())
    buf = bytearray(_BUF)
    out = getattr(eng, f"_op_{name}")(buf, 0, bytes(buf))
    assert bytes(out) == _BUF
    assert eng.f._op_declines[name] == 1
