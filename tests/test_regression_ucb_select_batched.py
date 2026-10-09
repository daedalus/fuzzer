"""UCBBase.select_op reads each arm's count once per call.

It used to call ``_arm_count`` three times per arm (unpulled scan, total,
score loop) and the Gaussian widths re-derived ``sqrt(xi * log_n)`` per
arm. Picks must match the old code exactly, ties included.
"""

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_ducb import DUCBScheduler
from fuzzer_tool.core.schedulers.op_kl_ducb import KL_DUCBScheduler
from fuzzer_tool.core.schedulers.op_kl_swucb import KL_SWUCBScheduler
from fuzzer_tool.core.schedulers.op_swucb import SWUCBScheduler
from fuzzer_tool.core.schedulers.ucb_common import MIN_LOG_ARG

_OPS = [f"op{i}" for i in range(40)]
_CLASSES = [DUCBScheduler, SWUCBScheduler, KL_DUCBScheduler, KL_SWUCBScheduler]


def _ref_select(s, ops):
    """Oracle: the pre-change UCBBase.select_op body."""
    if not ops:
        return ""
    if len(ops) == 1:
        return ops[0]
    unpulled = [op for op in ops if s._arm_count(op) <= 0.0]
    if unpulled:
        return s._rng.choice(unpulled)
    n_total = sum(s._arm_count(op) for op in ops)
    log_n = math.log(max(n_total, MIN_LOG_ARG))
    best_op, best_score = ops[0], -math.inf
    for op in ops:
        n = s._arm_count(op)
        mean = s._arm_mean(op, n)
        score = mean + s._width(mean, n, log_n)
        if score > best_score:
            best_score, best_op = score, op
    return best_op


def _trained(cls, pulls):
    s = cls(rng=RandPool(1))
    for op in _OPS:
        s.init_arm(op)
    rp = RandPool(2)
    for i in range(pulls):
        s.record(_OPS[i % len(_OPS)], rp.random() < 0.3, 1.0)
    return s


def _pair(cls, pulls, ops):
    """(new pick, oracle pick) on two identically built schedulers."""
    return _trained(cls, pulls).select_op(ops), _ref_select(_trained(cls, pulls), ops)


@pytest.mark.parametrize("cls", _CLASSES)
def test_control_oracle_matches_itself(cls):
    """Rule 46: the oracle agrees with a second run of itself."""
    a = _ref_select(_trained(cls, 500), _OPS)
    assert a == _ref_select(_trained(cls, 500), _OPS)


@pytest.mark.parametrize("cls", _CLASSES)
@pytest.mark.parametrize("pulls", [17, 500, 3000])
def test_pick_matches_oracle(cls, pulls):
    """Falsification: 17 pulls leaves unpulled arms; 500/3000 score them."""
    got, want = _pair(cls, pulls, _OPS)
    assert got == want


@pytest.mark.parametrize("cls", _CLASSES)
def test_ties_keep_first(cls):
    """Adversarial: identical arms tie; the first in *ops* order must win."""
    s = cls(rng=RandPool(1))
    for op in _OPS[:3]:
        s.init_arm(op)
        s.record(op, True, 1.0)
    assert s.select_op(_OPS[:3]) == _ref_select(s, _OPS[:3]) == _OPS[0]


@pytest.mark.parametrize("cls", _CLASSES)
def test_degenerate_inputs(cls):
    """Adversarial: empty and single-op lists short-circuit."""
    s = _trained(cls, 100)
    assert s.select_op([]) == ""
    assert s.select_op(["only"]) == "only"


@pytest.mark.parametrize("cls", _CLASSES)
def test_counts_read_once_per_arm(cls, monkeypatch):
    """Falsification: one _arm_count call per candidate arm."""
    s = _trained(cls, 500)
    calls = {"n": 0}
    real = s._arm_count

    def counting(op):
        calls["n"] += 1
        return real(op)

    monkeypatch.setattr(s, "_arm_count", counting)
    s.select_op(_OPS)
    assert calls["n"] == len(_OPS)
