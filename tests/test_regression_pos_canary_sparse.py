"""PositionCanaryScheduler.propose scanned every bin's posterior per call.

Unobserved bins all sit at the prior mean 0.5, so the worst bin is the
lexicographic min of (mean, bin) over the observed bins plus the first
unobserved one. 4M _mean calls per 3k --hail-mary execs before.
"""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.pos_canary import PositionCanaryScheduler


def _old_propose(s: PositionCanaryScheduler, data: bytes, buf_len: int):
    """The pre-change full scan, verbatim."""
    if buf_len <= 0:
        return None
    post = s._posterior_for(data)
    num_bins = max(1, -(-len(data) // post.width)) if data else 1
    worst_bin = 0
    worst_mean = s._mean(post, 0)
    for b in range(1, num_bins):
        mean = s._mean(post, b)
        if mean < worst_mean:
            worst_bin, worst_mean = b, mean
    return min(worst_bin * post.width, buf_len - 1)


def _trained(seed: int, data: bytes, rounds: int) -> PositionCanaryScheduler:
    rng = RandPool(seed)
    s = PositionCanaryScheduler()
    for _ in range(rounds):
        offs = [rng.randint(0, len(data) + 40) for _ in range(rng.randint(1, 4))]
        outcome = Outcome.GAIN if rng.random() < 0.3 else Outcome.MISS
        s.record(data, offs, outcome, weight=rng.random())
    return s


def test_regression_pos_canary_sparse():
    for seed in range(20):
        data = RandPool(seed).randbytes(50 + seed * 97)
        s = _trained(seed, data, rounds=seed * 5)
        for buf_len in (1, len(data) // 2, len(data), len(data) + 100):
            assert s.propose(data, buf_len) == _old_propose(s, data, buf_len)


def test_old_matches_itself():
    """Control (Hard Rule 46)."""
    data = RandPool(1).randbytes(300)
    s = _trained(1, data, 30)
    assert _old_propose(s, data, 300) == _old_propose(s, data, 300)


def test_ties_go_to_lowest_bin():
    """Adversarial: all bins at 0.5 (no data) -> bin 0, like the scan."""
    s = PositionCanaryScheduler()
    assert s.propose(b"x" * 500, 500) == 0
    s.record(b"x" * 500, [0], Outcome.GAIN, weight=1.0)  # bin 0 now above 0.5
    assert s.propose(b"x" * 500, 500) == _old_propose(s, b"x" * 500, 500) > 0


def test_empty_buffer_declines():
    assert PositionCanaryScheduler().propose(b"abc", 0) is None
