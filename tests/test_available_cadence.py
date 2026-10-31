"""OperatorRegistry.available: steady context gates on a cadence, geometric trickle.

* Context gates that never read the input are re-checked every
  ``PRED_REFRESH_CALLS`` calls per fuzzer (they were ~51 us of ~128 us per
  mutate in-run); gates that read the input still run on every call.
* The unseen-format trickle keeps each format at ``_FORMAT_BOOTSTRAP_RATE``
  independently but draws geometric gaps (~2 draws, not one per format).
"""

from __future__ import annotations

import math
import random

from fuzzer_tool.core import operator_registry as reg
from fuzzer_tool.core.chi_squared import chi_squared_pvalue
from fuzzer_tool.core.operator_registry import PRED_REFRESH_CALLS, REGISTRY, OperatorSpec
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng

DATA = b"plain bytes, no known format"
STEADY_OP = "regex_bomb"  # lambda f, _d: enable_regex_bomb
INPUT_OP = "afl_det"  # lambda f, d: bool(d) and op_afl_det
P = reg._FORMAT_BOOTSTRAP_RATE


def _fuzzer(seed: int = 1, **attrs):
    f = make_minimal_fuzzer(pool=RandPool(seed))
    for k, v in attrs.items():
        setattr(f, k, v)
    return f


# ── cadence ────────────────────────────────────────────────────────────


def test_steady_gate_change_lands_on_the_refresh_boundary():
    f = _fuzzer()
    assert STEADY_OP not in REGISTRY.available(f, DATA)  # evaluates, caches
    f.enable_regex_bomb = True

    seen = [STEADY_OP in REGISTRY.available(f, DATA) for _ in range(PRED_REFRESH_CALLS)]

    assert seen == [False] * (PRED_REFRESH_CALLS - 1) + [True]


def test_falsify_input_gate_reevaluated_every_call():
    """A gate reading the input must not be frozen by the cadence."""
    f = _fuzzer(op_afl_det=True)

    assert INPUT_OP not in REGISTRY.available(f, b"")
    assert INPUT_OP in REGISTRY.available(f, DATA)
    assert INPUT_OP not in REGISTRY.available(f, b"")


def test_adversarial_cache_never_crosses_fuzzers():
    on, off = _fuzzer(enable_regex_bomb=True), _fuzzer()
    for _ in range(3):
        assert STEADY_OP in REGISTRY.available(on, DATA)
        assert STEADY_OP not in REGISTRY.available(off, DATA)


def test_plan_rebuild_drops_the_cache():
    f = _fuzzer()
    REGISTRY.available(f, DATA)
    REGISTRY.register(OperatorSpec("cadence_probe", "byte", "_op_bit_flip", lambda fz, _d: True))
    try:
        assert "cadence_probe" in REGISTRY.available(f, DATA)
    finally:
        REGISTRY._ops.pop("cadence_probe")
        REGISTRY._version += 1


def test_reads_input_classification():
    def closure(f, d):
        return (lambda: bool(d))()

    assert not reg._reads_input(lambda f, _d: True)
    assert reg._reads_input(lambda f, d: bool(d))
    assert reg._reads_input(closure)
    assert reg._reads_input(print)  # not introspectable: per call, to be safe


# ── geometric trickle ──────────────────────────────────────────────────


def _gap(u: float) -> int:
    """Failures before the next success for one uniform draw (independent of reg)."""
    return int(math.log(1.0 - u) / math.log(1.0 - P))


def test_trickle_positions_follow_the_drawn_gaps():
    draws = [0.5, 0.99, 0.0, 0.999999]  # last gap overshoots k: the stop draw
    k = 300
    expect = [False] * k
    pos = -1
    for u in draws:
        pos += _gap(u) + 1
        if pos >= k:
            break
        expect[pos] = True

    assert reg._trickle(ScriptedRng(randoms=draws), k) == expect


def test_trickle_draws_successes_plus_one():
    class Spy:
        def __init__(self, rng: random.Random) -> None:
            self.rng, self.drawn = rng, 0

        def random(self) -> float:
            self.drawn += 1
            return self.rng.random()

    for k in (1, 54, 500):
        spy = Spy(random.Random(k))
        kept = sum(reg._trickle(spy, k))
        assert spy.drawn == kept + 1


def _bernoulli(rng: random.Random, k: int) -> list[bool]:
    return [rng.random() < P for _ in range(k)]


def _table(calls, k):
    """Kept count per call (0, 1, 2+) and kept position by eighths of k."""
    counts, where = [0, 0, 0], [0] * 8
    for row in calls:
        n = sum(row)
        counts[min(n, 2)] += 1
        for i, ok in enumerate(row):
            if ok:
                where[i * 8 // k] += 1
    return counts + where


def _homogeneity_p(a: list[int], b: list[int]) -> float:
    """Two-sample chi-squared homogeneity test (contingency, not GOF)."""
    keep = [(x, y) for x, y in zip(a, b, strict=True) if x + y]
    na, nb = sum(x for x, _ in keep), sum(y for _, y in keep)
    stat = 0.0
    for x, y in keep:
        col = x + y
        ea, eb = col * na / (na + nb), col * nb / (na + nb)
        stat += (x - ea) ** 2 / ea + (y - eb) ** 2 / eb
    return chi_squared_pvalue(stat, len(keep) - 1)


def test_trickle_matches_bernoulli_in_distribution():
    """Rule 46: the Bernoulli control against itself must pass first."""
    k, calls = 54, 20000
    ref_rng, ctl_rng, geo_rng = random.Random(1), random.Random(2), random.Random(3)
    ref = _table([_bernoulli(ref_rng, k) for _ in range(calls)], k)
    ctl = _table([_bernoulli(ctl_rng, k) for _ in range(calls)], k)
    geo = _table([reg._trickle(geo_rng, k) for _ in range(calls)], k)

    assert _homogeneity_p(ref, ctl) > 0.01
    assert _homogeneity_p(ref, geo) > 0.01
