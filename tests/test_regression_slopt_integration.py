"""SLOPT: the batch exponent is learned per (seed-size group, operator).

The first integration (aa6425d/c28f17c) computed the batch size from a
fixed formula, ``max(1, int((len/64) ** exp))`` with a constant exponent per
operator, and renamed the MC bandit's arms to ``f"{op}_{exp:.2f}"`` when
recording while it selected by plain operator name. Nothing was learned,
and under ``--mc-bandit`` 119.9 of 126.5 evidence units in a 120-round
campaign landed on arms the bandit could never select. Its tests
monkeypatched the formula and asserted the patched value came back, and
mocked the MC bandit, so neither defect could fail them.

These drive the real engine and a real Fuzzer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.slopt import EXPONENTS, SIZE_GROUP_EDGES, SloptBatchBandit, size_group
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer

# ---------------------------------------------------------------------------
# The bandit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "group"),
    [
        (0, 0),
        (99, 0),
        (100, 1),
        (999, 1),
        (1_000, 2),
        (9_999, 2),
        (10_000, 3),
        (99_999, 3),
        (100_000, 4),
        (10**7, 4),
    ],
)
def test_size_groups_are_the_papers(n, group):
    assert size_group(n) == group


def test_seven_arms_batch_sizes_two_to_128():
    assert EXPONENTS == (1, 2, 3, 4, 5, 6, 7)
    assert SIZE_GROUP_EDGES == (100, 1_000, 10_000, 100_000)


def test_learns_the_rewarded_exponent_for_that_instance_only():
    """Exponent 3 always pays for byte_flip on small seeds. That instance
    converges to it; byte_flip on large seeds and havoc on small seeds are
    separate instances and stay unlearned."""
    b = SloptBatchBandit(rng=RandPool(1))
    picks = []
    for _ in range(600):
        t = b.choose("byte_flip", 50)
        b.record("byte_flip", 50, t, success=(t == 3))
        picks.append(t)
    assert picks[-200:].count(3) > 180
    assert b.posterior_means("byte_flip", 5_000) == [0.5] * 7
    assert b.posterior_means("havoc", 50) == [0.5] * 7


def test_reward_is_fractional_and_clamped():
    b = SloptBatchBandit(rng=RandPool(1))
    b.record("op", 10, 2, True, weight=0.25)
    b.record("op", 10, 4, True, weight=7.0)
    b.record("op", 10, 5, True, weight=float("nan"))
    b.record("op", 10, 6, False)
    means = b.posterior_means("op", 10)
    assert means[1] == pytest.approx(1.25 / 3.0)  # alpha 1.25, beta 1.75
    assert means[3] == pytest.approx(2.0 / 3.0)
    assert means[4] == pytest.approx(1.0 / 3.0)
    assert means[5] == pytest.approx(1.0 / 3.0)


def test_rejects_an_exponent_that_is_not_an_arm():
    with pytest.raises(ValueError):
        SloptBatchBandit(rng=RandPool(1)).record("op", 10, 0, True)


def test_seeded_pool_reproduces_the_draws():
    def draws(seed):
        b = SloptBatchBandit(rng=RandPool(seed))
        return [b.choose("havoc", 300) for _ in range(50)]

    assert draws(4) == draws(4)
    assert draws(4) != draws(5)


def test_stats_name_each_instance_by_group():
    b = SloptBatchBandit(rng=RandPool(1))
    for _ in range(20):
        b.record("havoc", 20_000, 6, True)
    stats = b.stats()
    assert stats["slopt_pulls"] == 20
    assert stats["slopt_best_exponent"] == {"havoc@10000+": 6}


# ---------------------------------------------------------------------------
# The engine: one operator, 2**t applications, the arm published
# ---------------------------------------------------------------------------


class _FixedBandit:
    def __init__(self, t):
        self.t = t
        self.calls = []

    def choose(self, op, seed_len):
        self.calls.append((op, seed_len))
        return self.t


def _engine_with_counter(t, mutations=8):
    f = make_minimal_fuzzer(seed=3)
    f._use_slopt = True
    f._slopt = _FixedBandit(t)
    f.mutations_per_input = mutations
    f._last_perf_score = 800.0
    engine = OperatorEngine(f)
    applied = []

    def op(buf, _idx, _data):
        applied.append(len(buf))
        buf[0] ^= 1

    engine.build_ops = lambda data: ["counted"]
    f._op_dispatch = {"counted": op}
    return f, engine, applied


@pytest.mark.parametrize("t", [1, 4, 7])
def test_applies_one_operator_two_to_the_t_times(t):
    """-M and the perf score (800 -> x8) do not change the batch: the bandit
    is credited for the batch it chose, so the batch applied must be it."""
    f, engine, applied = _engine_with_counter(t)
    engine.mutate(b"A" * 300)
    assert len(applied) == 2**t
    assert f._slopt.calls == [("counted", 300)]
    assert f._last_slopt_arm == ("counted", 300, t)
    assert f._last_ops_used == ["counted"] * 2**t


def test_deterministic_stage_round_publishes_no_arm():
    f, engine, _ = _engine_with_counter(2)
    engine.mutate(b"A" * 40)
    assert f._last_slopt_arm is not None
    engine.maybe_deterministic_mutation = lambda data: b"det"
    engine.mutate(b"A" * 40)
    assert f._last_slopt_arm is None, "a stale arm would credit a round it never drew"


# ---------------------------------------------------------------------------
# A real Fuzzer
# ---------------------------------------------------------------------------

_TARGET = Path(__file__).resolve().parent.parent / "targets" / "test_target"


@pytest.mark.skipif(not _TARGET.exists(), reason="targets/test_target not built")
def test_mc_bandit_evidence_lands_on_arms_it_can_select(tmp_path):
    """--slopt --mc-bandit: every arm the MC bandit records is a registered
    operator, and the SLOPT bandit learns from the same rounds."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    (tmp_path / "c").mkdir()
    (tmp_path / "k").mkdir()
    f = Fuzzer(
        target=str(_TARGET),
        corpus_dir=str(tmp_path / "c"),
        crashes_dir=str(tmp_path / "k"),
        max_len=65536,
        use_coverage=True,
        mc_bandit=True,
        slopt=True,
        seed=5,
    )
    assert isinstance(f._slopt, SloptBatchBandit)
    assert f._slopt._rng is f._rng
    for i, n in enumerate([16, 200, 4000, 20000] * 10):
        f.fuzz_one(bytes([65 + i % 26]) * n)

    arms = set(f.mc.arm_alpha) | set(f.mc.arm_beta)
    assert arms <= set(REGISTRY.names()), sorted(arms - set(REGISTRY.names()))[:5]
    stats = f._slopt.stats()
    assert stats["slopt_pulls"] > 0
    assert stats["slopt_instances"] >= 2  # more than one size group was fuzzed
