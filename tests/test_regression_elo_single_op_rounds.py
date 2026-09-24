"""Single-operator rounds must still produce Elo operator matches.

SLOPT (``--slopt``, and therefore ``--hail-mary``) applies one operator
``2**t`` times per round, so every round's deduplicated operator set has
exactly one member. Both the call site in ``Fuzzer._record_outcome`` and
``RoundRecorderMixin.record_round`` required two or more operators, so a
SLOPT run recorded zero operator matches: ``BayesianEloTracker`` never
received a prediction error and ``_effective_k()`` stayed at ``_base_k``
(16) for the whole run. With the tau-floor steady state every strategy
reaches (sigma^2 ~= 1012.6 at beta=200, tau=5), that printed K = 0.40 on
every row of every strategy convergence table.
"""

import math

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker, EloTracker


def test_single_op_rounds_record_a_cross_round_match():
    t = EloTracker()
    t.record_round(["a"], {"a"})
    t.record_round(["b"], set())
    assert t._match_count.get("a", 0) == 1
    assert t._match_count.get("b", 0) == 1
    assert t.ratings["a"] > t.ratings["b"]


def test_single_op_rounds_feed_bayesian_prediction_errors():
    t = BayesianEloTracker()
    for i in range(40):
        op = f"op{i % 4}"
        t.record_round([op], {op} if i % 3 == 0 else set())
    assert len(t._prediction_errors) > 0
    assert t._effective_k() != t._base_k


def test_same_op_twice_is_not_a_match():
    """The same single operator in consecutive rounds has nothing unique
    on either side; the guard change must not pair it with itself."""
    t = EloTracker()
    t.record_round(["a"], {"a"})
    t.record_round(["a"], set())
    assert t._match_count.get("a", 0) == 0


def test_strategy_k_steady_state_is_the_tau_floor():
    """Pins why every row shows the same K once matches accumulate: the
    posterior variance update has a fixed point independent of the data,
    s* = (tau^2 + sqrt(tau^4 + 4 tau^2 beta^2)) / 2, and the fan-out puts
    every strategy in a match every round, so all of them reach it."""
    t = BayesianEloTracker()
    names = [f"s{i}" for i in range(6)]
    for r in range(200):
        a = names[r % len(names)]
        for b in names:
            if b != a:
                t.record_strategy_match(a, b, 0.0)
    tau2, beta2 = t.tau**2, t.beta**2
    s_star = (tau2 + math.sqrt(tau2 * tau2 + 4 * tau2 * beta2)) / 2
    for n in names:
        assert abs(t._strategy_sigma_sq[n] - s_star) < 1.0
    ks = {round(t.strategy_stats(n)["k"], 6) for n in names}
    assert len(ks) == 1
