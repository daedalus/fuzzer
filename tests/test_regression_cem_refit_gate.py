"""CEM refit ran on every call once the elite set held 10 inputs.

``maybe_refit`` returned early only when the interval was unspent *and* the
elite set was small, so a full elite set refit every call and
``refit_interval`` (adapted by ``_adapt_interval``) never gated anything.
Under --hail-mary that was ~13 s per 3k execs. Enough elites now only
triggers the first fit early.
"""

from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

INTERVAL = 50
ELITES = 15


def _fitted() -> MonteCarloScheduler:
    mc = MonteCarloScheduler(refit_interval=INTERVAL)
    for i in range(ELITES):
        mc.add_elite(bytes([65 + i % 3] * 8), score=i)
    mc.maybe_refit()
    return mc


def test_regression_cem_refit_gate():
    """Second call inside the interval keeps the fitted distribution."""
    mc = _fitted()
    fitted = mc.byte_freq
    mc.maybe_refit()
    assert mc.byte_freq is fitted


def test_first_fit_is_early():
    """Falsification: enough elites still fit before the first interval."""
    mc = _fitted()
    assert mc.cem_fitted
    assert mc.execs_since_refit == 0


def test_refits_once_interval_spent():
    """Adversarial: the gate delays refits, it must not stop them."""
    mc = _fitted()
    fitted = mc.byte_freq
    for _ in range(mc.refit_interval):
        mc.maybe_refit()
    assert mc.byte_freq is not fitted
