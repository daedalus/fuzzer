"""Regression: two misreports around the position-arena canary.

1. The uniform-floor check flagged pos_canary itself. The canary is built
   to lose, so every --position-arena run warned "pos_canary ... at or below
   uniform -- this proposer needs inspection" once it was rated.
2. The startup banner listed "canary" whenever the object existed, but only
   PositionArena.select reaches it, and only with --elo. --pos-canary alone,
   or --position-arena without --elo, advertised a proposer that never ran.
"""

import logging
from types import SimpleNamespace

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker
from fuzzer_tool.core.schedulers.pos_canary import PositionCanaryScheduler
from fuzzer_tool.services.fuzzer import Fuzzer, _active_position_schedulers

_UNIFORM_MU = 1500.0
_MATCHES = 5


def _rated_elo(ratings: dict[str, float]) -> BayesianEloTracker:
    elo = BayesianEloTracker(min_matches=1)
    for key, mu in ratings.items():
        elo._strategy_mu[key] = mu
        elo._strategy_match_count[key] = _MATCHES
    return elo


def _checker(elo: BayesianEloTracker) -> Fuzzer:
    f = Fuzzer.__new__(Fuzzer)
    f._elo = elo
    f._position_arena = object()
    f._pos_canary = PositionCanaryScheduler()
    f._use_seed_canary = False
    f._seed_canary = None
    return f


def _uniform_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "below uniform" in r.getMessage()]


def test_regression_pos_canary_not_flagged_below_uniform(caplog):
    """Falsification: the canary losing to uniform is its job, not a finding."""
    f = _checker(_rated_elo({"pos_uniform": _UNIFORM_MU, "pos_canary": _UNIFORM_MU - 100}))

    with caplog.at_level(logging.WARNING):
        f._check_canary_inspection()

    assert not any("canary" in m for m in _uniform_warnings(caplog))


def test_real_proposer_below_uniform_still_flagged(caplog):
    """Adversarial: skipping the canary must not silence real proposers."""
    elo = _rated_elo(
        {"pos_uniform": _UNIFORM_MU, "pos_canary": _UNIFORM_MU - 100, "pos_mi": _UNIFORM_MU - 50}
    )
    f = _checker(elo)

    with caplog.at_level(logging.WARNING):
        f._check_canary_inspection()

    flagged = _uniform_warnings(caplog)
    assert len(flagged) == 1
    assert "mi" in flagged[0] and "canary" not in flagged[0]


def _banner_f(arena: bool, elo: bool) -> SimpleNamespace:
    return SimpleNamespace(
        _pos_canary=PositionCanaryScheduler(),
        _position_arena=object() if arena else None,
        _use_elo=elo,
        _elo=object() if elo else None,
    )


def test_regression_banner_hides_idle_canary():
    """Falsification: no arena, or an arena without --elo, never runs it."""
    assert "canary" not in _active_position_schedulers(_banner_f(arena=False, elo=False))
    assert "canary" not in _active_position_schedulers(_banner_f(arena=False, elo=True))
    assert "canary" not in _active_position_schedulers(_banner_f(arena=True, elo=False))


def test_banner_shows_fielded_canary():
    """Adversarial: arena with --elo does field it."""
    assert "canary" in _active_position_schedulers(_banner_f(arena=True, elo=True))
