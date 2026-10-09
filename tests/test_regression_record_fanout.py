"""_record_schedulers builds its scheduler fan-out once per selector.

The enabled schedulers are fixed at setup; only the on-policy member moves
with ``_op_selector``. Rebuilding a ~50-slot tuple every exec was pure waste.
"""

from fuzzer_tool.services.fuzz_round import FuzzRound
from tests.test_regression_resume_state import _fuzzer

_ON = {"eps_greedy": True, "exp3": True, "ducb": True}
_REWARDS = [("havoc", True, 0.5), ("bit_flip", False, 1.0)]


def _round(f) -> FuzzRound:
    rnd = FuzzRound.__new__(FuzzRound)
    rnd._f = f
    return rnd


def _spy(f, names, log):
    """Replace each scheduler's record with one logging (name, op)."""
    for name in names:
        sched = getattr(f, f"_{name}")
        sched.record = lambda op, ok, weight=1.0, n=name: log.append((n, op, ok, weight))


def _count_builds(monkeypatch, f) -> dict:
    """Count fan-out builds for *f* only (other live fuzzers may run rounds)."""
    calls = {"n": 0}
    real = FuzzRound._fanout

    def counting(self, selector):
        calls["n"] += self._f is f
        return real(self, selector)

    monkeypatch.setattr(FuzzRound, "_fanout", counting)
    return calls


def test_fanout_built_once_per_selector(tmp_path, monkeypatch):
    """Falsification: repeated execs under one selector reuse the fan-out."""
    f = _fuzzer(tmp_path, **_ON)
    calls = _count_builds(monkeypatch, f)
    f._op_selector = "eps_greedy"

    for _ in range(5):
        _round(f)._record_schedulers(_REWARDS)
    assert calls["n"] == 1

    f._op_selector = "exp3"
    _round(f)._record_schedulers(_REWARDS)
    f._op_selector = "eps_greedy"
    _round(f)._record_schedulers(_REWARDS)
    assert calls["n"] == 2


def test_on_policy_follows_selector(tmp_path):
    """Adversarial: a cached fan-out must not leak exp3 into another round."""
    f = _fuzzer(tmp_path, **_ON)
    log: list = []
    _spy(f, ("exp3", "eps_greedy", "ducb"), log)

    f._op_selector = "exp3"
    _round(f)._record_schedulers(_REWARDS)
    f._op_selector = "eps_greedy"
    _round(f)._record_schedulers(_REWARDS)

    fed = [n for n, *_ in log]
    per_round = len(_REWARDS)
    # Declaration order: exp3 sits before eps_greedy and ducb.
    assert (
        fed
        == ["exp3"] * per_round
        + ["eps_greedy"] * per_round
        + ["ducb"] * per_round
        + ["eps_greedy"] * per_round
        + ["ducb"] * per_round
    )


def test_no_selector_feeds_off_policy_only(tmp_path):
    """Adversarial: selector None (random round) skips every on-policy arm."""
    f = _fuzzer(tmp_path, **_ON)
    log: list = []
    _spy(f, ("exp3", "eps_greedy", "ducb"), log)
    f._op_selector = None

    _round(f)._record_schedulers(_REWARDS)

    assert {n for n, *_ in log} == {"eps_greedy", "ducb"}
    assert [(op, ok, w) for _, op, ok, w in log[: len(_REWARDS)]] == _REWARDS
