"""available(): reuse mutate()'s context; fold steady gates into the format memo.

``mutate()`` builds the round's ``MutationContext`` and ``available()`` built a
second one for its class-based mutators (largest single callee in a fuzzgoat
cProfile). The steady gates' cached results were also fanned out slot by slot
on every call; they now ride in the memoized format mask, refreshed with the
gate cache every ``PRED_REFRESH_CALLS`` calls. Ops and RNG draws unchanged.
"""

import pytest

from fuzzer_tool.core import operator_registry as reg
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY, OperatorSpec
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.operator_env import make_minimal_fuzzer as make_fuzzer

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
_PLAIN = b"plain text"
_PROBE = "steady_fold_probe_op"
# Calls between flips of the probe gate: not a divisor of the refresh period,
# so flips land both inside and on epoch boundaries.
_FLIP_EVERY = 23


@pytest.fixture
def engine_fuzzer(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "corpus"
    (corpus / "seeds").mkdir(parents=True)
    (tmp_path / "crashes").mkdir()
    (corpus / "seeds" / "seed1").write_bytes(_PLAIN)
    f = Fuzzer(
        "/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=4096,
    )
    yield f
    if f._cmplog:
        f._cmplog.stop()


@pytest.fixture
def probe():
    """A steady (input-free) gate reading ``f._probe_on``."""
    REGISTRY.register(
        OperatorSpec(
            name=_PROBE,
            category="bit",
            handler_name="_op_x",
            available=lambda f, _data: getattr(f, "_probe_on", False),
        )
    )
    yield _PROBE
    REGISTRY._ops.pop(_PROBE, None)


def _count_contexts(monkeypatch) -> dict:
    calls = {"n": 0}
    real = MutationContext.from_fuzzer.__func__

    def counting(cls, f):
        calls["n"] += 1
        return real(cls, f)

    monkeypatch.setattr(MutationContext, "from_fuzzer", classmethod(counting))
    return calls


def test_regression_mutate_builds_context_once(engine_fuzzer, monkeypatch):
    calls = _count_contexts(monkeypatch)
    engine_fuzzer._operators.mutate(_PLAIN)

    assert calls["n"] == 1


def test_direct_available_still_builds_context(monkeypatch):
    """Falsification: a caller without a round context still gets a fresh one."""
    calls = _count_contexts(monkeypatch)
    f = make_fuzzer(seed=1)
    REGISTRY.available(f, _PLAIN)

    assert calls["n"] >= 1


def _sequence(memo: bool, seed: int) -> list:
    """available() over many calls while the probe gate flips; memo cleared when off."""
    f = make_fuzzer(seed=seed)
    f._rng = RandPool(seed)
    inputs = [_PLAIN, _PLAIN, _PNG, _PLAIN, _PNG, _PNG]
    out = []
    for k in range(reg.PRED_REFRESH_CALLS * 4):
        f._probe_on = (k // _FLIP_EVERY) % 2 == 1
        if not memo:
            f._format_memo = None
        out.append(REGISTRY.available(f, inputs[k % len(inputs)]))
    return out


def test_steady_fold_matches_unmemoized(probe):
    """Adversarial: gate flips inside and across epochs; same ops as the slow path."""
    for seed in range(3):
        want = _sequence(False, seed)
        assert _sequence(False, seed) == want  # control (Hard Rule 46)
        got = _sequence(True, seed)
        assert got == want
        assert any(probe in ops for ops in got) and any(probe not in ops for ops in got)


def test_steady_fanout_once_per_epoch(monkeypatch):
    """Falsification: memo hits inside one gate epoch skip the steady fan-out."""
    calls = {"n": 0}
    real = REGISTRY._apply_steady

    def counting(*a):
        calls["n"] += 1
        return real(*a)

    monkeypatch.setattr(REGISTRY, "_apply_steady", counting)
    f = make_fuzzer(seed=2)
    f._rng = RandPool(2)
    # Warm-up: the first call creates the memo table, the second its entry.
    REGISTRY.available(f, _PLAIN)
    REGISTRY.available(f, _PLAIN)
    warm = calls["n"]
    for _ in range(reg.PRED_REFRESH_CALLS // 2):
        REGISTRY.available(f, _PLAIN)

    assert calls["n"] == warm
