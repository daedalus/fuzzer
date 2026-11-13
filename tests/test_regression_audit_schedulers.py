"""Regressions from the 2026-11-13 audit: seed-scheduler ledgers grew per non-corpus parent."""

import pytest

from tests.test_os_net_scheduler_wiring import SEED_A, _real_fuzzer

_FLAGS = {
    "seed_canary_scheduler": "_seed_canary",
    "seed_round_robin_scheduler": "_seed_round_robin",
    "seed_drr_scheduler": "_seed_drr",
}
_SYNTHETIC_PARENTS = 50


@pytest.fixture
def fuzzer(tmp_path):
    return _real_fuzzer(tmp_path, **dict.fromkeys(_FLAGS, True))


def _ledgers(f):
    return [getattr(f, attr).bandit_stats() for attr in _FLAGS.values()]


def test_regression_synthetic_parents_leave_no_entry(fuzzer):
    """Markov / QEA parents are never candidates; each used to stay in the ledger forever."""
    for i in range(_SYNTHETIC_PARENTS):
        fuzzer._record_seed_floor_arms(b"synthetic-%d" % i, success=False, weight=0.0)

    assert _ledgers(fuzzer) == [{}, {}, {}]


def test_corpus_parent_reaches_every_arm(fuzzer):
    """Falsification: a live corpus parent is recorded by all three."""
    fuzzer.corpus.append(SEED_A)
    fuzzer.seed_meta[SEED_A] = {}
    fuzzer._record_seed_floor_arms(SEED_A, success=True, weight=1.0)

    key = fuzzer._seed_key(SEED_A)
    assert all(list(ledger) == [key] for ledger in _ledgers(fuzzer))


def test_disabled_arms_are_skipped(tmp_path):
    """Adversarial: no flag set, nothing built, the call is a no-op."""
    f = _real_fuzzer(tmp_path)
    f.corpus.append(SEED_A)
    f.seed_meta[SEED_A] = {}

    f._record_seed_floor_arms(SEED_A, success=True, weight=1.0)

    assert all(getattr(f, attr) is None for attr in _FLAGS.values())
