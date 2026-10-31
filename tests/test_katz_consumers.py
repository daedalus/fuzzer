"""Katz work only runs when something consumes it.

A default run (``--schedule base``, no Elo, no branch arm) built the node
channel, read its bitmap after every exec and computed ``katz_energy`` for
every scored round, and nothing used either (~9% of fuzz_one on fuzzgoat).
The channel is still built (resume contract and saved state unchanged); the
per-exec observe and the energy lookup are bound once, or not at all.
"""

from __future__ import annotations

from fuzzer_tool.services.fuzz_round import FuzzRound
from tests.test_regression_katz_resume_contract import (  # noqa: F401 - autouse fixture
    _FakeKatz,
    _fuzzer,
    _instrumented_target,
    _with_katz,
)


class _CountingKatz(_FakeKatz):
    def __init__(self):
        super().__init__()
        self.observed = 0

    def observe(self, seed_key=None) -> bool:
        self.observed += 1
        return True

    def seed_energy(self, _key) -> float:
        return 0.5

    def enable_branch(self) -> None:
        pass


def _built(tmp_path, monkeypatch, **kw):
    katz = _CountingKatz()
    _with_katz(monkeypatch, katz)
    return _fuzzer(tmp_path, **kw), katz


def test_default_run_skips_observe_and_energy(tmp_path, monkeypatch):
    f, katz = _built(tmp_path, monkeypatch)

    FuzzRound(f, b"x")._feed_katz()

    assert f._katz_channel is katz  # still built: resume contract unchanged
    assert katz.observed == 0
    assert f._katz_energy_of is None


def test_falsify_katz_schedule_observes_and_scores(tmp_path, monkeypatch):
    f, katz = _built(tmp_path, monkeypatch, schedule="katz")

    FuzzRound(f, b"x")._feed_katz()

    assert katz.observed == 1
    assert f._katz_energy_of("k") == 0.5


def test_adversarial_elo_arms_still_fed(tmp_path, monkeypatch):
    """Elo's katz/branch seed arms read the masks observe() builds."""
    f, katz = _built(tmp_path, monkeypatch, elo=True)

    FuzzRound(f, b"x")._feed_katz()

    assert katz.observed == 1
    assert f._katz_energy_of is None  # the power schedule still ignores it


def test_branch_scheduler_still_fed(tmp_path, monkeypatch):
    f, katz = _built(tmp_path, monkeypatch, seed_branch_scheduler=True)

    FuzzRound(f, b"x")._feed_katz()

    assert katz.observed == 1


def test_contract_unchanged_without_consumers(tmp_path, monkeypatch):
    from fuzzer_tool.services.corpus_manager import current_coverage_contract

    f, _katz = _built(tmp_path, monkeypatch)

    assert current_coverage_contract(f)["node_channel"] is True
