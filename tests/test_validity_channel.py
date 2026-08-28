"""Zest's second fitness channel: validity, not only coverage.

A coverage-only fuzzer cannot tell "reached new code" from "reached new
code in the error path". Zest (ISSTA'19) splits the two: the harness
reports whether the input was accepted by the parser, and coverage
reached on *valid* runs is tracked in its own map. An input that is
valid and covers something no valid input covered before is saved even
when its total coverage is old -- which is the only way past a syntax
check into the semantic stages behind it.

The channel is opt-in (``--reject-code``): without a harness convention
for rejection every exit code is UNKNOWN and the channel stays inert.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.validity import (
    VALID_SEED_BONUS,
    Validity,
    ValidityChannel,
)
from fuzzer_tool.services.seed_picker import SeedPicker

REJECT = 66


@pytest.fixture
def channel() -> ValidityChannel:
    return ValidityChannel(reject_code=REJECT)


class TestClassification:
    def test_reject_code_is_invalid(self, channel):
        assert channel.classify(REJECT) is Validity.INVALID

    def test_clean_exit_is_valid(self, channel):
        assert channel.classify(0) is Validity.VALID

    @pytest.mark.parametrize("rc", [-1, -2, -11, 1, 127])
    def test_everything_else_is_unknown(self, channel, rc):
        """A crash, a timeout, or an unexplained code says nothing about
        whether the parser accepted the input."""
        assert channel.classify(rc) is Validity.UNKNOWN

    def test_disabled_channel_classifies_nothing(self):
        c = ValidityChannel(reject_code=None)
        assert not c.enabled
        assert c.classify(0) is Validity.UNKNOWN
        assert c.classify(REJECT) is Validity.UNKNOWN


class TestValidCoverage:
    def test_valid_run_with_fresh_edges_is_new_valid_coverage(self, channel):
        assert channel.record(Validity.VALID, {1, 2, 3}) is True

    def test_the_same_valid_edges_twice_are_not_new(self, channel):
        channel.record(Validity.VALID, {1, 2, 3})
        assert channel.record(Validity.VALID, {1, 2}) is False

    def test_invalid_run_never_reports_valid_coverage(self, channel):
        """The falsification: brand-new edges, but reached while invalid."""
        assert channel.record(Validity.INVALID, {9, 10, 11}) is False
        # ...and they must not be banked either, or the next valid input to
        # reach them would be scored as covering nothing new.
        assert channel.record(Validity.VALID, {9, 10, 11}) is True

    def test_unknown_run_never_reports_valid_coverage(self, channel):
        assert channel.record(Validity.UNKNOWN, {4, 5}) is False

    def test_valid_run_with_no_edges_reports_nothing(self, channel):
        assert channel.record(Validity.VALID, set()) is False
        assert channel.record(Validity.VALID, None) is False

    def test_rate_counts_only_classified_runs(self, channel):
        channel.record(Validity.VALID, {1})
        channel.record(Validity.VALID, {2})
        channel.record(Validity.INVALID, {3})
        channel.record(Validity.UNKNOWN, {4})  # crash: not a verdict
        assert (channel.valid_count, channel.invalid_count) == (2, 1)
        assert channel.valid_rate == pytest.approx(2 / 3)

    def test_rate_is_zero_before_any_verdict(self, channel):
        assert channel.valid_rate == 0.0


class TestAdversarial:
    def test_reject_code_colliding_with_a_signal_is_refused(self):
        """-11 as a "rejection" would read every SIGSEGV as a parser verdict."""
        with pytest.raises(ValueError):
            ValidityChannel(reject_code=-11)

    def test_reject_code_out_of_exit_status_range_is_refused(self):
        with pytest.raises(ValueError):
            ValidityChannel(reject_code=256)

    def test_reject_code_zero_is_refused(self):
        """0 is how a harness says it accepted the input."""
        with pytest.raises(ValueError):
            ValidityChannel(reject_code=0)

    def test_valid_map_is_bounded(self, channel):
        for i in range(ValidityChannel.MAX_VALID_EDGES + 1000):
            channel.record(Validity.VALID, {i})
        assert len(channel.valid_edges) <= ValidityChannel.MAX_VALID_EDGES

    def test_bonus_is_a_boost_not_a_veto(self):
        """An invalid seed keeps its weight: validity ranks, never excludes."""
        assert VALID_SEED_BONUS > 1.0


class TestSeedWeighting:
    """The scheduler half: a valid seed outranks an equally-covered one."""

    def _weight(self, meta: dict) -> float:
        return SeedPicker.__dict__["_weight_validity"](None, meta, 1.0, None)

    def test_valid_seed_is_boosted(self):
        assert self._weight({"valid": True}) == pytest.approx(VALID_SEED_BONUS)

    def test_invalid_seed_keeps_its_weight(self):
        assert self._weight({"valid": False}) == 1.0

    def test_unclassified_seed_keeps_its_weight(self):
        """--reject-code unset: no seed carries the key, nothing shifts."""
        assert self._weight({}) == 1.0


class TestCliWiring:
    """The flag has to reach the Fuzzer, or the channel is dead code."""

    def test_reject_code_is_parsed_and_forwarded(self):
        import inspect

        from fuzzer_tool.cli import commands
        from fuzzer_tool.services.fuzzer import Fuzzer

        assert "reject_code" in inspect.signature(Fuzzer.__init__).parameters
        src = inspect.getsource(commands)
        assert "--reject-code" in src
        assert "reject_code=getattr(args" in src
