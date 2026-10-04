"""RejectStats wiring: FuzzRound feeds it, the report prints it."""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.core.reject_stats import NUM_BINS, RejectStats
from fuzzer_tool.core.validity import ValidityChannel
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.report import _reject_report

REJECT = 66


def _fuzzer(reject_code=REJECT, ops=("ins",)):
    return SimpleNamespace(
        _validity=ValidityChannel(reject_code),
        _validity_admits=0,
        _reject_stats=RejectStats(),
        _last_ops_used=list(ops),
        _current_edges_cache=frozenset({1, 2}),
    )


def _round(f, parent, mutant, rc):
    r = FuzzRound(f, parent)
    r._mutated = mutant
    r._returncode = rc
    r._valid_coverage()
    return r


def test_round_records_invalid_by_op_and_position():
    f = _fuzzer()
    _round(f, bytes(32), bytes(16) + b"\x01" + bytes(16), REJECT)
    (row,) = f._reject_stats.op_rows()
    assert (row.name, row.invalid, row.valid) == ("ins", 1, 0)
    assert f._reject_stats.bin_rows()[0].bin == NUM_BINS // 2


def test_round_records_valid_append_in_last_bin():
    f = _fuzzer(ops=("tail_append",))
    parent = bytes(32)
    _round(f, parent, parent + b"\x01", 0)
    assert f._reject_stats.bin_rows()[0].bin == NUM_BINS - 1


def test_crash_and_timeout_are_not_recorded():
    f = _fuzzer()
    for flag in ("_is_crash", "_is_timeout"):
        r = FuzzRound(f, b"ab")
        r._mutated = b"ac"
        r._returncode = REJECT
        setattr(r, flag, True)
        r._valid_coverage()
    assert f._reject_stats.op_rows() == []


def test_disabled_channel_records_nothing():
    f = _fuzzer(reject_code=None)
    _round(f, b"ab", b"ac", REJECT)
    assert f._reject_stats.op_rows() == []


def test_report_empty_when_channel_off_or_no_data():
    assert _reject_report(_fuzzer(reject_code=None)) == ""
    assert _reject_report(_fuzzer()) == ""


def test_report_lists_ops_and_position_bins():
    f = _fuzzer()
    parent = bytes(32)
    _round(f, parent, bytes(16) + b"\x01" + bytes(16), REJECT)
    _round(f, parent, parent + b"\x01", 0)
    out = _reject_report(f)
    assert "Rejection" in out
    assert "ins" in out
    assert "50%" in out
    assert "87%" in out


def test_report_survives_missing_stats_attribute():
    f = _fuzzer()
    del f._reject_stats
    assert _reject_report(f) == ""
