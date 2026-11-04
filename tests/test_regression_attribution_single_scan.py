"""``_record_attribution`` scanned the SHM table once per consumer.

Shapley, MI and TE each called ``_get_current_edge_set()`` (each builds a fresh
~thousand-element set from the table); they only read it, so one scan serves
all three.
"""

from unittest.mock import MagicMock

from fuzzer_tool.services.fuzz_round import FuzzRound


def _fuzzer(shapley: bool, mi: bool, te: bool, edges: set[int]) -> MagicMock:
    f = MagicMock()
    f._use_shapley, f._use_mi, f._use_transfer_entropy = shapley, mi, te
    f._get_current_edge_set.return_value = edges
    f._last_ops_used = ["op"]
    f.exec_count = 1
    f._te_history_max = 500
    f._te_input_history, f._te_edge_history, f._te_obs = [], [], 0
    return f


def _record(f: MagicMock) -> None:
    r = FuzzRound(f, b"seed")
    r._effective = None
    r._record_attribution()


def test_one_scan_feeds_all_consumers():
    edges = {1, 2, 3}
    f = _fuzzer(True, True, True, edges)
    _record(f)
    assert f._get_current_edge_set.call_count == 1
    assert f._shapley.record.call_args.args[2] is edges
    assert f._mi.record.call_args.args[1] is edges
    assert f._te_edge_history == [edges]
    assert edges == {1, 2, 3}


def test_no_scan_when_every_consumer_is_off():
    f = _fuzzer(False, False, False, {1})
    _record(f)
    f._get_current_edge_set.assert_not_called()


def test_empty_edges_still_records_zero_outcome_for_shapley_only():
    f = _fuzzer(True, True, True, set())
    _record(f)
    f._shapley.record.assert_called_once_with({"op"}, 0, set())
    f._mi.record.assert_not_called()
    assert f._te_edge_history == []
