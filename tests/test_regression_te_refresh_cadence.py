"""TE causal-map refresh ran every round once its history was full.

The "every 100 observations" gate read ``len(_te_input_history) % 100``; the
history is capped at 500, so from then on the length stayed 500 and the gate
fired every round -- ~1s of transfer entropy per round under --hail-mary.
"""

from unittest.mock import MagicMock

from fuzzer_tool.services.fuzz_round import FuzzRound

HISTORY_MAX = 500
REFRESH_EVERY = 100


def _te_fuzzer(observed: int) -> MagicMock:
    f = MagicMock()
    f._get_current_edge_set.return_value = {1, 2}
    f._te_history_max = HISTORY_MAX
    kept = min(observed, HISTORY_MAX)
    f._te_input_history = [b"x"] * kept
    f._te_edge_history = [{1}] * kept
    f._te_obs = observed
    return f


def _rounds(f: MagicMock, n: int) -> None:
    r = FuzzRound(f, b"seed")
    for _ in range(n):
        r._record_te()


def test_regression_te_refresh_cadence():
    """Full history: one refresh per REFRESH_EVERY rounds, not one per round."""
    f = _te_fuzzer(HISTORY_MAX)
    _rounds(f, REFRESH_EVERY)
    assert f._update_te_causal_map.call_count == 1
    assert f._update_causal_sector.call_count == 1
    assert len(f._te_input_history) == HISTORY_MAX


def test_filling_history_keeps_cadence():
    """Falsification: before the cap, refreshes land every REFRESH_EVERY rounds."""
    f = _te_fuzzer(0)
    _rounds(f, 3 * REFRESH_EVERY)
    assert f._update_te_causal_map.call_count == 3


def test_round_without_edges_is_not_counted():
    """Adversarial: an edgeless round neither records nor advances the cadence."""
    f = _te_fuzzer(REFRESH_EVERY - 1)
    f._get_current_edge_set.return_value = set()
    _rounds(f, REFRESH_EVERY)
    assert f._te_obs == REFRESH_EVERY - 1
    f._update_te_causal_map.assert_not_called()
