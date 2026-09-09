"""Regression test: causal_sector wiring reuses TE's observation cadence.

Phase B2 of docs/handover/handover_RoRd.md — ``StatsReporter.update_causal_sector``
must feed ``Fuzzer._causal_sector`` from the same ``_te_edge_history`` TE
already accumulates, via ``services.te_position.edge_sets_to_flow``, without
recomputing full pairwise transfer entropy anywhere else.
"""

from fuzzer_tool.core.causal_sector import CausalSectorGraph
from fuzzer_tool.services.stats import StatsReporter


class FakeTE:
    def __init__(self, return_val=0.05):
        self.return_val = return_val
        self.calls = 0

    def transfer_entropy(self, source, target):
        self.calls += 1
        return self.return_val


class FakeFuzzer:
    """Minimal duck-typed stand-in: only the attributes update_causal_sector reads."""

    def __init__(self, te, edge_history, causal_sector):
        self._te = te
        self._te_edge_history = edge_history
        self._causal_sector = causal_sector


class TestUpdateCausalSector:
    def test_observes_flow_into_graph(self):
        te = FakeTE(return_val=0.05)
        history = [{1, 2}, {1}, {1, 2}, {2}]
        sector = CausalSectorGraph()
        f = FakeFuzzer(te, history, sector)

        StatsReporter(f).update_causal_sector()

        assert len(sector) > 0
        assert sector.observation_windows == 1

    def test_no_flow_leaves_graph_empty_but_does_not_error(self):
        te = FakeTE(return_val=0.0)  # every TE value below threshold
        history = [{1, 2}, {1}, {1, 2}]
        sector = CausalSectorGraph()
        f = FakeFuzzer(te, history, sector)

        StatsReporter(f).update_causal_sector()

        assert len(sector) == 0
        assert sector.observation_windows == 0  # end_window() only runs when flow is non-empty

    def test_reuses_te_edge_history_not_a_fresh_computation(self):
        # Two calls over the same growing history should keep calling the
        # same TE object rather than constructing a second one -- the whole
        # point of reusing TE's cadence instead of a parallel TE instance.
        te = FakeTE(return_val=0.02)
        history = [{1, 2}, {1}, {1, 2}]
        sector = CausalSectorGraph()
        f = FakeFuzzer(te, history, sector)

        StatsReporter(f).update_causal_sector()
        calls_after_first = te.calls
        history.append({2})
        StatsReporter(f).update_causal_sector()

        assert te.calls > calls_after_first
