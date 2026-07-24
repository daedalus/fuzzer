"""Unit tests for core/distance.py — AFLGo directed distance computation."""

import pytest

from fuzzer_tool.core.distance import TargetDistance


class TestTargetDistanceUnit:
    """Test TargetDistance logic without requiring a real ELF binary."""

    def _make_td(self, functions=None, call_graph=None, distances=None, targets=None):
        """Build a TargetDistance with pre-populated internals for unit testing."""
        td = TargetDistance.__new__(TargetDistance)
        td.target = "/fake/binary"
        td.target_names = targets or []
        td.target_addrs = set()
        td.functions = functions or {}
        td.addr_to_func = {}
        td.call_graph = call_graph or {}
        td._distances = distances or {}
        td._bb_distances = {}
        td._loaded = True
        td._entry_addr = 0
        td._text_start = 0
        td._text_end = 0xFFFFFFFF
        td._base_addr = 0
        td._func_starts_np = None
        td._func_ends_np = None
        td._func_dists_np = None
        # Build addr_to_func from functions
        for fname, (start, _end) in td.functions.items():
            td.addr_to_func[start] = fname
        return td

    def test_addr_to_function_exact(self):
        td = self._make_td(
            functions={
                "main": (0x1000, 0x1100),
                "helper": (0x2000, 0x2100),
            }
        )
        assert td._addr_to_function(0x1000) == "main"
        assert td._addr_to_function(0x2050) == "helper"

    def test_addr_to_function_outside(self):
        td = self._make_td(functions={"main": (0x1000, 0x1100)})
        assert td._addr_to_function(0x5000) is None

    def test_addr_to_function_boundary(self):
        td = self._make_td(functions={"a": (0x1000, 0x1200), "b": (0x1200, 0x1400)})
        assert td._addr_to_function(0x11FF) == "a"
        assert td._addr_to_function(0x1200) == "b"

    def test_resolve_targets_by_name(self):
        td = self._make_td(
            functions={"target_func": (0x3000, 0x3100), "other": (0x4000, 0x4100)},
            targets=["target_func"],
        )
        td._resolve_targets()
        assert 0x3000 in td.target_addrs

    def test_resolve_targets_by_hex(self):
        td = self._make_td(targets=["0x5000"])
        td._resolve_targets()
        assert 0x5000 in td.target_addrs

    def test_resolve_targetsSubstring(self):
        td = self._make_td(
            functions={"__wrap_foo": (0x6000, 0x6100)},
            targets=["foo"],
        )
        td._resolve_targets()
        assert 0x6000 in td.target_addrs

    def test_reachable_from_linear(self):
        td = self._make_td(
            call_graph={
                "a": {"b"},
                "b": {"c"},
                "c": set(),
            }
        )
        assert td._reachable_from("a") == {"a", "b", "c"}

    def test_reachable_from_cycle(self):
        td = self._make_td(
            call_graph={
                "a": {"b"},
                "b": {"a", "c"},
                "c": set(),
            }
        )
        result = td._reachable_from("a")
        assert result == {"a", "b", "c"}

    def test_reachable_from_isolated(self):
        td = self._make_td(call_graph={"a": set(), "b": set()})
        assert td._reachable_from("a") == {"a"}

    def test_bb_distance_known_function(self):
        td = self._make_td(
            functions={"main": (0x1000, 0x1100)},
            distances={"main": 3.0},
        )
        assert td.bb_distance(0x1050) == 3.0

    def test_bb_distance_caches_result(self):
        td = self._make_td(
            functions={"main": (0x1000, 0x1100)},
            distances={"main": 2.0},
        )
        d1 = td.bb_distance(0x1050)
        d2 = td.bb_distance(0x1050)
        assert d1 == d2 == 2.0
        assert 0x1050 in td._bb_distances

    def test_bb_distance_unknown_function_heuristic(self):
        td = self._make_td(functions={"main": (0x1000, 0x1100)})
        # Address near a known function
        d = td.bb_distance(0x1100)
        assert 0 < d <= 20.0

    def test_seed_distance_empty_trace(self):
        td = self._make_td()
        assert td.seed_distance(set()) == 20.0

    def test_seed_distance_single_edge(self):
        td = self._make_td(
            functions={"main": (0x1000, 0x1100)},
            distances={"main": 5.0},
        )
        trace = {(0x0, 0x1050)}
        assert td.seed_distance(trace) == 5.0

    def test_seed_distance_average(self):
        td = self._make_td(
            functions={"a": (0x1000, 0x1100), "b": (0x2000, 0x2100)},
            distances={"a": 2.0, "b": 8.0},
        )
        trace = {(0x0, 0x1050), (0x1050, 0x2050)}
        avg = td.seed_distance(trace)
        assert avg == pytest.approx(5.0)

    def test_seed_distance_deduplicates_bbs(self):
        td = self._make_td(
            functions={"main": (0x1000, 0x1100)},
            distances={"main": 4.0},
        )
        # Same bb hit twice — should count once
        trace = {(0x0, 0x1050), (0x1050, 0x1050)}
        assert td.seed_distance(trace) == 4.0

    def test_max_distance_no_functions(self):
        td = self._make_td()
        assert td.max_distance == 10.0

    def test_max_distance_with_functions(self):
        td = self._make_td(distances={"a": 3.0, "b": 7.0})
        assert td.max_distance == 8.0  # max + 1

    def test_is_target_true(self):
        td = self._make_td(
            functions={"vuln": (0x3000, 0x3100)},
        )
        td.target_addrs = {0x3000}
        assert td.is_target(0x3050) is True

    def test_is_target_false_not_in_target(self):
        td = self._make_td(
            functions={"safe": (0x4000, 0x4100)},
        )
        td.target_addrs = {0x3000}
        assert td.is_target(0x4050) is False

    def test_is_target_false_unknown_addr(self):
        td = self._make_td(functions={})
        td.target_addrs = {0x3000}
        assert td.is_target(0x9999) is False

    def test_load_nonexistent_file(self):
        td = TargetDistance("/nonexistent/binary", targets=["main"])
        assert td.load() is False

    def test_load_not_elf(self, tmp_path):
        fake = tmp_path / "not_elf"
        fake.write_bytes(b"this is not an ELF")
        td = TargetDistance(str(fake), targets=["main"])
        assert td.load() is False

    def test_load_too_short(self, tmp_path):
        fake = tmp_path / "tiny"
        fake.write_bytes(b"\x7fELF")
        td = TargetDistance(str(fake), targets=["main"])
        assert td.load() is False


class TestDistanceAlgorithm:
    """Test the reverse-BFS distance algorithm with synthetic call graphs.

    Verifies that distance-to-target is computed correctly:
    - Functions that call target directly get distance 1
    - Functions on dead-end branches get penalty distance
    - Multiple targets work correctly
    """

    def _make_td_with_graph(self, functions, call_graph, target_addrs, addr_to_func):
        """Build a TargetDistance with a synthetic call graph."""
        td = TargetDistance.__new__(TargetDistance)
        td.functions = functions
        td.call_graph = call_graph
        td._distances = {}
        td.target_addrs = target_addrs
        td._addr_to_function = addr_to_func
        td._compute_distances()
        return td

    def test_dead_end_branch_gets_penalty(self):
        """B-F on dead-end branch should get penalty, not progressive distances."""
        td = self._make_td_with_graph(
            functions={f: None for f in ["main", "A", "B", "C", "D", "E", "F", "T"]},
            call_graph={
                "main": {"A"},
                "A": {"B", "T"},
                "B": {"C"},
                "C": {"D"},
                "D": {"E"},
                "E": {"F"},
            },
            target_addrs={0x1000},
            addr_to_func=lambda addr: "T" if addr == 0x1000 else None,
        )
        # T is target
        assert td._distances["T"] == 0.0
        # A calls T directly
        assert td._distances["A"] == 1.0
        # main calls A
        assert td._distances["main"] == 2.0
        # B-F are on dead-end branch (can't reach T) — all get penalty
        for f in ["B", "C", "D", "E", "F"]:
            assert td._distances[f] > 5.0, f"{f} should have penalty distance"

    def test_indirect_path_to_target(self):
        """B -> C -> T should give B distance 2, C distance 1."""
        td = self._make_td_with_graph(
            functions={f: None for f in ["main", "A", "B", "C", "T", "D"]},
            call_graph={
                "main": {"A", "B"},
                "A": {"T"},
                "B": {"C"},
                "C": {"T", "D"},
            },
            target_addrs={0x1000},
            addr_to_func=lambda addr: "T" if addr == 0x1000 else None,
        )
        assert td._distances["T"] == 0.0
        assert td._distances["A"] == 1.0  # A -> T
        assert td._distances["C"] == 1.0  # C -> T
        assert td._distances["B"] == 2.0  # B -> C -> T
        assert td._distances["main"] == 2.0  # main -> A -> T (or main -> B -> C -> T)
        assert td._distances["D"] > 5.0  # D can't reach T

    def test_multiple_targets(self):
        """Multiple targets: A->T1, B->T2, both should have distance 1."""
        td = self._make_td_with_graph(
            functions={f: None for f in ["main", "A", "B", "T1", "T2"]},
            call_graph={
                "main": {"A", "B"},
                "A": {"T1"},
                "B": {"T2"},
            },
            target_addrs={0x1000, 0x2000},
            addr_to_func=lambda addr: {0x1000: "T1", 0x2000: "T2"}.get(addr),
        )
        assert td._distances["T1"] == 0.0
        assert td._distances["T2"] == 0.0
        assert td._distances["A"] == 1.0
        assert td._distances["B"] == 1.0
        assert td._distances["main"] == 2.0

    def test_no_targets_gives_uniform_distance(self):
        """If no targets found, all functions should get distance 1."""
        td = self._make_td_with_graph(
            functions={"main": None, "A": None},
            call_graph={"main": {"A"}},
            target_addrs=set(),
            addr_to_func=lambda addr: None,
        )
        assert td._distances["main"] == 1.0
        assert td._distances["A"] == 1.0

    def test_target_is_zero(self):
        """Target function itself should always be distance 0."""
        td = self._make_td_with_graph(
            functions={"T": None},
            call_graph={},
            target_addrs={0x1000},
            addr_to_func=lambda addr: "T" if addr == 0x1000 else None,
        )
        assert td._distances["T"] == 0.0

    def test_diamond_graph(self):
        """Diamond: main -> A -> B -> T, main -> C -> B. B should be distance 1."""
        td = self._make_td_with_graph(
            functions={f: None for f in ["main", "A", "B", "C", "T"]},
            call_graph={
                "main": {"A", "C"},
                "A": {"B"},
                "C": {"B"},
                "B": {"T"},
            },
            target_addrs={0x1000},
            addr_to_func=lambda addr: "T" if addr == 0x1000 else None,
        )
        assert td._distances["T"] == 0.0
        assert td._distances["B"] == 1.0  # B -> T
        assert td._distances["A"] == 2.0  # A -> B -> T
        assert td._distances["C"] == 2.0  # C -> B -> T
        assert td._distances["main"] == 3.0  # main -> A -> B -> T
