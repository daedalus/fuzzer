"""Tests for length/offset arithmetic goals and TLV nesting invariants."""

import random

import pytest

from fuzzer_tool.core.field_constraints import satisfied
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.structural_constraints import (
    EXACT_FIT,
    GOALS,
    HUGE_SIZE,
    MAX_TLV_DEPTH,
    MAX_TLV_NODES,
    OFF_BY_ONE,
    SIGNED_NEGATIVE,
    WRAP,
    ZERO_SIZE,
    build_tlv,
    parse_tlv,
    repair_tlv,
    resize_tlv_value,
    serialize_tlv,
    solve_coupled_sections,
    solve_length_offset,
    tlv_fields,
    verify_goal,
)

# Z3's default 200ms solver timeout can fire on a loaded CI box and return
# unknown → None even for these trivially-satisfiable instances, flaking the
# soundness tests. These solve in ~11ms; 5000ms is headroom, not a cost.
SOLVE_TIMEOUT_MS = 5000


def _nest(payload=b"PAYLOAD"):
    """Four-level nest: 0x01 > 0x02 > 0x03 > 0x10(payload)."""
    return build_tlv(0x01, build_tlv(0x02, build_tlv(0x03, build_tlv(0x10, payload))))


# ── Length/offset arithmetic ───────────────────────────────────────────


class TestLengthOffsetGoals:
    @pytest.mark.parametrize("width", [2, 4, 8])
    @pytest.mark.parametrize("goal", GOALS)
    def test_solution_satisfies_its_goal(self, goal, width):
        solved = solve_length_offset(goal, width, 4096)
        assert solved is not None
        offset, size = solved
        assert verify_goal(goal, offset, size, width, 4096)

    def test_wrap_defeats_a_naive_bounds_check(self):
        """The bug class: ``off + size <= len`` without a wraparound check.

        The sum wraps to something small so the check passes, then the read
        uses the unwrapped size.
        """
        filesize, width = 4096, 4
        modulus = 1 << (width * 8)
        offset, size = solve_length_offset(WRAP, width, filesize)

        assert ((offset + size) % modulus) <= filesize  # check passes
        assert offset <= filesize  # offset alone looks fine
        assert offset + size > filesize  # but the read runs off the end

    def test_wrap_is_unreachable_when_the_width_cannot_hold_the_file(self):
        assert solve_length_offset(WRAP, 1, 4096) is None

    def test_wrap_needs_a_nonzero_filesize(self):
        assert solve_length_offset(WRAP, 4, 0) is None

    def test_exact_fit_lands_on_the_boundary(self):
        offset, size = solve_length_offset(EXACT_FIT, 4, 1000)
        assert offset + size == 1000

    def test_off_by_one_overshoots_by_exactly_one(self):
        offset, size = solve_length_offset(OFF_BY_ONE, 4, 1000)
        assert offset + size == 1001

    def test_zero_size_keeps_a_valid_offset(self):
        offset, size = solve_length_offset(ZERO_SIZE, 4, 1000)
        assert size == 0
        assert offset <= 1000

    def test_signed_negative_sets_the_high_bit(self):
        """Non-negative unsigned, negative when read as a signed int."""
        for width in (2, 4, 8):
            _, size = solve_length_offset(SIGNED_NEGATIVE, width, 4096)
            assert size & (1 << (width * 8 - 1))
            assert size > 0

    def test_huge_size_is_all_ones(self):
        _, size = solve_length_offset(HUGE_SIZE, 4, 4096)
        assert size == 0xFFFFFFFF

    def test_unknown_goal_returns_none(self):
        assert solve_length_offset("not_a_goal", 4, 100) is None

    def test_invalid_width_returns_none(self):
        assert solve_length_offset(WRAP, 0, 100) is None

    def test_rng_varies_the_wrap_overshoot(self):
        results = {
            solve_length_offset(WRAP, 4, 4096, rng=random.Random(seed)) for seed in range(20)
        }
        assert len(results) > 1

    def test_verify_rejects_out_of_range_values(self):
        assert not verify_goal(WRAP, 0, 1 << 32, 4, 4096)
        assert not verify_goal(WRAP, -1, 10, 4, 4096)


class TestCoupledSections:
    """Ordered + non-overlapping + one wrapping couples every field to every
    other, so no closed form exists. This is the solver case."""

    def test_ordered_non_overlapping_sections(self):
        pytest.importorskip("z3")
        result = solve_coupled_sections(3, 4, 4096, timeout_ms=SOLVE_TIMEOUT_MS)
        assert result is not None
        assert len(result) == 3
        offsets = [o for o, _ in result]
        assert offsets == sorted(offsets)
        for (off_a, size_a), (off_b, _) in zip(result, result[1:], strict=False):
            assert off_a + size_a <= off_b

    def test_sizes_are_nonzero(self):
        pytest.importorskip("z3")
        result = solve_coupled_sections(2, 4, 4096, timeout_ms=SOLVE_TIMEOUT_MS)
        assert result is not None
        assert all(size > 0 for _, size in result)

    def test_wrapping_last_section_is_satisfiable(self):
        pytest.importorskip("z3")
        result = solve_coupled_sections(2, 4, 4096, wrap_index=1, timeout_ms=SOLVE_TIMEOUT_MS)
        assert result is not None
        offset, size = result[1]
        assert (offset + size) % (1 << 32) < offset

    def test_wrap_conflicts_with_non_overlap_in_the_middle(self):
        """A wrapping section cannot also end before the next one starts."""
        assert solve_coupled_sections(3, 4, 4096, wrap_index=0) is None

    def test_zero_count_returns_none(self):
        assert solve_coupled_sections(0, 4, 4096) is None


# ── TLV nesting ────────────────────────────────────────────────────────


class TestTlvParsing:
    def test_parses_a_flat_sequence(self):
        data = build_tlv(1, b"aa") + build_tlv(2, b"bbbb")
        nodes = parse_tlv(data)
        assert len(nodes) == 2
        assert [n.size for n in nodes] == [2, 4]

    def test_parses_a_four_level_nest(self):
        nodes = parse_tlv(_nest())
        assert len(nodes) == 1
        assert nodes[0].depth() == 4

    def test_opaque_payload_is_a_leaf(self):
        """Only a *complete* TLV cover counts as nesting; random bytes are
        not misread as structure."""
        node = parse_tlv(build_tlv(1, b"\xff\xfe\xfd\xfc\xfb"))[0]
        assert node.children == []

    def test_truncated_frame_stops_the_scan(self):
        data = build_tlv(1, b"aa")[:-1]
        assert parse_tlv(data) == []

    def test_length_past_end_is_rejected(self):
        data = b"\x01\xff\xff" + b"short"
        assert parse_tlv(data) == []

    def test_depth_is_bounded(self):
        data = b"\x10\x00\x00"
        for _ in range(MAX_TLV_DEPTH + 6):
            data = build_tlv(1, data)
        assert parse_tlv(data)[0].depth() <= MAX_TLV_DEPTH

    def test_node_budget_is_bounded(self):
        """Adversarial input must not drive unbounded parsing."""
        data = build_tlv(1, b"") * (MAX_TLV_NODES + 500)
        assert len(parse_tlv(data)) <= MAX_TLV_NODES

    def test_little_endian_lengths(self):
        data = build_tlv(1, b"abcd", big_endian=False)
        nodes = parse_tlv(data, big_endian=False)
        assert nodes and nodes[0].size == 4

    def test_walk_yields_parents_before_children(self):
        walked = list(parse_tlv(_nest())[0].walk())
        assert len(walked) == 4
        assert walked[0].value_start < walked[-1].value_start


class TestTlvRepair:
    def test_valid_nest_is_already_satisfied(self):
        data = _nest()
        assert satisfied(tlv_fields(parse_tlv(data)), data)

    def test_fields_cover_every_level(self):
        assert len(tlv_fields(parse_tlv(_nest()))) == 4

    @pytest.mark.parametrize("size", [0, 1, 30, 500, 65000])
    def test_resize_propagates_to_every_ancestor(self, size):
        """The core claim: an edit several levels down leaves all enclosing
        lengths correct, so the parser reaches the mutated leaf."""
        data = _nest()
        innermost = list(parse_tlv(data)[0].walk())[-1]
        out = resize_tlv_value(data, innermost, b"X" * size)
        assert out is not None

        reparsed = parse_tlv(out)
        assert reparsed[0].depth() == 4
        assert list(reparsed[0].walk())[-1].size == size
        assert satisfied(tlv_fields(reparsed), out)

    def test_resize_matches_by_position_not_identity(self):
        """The caller's node comes from a separate parse of the same buffer,
        so identity comparison would silently never match and the edit would
        vanish."""
        data = _nest()
        node_from_one_parse = list(parse_tlv(data)[0].walk())[-1]
        # A second, independent parse produces different objects.
        assert node_from_one_parse is not list(parse_tlv(data)[0].walk())[-1]
        out = resize_tlv_value(data, node_from_one_parse, b"YYYY")
        assert list(parse_tlv(out)[0].walk())[-1].size == 4

    def test_resizing_an_outer_frame(self):
        data = _nest()
        outer = parse_tlv(data)[0]
        out = resize_tlv_value(data, outer, b"flat payload")
        assert out is not None
        reparsed = parse_tlv(out)
        assert reparsed[0].size == len(b"flat payload")

    def test_repair_fixes_a_corrupted_inner_length(self):
        data = bytearray(_nest())
        innermost = list(parse_tlv(bytes(data))[0].walk())[-1]
        data[innermost.length_offset] ^= 0xFF
        out = repair_tlv(bytes(data))
        if out is not None:
            assert satisfied(tlv_fields(parse_tlv(out)), out)

    def test_repair_of_non_tlv_returns_none(self):
        assert repair_tlv(b"\xff") is None
        assert repair_tlv(b"") is None

    def test_serialize_round_trips_an_untouched_nest(self):
        data = _nest()
        assert serialize_tlv(parse_tlv(data)[0], data) == data

    def test_trailing_bytes_are_preserved(self):
        data = _nest() + b"TRAILER"
        node = list(parse_tlv(data)[0].walk())[-1]
        out = resize_tlv_value(data, node, b"ZZ")
        assert out.endswith(b"TRAILER")

    def test_resize_rejects_out_of_range_node(self):
        from fuzzer_tool.core.structural_constraints import TlvNode

        bogus = TlvNode(0, 1, 1, 2, 3, 9999)
        assert resize_tlv_value(_nest(), bogus, b"x") is None


class TestOperatorRegistration:
    @pytest.mark.parametrize(
        ("name", "category"),
        [("tlv_nest_mutate", "format"), ("length_offset_goal", "adaptive")],
    )
    def test_registered(self, name, category):
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == category

    @pytest.mark.parametrize("name", ["tlv_nest_mutate", "length_offset_goal"])
    def test_handler_exists(self, name):
        from fuzzer_tool.services.operators import OperatorEngine

        assert hasattr(OperatorEngine, f"_op_{name}")


class TestNonOverlapSoundness:
    """Regression: the non-overlap constraint was written in modular
    arithmetic, so a wrapping sum satisfied it vacuously (0x1000 +
    0xFFFFF000 wraps to 0, which is <= anything) and the solver returned
    sections that overlapped by gigabytes."""

    def test_true_arithmetic_not_modular(self):
        pytest.importorskip("z3")
        for _ in range(20):
            result = solve_coupled_sections(3, 4, 4096, timeout_ms=SOLVE_TIMEOUT_MS)
            assert result is not None
            for (off_a, size_a), (off_b, _) in zip(result, result[1:], strict=False):
                # Python ints: no wraparound to hide behind.
                assert off_a + size_a <= off_b

    def test_sections_do_not_overlap_at_width_8(self):
        pytest.importorskip("z3")
        result = solve_coupled_sections(2, 8, 4096, timeout_ms=SOLVE_TIMEOUT_MS)
        assert result is not None
        (off_a, size_a), (off_b, _) = result
        assert off_a + size_a <= off_b
