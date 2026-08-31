"""Tests for the XOR-linear ("bitflip family") classification added to
operator_categories.py for handover doc item 3 -- see
docs/handover/handover_skittercreek_tailslayer_port.md.
"""

from __future__ import annotations

from fuzzer_tool.core.operator_categories import (
    OPERATOR_CATEGORIES,
    XOR_LINEAR_OPS,
    is_xor_linear,
)


def test_xor_linear_ops_contains_bit_and_byte_flip():
    assert XOR_LINEAR_OPS == frozenset({"bit_flip", "byte_flip"})


def test_is_xor_linear_true_for_family_members():
    assert is_xor_linear("bit_flip") is True
    assert is_xor_linear("byte_flip") is True


def test_is_xor_linear_false_for_non_linear_operators():
    # Same "bit" category as bit_flip, but not fixed-width XOR-linear in
    # the sense compose_linear_runs relies on (shifts/permutes bits).
    assert is_xor_linear("bit_rotate") is False
    assert is_xor_linear("bit_offset_flip") is False
    # Length-changing / structural: never linear.
    assert is_xor_linear("byte_insert") is False
    assert is_xor_linear("block_delete") is False


def test_is_xor_linear_false_for_unknown_name():
    assert is_xor_linear("not_a_real_operator") is False


def test_xor_linear_ops_is_subset_of_registered_bit_category():
    # bit_flip and byte_flip must actually be registered operators (in the
    # "bit"/"byte" categories respectively) -- a typo here would silently
    # make compose_linear_runs never fire.
    all_registered = {op for ops in OPERATOR_CATEGORIES.values() for op in ops}
    assert XOR_LINEAR_OPS <= all_registered
