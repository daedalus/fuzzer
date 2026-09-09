"""Unit tests for core.ro_rd primitives (no wiring)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.ro_rd import (
    LineageRecord,
    MutationStep,
    OrientationClass,
    classify_operator_name,
    is_invertible_operator,
    rd_reverse_lineage,
    tag_operator_record,
)


class TestClassify:
    def test_ro_markers(self):
        assert classify_operator_name("path_negation") is OrientationClass.RO
        assert classify_operator_name("smt_path_neg_z3") is OrientationClass.RO
        assert classify_operator_name("branch_invert_cmp") is OrientationClass.RO
        assert classify_operator_name("opposite_branch") is OrientationClass.RO

    def test_rd_default(self):
        assert classify_operator_name("bit_flip") is OrientationClass.RD
        assert classify_operator_name("havoc") is OrientationClass.RD
        assert classify_operator_name("dict_insert") is OrientationClass.RD

    def test_empty_neutral(self):
        assert classify_operator_name(None) is OrientationClass.NEUTRAL
        assert classify_operator_name("") is OrientationClass.NEUTRAL


class TestInvertible:
    def test_known_invertible(self):
        assert is_invertible_operator("bit_flip")
        assert is_invertible_operator("bit_flip_8")
        assert is_invertible_operator("arith_32")
        assert is_invertible_operator("interesting_16")
        assert is_invertible_operator("span_invert")

    def test_non_invertible(self):
        assert not is_invertible_operator("dict_insert")
        assert not is_invertible_operator("block_delete")
        assert not is_invertible_operator("splice")
        assert not is_invertible_operator(None)


class TestLineageRDReverse:
    def test_empty(self):
        r = rd_reverse_lineage(LineageRecord(steps=()))
        assert r.ok
        assert r.reversed_steps == ()

    def test_self_inverse_chain(self):
        steps = (
            MutationStep("bit_flip", site=0),
            MutationStep("arith_8", site=4),
            MutationStep("interesting_32", site=8),
        )
        lineage = LineageRecord(steps=steps)
        assert lineage.all_invertible()
        assert not lineage.has_ro()
        r = rd_reverse_lineage(lineage)
        assert r.ok
        assert [s.operator for s in r.reversed_steps] == [
            "interesting_32",
            "arith_8",
            "bit_flip",
        ]
        assert [s.site for s in r.reversed_steps] == [8, 4, 0]

    def test_refuses_non_invertible(self):
        lineage = LineageRecord(
            steps=(
                MutationStep("bit_flip", site=0),
                MutationStep("dict_insert", site=1),
            )
        )
        r = rd_reverse_lineage(lineage)
        assert not r.ok
        assert "non-invertible" in r.reason
        assert "dict_insert" in r.reason

    def test_refuses_ro_in_lineage(self):
        lineage = LineageRecord(
            steps=(
                MutationStep("bit_flip", site=0),
                MutationStep("path_negation", site=None),
            )
        )
        assert lineage.has_ro()
        r = rd_reverse_lineage(lineage)
        assert not r.ok
        assert "RO-class" in r.reason

    def test_orientation_profile(self):
        lineage = LineageRecord(
            steps=(
                MutationStep("bit_flip"),
                MutationStep("path_negation"),
                MutationStep("havoc"),
            )
        )
        profile = lineage.orientation_profile()
        assert profile[OrientationClass.RO] == 1
        assert profile[OrientationClass.RD] == 2


class TestTagRecord:
    def test_tag_adds_fields(self):
        tagged = tag_operator_record("bit_flip", {"reward": 1.0})
        assert tagged["orientation_class"] == "rd"
        assert tagged["invertible"] is True
        assert tagged["operator"] == "bit_flip"
        assert tagged["reward"] == 1.0

    def test_tag_does_not_mutate_input(self):
        original = {"reward": 0.5}
        tag_operator_record("path_negation", original)
        assert "orientation_class" not in original

    def test_tag_ro(self):
        tagged = tag_operator_record("path_negation")
        assert tagged["orientation_class"] == "ro"
        assert tagged["invertible"] is False
