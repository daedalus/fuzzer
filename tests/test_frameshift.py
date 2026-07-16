"""Tests for core/frameshift.py — automatic length field tracking."""

import pytest

from fuzzer_tool.core.frameshift import FrameShift, Relation


class TestRelation:
    def test_defaults(self):
        r = Relation(pos=0, size=4, anchor=0, insert_point=10, val=0)
        assert r.le is True
        assert r.enabled is True

    def test_custom_endian(self):
        r = Relation(pos=0, size=2, anchor=0, insert_point=5, val=100, le=False)
        assert r.le is False


class TestFrameShiftInit:
    def test_default_max_relations(self):
        fs = FrameShift()
        assert fs.max_relations == 64
        assert fs.relations == []
        assert fs.blocked_points == set()

    def test_custom_max(self):
        fs = FrameShift(max_relations=10)
        assert fs.max_relations == 10


class TestAddRelation:
    def test_add_under_limit(self):
        fs = FrameShift(max_relations=3)
        r = Relation(pos=0, size=4, anchor=0, insert_point=10, val=0)
        assert fs.add_relation(r) is True
        assert len(fs.relations) == 1

    def test_add_at_capacity(self):
        fs = FrameShift(max_relations=2)
        fs.add_relation(Relation(pos=0, size=2, anchor=0, insert_point=5, val=0))
        fs.add_relation(Relation(pos=10, size=2, anchor=0, insert_point=15, val=0))
        r3 = Relation(pos=20, size=2, anchor=0, insert_point=25, val=0)
        assert fs.add_relation(r3) is False
        assert len(fs.relations) == 2

    def test_blocked_points_populated(self):
        fs = FrameShift()
        r = Relation(pos=5, size=4, anchor=0, insert_point=10, val=0)
        fs.add_relation(r)
        assert fs.blocked_points == {5, 6, 7, 8}


class TestOnInsert:
    def test_insert_before_pos_shifts_pos(self):
        fs = FrameShift()
        # anchor=20 → insert at 5 is outside [20, 15] (empty range), val unchanged
        r = Relation(pos=10, size=4, anchor=20, insert_point=15, val=5)
        fs.add_relation(r)
        fs.on_insert(idx=5, data_size=3)
        assert r.pos == 13  # 10 + 3
        assert r.val == 5  # not in anchor..insert_point range

    def test_insert_inside_field_disables(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=0, insert_point=20, val=5)
        fs.add_relation(r)
        fs.on_insert(idx=12, data_size=2)  # inside [10, 14)
        assert r.enabled is False

    def test_insert_in_anchor_range_increments_val(self):
        fs = FrameShift()
        r = Relation(pos=0, size=4, anchor=0, insert_point=20, val=10)
        fs.add_relation(r)
        fs.on_insert(idx=5, data_size=3)  # 5 is in [0, 20]
        assert r.val == 13  # 10 + 3

    def test_insert_before_anchor_shifts_anchor(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=15, insert_point=30, val=10)
        fs.add_relation(r)
        fs.on_insert(idx=5, data_size=3)
        assert r.anchor == 18  # 15 + 3

    def test_insert_after_pos_does_not_shift(self):
        fs = FrameShift()
        r = Relation(pos=5, size=2, anchor=0, insert_point=10, val=0)
        fs.add_relation(r)
        fs.on_insert(idx=20, data_size=3)
        assert r.pos == 5  # unchanged

    def test_insert_at_pos_shifts(self):
        fs = FrameShift()
        r = Relation(pos=5, size=2, anchor=0, insert_point=10, val=0)
        fs.add_relation(r)
        fs.on_insert(idx=5, data_size=3)
        assert r.pos == 8  # 5 + 3

    def test_ignore_invalid_false_returns_false(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=0, insert_point=20, val=5)
        fs.add_relation(r)
        result = fs.on_insert(idx=12, data_size=2, ignore_invalid=False)
        assert result is False
        assert r.enabled is True  # not disabled

    def test_insert_after_insert_point_does_not_shift(self):
        fs = FrameShift()
        r = Relation(pos=5, size=2, anchor=0, insert_point=10, val=0)
        fs.add_relation(r)
        fs.on_insert(idx=15, data_size=3)
        assert r.insert_point == 10  # unchanged

    def test_insert_at_insert_point_shifts(self):
        fs = FrameShift()
        r = Relation(pos=5, size=2, anchor=0, insert_point=10, val=0)
        fs.add_relation(r)
        fs.on_insert(idx=10, data_size=3)
        assert r.insert_point == 13  # 10 + 3


class TestOnDelete:
    def test_delete_before_pos_shifts_pos(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=0, insert_point=20, val=10)
        fs.add_relation(r)
        fs.on_delete(idx=5, data_size=3)
        assert r.pos == 7  # 10 - 3

    def test_delete_overlapping_field(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=0, insert_point=20, val=10)
        fs.add_relation(r)
        fs.on_delete(idx=11, data_size=3)  # overlaps [10, 14)
        # No crash — _rel_on_remove returns True but caller doesn't check

    def test_delete_in_anchor_range_decrements_val(self):
        fs = FrameShift()
        r = Relation(pos=0, size=4, anchor=0, insert_point=20, val=10)
        fs.add_relation(r)
        fs.on_delete(idx=5, data_size=3)  # overlaps [0, 20)
        assert r.val == 7  # 10 - 3

    def test_delete_before_anchor_shifts_anchor(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=15, insert_point=30, val=10)
        fs.add_relation(r)
        fs.on_delete(idx=5, data_size=3)
        assert r.anchor == 12  # 15 - 3

    def test_delete_skips_disabled(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=0, insert_point=20, val=10)
        r.enabled = False
        fs.add_relation(r)
        fs.on_delete(idx=5, data_size=3)
        assert r.pos == 10  # unchanged


class TestApplyToBuffer:
    def test_le_2byte(self):
        fs = FrameShift()
        r = Relation(pos=4, size=2, anchor=0, insert_point=10, val=0x0201, le=True)
        fs.add_relation(r)
        buf = bytearray(10)
        fs.apply_to_buffer(buf)
        assert buf[4] == 0x01  # low byte
        assert buf[5] == 0x02  # high byte

    def test_be_2byte(self):
        fs = FrameShift()
        r = Relation(pos=4, size=2, anchor=0, insert_point=10, val=0x0201, le=False)
        fs.add_relation(r)
        buf = bytearray(10)
        fs.apply_to_buffer(buf)
        assert buf[4] == 0x02  # high byte first
        assert buf[5] == 0x01  # low byte

    def test_4byte(self):
        fs = FrameShift()
        r = Relation(pos=0, size=4, anchor=0, insert_point=20, val=0x04030201, le=True)
        fs.add_relation(r)
        buf = bytearray(20)
        fs.apply_to_buffer(buf)
        assert buf[0] == 0x01
        assert buf[1] == 0x02
        assert buf[2] == 0x03
        assert buf[3] == 0x04

    def test_skips_disabled(self):
        fs = FrameShift()
        r = Relation(pos=0, size=2, anchor=0, insert_point=5, val=99, enabled=False)
        fs.add_relation(r)
        buf = bytearray(10)
        fs.apply_to_buffer(buf)
        assert buf[0] == 0
        assert buf[1] == 0


class TestSaveRestore:
    def test_save_and_restore(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=5, insert_point=20, val=42)
        fs.add_relation(r)

        fs.save()
        # Mutate
        r.pos = 50
        r.val = 99
        r.anchor = 0
        r.insert_point = 100

        fs.restore()
        assert r.pos == 10
        assert r.val == 42
        assert r.anchor == 5
        assert r.insert_point == 20
        assert r.enabled is True

    def test_restore_reenables_disabled(self):
        fs = FrameShift()
        r = Relation(pos=10, size=4, anchor=5, insert_point=20, val=42)
        fs.add_relation(r)
        fs.save()
        r.enabled = False
        fs.restore()
        assert r.enabled is True


class TestDiscoverRelations:
    def test_empty_data(self):
        fs = FrameShift()
        assert fs.discover_relations(b"", lambda d: 0) == 0

    def test_discovers_relations(self):
        fs = FrameShift()
        # Every insertion changes the path → candidates become relations
        data = bytes(128)

        def exec_fn(d):
            return sum(d)

        count = fs.discover_relations(data, exec_fn, max_execs=100)
        assert count > 0

    def test_respects_max_relations(self):
        fs = FrameShift(max_relations=3)
        data = bytes(128)

        def exec_fn(d):
            return sum(d)

        count = fs.discover_relations(data, exec_fn, max_relations=2, max_execs=200)
        assert count <= 2

    def test_respects_max_execs(self):
        fs = FrameShift()
        data = bytes(256)
        exec_count = [0]

        def exec_fn(d):
            exec_count[0] += 1
            return sum(d)

        fs.discover_relations(data, exec_fn, max_execs=10)
        assert exec_count[0] <= 10 + 1
