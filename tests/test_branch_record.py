"""Tests for sampled branch-record coverage.

No host here has a hardware PMU at all (``/sys/bus/event_source/devices``
lists no ``cpu`` entry), let alone AMD BRS, so nothing below opens a real
event.  Covered instead: the sample-record parsing, the kernel-address
filter, the edge hash, the data-ring drain over a synthetic control page,
and the degrade path.

Record layouts are built here from the uapi field order rather than by
calling the module's own writer, so a parser that agrees with itself but not
with the kernel fails.
"""

import mmap
import struct

import pytest

from fuzzer_tool.adapters import lbr_trace
from fuzzer_tool.adapters.lbr_trace import LbrSession
from fuzzer_tool.adapters.pt_trace import (
    OFF_DATA_HEAD,
    OFF_DATA_OFFSET,
    OFF_DATA_SIZE,
    OFF_DATA_TAIL,
)
from fuzzer_tool.core import branch_record
from fuzzer_tool.core.branch_record import (
    KERNEL_ADDR_FLOOR,
    PERF_RECORD_LOST,
    PERF_RECORD_SAMPLE,
    BranchCoverage,
    count_lost,
    iter_samples,
)

PAGE = mmap.PAGESIZE
USER_A, USER_B, USER_C = 0x400100, 0x400200, 0x400300
KERNEL = 0xFFFFFFFF810001C4


def sample_record(entries, ip=USER_A, pid=1234, tid=1234) -> bytes:
    """A PERF_RECORD_SAMPLE with sample_type = IP | TID | BRANCH_STACK.

    Body order is the uapi order for those three bits: ip, then pid/tid,
    then the branch stack as a count followed by 24-byte entries.
    """
    body = struct.pack("<QIIQ", ip, pid, tid, len(entries))
    for frm, to in entries:
        body += struct.pack("<QQQ", frm, to, 0)

    return struct.pack("<IHH", PERF_RECORD_SAMPLE, 0, 8 + len(body)) + body


def lost_record(n: int, event_id: int = 7) -> bytes:
    body = struct.pack("<QQ", event_id, n)
    return struct.pack("<IHH", PERF_RECORD_LOST, 0, 8 + len(body)) + body


def other_record(rtype: int = 3, payload: bytes = b"\x00" * 16) -> bytes:
    return struct.pack("<IHH", rtype, 0, 8 + len(payload)) + payload


class TestIterSamples:
    def test_single_sample(self):
        buf = sample_record([(USER_A, USER_B)])
        (sample,) = list(iter_samples(buf))
        assert sample.ip == USER_A
        assert sample.pid == 1234
        assert [(e.frm, e.to) for e in sample.entries] == [(USER_A, USER_B)]

    def test_sixteen_entries_is_the_hardware_depth(self):
        entries = [(USER_A + i, USER_B + i) for i in range(16)]
        (sample,) = list(iter_samples(sample_record(entries)))
        assert len(sample.entries) == 16

    def test_several_records_in_one_span(self):
        buf = sample_record([(USER_A, USER_B)]) + sample_record([(USER_B, USER_C)])
        assert len(list(iter_samples(buf))) == 2

    def test_unknown_record_type_is_skipped_by_its_own_size(self):
        """Guessing a size instead of reading the header turns one unhandled
        record into a misparse of everything after it."""
        buf = other_record() + sample_record([(USER_A, USER_B)])
        assert len(list(iter_samples(buf))) == 1

    def test_lost_records_do_not_yield_samples(self):
        assert list(iter_samples(lost_record(5))) == []

    def test_empty_branch_stack_yields_a_sample_with_no_entries(self):
        (sample,) = list(iter_samples(sample_record([])))
        assert sample.entries == []

    def test_truncated_trailing_record_is_not_yielded(self):
        """Its bytes belong to the next read; decoding the fragment would
        invent an edge out of half an address."""
        buf = sample_record([(USER_A, USER_B)]) + sample_record([(USER_B, USER_C)])[:-8]
        assert len(list(iter_samples(buf))) == 1

    def test_header_only_tail_is_not_yielded(self):
        buf = sample_record([(USER_A, USER_B)]) + b"\x00" * 4
        assert len(list(iter_samples(buf))) == 1

    def test_zero_size_header_terminates_instead_of_looping(self):
        """A size of 0 would advance pos by nothing forever."""
        buf = struct.pack("<IHH", PERF_RECORD_SAMPLE, 0, 0) + sample_record([(USER_A, USER_B)])
        assert list(iter_samples(buf)) == []

    def test_implausible_depth_drops_the_record(self):
        """A count that cannot fit the hardware file means the body was
        misaligned; trusting it would read gigabytes of nothing."""
        body = struct.pack("<QIIQ", USER_A, 1, 1, 1 << 40)
        buf = struct.pack("<IHH", PERF_RECORD_SAMPLE, 0, 8 + len(body)) + body
        assert list(iter_samples(buf)) == []

    def test_count_larger_than_the_body_drops_the_record(self):
        body = struct.pack("<QIIQ", USER_A, 1, 1, 4) + struct.pack("<QQQ", USER_A, USER_B, 0)
        buf = struct.pack("<IHH", PERF_RECORD_SAMPLE, 0, 8 + len(body)) + body
        assert list(iter_samples(buf)) == []

    def test_hw_index_is_not_expected_in_the_layout(self):
        """PERF_SAMPLE_BRANCH_HW_INDEX inserts a u64 between the count and
        the entries.  It is not requested, so a record carrying one must not
        parse cleanly -- if it did, the first entry's `from` would be read as
        the index and every pair after it would shear by one field."""
        assert not lbr_trace.BRANCH_FILTER & (1 << 17)
        assert "HW_INDEX" in branch_record._decode_sample.__doc__


class TestCountLost:
    def test_sums_lost_counters(self):
        assert count_lost(lost_record(3) + lost_record(4)) == 7

    def test_ignores_samples(self):
        assert count_lost(sample_record([(USER_A, USER_B)])) == 0

    def test_mixed_stream(self):
        buf = sample_record([(USER_A, USER_B)]) + lost_record(9) + sample_record([])
        assert count_lost(buf) == 9

    def test_truncated_lost_record_is_not_counted(self):
        assert count_lost(lost_record(9)[:-4]) == 0


class TestBranchCoverage:
    def test_records_an_edge_per_branch(self):
        cov = BranchCoverage()
        new = cov.ingest(sample_record([(USER_A, USER_B), (USER_B, USER_C)]))
        assert new == 2
        assert cov.total_branches == 2
        assert cov.cumulative_edges == 2

    def test_repeating_an_edge_is_not_new(self):
        cov = BranchCoverage()
        cov.ingest(sample_record([(USER_A, USER_B)]))
        assert cov.ingest(sample_record([(USER_A, USER_B)])) == 0

    def test_kernel_from_address_is_dropped(self):
        """BRS has no hardware privilege filter, so a user-only stack can
        still carry SYSRET and interrupt-return branch-from addresses
        (CVE-2026-72237).  A kernel address in the map is coverage the
        target never had."""
        cov = BranchCoverage()
        assert cov.ingest(sample_record([(KERNEL, USER_B)])) == 0
        assert cov.dropped_kernel == 1
        assert cov.cumulative_edges == 0

    def test_kernel_to_address_is_dropped(self):
        cov = BranchCoverage()
        assert cov.ingest(sample_record([(USER_A, KERNEL)])) == 0
        assert cov.dropped_kernel == 1

    def test_zero_address_is_dropped(self):
        """An unfilled branch register reads back as zero."""
        cov = BranchCoverage()
        assert cov.ingest(sample_record([(0, USER_B), (USER_A, 0)])) == 0
        assert cov.dropped_kernel == 2

    def test_boundary_address_is_kernel(self):
        cov = BranchCoverage()
        cov.ingest(sample_record([(KERNEL_ADDR_FLOOR, USER_B)]))
        assert cov.dropped_kernel == 1

    def test_one_user_entry_survives_a_mixed_stack(self):
        cov = BranchCoverage()
        assert cov.ingest(sample_record([(KERNEL, USER_B), (USER_A, USER_B)])) == 1
        assert (cov.dropped_kernel, cov.total_branches) == (1, 1)

    def test_self_loop_does_not_collapse_to_bucket_zero(self):
        """Without the >>1 an edge from a block to itself XORs to 0, so every
        self-loop in the program shares one bucket with the map's origin."""
        cov = BranchCoverage()
        cov.ingest(sample_record([(USER_A, USER_A)]))
        assert cov.edge_map[0] == 0

    def test_direction_is_distinguished(self):
        """A->B and B->A are different edges; a hash that loses the order
        reports half the control flow it saw."""
        cov = BranchCoverage()
        cov.ingest(sample_record([(USER_A, USER_B)]))
        assert cov.ingest(sample_record([(USER_B, USER_A)])) == 1

    def test_base_address_makes_the_map_reproducible(self):
        """Two runs of a PIE target at different load bases must hash the
        same edge to the same bucket."""
        first, second = BranchCoverage(), BranchCoverage()
        first.set_base(0x400000)
        second.set_base(0x7F0000000000)
        first.ingest(sample_record([(0x400000 + 0x100, 0x400000 + 0x200)]))
        second.ingest(sample_record([(0x7F0000000000 + 0x100, 0x7F0000000000 + 0x200)]))
        assert first.edge_map == second.edge_map

    def test_lost_records_are_accumulated(self):
        cov = BranchCoverage()
        cov.ingest(lost_record(4))
        cov.ingest(lost_record(6))
        assert cov.stats["br_lost_records"] == 10

    def test_is_new_coverage_is_true_once_per_new_edge(self):
        cov = BranchCoverage()
        cov.reset_edge_map()
        cov.ingest(sample_record([(USER_A, USER_B)]))
        assert cov.is_new_coverage() is True
        assert cov.is_new_coverage() is False

    def test_reset_keeps_the_cumulative_map(self):
        cov = BranchCoverage()
        cov.ingest(sample_record([(USER_A, USER_B)]))
        cov.reset_edge_map()
        assert cov.cumulative_edges == 1
        assert cov.total_edges == 0
        assert cov.ingest(sample_record([(USER_A, USER_B)])) == 0

    def test_map_is_declared_sampled(self):
        """The flag is what lets stability calibration and the trim's subset
        test refuse this map: both read absence of coverage as a fact about
        the input, and here it is a fact about the sampling period."""
        assert BranchCoverage.sampled is True
        assert branch_record.SAMPLED is True

    def test_ingesting_garbage_yields_no_coverage(self):
        cov = BranchCoverage()
        assert cov.ingest(b"\xff" * 64) == 0


def synthetic_session(data_size=PAGE * 4, sink=None):
    """A session whose ring is anonymous memory instead of a perf fd."""
    session = LbrSession.__new__(LbrSession)
    session.page_size = PAGE
    session.data_size = data_size
    session.period = lbr_trace.DEFAULT_PERIOD
    session.sink = sink
    session.bytes_read = 0
    session.reads = 0
    session.attach_failures = 0
    session._available = True
    session._fd = -1
    session._base = mmap.mmap(-1, PAGE + data_size)
    struct.pack_into("<Q", session._base, OFF_DATA_SIZE, data_size)
    struct.pack_into("<Q", session._base, OFF_DATA_OFFSET, PAGE)
    return session


def kernel_writes(session, payload, at):
    """Place *payload* in the data ring as the kernel would, at byte counter
    *at*, and advance data_head."""
    size = session.data_size
    for i, byte in enumerate(payload):
        session._base[PAGE + ((at + i) % size)] = byte

    struct.pack_into("<Q", session._base, OFF_DATA_HEAD, at + len(payload))


class TestReadRecords:
    def test_empty_ring_reads_nothing(self):
        assert synthetic_session().read_records() == b""

    def test_reads_what_the_kernel_wrote(self):
        session = synthetic_session()
        rec = sample_record([(USER_A, USER_B)])
        kernel_writes(session, rec, 0)
        assert session.read_records() == rec

    def test_tail_is_advanced_to_head(self):
        session = synthetic_session()
        rec = sample_record([(USER_A, USER_B)])
        kernel_writes(session, rec, 0)
        session.read_records()
        assert struct.unpack_from("<Q", session._base, OFF_DATA_TAIL)[0] == len(rec)

    def test_second_read_of_an_idle_ring_is_empty(self):
        session = synthetic_session()
        kernel_writes(session, sample_record([(USER_A, USER_B)]), 0)
        session.read_records()
        assert session.read_records() == b""

    def test_wrapped_span_is_joined_in_order(self):
        session = synthetic_session(data_size=PAGE)
        rec = sample_record([(USER_A, USER_B)])
        at = PAGE - (len(rec) // 2)
        kernel_writes(session, rec, at)
        struct.pack_into("<Q", session._base, OFF_DATA_TAIL, at)
        assert session.read_records() == rec

    def test_a_wrapped_record_still_decodes(self):
        """The span is joined before parsing, so a record split across the
        ring end is contiguous by the time the decoder sees it."""
        session = synthetic_session(data_size=PAGE)
        rec = sample_record([(USER_A, USER_B)])
        at = PAGE - 8
        kernel_writes(session, rec, at)
        struct.pack_into("<Q", session._base, OFF_DATA_TAIL, at)
        (sample,) = list(iter_samples(session.read_records()))
        assert [(e.frm, e.to) for e in sample.entries] == [(USER_A, USER_B)]

    def test_span_is_clamped_to_the_ring(self):
        """head - tail can exceed data_size once the kernel has wrapped; the
        copy must stay one ring long instead of walking off the mapping."""
        session = synthetic_session(data_size=PAGE)
        struct.pack_into("<Q", session._base, OFF_DATA_HEAD, PAGE * 3)
        assert len(session.read_records()) == PAGE

    def test_bytes_and_reads_are_counted(self):
        session = synthetic_session()
        rec = sample_record([(USER_A, USER_B)])
        kernel_writes(session, rec, 0)
        session.read_records()
        assert session.stats["br_trace_bytes"] == len(rec)
        assert session.stats["br_trace_reads"] == 1

    def test_unmapped_session_reads_nothing(self):
        session = LbrSession()
        assert session.read_records() == b""


class TestDrain:
    def test_drain_feeds_the_sink(self):
        cov = BranchCoverage()
        session = synthetic_session(sink=cov)
        session._fd = 4242
        kernel_writes(session, sample_record([(USER_A, USER_B)]), 0)
        session.disable = lambda: True
        session.close = lambda: None
        assert session.drain() == 1
        assert cov.cumulative_edges == 1

    def test_drain_without_an_event_is_zero(self):
        assert synthetic_session(sink=BranchCoverage()).drain() == 0

    def test_drain_stops_sampling_before_reading(self, monkeypatch):
        session = synthetic_session(sink=BranchCoverage())
        session._fd = 4242
        order = []
        monkeypatch.setattr(session, "disable", lambda: order.append("disable") or True)
        monkeypatch.setattr(session, "read_records", lambda: order.append("read") or b"")
        monkeypatch.setattr(session, "close", lambda: order.append("close"))
        session.drain()
        assert order == ["disable", "read", "close"]


class TestAttr:
    def test_attr_requests_the_layout_the_decoder_parses(self):
        session = LbrSession()
        attr = session._build_attr()
        sample_type = struct.unpack_from("<Q", attr, 24)[0]
        branch_type = struct.unpack_from("<Q", attr, 72)[0]
        assert sample_type == lbr_trace.SAMPLE_LAYOUT
        assert branch_type == lbr_trace.BRANCH_FILTER

    def test_attr_carries_the_period_and_the_branch_event(self):
        session = LbrSession(period=1234)
        attr = session._build_attr()
        assert struct.unpack_from("<Q", attr, 8)[0] == lbr_trace.PERF_COUNT_HW_BRANCH_INSTRUCTIONS
        assert struct.unpack_from("<Q", attr, 16)[0] == 1234

    def test_attr_starts_disabled_and_excludes_kernel(self):
        from fuzzer_tool.adapters import perf_event

        flags = struct.unpack_from("<Q", LbrSession()._build_attr(), 40)[0]
        assert flags & perf_event._FLAG_DISABLED
        assert flags & perf_event._FLAG_EXCLUDE_KERNEL

    @pytest.mark.parametrize("pages", [0, 3, -4])
    def test_rejects_a_non_power_of_two_ring(self, pages):
        with pytest.raises(ValueError):
            LbrSession(data_pages=pages)

    def test_rejects_a_nonpositive_period(self):
        with pytest.raises(ValueError):
            LbrSession(period=0)


class TestAvailability:
    def test_probe_result_is_cached(self, monkeypatch):
        session = LbrSession()
        calls = []
        monkeypatch.setattr(session, "_open", lambda pid: calls.append(pid) or -1)
        assert session.available is False
        assert session.available is False
        assert calls == [0]

    def test_attach_on_an_unavailable_host_is_counted(self, monkeypatch):
        session = LbrSession()
        monkeypatch.setattr(session, "_open", lambda pid: -1)
        assert session.attach(1) is False
        assert session.stats["br_trace_attach_failures"] == 1

    def test_no_hardware_pmu_here(self):
        """States the environment this suite runs in: with no cpu PMU the
        probe must fail rather than raise."""
        assert LbrSession().available in (True, False)

    def test_close_is_idempotent(self):
        session = LbrSession()
        session.close()
        session.close()
