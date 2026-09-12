"""Tests for the Intel PT AUX ring session.

This host has no ``intel_pt`` PMU, so nothing here opens a real event.  What
is covered is everything that does not need one: the ring arithmetic, the
sysfs format parsing, the drain path driven over a synthetic control page and
AUX buffer, and -- where a compiler and the uapi header are present -- the
control-page offsets against the kernel's own struct.

The synthetic drain is the substantive one: it exercises the same
``_read_u64``/``ring_chunks``/tail-publish sequence the hardware path runs,
with the kernel's side of the protocol played by the test.
"""

import mmap
import os
import shutil
import struct
import subprocess
import textwrap

import pytest

from fuzzer_tool.adapters import pt_trace
from fuzzer_tool.adapters.pt_trace import (
    PtTraceSession,
    build_config,
    read_format_bits,
    read_pmu_type,
    ring_chunks,
)

PAGE = mmap.PAGESIZE


# --------------------------------------------------------------------------
# ring_chunks
# --------------------------------------------------------------------------


def reference_bytes(buf: bytes, tail: int, head: int, size: int) -> bytes:
    """The ring read written the obvious way, byte at a time.

    Independent of ring_chunks so it can disagree with it.
    """
    return bytes(buf[i % size] for i in range(tail, head))


def test_empty_ring_yields_nothing():
    assert ring_chunks(0, 0, 4096) == []
    assert ring_chunks(9000, 9000, 4096) == []


def test_single_range_when_no_wrap():
    assert ring_chunks(100, 200, 4096) == [(100, 100)]


def test_wrap_splits_into_two_ranges():
    # 4000..4196 over a 4096-byte ring: 96 bytes to the end, then 100 more.
    assert ring_chunks(4000, 4196, 4096) == [(4000, 96), (0, 100)]


def test_full_ring_is_not_read_as_empty():
    """head - tail == size puts both pointers on the same index.  Comparing
    them as indices reads that as an empty ring and drops a whole buffer."""
    chunks = ring_chunks(0, 4096, 4096)
    assert sum(length for _, length in chunks) == 4096

    chunks = ring_chunks(2048, 2048 + 4096, 4096)
    assert sum(length for _, length in chunks) == 4096
    assert chunks == [(2048, 2048), (0, 2048)]


def test_overflow_is_rejected_not_truncated():
    """More available than the ring holds means the caller failed to clamp;
    silently returning the last size bytes would hide the data loss."""
    with pytest.raises(ValueError, match="in a 4096-byte ring"):
        ring_chunks(0, 4097, 4096)


def test_nonpositive_size_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        ring_chunks(0, 10, 0)


@pytest.mark.parametrize("size", [16, 64, 4096])
def test_chunks_reassemble_like_the_byte_at_a_time_reference(size):
    buf = bytes((i * 7 + 3) % 256 for i in range(size))
    for tail in range(0, size * 2, 3):
        for avail in (0, 1, size // 2, size - 1, size):
            head = tail + avail
            got = b"".join(buf[off : off + length] for off, length in ring_chunks(tail, head, size))
            assert got == reference_bytes(buf, tail, head, size)


def test_chunk_offsets_stay_inside_the_ring():
    size = 512
    for tail in range(0, 4 * size, 7):
        for avail in (1, 200, size):
            for off, length in ring_chunks(tail, tail + avail, size):
                assert 0 <= off < size
                assert off + length <= size


# --------------------------------------------------------------------------
# sysfs discovery
# --------------------------------------------------------------------------


def write_fake_pmu(root, type_value="8", fields=None):
    fmt = root / "format"
    fmt.mkdir(parents=True)
    if type_value is not None:
        (root / "type").write_text(type_value + "\n")

    for name, spec in (fields or {}).items():
        (fmt / name).write_text(spec + "\n")

    return str(root)


def test_pmu_type_read_from_sysfs(tmp_path):
    root = write_fake_pmu(tmp_path / "intel_pt", type_value="11")
    assert read_pmu_type(root) == 11


def test_pmu_type_absent_is_none_not_an_error(tmp_path):
    assert read_pmu_type(str(tmp_path / "nope")) is None


def test_pmu_type_garbage_is_none(tmp_path):
    root = write_fake_pmu(tmp_path / "intel_pt", type_value="not a number")
    assert read_pmu_type(root) is None


def test_format_bits_parses_single_bits_and_ranges(tmp_path):
    root = write_fake_pmu(
        tmp_path / "intel_pt",
        fields={"pt": "config:0", "branch": "config:13", "psb_period": "config:24-27"},
    )
    assert read_format_bits(root) == {"pt": (0, 0), "branch": (13, 13), "psb_period": (24, 27)}


def test_format_bits_skips_other_config_registers(tmp_path):
    """config1 and config2 are separate 64-bit words.  Folding them into
    config would set unrelated bits at the same positions."""
    root = write_fake_pmu(
        tmp_path / "intel_pt",
        fields={"pt": "config:0", "addr_range": "config1:0-63", "weird": "config2:5"},
    )
    assert read_format_bits(root) == {"pt": (0, 0)}


def test_format_bits_skips_malformed_entries(tmp_path):
    root = write_fake_pmu(
        tmp_path / "intel_pt",
        fields={
            "pt": "config:0",
            "bad": "config:x-y",
            "empty": "config:",
            "backwards": "config:9-3",
        },
    )
    assert read_format_bits(root) == {"pt": (0, 0)}


def test_format_bits_missing_directory_is_empty(tmp_path):
    assert read_format_bits(str(tmp_path / "nope")) == {}


def test_build_config_sets_the_published_positions():
    bits = {"pt": (0, 0), "branch": (13, 13), "noretcomp": (11, 11)}
    assert build_config(bits) == (1 << 0) | (1 << 13) | (1 << 11)


def test_build_config_skips_unpublished_fields(caplog):
    """A field the PMU does not have must not be guessed at a fixed bit: the
    wrong bit selects a different trace mode, which decodes cleanly into
    wrong coverage rather than failing."""
    with caplog.at_level("WARNING"):
        config = build_config({"pt": (0, 0)})

    assert config == 1
    assert "noretcomp" in caplog.text


def test_build_config_requests_branch_and_noretcomp_by_default():
    """Without branch there are no TIP packets at all; with return
    compression on, every RET target is absent from the stream that
    core.intel_pt reads."""
    assert set(pt_trace.PT_CONFIG_FIELDS) == {"pt", "branch", "noretcomp"}


# --------------------------------------------------------------------------
# session lifecycle without a PMU
# --------------------------------------------------------------------------


def test_absent_pmu_degrades_instead_of_raising(tmp_path):
    session = PtTraceSession(aux_pages=1, root=str(tmp_path / "nope"))
    assert not session.available
    assert session.open_for_pid(os.getpid()) is False
    assert session.read_trace() == b""
    assert session.enable() is False
    assert session.disable() is False
    session.close()


def test_pmu_with_no_format_fields_is_unavailable(tmp_path):
    """A type file alone is not a usable PMU: without format bits the config
    word would be zero, which asks for no tracing at all."""
    root = write_fake_pmu(tmp_path / "intel_pt", type_value="8")
    assert not PtTraceSession(aux_pages=1, root=root).available


def test_aux_pages_must_be_a_power_of_two(tmp_path):
    for bad in (0, -4, 3, 100, 513):
        with pytest.raises(ValueError, match="power of two"):
            PtTraceSession(aux_pages=bad, root=str(tmp_path))

    for good in (1, 2, 512):
        PtTraceSession(aux_pages=good, root=str(tmp_path))


def test_rejected_constructor_still_finalizes_cleanly(tmp_path):
    """__del__ runs on a half-built object.  With the buffer attributes set
    after validation, close() raised AttributeError from the finalizer, which
    Python prints and discards -- so the ValueError the caller needs arrived
    next to an unrelated traceback."""
    with pytest.raises(ValueError, match="power of two"):
        PtTraceSession(aux_pages=3, root=str(tmp_path))

    partial = PtTraceSession.__new__(PtTraceSession)
    partial._fd = -1
    partial._base = None
    partial._aux = None
    partial.close()  # must not raise


def test_default_aux_ring_is_page_aligned_and_large():
    session = PtTraceSession(root="/nonexistent")
    assert session.aux_size == pt_trace.DEFAULT_AUX_PAGES * PAGE
    assert session.aux_size % PAGE == 0


# --------------------------------------------------------------------------
# the drain path, with the test playing the kernel
# --------------------------------------------------------------------------


def synthetic_session(aux_size=4096, fields=None):
    """A session whose rings are anonymous memory instead of a perf fd."""
    session = PtTraceSession.__new__(PtTraceSession)
    session.page_size = PAGE
    session.aux_size = aux_size
    session.data_size = PAGE
    session.bytes_read = 0
    session.lost_bytes = 0
    session.overflows = 0
    session.reads = 0
    session.attach_failures = 0
    session.sink = None
    session._fd = -1
    session._pmu_type = 8
    session._format_bits = fields or {"pt": (0, 0), "branch": (13, 13), "noretcomp": (11, 11)}
    session._base = mmap.mmap(-1, PAGE * 2)
    session._aux = mmap.mmap(-1, aux_size)
    return session


def kernel_writes(session, payload, at):
    """Place *payload* in the ring as the kernel would, at byte counter *at*,
    and advance aux_head."""
    size = session.aux_size
    for i, byte in enumerate(payload):
        session._aux[(at + i) % size] = byte

    struct.pack_into("<Q", session._base, pt_trace.OFF_AUX_HEAD, at + len(payload))


def test_drain_returns_what_the_kernel_wrote():
    session = synthetic_session()
    kernel_writes(session, b"PTBYTES!", at=0)

    assert session.read_trace() == b"PTBYTES!"
    assert session.bytes_read == 8
    assert session.reads == 1


def test_drain_publishes_the_tail_so_the_kernel_can_reuse_the_space():
    session = synthetic_session()
    kernel_writes(session, b"abcd", at=0)
    session.read_trace()

    tail = struct.unpack_from("<Q", session._base, pt_trace.OFF_AUX_TAIL)[0]
    assert tail == 4


def test_second_drain_returns_only_the_new_bytes():
    session = synthetic_session()
    kernel_writes(session, b"first", at=0)
    assert session.read_trace() == b"first"

    kernel_writes(session, b"second", at=5)
    assert session.read_trace() == b"second"
    assert session.reads == 2
    assert session.bytes_read == 11


def test_drain_with_nothing_new_is_empty_and_not_counted():
    session = synthetic_session()
    kernel_writes(session, b"x", at=0)
    session.read_trace()

    assert session.read_trace() == b""
    assert session.reads == 1


def test_drain_reassembles_across_a_wrap():
    session = synthetic_session(aux_size=64)
    payload = bytes(range(32))
    # Starts 48 bytes in, so 16 bytes land at the top and 16 at the bottom.
    kernel_writes(session, payload, at=48)
    struct.pack_into("<Q", session._base, pt_trace.OFF_AUX_TAIL, 48)

    assert session.read_trace() == payload


def test_overflow_is_counted_and_the_oldest_bytes_are_dropped():
    """head - tail > size means the kernel outran us.  The read must clamp to
    the last size bytes and say so, not hand back a mis-assembled buffer."""
    session = synthetic_session(aux_size=64)
    session._aux[:] = bytes(range(64))
    struct.pack_into("<Q", session._base, pt_trace.OFF_AUX_TAIL, 0)
    struct.pack_into("<Q", session._base, pt_trace.OFF_AUX_HEAD, 100)

    data = session.read_trace()

    assert len(data) == 64
    assert session.overflows == 1
    assert session.lost_bytes == 36
    # The surviving window is counters 36..100, i.e. index 36 onward then wrap.
    assert data == bytes(range(36, 64)) + bytes(range(0, 36))


def test_stats_report_loss_separately_from_volume():
    session = synthetic_session(aux_size=64)
    struct.pack_into("<Q", session._base, pt_trace.OFF_AUX_HEAD, 100)
    session.read_trace()

    stats = session.stats
    assert stats["pt_trace_overflows"] == 1
    assert stats["pt_trace_lost_bytes"] == 36
    assert stats["pt_trace_bytes"] == 64
    assert stats["pt_aux_size"] == 64


def test_drained_bytes_feed_the_decoder():
    """End to end against core.intel_pt: a PSB written into the ring comes
    back out as a packet the decoder recognises."""
    from fuzzer_tool.core.intel_pt import PSB, PtCoverage

    session = synthetic_session()
    # PSB, then a TIP with a full 6-byte address (ipc=3 -> 0x00007fff1234).
    stream = PSB + bytes([0x0D | (3 << 5), 0x34, 0x12, 0xFF, 0x7F, 0x00, 0x00])
    kernel_writes(session, stream, at=0)

    cov = PtCoverage()
    new = cov.ingest(session.read_trace())

    assert new == 1
    assert cov.total_blocks == 1


# --------------------------------------------------------------------------
# control page offsets against the kernel's own struct
# --------------------------------------------------------------------------

OFFSET_ORACLE = textwrap.dedent(
    """
    #include <linux/perf_event.h>
    #include <stddef.h>
    #include <stdio.h>
    int main(void) {
        printf("%zu %zu %zu %zu %zu %zu %zu %zu\\n",
            offsetof(struct perf_event_mmap_page, data_head),
            offsetof(struct perf_event_mmap_page, data_tail),
            offsetof(struct perf_event_mmap_page, data_offset),
            offsetof(struct perf_event_mmap_page, data_size),
            offsetof(struct perf_event_mmap_page, aux_head),
            offsetof(struct perf_event_mmap_page, aux_tail),
            offsetof(struct perf_event_mmap_page, aux_offset),
            offsetof(struct perf_event_mmap_page, aux_size));
        return 0;
    }
    """
)


def test_control_page_offsets_match_the_uapi_header(tmp_path):
    """Derive the offsets rather than re-assert the module's own numbers.

    The reserved gap before data_head is 928 bytes; miscounting it reads
    every pointer off the wrong word, and a wrong aux_head reads as an empty
    ring forever -- no error, just no coverage.
    """
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None or not os.path.exists("/usr/include/linux/perf_event.h"):
        pytest.skip("needs a C compiler and the uapi perf_event header")

    src = tmp_path / "off.c"
    src.write_text(OFFSET_ORACLE)
    exe = tmp_path / "off"
    build = subprocess.run([cc, "-o", str(exe), str(src)], capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"oracle did not build: {build.stderr.strip()[:200]}")

    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    expected = [int(x) for x in out.split()]

    assert expected == [
        pt_trace.OFF_DATA_HEAD,
        pt_trace.OFF_DATA_TAIL,
        pt_trace.OFF_DATA_OFFSET,
        pt_trace.OFF_DATA_SIZE,
        pt_trace.OFF_AUX_HEAD,
        pt_trace.OFF_AUX_TAIL,
        pt_trace.OFF_AUX_OFFSET,
        pt_trace.OFF_AUX_SIZE,
    ]
