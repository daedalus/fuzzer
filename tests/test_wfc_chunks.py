"""Tests for ``core/wfc_chunks.py`` -- learned-adjacency WFC chunk reordering.

Covers the P2-1 item from
``docs/handover/handover_generators_2026-09-20.md``: a per-format
``AdjacencyTable`` learned from admitted corpus seeds (``WfcChunkTableStore``),
applied to isobmff/webp/riff/gif top-level chunk sequences
(``wfc_reorder_chunks``), and wired as a selectable operator
(``WfcChunkMutator`` / ``wfc_reorder_learned``).

Acceptance criteria from the handover: output parses with the format's own
parser; differs from the parent >= 95% of calls; <= 10 ms; strict mode never
emits an unobserved adjacency; violate mode emits exactly one.
"""

from __future__ import annotations

import dataclasses
import io
import struct
import time
import zipfile
from typing import Any

from fuzzer_tool.core.mutations.asf import (
    DATA_OBJECT_GUID,
    HEADER_OBJECT_GUID,
    AsfObject,
    parse_asf_objects,
    serialize_asf_objects,
)
from fuzzer_tool.core.mutations.flv import FlvTag, parse_flv, serialize_flv
from fuzzer_tool.core.mutations.gif import parse_gif, serialize_gif
from fuzzer_tool.core.mutations.isobmff import parse_boxes, serialize_boxes
from fuzzer_tool.core.mutations.mpegts import TsPacket, parse_ts_packets, serialize_ts_packets
from fuzzer_tool.core.mutations.nal import parse_nal_units
from fuzzer_tool.core.mutations.ogg import parse_ogg_pages
from fuzzer_tool.core.mutations.riff import parse_riff_chunks
from fuzzer_tool.core.mutations.webp import parse_webp
from fuzzer_tool.core.mutations.zip import parse_zip
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.wfc import AdjacencyTable
from fuzzer_tool.core.wfc_chunks import (
    ASF_FORMAT,
    FLV_FORMAT,
    GIF_FORMAT,
    ISOBMFF_FORMAT,
    MPEGTS_FORMAT,
    NAL_FORMAT,
    OGG_FORMAT,
    RIFF_FORMAT,
    WEBP_FORMAT,
    ZIP_FORMAT,
    VIOLATE_RATE,
    ChunkFormat,
    WfcChunkMutator,
    WfcChunkTableStore,
    _FORMATS,
    _table_pairs,
    _try_asf,
    _try_flv,
    _try_gif,
    _try_isobmff,
    _try_mpegts,
    _try_nal,
    _try_ogg,
    _try_riff,
    _try_webp,
    _try_zip,
    kind_sequence,
    wfc_reorder_chunks,
)

# ═══════════════════════════════════════════════════════════════════
# Synthetic format bodies (well-formed, >=3 top-level chunks each)
# ═══════════════════════════════════════════════════════════════════


def _box(t: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + t + payload


def isobmff_sample() -> bytes:
    return (
        _box(b"ftyp", b"isom" + bytes(4))
        + _box(b"free", bytes(4))
        + _box(b"mdat", b"X" * 40)
        + _box(b"free", bytes(4))
        + _box(b"skip", bytes(8))
    )


def gif_sample() -> bytes:
    lsd = bytes([10, 0, 10, 0, 0x80, 0, 0])
    gct = b"\x00\x00\x00\xff\xff\xff" * 2
    gce = b"\x21\xf9\x04\x00\x00\x00\x00\x00\x00"
    image = b"\x2c" + bytes([0, 0, 0, 0, 10, 0, 10, 0, 0]) + b"\x02" + b"\x02\x4c\x01\x00"
    return b"GIF89a" + lsd + gct + gce + image + b"\x3b"


def _wchunk(fourcc: bytes, payload: bytes) -> bytes:
    out = fourcc + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        out += b"\x00"
    return out


def webp_sample() -> bytes:
    body = (
        _wchunk(b"VP8X", bytes(10))
        + _wchunk(b"ANIM", bytes(6))
        + _wchunk(b"ICCP", b"Y" * 20)
        + _wchunk(b"EXIF", b"E" * 12)
    )
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body


def riff_sample() -> bytes:
    body = _wchunk(b"fmt ", bytes(16)) + _wchunk(b"data", b"Z" * 30) + _wchunk(b"JUNK", bytes(4))
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body



_OGG_S1, _OGG_S2 = 0x1111, 0x2222


def _ogg_page(header_type: int, serial: int, seq: int, payload: bytes = b"pg") -> bytes:
    return (
        b"OggS"
        + bytes([0, header_type])
        + bytes(8)
        + serial.to_bytes(4, "little")
        + seq.to_bytes(4, "little")
        + bytes(4)
        + bytes([1, len(payload)])
        + payload
    )


def ogg_sample() -> bytes:
    """Two multiplexed logical streams: BOS x2, interleaved data, EOS x2."""
    a, b = _OGG_S1, _OGG_S2
    return (
        _ogg_page(0x02, a, 0)
        + _ogg_page(0x02, b, 0)
        + _ogg_page(0x00, a, 1, b"a1")
        + _ogg_page(0x00, b, 1, b"b1")
        + _ogg_page(0x00, a, 2, b"a2")
        + _ogg_page(0x00, b, 2, b"b2")
        + _ogg_page(0x04, a, 3)
        + _ogg_page(0x04, b, 3)
    )


_FLV_HEADER = b"FLV\x01\x05\x00\x00\x00\x09"


def flv_sample() -> bytes:
    def tag(prev: int, kind: int, body: bytes) -> FlvTag:
        return FlvTag(prev, kind, len(body), 0, 0, body)

    tags = [
        tag(0, 18, b"meta-meta"),
        tag(20, 9, b"video-1"),
        tag(18, 8, b"audio-1"),
        tag(18, 9, b"video-2"),
        tag(18, 8, b"audio-2"),
        tag(18, 9, b"video-3"),
    ]
    return serialize_flv(_FLV_HEADER, tags, 18)


def nal_sample() -> bytes:
    sc4, sc3 = b"\x00\x00\x00\x01", b"\x00\x00\x01"
    return (
        sc4 + b"\x67\x11\x22"  # SPS
        + sc4 + b"\x68\x33\x44"  # PPS
        + sc4 + b"\x65\x55\x66"  # IDR
        + sc3 + b"\x41\x77\x88"  # non-IDR slice
        + sc3 + b"\x41\x99\xaa"  # non-IDR slice
        + sc4 + b"\x06\xbb\xcc"  # SEI
    )


_ASF_INDEX_GUID = bytes.fromhex("90080033B1E5CF1189F400A0C90349CB")
_ASF_PADDING_GUID = bytes.fromhex("1806D474CACF11B1A50C00A0C9034A15")
_ASF_MEDIA_INDEX_GUID = bytes.fromhex("F803B1FEAD12644CA1F7E5AEE8BF7B0D")
_ASF_TIMECODE_INDEX_GUID = bytes.fromhex("CFB1FA3C98E7484EAC19ABBB0EA8D4AA")
_ASF_METADATA_GUID = bytes.fromhex("EAD2C3B7D2114F4CA0A3D5C6E9E5E4B1")


def asf_sample() -> bytes:
    """Header + data + five trailing objects.

    The header is pinned, so only the objects after it can move: with three
    of them a random order equals the original 1 time in 6, which alone caps
    the "differs from the parent" rate near 83% and says nothing about the
    operator. Real ASF files carry index, media-object-index, timecode-index
    and padding objects after the data object.
    """

    def obj(guid: bytes, body: bytes) -> AsfObject:
        return AsfObject(guid, 24 + len(body), body)

    return serialize_asf_objects(
        [
            obj(HEADER_OBJECT_GUID, b"H" * 8),
            obj(DATA_OBJECT_GUID, b"D" * 12),
            obj(_ASF_INDEX_GUID, b"I" * 6),
            obj(_ASF_MEDIA_INDEX_GUID, b"M" * 5),
            obj(_ASF_TIMECODE_INDEX_GUID, b"T" * 7),
            obj(_ASF_METADATA_GUID, b"X" * 3),
            obj(_ASF_PADDING_GUID, b"P" * 4),
        ]
    )


def mpegts_sample() -> bytes:
    pids = [0, 0x100, 0x101, 0x100, 0x101, 0x100, 0x11, 0x101]
    pkts = [
        TsPacket(0, 0, 0, pid, 0, 1, i & 0xF, b"", bytes([pid & 0xFF]) * 184)
        for i, pid in enumerate(pids)
    ]
    return serialize_ts_packets(pkts)


def zip_sample() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("mimetype", b"application/epub+zip")
        z.writestr("META-INF/container.xml", b"<container/>")
        z.writestr("a.xml", b"<a/>")
        z.writestr("b.xml", b"<b/>")
        z.writestr("c.png", b"png-bytes")
        z.writestr("d.png", b"png-bytes-2")
    return buf.getvalue()


_FORMATS_UNDER_TEST = [
    ("isobmff", isobmff_sample(), _try_isobmff, ISOBMFF_FORMAT),
    ("gif", gif_sample(), _try_gif, GIF_FORMAT),
    ("webp", webp_sample(), _try_webp, WEBP_FORMAT),
    ("riff", riff_sample(), _try_riff, RIFF_FORMAT),
    ("ogg", ogg_sample(), _try_ogg, OGG_FORMAT),
    ("flv", flv_sample(), _try_flv, FLV_FORMAT),
    ("nal", nal_sample(), _try_nal, NAL_FORMAT),
    ("asf", asf_sample(), _try_asf, ASF_FORMAT),
    ("mpegts", mpegts_sample(), _try_mpegts, MPEGTS_FORMAT),
    ("zip", zip_sample(), _try_zip, ZIP_FORMAT),
]


def _train(store: WfcChunkTableStore, fmt: ChunkFormat, data: bytes, variants: int = 40) -> None:
    """Feed *store* several shuffled re-orderings of *data* so its table has
    more than one observed adjacency to choose from."""
    chunks = fmt.parse(data)
    kinds = [fmt.kind(c) for c in chunks]
    rng = RandPool(seed=123)
    seen = {tuple(kinds)}
    store.observe(fmt, data)
    for _ in range(variants):
        order = list(chunks)
        rng.shuffle(order)
        key = tuple(fmt.kind(c) for c in order)
        if key in seen:
            continue
        seen.add(key)
        try:
            out = fmt.serialize(order)
        except Exception:
            continue
        store.observe(fmt, out)


# ═══════════════════════════════════════════════════════════════════
# WfcChunkTableStore
# ═══════════════════════════════════════════════════════════════════


class TestWfcChunkTableStore:
    def test_table_for_creates_and_caches(self):
        store = WfcChunkTableStore()
        t1 = store.table_for("isobmff")
        t2 = store.table_for("isobmff")
        assert t1 is t2

    def test_observe_learns_bigrams(self):
        store = WfcChunkTableStore()
        store.observe(ISOBMFF_FORMAT, isobmff_sample())
        table = store.table_for("isobmff")
        assert table.compatible(b"ftyp", b"free", "right")

    def test_observe_never_raises_on_garbage(self):
        store = WfcChunkTableStore()
        store.observe(ISOBMFF_FORMAT, b"not an isobmff file at all")
        store.observe(GIF_FORMAT, b"")
        store.observe(RIFF_FORMAT, b"\x00" * 4)
        # No exception, and no bogus table created for garbage input.
        assert not store.table_for("isobmff").has_tile(b"not ")


# ═══════════════════════════════════════════════════════════════════
# wfc_reorder_chunks: per-format round-trip and acceptance criteria
# ═══════════════════════════════════════════════════════════════════


class TestWfcReorderChunksPerFormat:
    def test_output_always_reparses(self):
        """Output parses with the format's own parser, every call."""
        rng = RandPool(seed=1)
        for name, data, try_parse, fmt in _FORMATS_UNDER_TEST:
            store = WfcChunkTableStore()
            _train(store, fmt, data)
            table = store.table_for(name)
            chunks = fmt.parse(data)
            for mode in ("strict", "violate"):
                for _ in range(20):
                    out = wfc_reorder_chunks(fmt, chunks, table, rng, mode=mode)
                    assert try_parse(out) is not None, (
                        f"{name}/{mode} produced output its own parser rejects"
                    )

    def test_differs_from_parent_at_least_95_percent(self):
        """Across repeated calls, output differs from the input often --
        the 95% bar from the handover's acceptance criteria, checked on a
        trained table (an untrained one leans on the shuffle fallback,
        which is not what this bar is measuring)."""
        rng = RandPool(seed=2)
        for name, data, _try_parse, fmt in _FORMATS_UNDER_TEST:
            store = WfcChunkTableStore()
            _train(store, fmt, data)
            table = store.table_for(name)
            chunks = fmt.parse(data)
            n = 200
            changed = 0
            for _ in range(n):
                out = wfc_reorder_chunks(fmt, chunks, table, rng, mode="strict")
                if out != data:
                    changed += 1
            assert changed / n >= 0.95, f"{name}: only {changed}/{n} calls changed the input"

    def test_under_10ms_per_call(self):
        for name, data, _try_parse, fmt in _FORMATS_UNDER_TEST:
            store = WfcChunkTableStore()
            _train(store, fmt, data)
            table = store.table_for(name)
            chunks = fmt.parse(data)
            rng = RandPool(seed=3)
            n = 50
            t0 = time.time()
            for _ in range(n):
                wfc_reorder_chunks(fmt, chunks, table, rng, mode="strict")
            elapsed_ms = (time.time() - t0) / n * 1000
            assert elapsed_ms <= 10.0, f"{name}: {elapsed_ms:.2f} ms/call"

    def test_strict_mode_never_emits_unobserved_adjacency(self):
        rng = RandPool(seed=4)
        for name, data, _try_parse, fmt in _FORMATS_UNDER_TEST:
            store = WfcChunkTableStore()
            _train(store, fmt, data)
            table = store.table_for(name)
            chunks = fmt.parse(data)
            kinds = [fmt.kind(c) for c in chunks]
            for _ in range(30):
                out = wfc_reorder_chunks(fmt, chunks, table, rng, mode="strict")
                out_chunks = fmt.parse(out)
                assert out_chunks is not None
                out_kinds = [fmt.kind(c) for c in out_chunks]
                for a, b in zip(out_kinds, out_kinds[1:]):
                    if a not in kinds or b not in kinds:
                        continue  # shuffle-fallback path, not a table claim
                    assert table.compatible(a, b, "right") or not table.has_tile(a), (
                        f"{name}: strict mode emitted an unobserved adjacency "
                        f"{a!r} -> {b!r}"
                    )

    def test_violate_mode_emits_exactly_one_unobserved_pair(self):
        """``_add_one_unobserved_pair`` is the unit under test here directly
        -- the module-level helper the handover's 'violate' mode relies on."""
        from fuzzer_tool.core.wfc_chunks import _add_one_unobserved_pair

        table = AdjacencyTable()
        table.add_forward(b"A", b"B")
        table.add_forward(b"B", b"C")
        kinds = [b"A", b"B", b"C", b"D"]
        rng = RandPool(seed=5)
        for _ in range(20):
            new_table, pair = _add_one_unobserved_pair(table, kinds, rng)
            if pair is None:
                continue
            a, b = pair
            assert not table.compatible(a, b, "right"), "violate pair was already observed"
            observed_before = _table_pairs(table, kinds)
            observed_after = _table_pairs(new_table, kinds)
            assert observed_after - observed_before == {(a, b)}, (
                "violate mode must add exactly one new pair"
            )


# ═══════════════════════════════════════════════════════════════════
# kind_sequence
# ═══════════════════════════════════════════════════════════════════


class TestKindSequence:
    def test_kind_sequence_matches_parser(self):
        for name, data, _try_parse, fmt in _FORMATS_UNDER_TEST:
            seq = kind_sequence(fmt, data)
            chunks = fmt.parse(data)
            assert seq == [fmt.kind(c) for c in chunks], name

    def test_kind_sequence_none_on_unparseable(self):
        assert kind_sequence(ISOBMFF_FORMAT, b"garbage") is None
        assert kind_sequence(GIF_FORMAT, b"not a gif") is None


# ═══════════════════════════════════════════════════════════════════
# WfcChunkMutator: registry wiring and gating
# ═══════════════════════════════════════════════════════════════════


class TestWfcChunkMutator:
    def test_registered_under_expected_name_and_category(self):
        assert "wfc_reorder_learned" in REGISTRY.names()
        assert REGISTRY.category_of("wfc_reorder_learned") == "format"

    def test_unavailable_when_wfc_disabled(self):
        m = WfcChunkMutator()
        ctx = MutationContext(wfc_enabled=False)
        assert not m.is_available(ctx, isobmff_sample())

    def test_unavailable_on_non_matching_data(self):
        m = WfcChunkMutator()
        ctx = MutationContext(wfc_enabled=True)
        assert not m.is_available(ctx, b"plain text, no container magic here")

    def test_available_when_enabled_and_format_sniffed(self):
        m = WfcChunkMutator()
        ctx = MutationContext(wfc_enabled=True)
        for _name, data, _try_parse, _fmt in _FORMATS_UNDER_TEST:
            assert m.is_available(ctx, data)

    def test_mutate_declines_without_context_data(self):
        m = WfcChunkMutator()
        rng = RandPool(seed=6)
        assert m.mutate(b"", rng) is None

    def test_mutate_reorders_and_reparses(self):
        m = WfcChunkMutator()
        ctx = MutationContext(wfc_enabled=True)
        rng = RandPool(seed=7)
        for _name, data, try_parse, _fmt in _FORMATS_UNDER_TEST:
            for _ in range(10):
                m.on_new_coverage(data, 1)
            changed = 0
            for _ in range(40):
                out = m.mutate(data, rng, context=ctx)
                if out is None:
                    continue
                changed += 1
                assert try_parse(out) is not None
            assert changed > 0

    def test_on_new_coverage_populates_the_store(self):
        m = WfcChunkMutator()
        m.on_new_coverage(isobmff_sample(), 3)
        table = m.store.table_for("isobmff")
        assert table.compatible(b"ftyp", b"free", "right")

    def test_on_new_coverage_ignores_empty_and_garbage(self):
        m = WfcChunkMutator()
        m.on_new_coverage(b"", 0)  # must not raise
        m.on_new_coverage(b"not any known container", 1)  # must not raise

    def test_violate_rate_is_a_probability(self):
        assert 0.0 < VIOLATE_RATE < 1.0


# ═══════════════════════════════════════════════════════════════════
# Rollout beyond isobmff/webp/riff/gif: ogg, flv, nal, asf, mpegts, zip
# ═══════════════════════════════════════════════════════════════════

_ROLLOUT = ("ogg", "flv", "nal", "asf", "mpegts", "zip")
_ROLLOUT_UNDER_TEST = [row for row in _FORMATS_UNDER_TEST if row[0] in _ROLLOUT]


def _trained_mutator(fmt: ChunkFormat, data: bytes) -> WfcChunkMutator:
    m = WfcChunkMutator()
    _train(m.store, fmt, data)
    return m


class TestRolloutSamples:
    """Guard the fixtures: a sample its own parser rejects proves nothing."""

    def test_samples_have_at_least_three_chunks(self):
        for name, data, _try, fmt in _ROLLOUT_UNDER_TEST:
            chunks = fmt.parse(data)
            assert chunks is not None and len(chunks) >= 3, name

    def test_each_sample_matches_exactly_one_registered_format(self):
        """Adversarial: overlapping sniffers would let the first entry in
        ``_FORMATS`` silently claim another format's input."""
        for name, data, _try, _fmt in _FORMATS_UNDER_TEST:
            hits = [n for n, sniff, _ in _FORMATS if sniff(data)]
            assert hits == [name], f"{name} sample sniffed as {hits}"

    def test_format_names_are_unique(self):
        names = [n for n, _, _ in _FORMATS]
        assert len(names) == len(set(names))
        assert set(_ROLLOUT) <= set(names)


class TestRolloutKinds:
    def test_ogg_kind_separates_streams_and_page_roles(self):
        kinds = kind_sequence(OGG_FORMAT, ogg_sample())
        assert kinds is not None
        assert len(set(kinds)) == 6  # (BOS|data|EOS) x (stream 1|stream 2)
        assert kinds[0] != kinds[1]  # same role, different stream
        assert kinds[2] != kinds[6]  # same stream, different role

    def test_ogg_continuation_bit_does_not_change_the_kind(self):
        """Only BOS/EOS carry sequence meaning; 'continued packet' is payload."""
        plain = parse_ogg_pages(_ogg_page(0x00, 1, 1))[0]
        cont = parse_ogg_pages(_ogg_page(0x01, 1, 1))[0]
        assert OGG_FORMAT.kind(plain) == OGG_FORMAT.kind(cont)

    def test_flv_kind_is_the_tag_type(self):
        kinds = kind_sequence(FLV_FORMAT, flv_sample())
        assert kinds == [bytes([t]) for t in (18, 9, 8, 9, 8, 9)]

    def test_nal_kind_is_the_unit_type(self):
        kinds = kind_sequence(NAL_FORMAT, nal_sample())
        assert kinds == [bytes([t]) for t in (7, 8, 5, 1, 1, 6)]

    def test_asf_kind_is_the_object_guid(self):
        kinds = kind_sequence(ASF_FORMAT, asf_sample())
        assert kinds[0] == HEADER_OBJECT_GUID and kinds[1] == DATA_OBJECT_GUID

    def test_mpegts_kind_is_the_pid(self):
        kinds = kind_sequence(MPEGTS_FORMAT, mpegts_sample())
        assert kinds[:3] == [b"\x00\x00", b"\x01\x00", b"\x01\x01"]

    def test_zip_kind_groups_by_role_not_by_full_name(self):
        kinds = kind_sequence(ZIP_FORMAT, zip_sample())
        assert kinds[2] == kinds[3]  # a.xml, b.xml
        assert kinds[4] == kinds[5]  # c.png, d.png
        assert kinds[2] != kinds[4]
        assert kinds[0] not in (kinds[2], kinds[4])  # `mimetype` is its own tile
        assert kinds[1] != kinds[2]  # META-INF/* is not just another .xml


class TestRolloutInvariants:
    def test_ogg_bos_stays_first_and_eos_stays_last(self):
        data = ogg_sample()
        pages = OGG_FORMAT.parse(data)
        first, last = OGG_FORMAT.kind(pages[0]), OGG_FORMAT.kind(pages[-1])
        store = WfcChunkTableStore()
        _train(store, OGG_FORMAT, data)
        rng = RandPool(seed=12)
        for mode in ("strict", "violate"):
            for _ in range(60):
                out = OGG_FORMAT.parse(
                    wfc_reorder_chunks(OGG_FORMAT, pages, store.table_for("ogg"), rng, mode=mode)
                )
                assert OGG_FORMAT.kind(out[0]) == first, mode
                assert OGG_FORMAT.kind(out[-1]) == last, mode

    def test_asf_header_object_stays_first(self):
        data = asf_sample()
        objs = ASF_FORMAT.parse(data)
        store = WfcChunkTableStore()
        _train(store, ASF_FORMAT, data)
        rng = RandPool(seed=13)
        for mode in ("strict", "violate"):
            for _ in range(60):
                out = wfc_reorder_chunks(ASF_FORMAT, objs, store.table_for("asf"), rng, mode=mode)
                assert out[:16] == HEADER_OBJECT_GUID, mode
                assert parse_asf_objects(out) is not None, mode

    def test_flv_mutator_keeps_the_real_header_and_recomputes_the_trailer(self):
        """The bound serializer must use *this* file's header, not the
        placeholder the shared FLV_FORMAT carries for table learning. The
        parser does not keep the final PreviousTagSize (``serialize_flv``
        regenerates it from the last tag), so after a reorder the trailer must
        describe whichever tag is now last."""
        header = b"FLV\x01\x01\x00\x00\x00\x0b\xaa\xbb"  # 11-byte header, odd flags
        _, tags, _ = parse_flv(flv_sample())
        data = serialize_flv(header, tags, None)
        assert parse_flv(data)[0] == header
        m = _trained_mutator(FLV_FORMAT, data)
        ctx = MutationContext(wfc_enabled=True)
        rng = RandPool(seed=14)
        outs = [o for o in (m.mutate(data, rng, context=ctx) for _ in range(40)) if o]
        assert outs
        for out in outs:
            assert out.startswith(header)
            last = parse_flv(out)[1][-1]
            assert out.endswith((11 + len(last.data)).to_bytes(4, "big"))

    def test_zip_mutator_keeps_the_central_directory_consistent(self):
        """Entries move, but offsets/counts are recomputed, the EOCD comment
        survives, and every entry keeps its own bytes."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for n in ("mimetype", "META-INF/container.xml", "a.xml", "b.xml", "c.png", "d.png"):
                z.writestr(n, b"body-of-" + n.encode())
            z.comment = b"keep-me"
        data = buf.getvalue()
        before = {e.name: e.data for e in parse_zip(data).entries}
        m = _trained_mutator(ZIP_FORMAT, data)
        ctx = MutationContext(wfc_enabled=True)
        rng = RandPool(seed=15)
        outs = [o for o in (m.mutate(data, rng, context=ctx) for _ in range(40)) if o]
        assert outs
        for out in outs:
            doc = parse_zip(out)
            assert doc is not None
            assert doc.eocd_comment == b"keep-me"
            assert {e.name: e.data for e in doc.entries} == before
            with zipfile.ZipFile(io.BytesIO(out)) as z:  # stdlib agrees on the layout
                assert sorted(z.namelist()) == sorted(k.decode() for k in before)

    def test_mpegts_output_stays_188_aligned(self):
        data = mpegts_sample()
        m = _trained_mutator(MPEGTS_FORMAT, data)
        ctx = MutationContext(wfc_enabled=True)
        rng = RandPool(seed=16)
        outs = [o for o in (m.mutate(data, rng, context=ctx) for _ in range(40)) if o]
        assert outs
        for out in outs:
            assert len(out) == len(data) and len(out) % 188 == 0
            assert parse_ts_packets(out) is not None


class TestRolloutGating:
    def test_available_for_each_new_format_only_when_wfc_is_on(self):
        m = WfcChunkMutator()
        on, off = MutationContext(wfc_enabled=True), MutationContext(wfc_enabled=False)
        for name, data, _try, _fmt in _ROLLOUT_UNDER_TEST:
            assert m.is_available(on, data), name
            assert not m.is_available(off, data), name

    def test_on_new_coverage_learns_each_new_format(self):
        m = WfcChunkMutator()
        for name, data, _try, fmt in _ROLLOUT_UNDER_TEST:
            m.on_new_coverage(data, 1)
            kinds = kind_sequence(fmt, data)
            assert m.store.table_for(name).compatible(kinds[0], kinds[1], "right"), name

    def test_truncated_new_format_inputs_decline_without_raising(self):
        """Adversarial: half a file must never make ``mutate`` raise."""
        m = WfcChunkMutator()
        ctx = MutationContext(wfc_enabled=True)
        rng = RandPool(seed=17)
        for _name, data, _try, _fmt in _ROLLOUT_UNDER_TEST:
            for cut in (1, 5, len(data) // 2, len(data) - 1):
                m.mutate(data[:cut], rng, context=ctx)  # must not raise
                m.on_new_coverage(data[:cut], 1)  # must not raise


# ═══════════════════════════════════════════════════════════════════
# Shared reorder core: bounds and pin handling
#
# Found while rolling the operator out to six more formats: the violate-mode
# pair placement indexed one cell past the grid when nothing was pinned last
# (`mutate` swallowed the IndexError, so a fraction of violate calls silently
# declined), and overwrote the pinned last cell when something was; and
# leftovers from an over-subscribed kind were appended *after* the pinned-last
# chunk. Toy formats keep the property independent of any real container.
# ═══════════════════════════════════════════════════════════════════


def _toy_format(pin_first: bool = False, pin_last: bool = False) -> ChunkFormat:
    return ChunkFormat(
        name="toy",
        parse=lambda d: [d[i : i + 2] for i in range(0, len(d), 2)],
        serialize=lambda cs: b"".join(cs),
        kind=lambda c: c[:1],
        pin_first=pin_first,
        pin_last=pin_last,
    )


def _toy_table(*pairs: tuple[bytes, bytes]) -> AdjacencyTable:
    table = AdjacencyTable()
    for a, b in pairs:
        table.add_forward(a, b)
    return table


class TestReorderPinsAndBounds:
    def test_violate_mode_never_indexes_past_the_grid(self):
        """Unpinned format, sparse table (so an unobserved pair always
        exists): the pair may land on the last two cells without raising."""
        fmt = _toy_format()
        chunks = [b"A1", b"B1", b"C1", b"D1"]
        table = _toy_table((b"A", b"B"), (b"B", b"C"), (b"C", b"D"))
        for seed in range(400):
            wfc_reorder_chunks(fmt, chunks, table, RandPool(seed=seed), mode="violate")

    def test_violate_pair_never_overwrites_the_pinned_last_cell(self):
        fmt = _toy_format(pin_first=True, pin_last=True)
        chunks = [b"H1", b"A1", b"B1", b"C1", b"T1"]
        table = _toy_table(
            (b"H", b"A"), (b"A", b"B"), (b"B", b"C"), (b"C", b"T"), (b"A", b"C"), (b"B", b"T")
        )
        for seed in range(400):
            out = wfc_reorder_chunks(fmt, chunks, table, RandPool(seed=seed), mode="violate")
            assert out[:2] == b"H1", seed
            assert out[-2:] == b"T1", seed

    def test_leftovers_never_displace_the_pinned_last_chunk(self):
        """An over-subscribed kind leaves chunks unplaced; they must go
        before the pinned-last chunk, not after it."""
        fmt = _toy_format(pin_first=True, pin_last=True)
        chunks = [b"S1", b"X1", b"X2", b"Y1", b"E1"]
        table = _toy_table(
            (b"S", b"X"),
            (b"S", b"Y"),
            (b"X", b"Y"),
            (b"Y", b"X"),
            (b"X", b"E"),
            (b"Y", b"E"),
        )
        for seed in range(400):
            out = wfc_reorder_chunks(fmt, chunks, table, RandPool(seed=seed), mode="strict")
            assert out[:2] == b"S1", seed
            assert out[-2:] == b"E1", seed
            assert sorted(out[i : i + 2] for i in range(0, len(out), 2)) == sorted(chunks), seed

    def test_every_chunk_appears_exactly_once_on_every_real_format(self):
        """Reordering may not lose or invent chunks. Checked on the chunk
        objects handed to the serializer, not on a re-parse: GIF's global
        colour table is positional, so a re-parse legitimately re-segments a
        reordered file and says nothing about what the operator did."""
        for name, data, _try, fmt in _FORMATS_UNDER_TEST:
            store = WfcChunkTableStore()
            _train(store, fmt, data)
            chunks = fmt.parse(data)
            seen: list[list[Any]] = []

            def capture(cs, real=fmt.serialize, sink=seen):
                sink.append(list(cs))
                return real(cs)

            spy = dataclasses.replace(fmt, serialize=capture)
            for mode in ("strict", "violate"):
                for seed in range(60):
                    seen.clear()
                    wfc_reorder_chunks(spy, chunks, store.table_for(name), RandPool(seed=seed), mode=mode)
                    assert sorted(map(id, seen[-1])) == sorted(map(id, chunks)), (name, mode, seed)


class TestLeftoverPlacement:
    """WFC has no cardinality constraint, so a collapse can place a kind more
    often than the input has chunks of it and skip another kind entirely. The
    skipped chunks used to be appended at the end, which broke strict mode's
    'never emits an unobserved adjacency' guarantee at the chunk level. (The
    old adjacency test re-parsed the output, and on positional formats a
    re-parse re-segments the bytes, so it could not see this.)"""

    @staticmethod
    def _emitted_kinds(fmt, chunks, table, seed, mode):
        seen: list[list[Any]] = []
        spy = dataclasses.replace(fmt, serialize=lambda cs: (seen.append(list(cs)), fmt.serialize(cs))[1])
        wfc_reorder_chunks(spy, chunks, table, RandPool(seed=seed), mode=mode)
        return [fmt.kind(c) for c in seen[-1]]

    def test_skipped_kind_is_placed_at_an_observed_slot(self):
        fmt = _toy_format(pin_first=True, pin_last=True)
        chunks = [b"S1", b"X1", b"X2", b"Z1", b"E1"]
        table = _toy_table(
            (b"S", b"X"), (b"X", b"X"), (b"X", b"E"), (b"S", b"Z"), (b"Z", b"X")
        )
        for seed in range(400):
            ks = self._emitted_kinds(fmt, chunks, table, seed, "strict")
            for a, b in zip(ks, ks[1:]):
                assert table.compatible(a, b, "right"), (seed, ks)

    def test_strict_mode_is_table_legal_at_chunk_level_on_every_format(self):
        for name, data, _try, fmt in _FORMATS_UNDER_TEST:
            store = WfcChunkTableStore()
            _train(store, fmt, data)
            table = store.table_for(name)
            chunks = fmt.parse(data)
            for seed in range(60):
                ks = self._emitted_kinds(fmt, chunks, table, seed, "strict")
                for a, b in zip(ks, ks[1:]):
                    assert table.compatible(a, b, "right"), (name, seed, a, b)

    def test_unplaceable_leftover_is_kept_not_dropped(self):
        """Falsification: when no slot is legal the chunk must still appear
        (ahead of the pinned tail) -- placement may not turn into deletion."""
        fmt = _toy_format(pin_first=True, pin_last=True)
        chunks = [b"S1", b"X1", b"Q1", b"E1"]  # Q has no observed neighbours at all
        table = _toy_table((b"S", b"X"), (b"X", b"X"), (b"X", b"E"), (b"S", b"E"))
        for seed in range(100):
            out = wfc_reorder_chunks(fmt, chunks, table, RandPool(seed=seed), mode="strict")
            assert sorted(out[i : i + 2] for i in range(0, len(out), 2)) == sorted(chunks), seed
            assert out[-2:] == b"E1", seed
