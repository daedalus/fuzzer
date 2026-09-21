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

import struct
import time

from fuzzer_tool.core.mutations.gif import parse_gif, serialize_gif
from fuzzer_tool.core.mutations.isobmff import parse_boxes, serialize_boxes
from fuzzer_tool.core.mutations.riff import parse_riff_chunks
from fuzzer_tool.core.mutations.webp import parse_webp
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.wfc import AdjacencyTable
from fuzzer_tool.core.wfc_chunks import (
    GIF_FORMAT,
    ISOBMFF_FORMAT,
    RIFF_FORMAT,
    WEBP_FORMAT,
    VIOLATE_RATE,
    ChunkFormat,
    WfcChunkMutator,
    WfcChunkTableStore,
    _table_pairs,
    _try_gif,
    _try_isobmff,
    _try_riff,
    _try_webp,
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


_FORMATS_UNDER_TEST = [
    ("isobmff", isobmff_sample(), _try_isobmff, ISOBMFF_FORMAT),
    ("gif", gif_sample(), _try_gif, GIF_FORMAT),
    ("webp", webp_sample(), _try_webp, WEBP_FORMAT),
    ("riff", riff_sample(), _try_riff, RIFF_FORMAT),
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
