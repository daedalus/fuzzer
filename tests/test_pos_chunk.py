"""PositionChunkScheduler: land mutations on container chunk headers.

Covers core/schedulers/pos_chunk.py and ``wfc_chunks.detect_chunks``.
"""

import struct

from fuzzer_tool.core.mutations.png import PngChunk, serialize_png_chunks
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_chunk import (
    EPSILON,
    HEADER_PAD,
    MAX_SEEDS,
    MAX_SPANS,
    PARSE_CAP,
    PositionChunkScheduler,
)
from fuzzer_tool.core.wfc_chunks import detect_chunks

NO_ESCAPE = 0.99  # random() draw above EPSILON


class ScriptedRng:
    """Scripted random()/randint(); randint asserts and logs its bounds."""

    def __init__(self, randoms=(), ints=()):
        self._randoms = list(randoms)
        self._ints = list(ints)
        self.bounds = []

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_ESCAPE

    def randint(self, a, b):
        self.bounds.append((a, b))
        v = self._ints.pop(0) if self._ints else a
        assert a <= v <= b, f"scripted randint {v} outside [{a}, {b}]"
        return v


def _riff(chunks):
    body = b"WAVE" + b"".join(
        cid + struct.pack("<I", len(d)) + d + (b"\x00" if len(d) % 2 else b"") for cid, d in chunks
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


PNG = serialize_png_chunks(
    [PngChunk(b"IHDR", bytes(13)), PngChunk(b"IDAT", b"\x01" * 20), PngChunk(b"IEND", b"")]
)
WAV = _riff([(b"fmt ", bytes(16)), (b"data", b"\x02" * 12)])


def _kind_offsets(data, kinds):
    """Independent oracle: each kind's first occurrence after the previous one."""
    out, cur = [], 0
    for k in kinds:
        i = data.find(k, cur)
        out.append(i)
        cur = i + len(k)
    return out


class TestDetect:
    def test_riff_detected(self):
        fmt, chunks = detect_chunks(WAV)
        assert fmt.name == "riff"
        assert [fmt.kind(c) for c in chunks] == [b"fmt ", b"data"]

    def test_unknown_is_none(self):
        assert detect_chunks(b"hello world") is None


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(PositionChunkScheduler(ScriptedRng()), PositionScheduler)

    def test_record_is_a_noop(self):
        s = PositionChunkScheduler(ScriptedRng(ints=[0, 0]))
        s.record(PNG, [5], Outcome.GAIN, 1.0)
        assert s.propose(PNG, len(PNG)) == PNG.find(b"IHDR") - HEADER_PAD

    def test_active_after_first_parsed_seed(self):
        s = PositionChunkScheduler(ScriptedRng())
        assert not s.active()
        s.propose(b"plain text", 10)
        assert not s.active()
        s.propose(PNG, len(PNG))
        assert s.active()


class TestDeclines:
    def test_unknown_format(self):
        assert PositionChunkScheduler(ScriptedRng()).propose(b"x" * 64, 64) is None

    def test_empty_data_or_buffer(self):
        s = PositionChunkScheduler(ScriptedRng())
        assert s.propose(b"", 4) is None
        assert s.propose(PNG, 0) is None

    def test_epsilon_escape(self):
        s = PositionChunkScheduler(ScriptedRng(randoms=[EPSILON / 2]))
        assert s.propose(PNG, len(PNG)) is None

    def test_adversarial_oversized_seed_is_not_parsed(self):
        big = PNG + bytes(PARSE_CAP)
        assert PositionChunkScheduler(ScriptedRng()).propose(big, len(big)) is None

    def test_adversarial_truncated_container(self):
        # A header that claims more than the file holds must not raise.
        bad = PNG[:20]
        PositionChunkScheduler(ScriptedRng()).propose(bad, len(bad))


class TestPicks:
    def test_spans_cover_each_chunk_header(self):
        s = PositionChunkScheduler(ScriptedRng())
        s.propose(PNG, len(PNG))
        offs = _kind_offsets(PNG, [b"IHDR", b"IDAT", b"IEND"])
        assert s.spans(PNG) == [(o - HEADER_PAD, o + 4 + HEADER_PAD) for o in offs]

    def test_riff_spans(self):
        s = PositionChunkScheduler(ScriptedRng())
        s.propose(WAV, len(WAV))
        offs = _kind_offsets(WAV, [b"fmt ", b"data"])
        assert [a for a, _ in s.spans(WAV)] == [o - HEADER_PAD for o in offs]

    def test_falsification_only_header_bytes(self):
        s = PositionChunkScheduler(ScriptedRng())
        s.propose(PNG, len(PNG))
        spans = s.spans(PNG)
        inside = {i for a, b in spans for i in range(a, min(b, len(PNG)))}
        picks = set()
        for i, (a, b) in enumerate(spans):
            for k in range(min(b, len(PNG)) - a):
                picks.add(PositionChunkScheduler(ScriptedRng(ints=[i, k])).propose(PNG, len(PNG)))
        assert picks == inside
        assert len(inside) < len(PNG)  # not uniform over the file

    def test_adversarial_shrunk_buffer_drops_later_spans(self):
        cut = PNG.find(b"IDAT")
        rng = ScriptedRng(ints=[1, 0])
        s = PositionChunkScheduler(rng)
        assert s.propose(PNG, cut) == cut - HEADER_PAD
        assert rng.bounds[0] == (0, 1)  # IHDR and IDAT spans only


class TestBounds:
    def test_spans_capped(self):
        many = [PngChunk(b"tEXt", b"a") for _ in range(MAX_SPANS * 2)]
        data = serialize_png_chunks([PngChunk(b"IHDR", bytes(13)), *many])
        s = PositionChunkScheduler(ScriptedRng())
        s.propose(data, len(data))
        spans = s.spans(data)
        assert len(spans) == MAX_SPANS
        assert spans[-1][0] > len(data) // 2

    def test_seed_cache_bounded(self):
        s = PositionChunkScheduler(ScriptedRng())
        for i in range(MAX_SEEDS + 2):
            d = PNG + i.to_bytes(2, "big")
            s.propose(d, len(d))
        assert s.cached_seeds() == MAX_SEEDS
