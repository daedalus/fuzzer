"""Tests for ``mkv_elem_grow``: insert missing required Matroska elements.

An empty-corpus GStreamer run built Segment/Tracks/Cluster but never
CodecPrivate, Audio, BlockGroup, Cues or SeekHead (0 of 327 MKV inputs).
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.mkv_grow import (
    AUDIO,
    CLUSTER,
    CODEC_PRIVATE,
    CUES,
    EBML,
    SEEK_HEAD,
    SEGMENT,
    TRACK_TYPE,
    TRACKS,
    VIDEO,
    MkvKind,
    _candidates,
    grow_mkv,
)
from fuzzer_tool.core.mutations.webm import Element, parse_webm, serialize_webm
from fuzzer_tool.core.operator_registry import REGISTRY, format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng

MAX_LEN = 1 << 16
HOSTILE_SEEDS = range(24)


def _vint_size(n: int) -> bytes:
    return bytes([0x80 | n]) if n < 0x7F else (0x4000 | n).to_bytes(2, "big")


def _el(elem_id: int, payload: bytes) -> bytes:
    return (
        elem_id.to_bytes((elem_id.bit_length() + 7) // 8, "big")
        + _vint_size(len(payload))
        + payload
    )


EBML_ONLY = _el(EBML, _el(0x4282, b"matroska"))


def _ids(data: bytes) -> set[int]:
    """Every element id in the parsed tree (leaves included)."""
    els = parse_webm(data) or []
    out: set[int] = set()
    stack: list[Element] = list(els)
    while stack:
        e = stack.pop()
        out.add(e.elem_id)
        stack += e.children
    return out


def _grow(data: bytes, elem_id: int, *kind_idx: int) -> bytes:
    idx = [t for _, t in _candidates(data)].index(elem_id)
    out = grow_mkv(data, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[idx, *kind_idx]))
    assert out is not None
    return out


def _complete() -> bytes:
    data = EBML_ONLY
    while (out := grow_mkv(data, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0, 0]))) is not None:
        data = out
    return data


def test_ebml_grows_segment():
    """A bare EBML header grows a Segment down to a Cluster with a block."""
    out = _grow(EBML_ONLY, SEGMENT, 0)
    assert parse_webm(out) is not None
    assert {SEGMENT, TRACKS, CLUSTER} <= _ids(out)


def test_audio_track_kind_inferred():
    """TrackType 2 (audio): CodecPrivate grows from the audio template, no draw."""
    entry = _el(0xAE, _el(TRACK_TYPE, b"\x02"))
    data = EBML_ONLY + _el(SEGMENT, _el(TRACKS, entry))
    out = _grow(data, CODEC_PRIVATE)
    assert b"\x11\x88" in out  # AAC AudioSpecificConfig


def test_no_type_draws_kind():
    """No TrackType: a grown TrackEntry takes the drawn kind, coherently."""
    data = EBML_ONLY + _el(SEGMENT, _el(TRACKS, b""))
    out = _grow(data, 0xAE, list(MkvKind).index(MkvKind.AUDIO))
    assert AUDIO in _ids(out) and VIDEO not in _ids(out)
    assert b"A_AAC" in out


def test_cluster_after_tracks():
    """Cluster and Cues are appended to the Segment, after Tracks."""
    full = _complete()
    seg = parse_webm(full)[1]
    order = [c.elem_id for c in seg.children]
    assert order.index(TRACKS) < order.index(CLUSTER)
    assert {CUES, SEEK_HEAD} <= set(order)


def test_complete_declines():
    """Falsification: nothing missing, nothing to grow."""
    full = _complete()
    assert grow_mkv(full, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0, 0])) is None


def test_tail_bytes_kept():
    """Bytes after a known-size Segment survive the grow."""
    tail = b"\xde\xad\xbe\xef"
    data = EBML_ONLY + _el(SEGMENT, _el(TRACKS, b"")) + tail
    out = _grow(data, 0xAE, 0)
    assert out.endswith(tail)


def test_max_len_declines():
    rng = ScriptedRng(choice_idxs=[0, 0])
    assert grow_mkv(EBML_ONLY, max_len=len(EBML_ONLY) + 8, rng=rng) is None


@pytest.mark.parametrize("seed", HOSTILE_SEEDS)
def test_hostile_input(seed):
    """Adversarial: corrupted/truncated files -> None or a parseable result."""
    rng = RandPool(seed=seed)
    base = bytearray(_complete())
    pos = rng.randint(0, len(base) - 4)
    base[pos : pos + 4] = rng.randbytes(4)
    data = bytes(base[: rng.randint(4, len(base))])
    out = grow_mkv(data, max_len=MAX_LEN, rng=RandPool(seed=seed))
    if out is not None and parse_webm(data) is not None:
        assert parse_webm(out) is not None
        assert serialize_webm(parse_webm(out)) == out[: len(serialize_webm(parse_webm(out)))]


def test_wiring():
    assert "mkv_elem_grow" in REGISTRY.names()
    assert format_gate_matches("mkv_elem_grow", EBML_ONLY) is True
    assert format_gate_matches("mkv_elem_grow", b"RIFF" + bytes(12)) is False
    assert callable(getattr(OperatorEngine, "_op_mkv_elem_grow", None))
