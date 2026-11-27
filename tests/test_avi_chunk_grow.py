"""Tests for ``avi_chunk_grow``: insert missing required AVI chunks.

An empty-corpus GStreamer run produced one RIFF/AVI header in 2,634 inputs
and no parseable AVI chunk: avidemux beyond the header stayed dark.
"""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.avi_grow import AviKind, _candidates, grow_avi
from fuzzer_tool.core.mutations.riff_fields import walk_riff
from fuzzer_tool.core.operator_registry import REGISTRY, format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng

MAX_LEN = 1 << 16
HOSTILE_SEEDS = range(24)
HDR = 8


def _chunk(fcc: bytes, body: bytes) -> bytes:
    pad = b"\x00" if len(body) & 1 else b""
    return fcc + struct.pack("<I", len(body)) + body + pad


def _list(kind: bytes, body: bytes) -> bytes:
    return _chunk(b"LIST", kind + body)


def _riff(body: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"AVI " + body


BARE = _riff(b"")


def _sizes_ok(data: bytes, pos: int = 12, end: int | None = None) -> bool:
    """RIFF size matches the file; every chunk fits its parent; LISTs tile."""
    if end is None:
        if struct.unpack_from("<I", data, 4)[0] != len(data) - HDR:
            return False
        end = len(data)
    while pos < end:
        if pos + HDR > end:
            return False
        size = struct.unpack_from("<I", data, pos + 4)[0]
        nxt = pos + HDR + size + (size & 1)
        if nxt > end:
            return False
        if data[pos : pos + 4] == b"LIST" and not _sizes_ok(data, pos + 12, pos + HDR + size):
            return False
        pos = nxt
    return pos == end


def _fccs(data: bytes) -> set[bytes]:
    out = {c.fourcc for c in walk_riff(data)}
    out |= {data[c.off : c.off + 4] for c in walk_riff(data) if c.fourcc == b"LIST"}
    return out


def _grow(data: bytes, what: bytes, *kind_idx: int) -> bytes:
    idx = [t for _, t in _candidates(data)].index(what)
    out = grow_avi(data, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[idx, *kind_idx]))
    assert out is not None and _sizes_ok(out)
    return out


def _complete() -> bytes:
    data = BARE
    while (out := grow_avi(data, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0, 0]))) is not None:
        data = out
    return data


def test_bare_header_grows_hdrl():
    """A bare RIFF/AVI header grows hdrl down to strh/strf."""
    out = _grow(BARE, b"hdrl", 0)
    assert {b"hdrl", b"avih", b"strl", b"strh", b"strf"} <= _fccs(out)


def test_audio_strh_infers_kind():
    """strh fccType auds: strf grows a WAVEFORMATEX (no draw)."""
    strh = _chunk(b"strh", b"auds" + bytes(52))
    data = _riff(_list(b"hdrl", _list(b"strl", strh)))
    out = _grow(data, b"strf")
    strf = [c for c in walk_riff(out) if c.fourcc == b"strf"][0]
    assert strf.size == 18  # WAVEFORMATEX with cbSize 0


def test_draw_picks_kind():
    """No strh: the drawn kind decides strh and strf together."""
    data = _riff(_list(b"hdrl", _chunk(b"avih", bytes(56))))
    out = _grow(data, b"strl", list(AviKind).index(AviKind.VIDEO))
    assert b"vids" in out and b"auds" not in out


def test_complete_declines():
    """Falsification: nothing missing, nothing to grow."""
    full = _complete()
    assert _sizes_ok(full)
    assert {b"movi", b"00dc", b"idx1"} <= _fccs(full)
    assert grow_avi(full, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0, 0])) is None


def test_max_len_declines():
    rng = ScriptedRng(choice_idxs=[0, 0])
    assert grow_avi(BARE, max_len=len(BARE) + 8, rng=rng) is None


@pytest.mark.parametrize("seed", HOSTILE_SEEDS)
def test_hostile_input(seed):
    """Adversarial: corrupted/truncated files -> None or longer; valid stays valid."""
    rng = RandPool(seed=seed)
    base = bytearray(_complete())
    pos = rng.randint(0, len(base) - 4)
    base[pos : pos + 4] = rng.randbytes(4)
    data = bytes(base[: rng.randint(12, len(base))])
    out = grow_avi(data, max_len=MAX_LEN, rng=RandPool(seed=seed))
    assert out is None or len(out) > len(data)
    if out is not None and _sizes_ok(data):
        assert _sizes_ok(out)


def test_wiring():
    assert "avi_chunk_grow" in REGISTRY.names()
    assert format_gate_matches("avi_chunk_grow", BARE) is True
    assert format_gate_matches("avi_chunk_grow", b"RIFF\x04\x00\x00\x00WAVE") is False
    assert callable(getattr(OperatorEngine, "_op_avi_chunk_grow", None))
