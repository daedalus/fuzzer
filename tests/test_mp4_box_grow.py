"""Tests for ``mp4_box_grow``: insert a missing required child box.

From an empty corpus no file ever held ``stbl``: the edit/transplant ops
need the box to exist, so the tree stalled at ``mdia`` (GStreamer qtdemux:
~80% of a real MP4's edges). Growing fills the chain in one draw.
"""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.mutations.mp4_grow import KINDS, REQUIRED, TrackKind, _candidates, grow_mp4
from fuzzer_tool.core.mutations.mp4_path import find_boxes
from fuzzer_tool.core.operator_registry import REGISTRY, format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.mp4_build import box, mp4, stsd
from .support.scripted_rng import ScriptedRng

HDR = 8
MAX_LEN = 1 << 16
FTYP = box(b"ftyp", b"isom" + bytes(4) + b"isom")
SAMPLE_TABLE = (b"stsd", b"stts", b"stsc", b"stsz", b"stco")
CONTAINERS = (b"moov", b"trak", b"mdia", b"minf", b"dinf", b"stbl")
HOSTILE_SEEDS = range(24)


def _tree_ok(data: bytes, pos: int = 0, end: int | None = None) -> bool:
    """Every box fits its parent and children tile each container exactly."""
    end = len(data) if end is None else end
    while pos < end:
        if pos + HDR > end:
            return False
        size = struct.unpack_from(">I", data, pos)[0]
        if size < HDR or pos + size > end:
            return False
        if data[pos + 4 : pos + 8] in CONTAINERS and not _tree_ok(data, pos + HDR, pos + size):
            return False
        pos += size
    return pos == end


def _types(data: bytes) -> set[bytes]:
    hits = find_boxes(data, frozenset(REQUIRED) | frozenset(SAMPLE_TABLE))
    return {data[h.start + 4 : h.start + 8] for h in hits}


def _complete() -> bytes:
    """Grow from ftyp alone until nothing is missing (always the first candidate)."""
    data = FTYP
    while (out := grow_mp4(data, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0, 0]))) is not None:
        data = out
    return data


def test_ftyp_grows_moov():
    """One draw on a bare ftyp yields moov down to the sample tables."""
    out = grow_mp4(FTYP, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0, 0]))
    assert out is not None and _tree_ok(out)
    assert out.startswith(FTYP)
    assert set(SAMPLE_TABLE) <= _types(out)


def test_stbl_under_minf():
    """minf without stbl: drawing the stbl candidate places it under that minf."""
    stub = FTYP + box(b"moov", box(b"trak", box(b"mdia", box(b"minf", b""))))
    idx = [t for _, t in _candidates(stub)].index(b"stbl")
    out = grow_mp4(stub, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[idx, 0]))
    assert out is not None and _tree_ok(out)
    (stbl,) = find_boxes(out, frozenset({b"stbl"}))
    (minf,) = find_boxes(out, frozenset({b"minf"}))
    assert minf.start in stbl.ancestors
    assert set(SAMPLE_TABLE) <= _types(out)


def test_draw_selects_candidate():
    """Scripted index picks which missing child grows (Hard Rule 39)."""
    stub = FTYP + box(b"moov", box(b"trak", b""))
    first = grow_mp4(stub, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0]))
    second = grow_mp4(stub, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[1]))
    assert first != second
    assert _tree_ok(first) and _tree_ok(second)


def test_complete_file_declines():
    """Falsification: nothing missing, nothing to grow."""
    full = _complete()
    assert _tree_ok(full)
    assert grow_mp4(full, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[0, 0])) is None


def test_chunk_offsets_shift():
    """Growing inside moov shifts an existing stco that points past it."""
    data = mp4(stsd(b"mp4v", b""))
    old = data.index(b"\xaa" * 16)
    idx = [t for _, t in _candidates(data)].index(b"mvhd")
    out = grow_mp4(data, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[idx]))
    assert out is not None and _tree_ok(out)
    stco = find_boxes(out, frozenset({b"stco"}))[0]
    offset = struct.unpack_from(">I", out, stco.start + HDR + 8)[0]
    assert out[offset : offset + 16] == b"\xaa" * 16
    assert offset > old


def test_max_len_declines():
    assert grow_mp4(FTYP, max_len=len(FTYP) + 16, rng=ScriptedRng(choice_idxs=[0, 0])) is None


@pytest.mark.parametrize("seed", HOSTILE_SEEDS)
def test_hostile_input(seed):
    """Adversarial: lying sizes, truncation, junk tails -> None or a valid tree prefix."""
    rng = RandPool(seed=seed)
    base = bytearray(_complete())
    pos = rng.randint(0, len(base) - 4)
    base[pos : pos + 4] = rng.randbytes(4)
    data = bytes(base[: rng.randint(HDR, len(base))])
    out = grow_mp4(data, max_len=MAX_LEN, rng=RandPool(seed=seed))
    assert out is None or len(out) > len(data)
    if out is not None and _tree_ok(data):
        assert _tree_ok(out)  # a well-formed input stays well-formed


def test_wiring():
    """Wiring: in REGISTRY, gated to ISO-BMFF heads, handler on OperatorEngine."""
    assert "mp4_box_grow" in REGISTRY.names()
    assert format_gate_matches("mp4_box_grow", FTYP) is True
    assert format_gate_matches("mp4_box_grow", b"RIFF" + bytes(12)) is False
    assert callable(getattr(OperatorEngine, "_op_mp4_box_grow", None))


def _handler(kind_fourcc: bytes) -> bytes:
    """hdlr box with *kind_fourcc* as handler_type."""
    return box(b"hdlr", bytes(8) + kind_fourcc + bytes(12) + b"\x00")


def _grow_type(data: bytes, typ: bytes, *kind_idx: int) -> bytes:
    idx = [t for _, t in _candidates(data)].index(typ)
    out = grow_mp4(data, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[idx, *kind_idx]))
    assert out is not None and _tree_ok(out)
    return out


def test_soun_grows_audio():
    """hdlr soun: minf grows smhd + mp4a/esds, no kind draw needed."""
    stub = FTYP + box(b"moov", box(b"trak", box(b"mdia", _handler(b"soun"))))
    out = _grow_type(stub, b"minf")
    types = _types(out) | {t for t in (b"smhd", b"mp4a", b"esds", b"vmhd") if t in out}
    assert {b"smhd", b"mp4a", b"esds"} <= types
    assert b"vmhd" not in types


def test_vide_draws_kind():
    """hdlr vide: the kind draw ranges over video kinds; index picks avc1 + avcC."""
    stub = FTYP + box(b"moov", box(b"trak", box(b"mdia", _handler(b"vide") + box(b"minf", b""))))
    avc = KINDS[b"vide"].index(TrackKind.AVC1)
    out = _grow_type(stub, b"stbl", avc)
    assert b"avc1" in out and b"avcC" in out and b"mp4a" not in out


def test_no_hdlr_coherent():
    """No hdlr: a grown trak is internally consistent (soun + smhd + mp4a)."""
    stub = FTYP + box(b"moov", b"")
    out = _grow_type(stub, b"trak", list(TrackKind).index(TrackKind.MP4A))
    assert b"soun" in out and b"smhd" in out and b"mp4a" in out and b"vide" not in out


def test_fragment_boxes_grow():
    """mvex under moov; moof (mfhd, traf(tfhd, tfdt, trun)) appended at the end."""
    full = _complete()
    assert {b"mvex", b"moof"} <= {
        full[h.start + 4 : h.start + 8] for h in find_boxes(full, frozenset({b"mvex", b"moof"}))
    }
    moof = find_boxes(full, frozenset({b"moof"}))[0]
    assert moof.start + moof.size == len(full)
    assert all(t in full for t in (b"trex", b"mfhd", b"tfhd", b"tfdt", b"trun"))


def test_regression_stco_at_eof():
    """A truncated stco ending the buffer: resizing declines, never raises."""
    stub = FTYP + box(
        b"moov", box(b"trak", box(b"mdia", box(b"minf", box(b"stbl", box(b"stco", bytes(4))))))
    )
    for idx in range(len(_candidates(stub))):
        out = grow_mp4(stub, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[idx, 0]))
        assert out is None or _tree_ok(out)


@pytest.mark.parametrize("anchor", [b"moof", b"sidx"])
def test_moov_not_before_offsets(anchor):
    """moof/sidx carry offsets a root moov insert would shift: decline it."""
    stub = FTYP + box(anchor, bytes(16))
    idx = [t for _, t in _candidates(stub)].index(b"moov")
    assert grow_mp4(stub, max_len=MAX_LEN, rng=ScriptedRng(choice_idxs=[idx, 0])) is None
