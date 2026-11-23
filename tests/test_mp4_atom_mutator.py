"""Tests for atom-scoped mutation and transplant (`mp4_atom_mutate`, `mp4_atom_transplant`)."""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.mp4_atom import (
    Havoc,
    mutate_mp4_atom,
    transplant_atom,
)
from fuzzer_tool.core.mutations.mp4_path import find_boxes
from fuzzer_tool.core.operator_registry import format_gate_matches
from fuzzer_tool.core.rand_pool import RandPool

from .support.mp4_build import avcc, box, esds, mp4, stco, stsd
from .support.scripted_rng import ScriptedRng

SPS = b"\x67\x42\xc0\x1e\xed\x08\xf2"
PPS = b"\x68\xce\x0f\xc8"
FIX, KEEP = 1, 0
HDR = 8
STSD_FIRST = 0  # find_boxes order: stsd before stco in mp4()
MAX_LEN = 1 << 16
HOSTILE_SEEDS = range(24)
VIDEO = mp4(stsd(b"avc1", avcc(SPS, PPS)))
AUDIO = mp4(stsd(b"mp4a", esds(b"\x12\x10")))


def _atom(data: bytes, typ: bytes = b"stsd") -> tuple[int, int]:
    (hit,) = find_boxes(data, frozenset({typ}))
    return hit.start, hit.size


def _payload(data: bytes, typ: bytes = b"stsd") -> bytes:
    start, size = _atom(data, typ)
    return data[start + HDR : start + size]


def _tree_ok(data: bytes, pos: int = 0, end: int | None = None) -> bool:
    end = len(data) if end is None else end
    while pos < end:
        size = struct.unpack_from(">I", data, pos)[0]
        if size < HDR or pos + size > end:
            return False
        nested = data[pos + 4 : pos + 8] in (b"moov", b"trak", b"mdia", b"minf", b"stbl")
        if nested and not _tree_ok(data, pos + HDR, pos + size):
            return False
        pos += size
    return pos == end


def _stbl(leaf: bytes) -> bytes:
    """moov > trak > mdia > minf > stbl > leaf, nothing else."""
    for typ in (b"stbl", b"minf", b"mdia", b"trak", b"moov"):
        leaf = box(typ, leaf)
    return leaf


def _run(data: bytes, op: Havoc, *draws: int, **kw) -> bytes | None:
    rng = ScriptedRng(randints=[STSD_FIRST, int(op), *draws], **kw)
    return mutate_mp4_atom(data, MAX_LEN, rng=rng)


class TestHavoc:
    def test_flip_bit(self):
        out = _run(VIDEO, Havoc.FLIP, 0, 0, FIX)
        assert _payload(out)[0] == _payload(VIDEO)[0] ^ 1
        assert len(out) == len(VIDEO)

    def test_byte_edge(self):
        out = _run(VIDEO, Havoc.BYTE, 3, FIX, choice_idxs=[4])
        assert _payload(out)[3] == 0xFF

    def test_word_edge(self):
        out = _run(VIDEO, Havoc.WORD, 4, FIX, choice_idxs=[0, 0])
        assert _payload(out)[4:6] == b"\x00\x00"

    def test_insert_grows_everything(self):
        out = _run(VIDEO, Havoc.INSERT, 0, 3, FIX, randbytes=[b"\xaa\xbb\xcc"])
        assert len(out) == len(VIDEO) + 3
        assert _payload(out)[:3] == b"\xaa\xbb\xcc"
        assert _tree_ok(out)

    def test_insert_shifts_chunk_offsets(self):
        out = _run(VIDEO, Havoc.INSERT, 0, 3, FIX, randbytes=[b"\xaa\xbb\xcc"])
        old = struct.unpack_from(">I", VIDEO, VIDEO.index(b"stco") + 12)[0]
        new = struct.unpack_from(">I", out, out.index(b"stco") + 12)[0]
        assert new == old + 3

    def test_insert_keep_restores_size(self):
        out = _run(VIDEO, Havoc.INSERT, 0, 3, KEEP, randbytes=[b"\xaa\xbb\xcc"])
        start, _ = _atom(VIDEO)
        assert struct.unpack_from(">I", out, start)[0] == _atom(VIDEO)[1]
        assert len(out) == len(VIDEO) + 3

    def test_delete(self):
        out = _run(VIDEO, Havoc.DELETE, 0, 2, FIX)
        assert len(out) == len(VIDEO) - 2
        assert _tree_ok(out)

    def test_dup_span(self):
        out = _run(VIDEO, Havoc.DUP, 0, 4, FIX)
        pay = _payload(out)
        assert pay[:8] == _payload(VIDEO)[:4] * 2
        assert _tree_ok(out)

    def test_fourcc_preserved(self):
        for op, draws in ((Havoc.FLIP, (0, 0, FIX)), (Havoc.INSERT, (0, 1, FIX))):
            out = _run(VIDEO, op, *draws, randbytes=[b"\x00"])
            assert out[_atom(out)[0] + 4 : _atom(out)[0] + 8] == b"stsd"


class TestDecline:
    def test_no_atoms(self):
        rng = ScriptedRng(randints=[])
        assert mutate_mp4_atom(b"\x00" * 32, MAX_LEN, rng=rng) is None

    def test_word_in_short_payload(self):
        tiny = mp4(box(b"stsd", b"\x00"))
        assert _run(tiny, Havoc.WORD, 0, FIX, choice_idxs=[1, 0]) is None


class TestTransplant:
    def test_donor_atom_replaces_base(self):
        rng = ScriptedRng(randints=[0, 0, FIX])
        out = transplant_atom(VIDEO, AUDIO, MAX_LEN, rng=rng)
        assert _payload(out) == _payload(AUDIO)
        assert _tree_ok(out)

    def test_chunk_offsets_follow(self):
        rng = ScriptedRng(randints=[0, 0, FIX])
        out = transplant_atom(VIDEO, AUDIO, MAX_LEN, rng=rng)
        delta = len(AUDIO) - len(VIDEO)
        old = struct.unpack_from(">I", VIDEO, VIDEO.index(b"stco") + 12)[0]
        new = struct.unpack_from(">I", out, out.index(b"stco") + 12)[0]
        assert new == old + delta

    def test_no_shared_type_declines(self):
        base = _stbl(box(b"stsd", bytes(8)))
        rng = ScriptedRng(randints=[0, 0, 0])
        assert transplant_atom(base, _stbl(stco([0])), MAX_LEN, rng=rng) is None

    def test_donor_without_atoms_declines(self):
        rng = ScriptedRng(randints=[0, 0, 0])
        assert transplant_atom(VIDEO, b"\x00" * 16, MAX_LEN, rng=rng) is None

    def test_identical_donor_is_noop_declined(self):
        rng = ScriptedRng(randints=[0, 0])
        assert transplant_atom(VIDEO, VIDEO, MAX_LEN, rng=rng) is None


class TestGate:
    def test_matches(self):
        assert format_gate_matches("mp4_atom_mutate", VIDEO)
        assert format_gate_matches("mp4_atom_transplant", AUDIO)

    def test_rejects(self):
        assert not format_gate_matches("mp4_atom_mutate", b"\x00" * 32)


class TestAdversarial:
    def test_hostile_inputs_never_raise(self):
        hostile = [
            mp4(box(b"stsd", b"")),
            mp4(box(b"stsd", struct.pack(">I", 0xFFFFFFFF) * 4)),
            mp4(box(b"stts", bytes(4) + struct.pack(">I", 0xFFFFFFFF))),
            VIDEO[: len(VIDEO) // 2],
        ]
        for seed in HOSTILE_SEEDS:
            rng = RandPool(seed=seed)
            for data in [*hostile, VIDEO, AUDIO]:
                out = mutate_mp4_atom(data, MAX_LEN, rng=rng)
                assert out is None or len(out) <= MAX_LEN
                out = transplant_atom(data, AUDIO, MAX_LEN, rng=rng)
                assert out is None or len(out) <= MAX_LEN

    def test_max_len_caps(self):
        out = mutate_mp4_atom(VIDEO, 40, rng=RandPool(seed=5))
        assert out is None or len(out) <= 40

    def test_same_seed_same_output(self):
        a = mutate_mp4_atom(VIDEO, MAX_LEN, rng=RandPool(seed=9))
        b = mutate_mp4_atom(VIDEO, MAX_LEN, rng=RandPool(seed=9))
        assert a == b
