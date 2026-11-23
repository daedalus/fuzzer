"""Dispatch wiring of the MP4 stsd / atom operators (live engine, no mocks)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

from .support.mp4_build import avcc, esds, mp4, stsd

SPS = b"\x67\x42\xc0\x1e\xed\x08\xf2"
PPS = b"\x68\xce\x0f\xc8"
VIDEO = mp4(stsd(b"avc1", avcc(SPS, PPS)))
AUDIO = mp4(stsd(b"mp4a", esds(b"\x12\x10")))
TARGET = Path(__file__).resolve().parent.parent / "targets" / "test_target"
MP4_OPS = ("mp4_stsd_mutate", "mp4_atom_mutate", "mp4_atom_transplant")


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory() as tmp:
        corpus, crashes = Path(tmp) / "corpus", Path(tmp) / "crashes"
        corpus.mkdir()
        crashes.mkdir()
        yield Fuzzer(
            target=str(TARGET), corpus_dir=str(corpus), crashes_dir=str(crashes), max_len=4096
        )


def _declines(f, op: str) -> int:
    return f._op_declines.get(op, 0)


def test_ops_are_dispatched(fuzzer):
    for op in MP4_OPS:
        assert op in fuzzer._op_dispatch


def test_stsd_op_changes_mp4(fuzzer):
    out = fuzzer._op_dispatch["mp4_stsd_mutate"](bytearray(VIDEO), 0, VIDEO)
    assert bytes(out) != VIDEO or _declines(fuzzer, "mp4_stsd_mutate") == 1


def test_transplant_uses_corpus_donor(fuzzer):
    fuzzer.corpus[:] = [VIDEO, AUDIO]
    out = fuzzer._op_dispatch["mp4_atom_transplant"](bytearray(VIDEO), 0, VIDEO)
    assert bytes(out) != VIDEO
    assert _declines(fuzzer, "mp4_atom_transplant") == 0


def test_transplant_declines_without_donor(fuzzer):
    fuzzer.corpus[:] = [VIDEO]
    out = fuzzer._op_dispatch["mp4_atom_transplant"](bytearray(VIDEO), 0, VIDEO)
    assert bytes(out) == VIDEO
    assert _declines(fuzzer, "mp4_atom_transplant") == 1


def test_non_mp4_input_declines(fuzzer):
    junk = b"\x00\x01\x02\x03" * 8
    for op in ("mp4_stsd_mutate", "mp4_atom_mutate"):
        out = fuzzer._op_dispatch[op](bytearray(junk), 0, junk)
        assert bytes(out) == junk
        assert _declines(fuzzer, op) == 1
