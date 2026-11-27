"""Tests for tools/corpus_mp4.py: ffmpeg-built MP4/MOV/3GP seeds."""

from __future__ import annotations

import shutil

import pytest
import tools.corpus_mp4 as gen

from fuzzer_tool.core.mutations.mp4_path import find_boxes

ISO_HEADS = {b"ftyp", b"moov", b"mdat", b"free", b"wide", b"styp", b"moof"}
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _has(data: bytes, typ: bytes) -> bool:
    return bool(find_boxes(data, frozenset({typ})))


def test_variant_table():
    """The table covers mp4, mov, 3gp, fragmented and faststart layouts."""
    fmts = {v.fmt for v in gen.VARIANTS}
    assert {"mp4", "mov", "3gp"} <= fmts
    flags = " ".join(" ".join(v.extra) for v in gen.VARIANTS)
    assert "frag_keyframe" in flags and "faststart" in flags
    assert len({v.name for v in gen.VARIANTS}) == len(gen.VARIANTS)


@needs_ffmpeg
def test_seeds_valid(tmp_path):
    """Every written seed is ISO-BMFF, bounded, and the set reaches stbl and moof."""
    written = gen.generate(tmp_path / "seeds")
    assert written
    blobs = [p.read_bytes() for p in written]
    assert all(b[4:8] in ISO_HEADS for b in blobs)
    assert all(len(b) <= gen.MAX_SEED_BYTES for b in blobs)
    assert any(_has(b, b"stbl") for b in blobs)
    if any(p.name.endswith("_frag") for p in written):  # needs libx264
        assert any(_has(b, b"moof") for b in blobs)


def test_missing_encoder(tmp_path, monkeypatch):
    """Falsification: a variant whose encoder is absent is skipped, never run."""
    ran: list[str] = []
    monkeypatch.setattr(gen, "_encoders", lambda: frozenset())
    monkeypatch.setattr(gen, "_encode", lambda v, out: ran.append(v.name) or None)
    assert gen.generate(tmp_path / "seeds") == []
    assert ran == []


def test_failed_encode(tmp_path, monkeypatch):
    """Adversarial: one failing or oversized encode does not stop the others."""
    first, second, third = gen.VARIANTS[:3]
    outputs = {first.name: None, second.name: b"\0" * (gen.MAX_SEED_BYTES + 1), third.name: b"x"}
    monkeypatch.setattr(gen, "VARIANTS", (first, second, third))
    monkeypatch.setattr(gen, "_encoders", lambda: gen.needed_encoders())
    monkeypatch.setattr(gen, "_encode", lambda v, out: outputs[v.name])
    written = gen.generate(tmp_path / "seeds")
    assert [p.name for p in written] == [gen.PREFIX + third.name]
