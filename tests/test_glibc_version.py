"""glibc little-chunk exposure model (Phrack #73 "Poor Little Chunks").

Pins the size-class/version matrix the severity multiplier gates on:
  - fallen_seven  (0x20-0x80, 7 sizes): lost its fences at glibc 2.43.
  - middle_fiftyfive (0x90-0x3F0, 55 sizes): lost its fences at glibc 2.41.

Detection is host-independent here: `_query_gnu_libc_version` is monkeypatched
and `detect(_refresh=True)` bypasses the process cache, so the tests assert the
model, not whatever glibc this box happens to run.
"""

import pytest

from fuzzer_tool.core import glibc_version as gv
from fuzzer_tool.core.glibc_version import (
    FASTBIN_KILLED,
    UNSORTED_CHECKS_DROPPED,
    LittleChunkExposure,
    detect,
    little_chunk_exposure,
    version_note,
)

# Version thresholds, as independent literals (not echoed from the module).
_FALLEN_V = (2, 43)
_MIDDLE_V = (2, 41)


def test_thresholds_match_phrack():
    assert UNSORTED_CHECKS_DROPPED == (2, 41)
    assert FASTBIN_KILLED == (2, 43)


# ── detect() ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2.41", (2, 41)),
        ("2.38", (2, 38)),
        ("2.41.9000", (2, 41)),  # leading major.minor taken
        ("glibc-nonsense", None),
        ("2", None),
        (None, (None)),  # query returned nothing -> unknown
    ],
)
def test_detect_parses_runtime_string(monkeypatch, raw, expected):
    monkeypatch.setattr(gv, "_query_gnu_libc_version", lambda: raw)
    assert detect(_refresh=True) == expected


def test_detect_is_cached(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return "2.40"

    monkeypatch.setattr(gv, "_query_gnu_libc_version", fake)
    first = detect(_refresh=True)
    second = detect()  # cached: no second query
    assert first == second == (2, 40)
    assert len(calls) == 1


# ── little_chunk_exposure: the size-class / version split ──────────────────


@pytest.mark.parametrize("size", [0x20, 0x40, 0x80])
def test_fallen_seven_only_exposed_at_2_43(size):
    # The seven fastbin sizes keep their fences on 2.41/2.42, lose them at 2.43.
    assert little_chunk_exposure(size, (2, 41)) is None
    assert little_chunk_exposure(size, (2, 42)) is None
    exp = little_chunk_exposure(size, _FALLEN_V)
    assert isinstance(exp, LittleChunkExposure)
    assert exp.size_class == "fallen_seven"
    assert exp.lost  # non-empty list of dropped protections


@pytest.mark.parametrize("size", [0x90, 0x200, 0x3F0])
def test_middle_fiftyfive_exposed_at_2_41(size):
    assert little_chunk_exposure(size, (2, 40)) is None
    exp = little_chunk_exposure(size, _MIDDLE_V)
    assert isinstance(exp, LittleChunkExposure)
    assert exp.size_class == "middle_fiftyfive"
    # Still exposed on a later glibc.
    assert little_chunk_exposure(size, (2, 43)) is not None


@pytest.mark.parametrize("size", [0x00, 0x10, 0x1F, 0x88, 0x400, 0x1000, -16])
def test_sizes_outside_little_chunk_range(size):
    # < MINSIZE, non-16-aligned-between-classes, or large chunk (>= 0x400).
    assert little_chunk_exposure(size, (2, 43)) is None


def test_none_version_is_never_exposed():
    assert little_chunk_exposure(0x200, None) is None
    assert little_chunk_exposure(0x40, None) is None


# ── falsification: a fastbin-size crash must NOT read as exposed on 2.41 ────


def test_fastbin_size_not_exposed_on_2_41():
    # The whole point of the size-class split: 0x40 on 2.41 is still fenced.
    assert little_chunk_exposure(0x40, (2, 41)) is None
    # ... while the same glibc exposes the middle range.
    assert little_chunk_exposure(0x200, (2, 41)) is not None


# ── version_note: size-unaware summary ─────────────────────────────────────


def test_version_note_pre_regression_empty():
    assert version_note((2, 40)) == ""
    assert version_note(None) == ""


def test_version_note_2_41_middle_only():
    note = version_note((2, 41))
    assert "0x90-0x3F0" in note
    assert "0x20-0x80" not in note  # fastbin still fenced on 2.41


def test_version_note_2_43_both_classes():
    note = version_note((2, 43))
    assert "0x20-0x80" in note
    assert "0x90-0x3F0" in note
