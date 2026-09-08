"""Regression tests for finding #23 — unguarded section-header reads.

``branch_density``/``_text_size``/``extract_constants_pure``/
``extract_div_constants`` derived every section offset from a header field
without checking it against the buffer, so a malformed target ELF raised
``struct.error`` out of fuzzer startup instead of being declined.

Each test here crafts a header that is well-formed enough to reach the read
that used to blow up, then asserts the function returns its documented
"analysis failed" value.  The functions are also exercised against a real
binary so the guards cannot pass by rejecting everything.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

from fuzzer_tool.core.elf import (
    _find_text_section,
    _text_size,
    branch_density,
    extract_constants_pure,
    extract_div_constants,
)

# (callable, value returned when the image cannot be analysed)
ANALYSERS = [
    (branch_density, None),
    (_text_size, None),
    (extract_constants_pure, []),
    (extract_div_constants, ({}, set())),
]

ANALYSER_IDS = ["branch_density", "_text_size", "extract_constants_pure", "extract_div_constants"]


def _elf64(
    *,
    e_shoff: int,
    e_shnum: int,
    e_shentsize: int,
    e_shstrndx: int,
    body: bytes = b"",
) -> bytes:
    """Build a minimal ELF64 LSB header with the given section-table fields."""
    elf = bytearray(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56)
    struct.pack_into("<Q", elf, 40, e_shoff)
    struct.pack_into("<H", elf, 58, e_shentsize)
    struct.pack_into("<H", elf, 60, e_shnum)
    struct.pack_into("<H", elf, 62, e_shstrndx)
    return bytes(elf) + body


@pytest.fixture
def write_elf(tmp_path: Path):
    counter = [0]

    def _write(data: bytes) -> str:
        counter[0] += 1
        p = tmp_path / f"target{counter[0]}.elf"
        p.write_bytes(data)
        return str(p)

    return _write


# --- e_shoff far past end of file ------------------------------------------
# This is the exact shape from the finding: the shstrtab entry is read at
# e_shoff + e_shstrndx * e_shentsize + 24, and nothing bounded e_shoff.


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_shoff_past_eof_does_not_raise(fn, sentinel, write_elf):
    path = write_elf(_elf64(e_shoff=0xDEADBEEF, e_shnum=5, e_shentsize=64, e_shstrndx=1))
    assert fn(path) == sentinel


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_shoff_just_past_eof_does_not_raise(fn, sentinel, write_elf):
    # One byte over is as fatal as four billion, and is the case an
    # "is it plausible?" heuristic would wave through.
    body = b"\x00" * 256
    path = write_elf(
        _elf64(e_shoff=64 + len(body), e_shnum=1, e_shentsize=64, e_shstrndx=0, body=body)
    )
    assert fn(path) == sentinel


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_table_overruns_eof_does_not_raise(fn, sentinel, write_elf):
    # e_shoff itself is inside the file; e_shnum * e_shentsize is not.
    body = b"\x00" * 128
    path = write_elf(_elf64(e_shoff=64, e_shnum=4096, e_shentsize=64, e_shstrndx=0, body=body))
    assert fn(path) == sentinel


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_undersized_shentsize_does_not_raise(fn, sentinel, write_elf):
    # The per-entry `sh + e_shentsize > len(elf)` guard the old loops had
    # passes here for every entry, while `sh + 32` still reads past the end.
    body = b"\x00" * 16
    path = write_elf(_elf64(e_shoff=64, e_shnum=2, e_shentsize=8, e_shstrndx=0, body=body))
    assert fn(path) == sentinel


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_shoff_zero_does_not_raise(fn, sentinel, write_elf):
    # No section headers at all: the old code read the shstrtab entry out of
    # the ELF header's own bytes.
    path = write_elf(_elf64(e_shoff=0, e_shnum=3, e_shentsize=64, e_shstrndx=0))
    assert fn(path) == sentinel


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_truncated_file_does_not_raise(fn, sentinel, write_elf):
    path = write_elf(b"\x7fELF\x02\x01" + b"\x00" * 10)
    assert fn(path) == sentinel


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_not_an_elf_does_not_raise(fn, sentinel, write_elf):
    path = write_elf(b"MZ" + b"\x00" * 512)
    assert fn(path) == sentinel


@pytest.mark.parametrize("fn,sentinel", ANALYSERS, ids=ANALYSER_IDS)
def test_missing_file_does_not_raise(fn, sentinel, tmp_path):
    assert fn(str(tmp_path / "nope")) == sentinel


# --- the guards must not simply reject everything ---------------------------


def test_real_binary_still_analysable():
    """A guard that declines every image would pass every test above."""
    real = sys.executable
    found = _find_text_section(Path(real).read_bytes())
    assert found is not None, "failed to locate .text in the running interpreter"
    text_data, sh_addr, sh_size = found
    assert len(text_data) > 0
    assert sh_size == len(text_data)
    assert sh_addr > 0

    assert _text_size(real) == sh_size
    density = branch_density(real)
    assert density is not None and density > 0


def test_shstrtab_offset_past_eof_finds_nothing(write_elf):
    """A bogus shstrtab offset makes names read empty, not raise."""
    # One section header, sh_type = SHT_PROGBITS, with sh_name pointing into
    # a string table that lives past the end of the file.
    shdr = bytearray(64)
    struct.pack_into("<I", shdr, 0, 4)  # sh_name
    struct.pack_into("<I", shdr, 4, 1)  # sh_type = SHT_PROGBITS
    struct.pack_into("<Q", shdr, 24, 0xFFFFFFFF)  # sh_offset, way past EOF
    path = write_elf(_elf64(e_shoff=64, e_shnum=1, e_shentsize=64, e_shstrndx=0, body=bytes(shdr)))
    assert _find_text_section(Path(path).read_bytes()) is None
