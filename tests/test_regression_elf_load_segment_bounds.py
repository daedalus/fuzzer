"""Regression: `find_load_segment` read program headers past the buffer.

Found by tools/fuzz.py: an e_phoff/e_phnum pointing outside the image
raised OverflowError/struct.error instead of returning None.
"""

from __future__ import annotations

import struct

from fuzzer_tool.core.elf import find_load_segment

E_PHOFF = 32
E_PHENTSIZE = 54
E_PHNUM = 56
PHDR_SIZE = 56
PT_LOAD = 1
VADDR = 0x400000


def _elf(phoff: int, phentsize: int, phnum: int, size: int = 64) -> bytearray:
    buf = bytearray(size)
    buf[:4] = b"\x7fELF"
    struct.pack_into("<Q", buf, E_PHOFF, phoff)
    struct.pack_into("<H", buf, E_PHENTSIZE, phentsize)
    struct.pack_into("<H", buf, E_PHNUM, phnum)
    return buf


def test_regression_phoff_huge_returns_none():
    assert find_load_segment(bytes(_elf((1 << 64) - 1, PHDR_SIZE, 1)), VADDR) is None


def test_regression_phdr_past_end_returns_none():
    assert find_load_segment(bytes(_elf(60, PHDR_SIZE, 3)), VADDR) is None


def test_falsify_in_bounds_load_still_found():
    # One valid PT_LOAD right after the header: bounds check must not hide it.
    buf = _elf(64, PHDR_SIZE, 1, 64 + PHDR_SIZE)
    struct.pack_into("<I", buf, 64, PT_LOAD)
    struct.pack_into("<QQQQQ", buf, 64 + 8, 0, VADDR, VADDR, 0x100, 0x200)

    assert find_load_segment(bytes(buf), VADDR + 1) == (VADDR, 0x100, 0x200)


def test_adversarial_all_ff_header():
    assert find_load_segment(b"\x7fELF" + b"\xff" * 200, VADDR) is None
