"""Byte classes shared by content-aware position schedulers.

Seven disjoint classes cover every byte value::

    0x00        CLS_ZERO      NUL padding / zeroed fields
    0xFF        CLS_FF        all-ones sentinels, -1
    0x01-0x1F   CLS_CONTROL   control bytes; 0x7F (DEL) is pinned here too
    0x30-0x39   CLS_DIGIT     ASCII digits
    A-Z a-z     CLS_ALPHA     ASCII letters
    other       CLS_PUNCT     remaining printable 0x20-0x7E (space, punctuation)
    0x80-0xFE   CLS_HIGH      high bytes

``BYTE_CLASS[b]`` is a 256-entry lookup table; index it directly on hot paths.
"""

from __future__ import annotations

CLS_ZERO = 0
CLS_FF = 1
CLS_CONTROL = 2
CLS_DIGIT = 3
CLS_ALPHA = 4
CLS_PUNCT = 5
CLS_HIGH = 6
NUM_CLASSES = 7

_DEL = 0x7F
_HIGH_START = 0x80
_SPACE = 0x20


def _classify(b: int) -> int:
    if b == 0x00:
        return CLS_ZERO
    if b == 0xFF:
        return CLS_FF
    if b >= _HIGH_START:
        return CLS_HIGH
    if b < _SPACE or b == _DEL:
        return CLS_CONTROL
    if ord("0") <= b <= ord("9"):
        return CLS_DIGIT
    if ord("A") <= b <= ord("Z") or ord("a") <= b <= ord("z"):
        return CLS_ALPHA
    return CLS_PUNCT


BYTE_CLASS = bytes(_classify(b) for b in range(256))


def byte_class(b: int) -> int:
    """Class of byte value ``b`` (0-255)."""
    return BYTE_CLASS[b]
