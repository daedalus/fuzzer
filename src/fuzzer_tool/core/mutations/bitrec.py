"""Bit-record helpers: parse a header into fields, edit one, re-pack.

Fields are fixed-width (MSB-first) or exp-Golomb (``ue(v)``/``se(v)``,
H.264 §9.1). Bits past the last parsed field are kept verbatim as a
``'0'/'1'`` string, so a re-pack after an edit shifts them intact:

  [f0][f1]...[fn] | rest ............ | tail
                    kept verbatim       RBSP: stop bit + zero pad
                                        NONE: zero pad only

Strings, not ints: headers are tens of bytes, and slicing a bit string
keeps the reader and the re-pack trivially consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

MAX_UE_ZEROS = 31  # ue(v) codewords fit in 32-bit values


class Kind(IntEnum):
    """Field coding."""

    FIXED = 0
    UE = 1
    SE = 2


class Tail(IntEnum):
    """What follows the rest bits when re-packing."""

    NONE = 0  # zero pad to a byte
    RBSP = 1  # rbsp_stop_one_bit, then zero pad


@dataclass
class Field:
    """One parsed field; *width* is used only by ``Kind.FIXED``."""

    name: str
    kind: Kind
    width: int
    value: int


def ue_bits(value: int) -> str:
    """ue(v) codeword. Example: ``ue_bits(3) == "00100"``."""
    code = bin(value + 1)[2:]
    return "0" * (len(code) - 1) + code


def se_bits(value: int) -> str:
    """se(v) codeword: k > 0 -> 2k - 1, k <= 0 -> -2k."""
    return ue_bits(2 * value - 1 if value > 0 else -2 * value)


def _field_bits(f: Field) -> str:
    if f.kind is Kind.UE:
        return ue_bits(f.value)
    if f.kind is Kind.SE:
        return se_bits(f.value)
    return format(f.value & ((1 << f.width) - 1), f"0{f.width}b")


def pack(fields: list[Field], rest: str, tail: Tail) -> bytes:
    """Serialize *fields*, then *rest*, then the *tail* framing."""
    bits = "".join(_field_bits(f) for f in fields) + rest
    if tail is Tail.RBSP:
        bits += "1"
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""


class BitReader:
    """MSB-first reader that records nothing; raises ValueError past the end."""

    def __init__(self, data: bytes) -> None:
        self._bits = "".join(format(b, "08b") for b in data)
        self.pos = 0

    def read(self, width: int) -> int:
        end = self.pos + width
        if end > len(self._bits):
            raise ValueError("read past end")
        value = int(self._bits[self.pos : end], 2) if width else 0
        self.pos = end
        return value

    def ue(self) -> int:
        one = self._bits.find("1", self.pos)
        zeros = one - self.pos
        if one < 0 or zeros > MAX_UE_ZEROS:
            raise ValueError("bad exp-Golomb codeword")
        self.pos = one + 1
        return (1 << zeros) - 1 + self.read(zeros)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)

    def rest(self, tail: Tail) -> str:
        """Unread bits; for RBSP without trailing zeros and the stop bit."""
        bits = self._bits[self.pos :]
        if tail is Tail.NONE:
            return bits
        bits = bits.rstrip("0")
        return bits[:-1] if bits.endswith("1") else bits


class FieldParser:
    """BitReader that appends every value it reads to ``fields``."""

    def __init__(self, data: bytes) -> None:
        self.reader = BitReader(data)
        self.fields: list[Field] = []

    def u(self, name: str, width: int) -> int:
        value = self.reader.read(width)
        self.fields.append(Field(name, Kind.FIXED, width, value))
        return value

    def ue(self, name: str) -> int:
        value = self.reader.ue()
        self.fields.append(Field(name, Kind.UE, 0, value))
        return value

    def se(self, name: str) -> int:
        value = self.reader.se()
        self.fields.append(Field(name, Kind.SE, 0, value))
        return value
