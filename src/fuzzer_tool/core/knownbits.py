"""Known-bits abstract domain with a logarithmic barrel-shifter ``shl``.

Each bit of a value is 0, 1 or ? (unknown). ``0000?101 << ?0`` is
``00???10?``: the union of ``0000?101`` (shift 0) and ``00?10100`` (shift 2).

``shl`` mirrors a hardware barrel shifter: one stage per shift-amount bit,
each stage a mux between "shifted" and "copy". An unknown control bit
unions both branches. Cost is log2(width) stages, not width shifts, and the
result equals the optimal abstraction (see tests/test_knownbits.py).

    amount bit i ──► mux ──► next stage
                      ▲ ▲
         value << 2**i  └ value
"""

from __future__ import annotations

from dataclasses import dataclass

_UNKNOWN_CHAR = "?"


def _mask(width: int) -> int:
    return (1 << width) - 1


@dataclass(frozen=True)
class KnownBits:
    """Unsigned fixed-width value; ``ones`` and ``unknowns`` never overlap."""

    ones: int
    unknowns: int
    width: int

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError("width must be positive")
        full = _mask(self.width)
        if self.ones & ~full or self.unknowns & ~full:
            raise ValueError("bits exceed width")
        if self.ones & self.unknowns:
            raise ValueError("bit both one and unknown")

    @staticmethod
    def const(value: int, width: int) -> KnownBits:
        return KnownBits(value & _mask(width), 0, width)

    @staticmethod
    def all_unknown(width: int) -> KnownBits:
        return KnownBits(0, _mask(width), width)

    @staticmethod
    def from_str(text: str) -> KnownBits:
        """'0', '1', '?' chars, MSB first; width is the string length."""
        ones = unknowns = 0
        for ch in text:
            if ch not in "01" + _UNKNOWN_CHAR:
                raise ValueError(f"bad char {ch!r}")
            ones = ones << 1 | (ch == "1")
            unknowns = unknowns << 1 | (ch == _UNKNOWN_CHAR)
        return KnownBits(ones, unknowns, len(text))

    def __str__(self) -> str:
        out = []
        for i in reversed(range(self.width)):
            if self.unknowns >> i & 1:
                out.append(_UNKNOWN_CHAR)
                continue
            out.append("1" if self.ones >> i & 1 else "0")
        return "".join(out)

    def contains(self, value: int) -> bool:
        known = ~self.unknowns & _mask(self.width)
        return value & known == self.ones

    def union(self, other: KnownBits) -> KnownBits:
        """Tightest KnownBits holding both: keep only bits that agree."""
        known = ~(self.unknowns | other.unknowns | (self.ones ^ other.ones))
        known &= _mask(self.width)
        return KnownBits(self.ones & known, ~known & _mask(self.width), self.width)

    def bit(self, i: int) -> KnownBits:
        """Bit i as a width-1 value; bits past the width are known 0."""
        return KnownBits(self.ones >> i & 1, self.unknowns >> i & 1, 1)

    def shl_const(self, dist: int) -> KnownBits:
        dist = min(dist, self.width)
        full = _mask(self.width)
        return KnownBits((self.ones << dist) & full, (self.unknowns << dist) & full, self.width)

    def _mux(self, sel: KnownBits, shifted: KnownBits) -> KnownBits:
        """self if sel is 0, shifted if 1, union if unknown."""
        if sel.unknowns:
            return self.union(shifted)
        return shifted if sel.ones else self

    def _in_range(self, limit: int) -> KnownBits:
        """Is this unsigned value < limit? Width-1 result."""
        if self.ones | self.unknowns < limit:
            return KnownBits(1, 0, 1)
        if self.ones >= limit:
            return KnownBits(0, 0, 1)
        return KnownBits(0, 1, 1)

    def shl(self, amount: KnownBits) -> KnownBits:
        """Abstract ``self << amount`` truncated to ``self.width``."""
        out = self
        for i in range((self.width - 1).bit_length()):
            out = out._mux(amount.bit(i), out.shl_const(1 << i))

        zero = KnownBits.const(0, self.width)
        # Amounts >= width shift everything out: result is 0.
        return zero._mux(amount._in_range(self.width), out)
