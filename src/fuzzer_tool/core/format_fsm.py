"""Constraint-labelled FSM message formats.

StateLifter (Shi, Xu, Zhang, USENIX Sec '23, arXiv:2305.13483) infers a
protocol's message format as a finite state machine whose transitions are
labelled with byte constraints -- a regular expression with constraints
attached ("ce-regex"), e.g. ``(a|b)+c`` with ``a % 10 == 4``. §6.3 feeds
the FSM to fuzzers by solving transition constraints into valid messages.

This module is that consumer half: parse an FSM spec, generate messages by
walking it (each label sampled exactly, no rejection), and re-walk an input
to keep a valid prefix and regenerate the tail. The inference
half (static loop analysis over LLVM bitcode) is not here; a spec comes from
StateLifter's output or is written by hand.

Spec, one directive per line, ``#`` comments::

    start A            # one or more start states
    final D            # one or more final states
    A -> B : 'a' | 'b' # byte set: 'c', 0x61, \\x61, ranges 'a'-'z', any
    B -> C : "GET "    # literal byte string
    C -> D : u16le[11,65535]%10=4   # integer: u8 u16le u16be u32le u32be

Walk::

    start ──label──► s1 ──label──► ... ──► final   (stop, or continue)
             │
             └─ only labels whose cheapest completion still fits max_len

Not modelled: StateLifter's induction states (``A_k``, "loop k times") and
constraints across transitions (a length field sizing a later payload).
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from fuzzer_tool.core.grammar import decode_quoted_literal

BYTE_VALUES = 256
_ARROW = "->"
_START = "start"
_FINAL = "final"


class Endian(Enum):
    LITTLE = "little"
    BIG = "big"


# Integer kind -> (width in bytes, byte order).
_INT_KINDS: dict[str, tuple[int, Endian]] = {
    "u8": (1, Endian.LITTLE),
    "u16le": (2, Endian.LITTLE),
    "u16be": (2, Endian.BIG),
    "u32le": (4, Endian.LITTLE),
    "u32be": (4, Endian.BIG),
}

_INT_RE = re.compile(
    r"(u8|u16le|u16be|u32le|u32be)\[\s*(\w+)\s*,\s*(\w+)\s*\]"
    r"(?:\s*%\s*(\w+)\s*=\s*(\w+))?"
)
_ATOM = r"(?:'(?:\\.|[^'\\])'|0x[0-9a-fA-F]{1,2}|\\x[0-9a-fA-F]{2})"
_SET_ITEM_RE = re.compile(rf"\s*(?:(any)|({_ATOM})(?:\s*-\s*({_ATOM}))?)\s*(\||$)")
_EDGE_RE = re.compile(r"(\w+)\s*->\s*(\w+)\s*:\s*(.+)")


def _index(n: int, rng) -> int:
    """Uniform index in [0, n); a forced choice (n == 1) draws nothing."""
    return 0 if n == 1 else rng.randint(0, n - 1)


def _pick(seq, rng):
    return seq[_index(len(seq), rng)]


@dataclass(frozen=True)
class ByteSet:
    """One byte drawn from a fixed set."""

    values: bytes
    # 256-entry membership table: O(1) match instead of scanning values.
    mask: bytes = b""

    width = 1

    def __post_init__(self):
        table = bytearray(BYTE_VALUES)
        for v in self.values:
            table[v] = 1
        object.__setattr__(self, "mask", bytes(table))

    def sample(self, rng) -> bytes:
        if len(self.values) == 1:
            return self.values
        return bytes((rng.choice(self.values),))

    def match(self, data: bytes, pos: int) -> int:
        """Bytes consumed at ``pos``, or -1."""
        if pos < len(data) and self.mask[data[pos]]:
            return 1
        return -1


@dataclass(frozen=True)
class Literal:
    """A fixed byte string."""

    value: bytes

    @property
    def width(self) -> int:
        return len(self.value)

    def sample(self, _rng) -> bytes:
        return self.value

    def match(self, data: bytes, pos: int) -> int:
        return len(self.value) if data.startswith(self.value, pos) else -1


@dataclass(frozen=True)
class IntRange:
    """A fixed-width integer in [lo, hi] with ``v % mod == rem``."""

    width: int
    endian: Endian
    lo: int
    hi: int
    mod: int
    rem: int

    @property
    def first(self) -> int:
        return self.lo + (self.rem - self.lo) % self.mod

    @property
    def count(self) -> int:
        return max(0, (self.hi - self.first) // self.mod + 1)

    def sample(self, rng) -> bytes:
        # Direct index into the arithmetic progression: exact, one draw.
        v = self.first + self.mod * _index(self.count, rng)
        return v.to_bytes(self.width, self.endian.value)

    def match(self, data: bytes, pos: int) -> int:
        end = pos + self.width
        if end > len(data):
            return -1
        v = int.from_bytes(data[pos:end], self.endian.value)
        if self.lo <= v <= self.hi and v % self.mod == self.rem % self.mod:
            return self.width
        return -1


Label = ByteSet | Literal | IntRange


@dataclass(frozen=True)
class Transition:
    src: str
    dst: str
    label: Label


class FormatFsm:
    """A parsed FSM: generate, match and regenerate messages."""

    def __init__(self, starts: list[str], finals: set[str], edges: list[Transition]):
        self.starts = starts
        self.finals = finals
        self.edges = edges
        self._out: dict[str, list[Transition]] = {}
        for t in edges:
            self._out.setdefault(t.src, []).append(t)

        self._cost = self._costs()
        self._live = {s: [t for t in ts if t.dst in self._cost] for s, ts in self._out.items()}
        self._starts_live = [s for s in starts if s in self._cost]
        # Largest "label + cheapest completion" per state: with that much
        # room every live label fits and the per-step filter is skipped.
        self._max_need = {
            s: max((t.label.width + self._cost[t.dst] for t in ts), default=0)
            for s, ts in self._live.items()
        }
        if not self._starts_live:
            raise ValueError("no start state reaches a final state")

    def _costs(self) -> dict[str, int]:
        """Fewest bytes from each state to a final (Dijkstra on reversed
        edges, weight = label width); absent = dead state."""
        back: dict[str, list[Transition]] = {}
        for t in self.edges:
            back.setdefault(t.dst, []).append(t)

        cost: dict[str, int] = {}
        heap = [(0, f) for f in self.finals]
        while heap:
            c, s = heapq.heappop(heap)
            if s in cost:
                continue
            cost[s] = c
            for t in back.get(s, ()):
                if t.src not in cost:
                    heapq.heappush(heap, (c + t.label.width, t.src))
        return cost

    def generate(self, rng, max_len: int) -> bytes | None:
        """One message of at most ``max_len`` bytes; None if none fits."""
        starts = [s for s in self._starts_live if self._cost[s] <= max_len]
        if not starts:
            return None
        return self._walk(_pick(starts, rng), bytearray(), rng, max_len)

    def _walk(self, s: str, out: bytearray, rng, max_len: int) -> bytes:
        """Walk to a final state without overrunning ``max_len``.

        Caller guarantees ``len(out) + cost[s] <= max_len``. Only labels
        whose cheapest completion still fits are offered, so the invariant
        holds every step and a non-final state always has an option.
        """
        while True:
            room = max_len - len(out)
            fits = self._live.get(s, [])
            if room < self._max_need.get(s, 0):
                fits = [t for t in fits if t.label.width + self._cost[t.dst] <= room]

            # A final state offers "stop" as one more option.
            n = len(fits) + (1 if s in self.finals else 0)
            k = _index(n, rng)
            if k == len(fits):
                return bytes(out)
            t = fits[k]
            out += t.label.sample(rng)
            s = t.dst

    def prefix_states(self, data: bytes) -> list[tuple[int, str]]:
        """Every (offset, state) reachable by consuming ``data[:offset]``.

        Offsets only grow, so one sweep over positions settles each pair
        once: at most ``(len(data) + 1) * n_states`` pairs.
        """
        out = self._out
        frontier: dict[int, set[str]] = {0: set(self.starts)}
        pairs: list[tuple[int, str]] = []
        n = len(data)
        for pos in range(n + 1):
            if not frontier:
                break
            states = frontier.pop(pos, None)
            if not states:
                continue

            for s in states if len(states) == 1 else sorted(states):
                pairs.append((pos, s))
                for t in out.get(s, ()):
                    w = t.label.match(data, pos)
                    if w < 0:
                        continue
                    frontier.setdefault(pos + w, set()).add(t.dst)
        return pairs

    def accepts(self, data: bytes) -> bool:
        n = len(data)
        return any(p == n and s in self.finals for p, s in self.prefix_states(data))

    def regenerate(self, data: bytes, rng, max_len: int) -> bytes | None:
        """Keep a random valid prefix of ``data``, regenerate a valid tail.

        The prefix is picked uniformly, not the longest: always taking the
        longest only ever appends to a valid message. Pairs that are dead,
        or whose cheapest completion overruns ``max_len``, are skipped; if
        none is left the message is generated from scratch (None if even
        that cannot fit).
        """
        cost = self._cost
        pairs = [
            (p, s) for p, s in self.prefix_states(data) if s in cost and p + cost[s] <= max_len
        ]
        if not pairs:
            return self.generate(rng, max_len)

        pos, s = _pick(pairs, rng)
        return self._walk(s, bytearray(data[:pos]), rng, max_len)


def _atom_byte(atom: str) -> int:
    if atom.startswith("'"):
        b = decode_quoted_literal(atom[1:-1])
    elif atom.startswith("0x"):
        b = bytes((int(atom, 16),))
    else:
        b = bytes((int(atom[2:], 16),))
    if len(b) != 1:
        raise ValueError(f"not a single byte: {atom}")
    return b[0]


def _parse_set(text: str) -> ByteSet:
    """``'a' | 'b'-'z' | 0x00 | any`` -> ByteSet."""
    seen = bytearray(BYTE_VALUES)
    pos = 0
    n = len(text)
    while pos < n:
        m = _SET_ITEM_RE.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"bad byte set: {text!r}")
        pos = m.end()

        # A trailing '|' with nothing after it is a syntax error.
        if m.group(4) == "|" and pos >= n:
            raise ValueError(f"dangling '|': {text!r}")

        if m.group(1):
            seen[:] = b"\x01" * BYTE_VALUES
            continue
        lo = _atom_byte(m.group(2))
        hi = _atom_byte(m.group(3)) if m.group(3) else lo
        if lo > hi:
            raise ValueError(f"inverted range: {m.group(0).strip()}")
        seen[lo : hi + 1] = b"\x01" * (hi - lo + 1)

    values = bytes(i for i in range(BYTE_VALUES) if seen[i])
    if not values:
        raise ValueError(f"empty byte set: {text!r}")
    return ByteSet(values)


def _parse_int(m: re.Match) -> IntRange:
    width, endian = _INT_KINDS[m.group(1)]
    lo, hi = int(m.group(2), 0), int(m.group(3), 0)
    mod = int(m.group(4), 0) if m.group(4) else 1
    rem = int(m.group(5), 0) if m.group(5) else 0
    top = (1 << (8 * width)) - 1
    if not 0 <= lo <= hi <= top or mod < 1:
        raise ValueError(f"bad integer range: {m.group(0)}")

    r = IntRange(width, endian, lo, hi, mod, rem % mod)
    if r.count == 0:
        raise ValueError(f"no value satisfies: {m.group(0)}")
    return r


def _parse_label(text: str) -> Label:
    text = text.strip()
    m = _INT_RE.fullmatch(text)
    if m:
        return _parse_int(m)

    if text.startswith('"'):
        if len(text) < 2 or not text.endswith('"'):
            raise ValueError(f"bad literal: {text}")
        value = decode_quoted_literal(text[1:-1])
        if not value:
            raise ValueError("empty literal")
        return Literal(value)

    return _parse_set(text)


def _strip_comment(line: str) -> str:
    """Cut at the first ``#`` outside quotes: ``'#'`` and ``"a#b"`` stay."""
    quote = ""
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if quote and c == "\\":
            i += 2
            continue
        if c in "'\"" and (not quote or c == quote):
            quote = "" if quote else c
        elif c == "#" and not quote:
            return line[:i]
        i += 1
    return line


def parse_fsm(spec: str) -> FormatFsm:
    """Parse the spec format described in the module docstring."""
    starts: list[str] = []
    finals: set[str] = set()
    edges: list[Transition] = []

    for raw in spec.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue

        head, _, rest = line.partition(" ")
        if head == _START:
            starts.extend(rest.split())
            continue
        if head == _FINAL:
            finals.update(rest.split())
            continue

        m = _EDGE_RE.fullmatch(line)
        if not m or _ARROW not in line:
            raise ValueError(f"bad line: {line!r}")
        edges.append(Transition(m.group(1), m.group(2), _parse_label(m.group(3))))

    if not starts or not finals:
        raise ValueError("spec needs 'start' and 'final' lines")
    return FormatFsm(starts, finals, edges)


def load_fsm(path: str | Path) -> FormatFsm:
    return parse_fsm(Path(path).read_text())
