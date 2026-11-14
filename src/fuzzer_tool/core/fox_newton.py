"""FOX secant step: zero-exec fit of a compare operand against an input field.

After FOX (She et al., CCS '24, arXiv 2406.04517): learn a linear model of a
branch's operand from inputs that already reached it, then root-find the
field value that moves the operand onto the branch boundary.

``--affine-mod`` / ``--derived-cmp`` fit richer models but spend probe execs.
This fit spends none: it reuses the (input, PC, operand) records the cmplog
drain already parsed. Records are gated on to collect rounds only, so one
drain is one exec::

    run 1: field x=10  ->  PC 0x4010: 37 vs 1000     y = 3x + 7
    run 2: field x=20  ->  PC 0x4010: 67 vs 1000
    secant: k = (67-37)/(20-10) = 3;  x* = 10 + (1000-37)/3 = 331
    run 3 (optional) checks the line; x*x fails it and the site declines.

The window is where the two inputs differ (<= compare width): a LE field
starts at the first changed byte, a BE field ends at the last. Goals sit on
the sides of the constant no run has reached (``flip_goals``), which covers
==, <, > and their negations.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from itertools import islice
from typing import NamedTuple

import numpy as np

#: Executed inputs kept for fitting; bounds memory to MAX_INPUTS * max_len.
MAX_INPUTS = 16
#: Compare sites tracked (LRU).
MAX_SITES = 512
#: Records kept per site: two to fit, the rest to check the line.
OBS_PER_SITE = 4

# trace-cmp integer widths; byte compares carry no PC and are skipped.
_INT_WIDTHS = frozenset((1, 2, 4, 8))
# Signs of (operand - constant); the boundary value on side s is const + s.
_SIDES = (-1, 0, 1)


class _Obs(NamedTuple):
    gen: int
    src: int
    const: int
    width: int


class _Window(NamedTuple):
    start: int
    order: str


def flip_goals(signs: set[int], const: int, width: int) -> tuple[int, ...]:
    """Operand values that reach a side of *const* no run has reached.

    Example: only ``a < b`` seen -> ``(b, b + 1)`` (``==`` and ``>``).
    Every side seen -> all three boundary values.
    """
    unseen = [s for s in _SIDES if s not in signs] or _SIDES
    mask = (1 << 8 * width) - 1
    return tuple((const + s) & mask for s in unseen)


def _span(a: bytes, b: bytes) -> tuple[int, int] | None:
    """First and last differing offset of two equal-length inputs."""
    diff = np.flatnonzero(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8))
    if not diff.size:
        return None
    return int(diff[0]), int(diff[-1])


def _windows(first: int, last: int, width: int, n: int) -> list[_Window]:
    """LE field starting at *first*, BE field ending at *last*."""
    if last - first + 1 > width:
        return []
    cands = [_Window(first, "little"), _Window(last - width + 1, "big")]
    return [w for w in cands if 0 <= w.start <= n - width]


def _read(data: bytes, win: _Window, width: int) -> int:
    return int.from_bytes(data[win.start : win.start + width], win.order)


class SecantLog:
    """Per-site operand records tied to the inputs that produced them."""

    def __init__(self) -> None:
        self._gen = 0
        self._inputs: dict[int, bytes] = {}
        self._sites: dict[int, deque[_Obs]] = {}
        self._ready: list[int] | None = None

    # ── recording ────────────────────────────────────────────────────

    def observe(self, data: bytes, conds) -> None:
        """Log one exec's integer compares (``CondStmt`` list), first per PC."""
        recs = self._records(conds)
        if not recs:
            return

        self._gen += 1
        self._inputs[self._gen] = data
        if len(self._inputs) > MAX_INPUTS:
            del self._inputs[next(iter(self._inputs))]

        self._push(recs)
        self._ready = None

    def _records(self, conds) -> dict[int, _Obs]:
        # Hot per collect round: locals bound once (+17-39% at 100-5000 conds).
        recs: dict[int, _Obs] = {}
        gen = self._gen + 1
        widths = _INT_WIDTHS
        decode = int.from_bytes
        for cond in conds:
            base = cond.base
            pc = base.pc
            if pc is None or pc in recs:
                continue
            width = base.width
            a, b = base.op_a, base.op_b
            if width not in widths or len(a) != width or len(b) != width:
                continue
            recs[pc] = _Obs(gen, decode(a, "little"), decode(b, "little"), width)
        return recs

    def _push(self, recs: dict[int, _Obs]) -> None:
        # LRU: re-insert each site at the end, then drop the oldest past the
        # cap once per batch (+7-34% over evicting per insert, same result).
        sites = self._sites
        pop = sites.pop
        for pc, obs in recs.items():
            ring = pop(pc, None)
            if ring is None:
                ring = deque(maxlen=OBS_PER_SITE)
            ring.append(obs)
            sites[pc] = ring

        excess = len(sites) - MAX_SITES
        for pc in list(islice(sites, max(excess, 0))):
            del sites[pc]

    # ── queries ──────────────────────────────────────────────────────

    def live_inputs(self) -> int:
        return len(self._inputs)

    def site_count(self) -> int:
        return len(self._sites)

    def ready(self) -> bool:
        return bool(self._ready_sites())

    def _ready_sites(self) -> list[int]:
        if self._ready is None:
            self._ready = [pc for pc in self._sites if self._fit_set(pc)]
        return self._ready

    def _fit_set(self, pc: int) -> list[_Obs] | None:
        """Live records, oriented so ``src`` varies and ``const`` does not.

        Newest two (equal input length, distinct inputs) fit; older live
        ones of that length check. None when nothing fits.
        """
        live = [o for o in reversed(self._sites[pc]) if o.gen in self._inputs]
        if len(live) < 2:
            return None
        n = len(self._inputs[live[0].gen])
        same = [o for o in live if len(self._inputs[o.gen]) == n]
        if len(same) < 2 or self._inputs[same[0].gen] == self._inputs[same[1].gen]:
            return None

        o1, o2 = same[0], same[1]
        if o1.const != o2.const:
            # The input-derived side may be the right operand.
            same = [o._replace(src=o.const, const=o.src) for o in same]
            o1, o2 = same[0], same[1]
        if o1.const != o2.const or o1.src == o2.src:
            return None
        return same

    # ── solving ──────────────────────────────────────────────────────

    def propose(self, buf: bytes, rng) -> bytes | None:
        """*buf* with one site's field set to its secant root, else None.

        Draws: ``choice`` site, ``choice`` window, ``choice`` goal.
        """
        ready = self._ready_sites()
        if not ready:
            return None
        obs = self._fit_set(rng.choice(ready))
        if obs is None:
            return None

        fits = self._valid_windows(obs)
        if not fits:
            return None
        win = rng.choice(fits)

        width, const = obs[0].width, obs[0].const
        signs = {(o.src > const) - (o.src < const) for o in obs}
        goal = rng.choice(flip_goals(signs, const, width))
        return self._apply(buf, obs, win, goal)

    def _valid_windows(self, obs: list[_Obs]) -> list[_Window]:
        """Windows on which every checking record lies on the secant line."""
        d1, d2 = self._inputs[obs[0].gen], self._inputs[obs[1].gen]
        span = _span(d1, d2)
        if span is None:
            return []
        width = obs[0].width
        return [w for w in _windows(*span, width, len(d1)) if self._on_line(obs, w) is not None]

    def _on_line(self, obs: list[_Obs], win: _Window) -> Fraction | None:
        """Slope dy/dx of the newest two, if the older records agree."""
        width = obs[0].width
        xs = [_read(self._inputs[o.gen], win, width) for o in obs]
        if xs[0] == xs[1]:
            return None
        slope = Fraction(obs[0].src - obs[1].src, xs[0] - xs[1])
        for x, o in zip(xs[2:], obs[2:], strict=True):
            if obs[0].src + slope * (x - xs[0]) != o.src:
                return None
        return slope

    def _apply(self, buf: bytes, obs: list[_Obs], win: _Window, goal: int) -> bytes | None:
        """Newton/secant root ``x* = x1 + (goal - y1) / k``, rounded."""
        width = obs[0].width
        if len(buf) < win.start + width:
            return None
        slope = self._on_line(obs, win)
        x1 = _read(self._inputs[obs[0].gen], win, width)
        root = round(x1 + (goal - obs[0].src) / slope)
        if not 0 <= root < 1 << 8 * width:
            return None

        out = bytearray(buf)
        out[win.start : win.start + width] = root.to_bytes(width, win.order)
        out = bytes(out)
        return out if out != buf else None
