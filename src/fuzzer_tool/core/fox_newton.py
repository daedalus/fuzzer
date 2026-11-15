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


class _Fit(NamedTuple):
    """Newest-first fitting pair, oriented, with the windows it fits on."""

    new: _Obs
    old: _Obs
    windows: tuple[_Window, ...]
    signs: frozenset[int]


def _orient(new: _Obs, old: _Obs) -> tuple[_Obs, _Obs] | None:
    """Pair with ``src`` varying and ``const`` fixed; swap if the input side is b."""
    if new.const == old.const and new.src != old.src:
        return new, old
    if new.src == old.src and new.const != old.const:
        return new._replace(src=new.const, const=new.src), old._replace(
            src=old.const, const=old.src
        )
    return None


def _sign(obs: _Obs) -> int:
    return (obs.src > obs.const) - (obs.src < obs.const)


class SecantLog:
    """Per-site operand records tied to the inputs that produced them.

    ``observe`` refits only the sites it touched, so ``ready`` is a cached
    list and ``propose`` never searches::

        observe(run)  ->  spans(run vs each live input)  ->  refit touched PCs
        propose       ->  pick fitted PC  ->  window  ->  goal  ->  root
    """

    def __init__(self) -> None:
        self._gen = 0
        self._inputs: dict[int, bytes] = {}
        self._sites: dict[int, deque[_Obs]] = {}
        # (older gen, newer gen) -> first/last differing byte; same-length only.
        self._spans: dict[tuple[int, int], tuple[int, int]] = {}
        self._fits: dict[int, _Fit] = {}
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
            self._drop_input(next(iter(self._inputs)))
        self._add_spans(self._gen)

        self._push(recs)
        for pc in recs:
            self._refit(pc)
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

    def _drop_input(self, gen: int) -> None:
        del self._inputs[gen]
        for key in [k for k in self._spans if gen in k]:
            del self._spans[key]

    def _add_spans(self, gen: int) -> None:
        # One numpy compare per live input (<= MAX_INPUTS), shared by all sites.
        data = self._inputs[gen]
        n = len(data)
        for old, other in self._inputs.items():
            if old == gen or len(other) != n:
                continue
            span = _span(other, data)
            if span is not None:
                self._spans[(old, gen)] = span

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
            self._fits.pop(pc, None)

    # ── fitting ──────────────────────────────────────────────────────

    def _refit(self, pc: int) -> None:
        fit = self._fit(pc)
        if fit is None:
            self._fits.pop(pc, None)
        else:
            self._fits[pc] = fit

    def _fit(self, pc: int) -> _Fit | None:
        """Newest pair one field-edit apart whose secant the checks accept."""
        live = [o for o in reversed(self._sites[pc]) if o.gen in self._inputs]
        n = len(live)
        for i in range(n):
            for j in range(i + 1, n):
                fit = self._fit_pair(live[i], live[j], live)
                if fit is not None:
                    return fit
        return None

    def _fit_pair(self, new: _Obs, old: _Obs, live: list[_Obs]) -> _Fit | None:
        span = self._spans.get((old.gen, new.gen))
        pair = _orient(new, old) if span is not None else None
        if pair is None:
            return None

        # Checks read the same side as the pair.
        if pair[0] is not new:
            live = [o._replace(src=o.const, const=o.src) for o in live]
        width, size = new.width, len(self._inputs[new.gen])
        wins = tuple(w for w in _windows(*span, width, size) if self._agrees(pair, w, live))
        if not wins:
            return None
        return _Fit(*pair, wins, frozenset((_sign(pair[0]), _sign(pair[1]))))

    def _agrees(self, pair: tuple[_Obs, _Obs], win: _Window, live: list[_Obs]) -> bool:
        """Records whose input differs from the pair's only inside *win* lie on the line.

        Example: runs x=10, 20, 30 at one field; a line through two of them
        must predict the third (x*x fails). A run that also changed other
        bytes says nothing about this field and is skipped.
        """
        new, old = pair
        slope = self._slope(new, old, win)
        if slope is None:
            return False

        lo, hi = win.start, win.start + new.width - 1
        x_new = _read(self._inputs[new.gen], win, new.width)
        for o in live:
            if o.gen in (new.gen, old.gen) or not self._inside(o.gen, new.gen, lo, hi):
                continue
            x = _read(self._inputs[o.gen], win, new.width)
            if new.src + slope * (x - x_new) != o.src:
                return False
        return True

    def _inside(self, a: int, b: int, lo: int, hi: int) -> bool:
        span = self._spans.get((min(a, b), max(a, b)))
        return span is not None and lo <= span[0] and span[1] <= hi

    def _slope(self, new: _Obs, old: _Obs, win: _Window) -> Fraction | None:
        x_new = _read(self._inputs[new.gen], win, new.width)
        x_old = _read(self._inputs[old.gen], win, new.width)
        if x_new == x_old:
            return None
        return Fraction(new.src - old.src, x_new - x_old)

    # ── queries ──────────────────────────────────────────────────────

    def live_inputs(self) -> int:
        return len(self._inputs)

    def site_count(self) -> int:
        return len(self._sites)

    def ready(self) -> bool:
        return bool(self._ready_sites())

    def _ready_sites(self) -> list[int]:
        # Fits go stale only when observe evicts an input; observe resets this.
        if self._ready is None:
            live = self._inputs
            stale = [
                pc for pc, f in self._fits.items() if f.new.gen not in live or f.old.gen not in live
            ]
            for pc in stale:
                del self._fits[pc]
            self._ready = list(self._fits)
        return self._ready

    # ── solving ──────────────────────────────────────────────────────

    def propose(self, buf: bytes, rng) -> bytes | None:
        """*buf* with one site's field set to its secant root, else None.

        Draws: ``choice`` site, ``choice`` window, ``choice`` goal.
        """
        ready = self._ready_sites()
        if not ready:
            return None
        fit = self._fits[rng.choice(ready)]
        win = rng.choice(fit.windows)
        goal = rng.choice(flip_goals(set(fit.signs), fit.new.const, fit.new.width))
        return self._apply(buf, fit, win, goal)

    def _apply(self, buf: bytes, fit: _Fit, win: _Window, goal: int) -> bytes | None:
        """Newton/secant root ``x* = x1 + (goal - y1) / k``, rounded."""
        width = fit.new.width
        if len(buf) < win.start + width:
            return None
        slope = self._slope(fit.new, fit.old, win)
        x1 = _read(self._inputs[fit.new.gen], win, width)
        root = round(x1 + (goal - fit.new.src) / slope)
        if not 0 <= root < 1 << 8 * width:
            return None

        out = bytearray(buf)
        out[win.start : win.start + width] = root.to_bytes(width, win.order)
        out = bytes(out)
        return out if out != buf else None
