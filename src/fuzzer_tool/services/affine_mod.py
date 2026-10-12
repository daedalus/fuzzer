"""--affine-mod: solve ``(a*x + b) % m == t`` compares; queue what settles.

On each seed's first round the input is probed under cmplog. A pair whose
operands are not in the input (Redqueen has nothing to patch) may hide a
modular map of some field. Locate the field, fit the map
(``core.affine_mod``), and queue the field values that satisfy the compare;
``OperatorEngine.mutate`` hands one per round to the normal pipeline, off the
bandit tournament (same path as ``--i2s-fixpoint``), so coverage decides.

    pair (v, t) ──screen──▶ bytes that move v ──▶ window (offset, width)
        ──fit (LE, then BE)──▶ v = (a*x + b) % m ──solve──▶ x with v == t
"""

from collections import Counter, deque
from enum import Enum

from fuzzer_tool.core import affine_mod as core
from fuzzer_tool.core.i2s_fixpoint import MIN_OPERAND, Pair, Probe

# Pending candidates; oldest dropped first.
QUEUE_MAX = 64

# Bytes screened for influence on an operand: one exec each.
SCREEN_BYTES = 32

# Pairs / windows examined per search. Each window costs up to two fits.
MAX_PAIRS = 4
MAX_WINDOWS = 4

# Field widths in bytes, as a compare operand can be.
FIELD_WIDTHS = (1, 2, 4, 8)

# Probe budget per search (screen + fits + one verify per window).
MAX_SEARCH_EXECS = 400


class Order(Enum):
    LITTLE = "little"
    BIG = "big"


class Side(Enum):
    """Which pair operand moves with the field (the other is the constant)."""

    FIRST = 0
    SECOND = 1


class _Window:
    """One candidate field and the compare it feeds."""

    def __init__(self, start: int, width: int, const: bytes, side: Side):
        self.start = start
        self.width = width
        self.const = const
        self.side = side


def _read(pairs: list[Pair] | None, const: bytes, side: Side) -> int | None:
    """The moving operand of the first pair carrying *const*; None if absent."""
    if pairs is None:
        return None

    moving = side.value
    for pair in pairs:
        if pair[1 - moving] == const and len(pair[moving]) == len(const):
            return int.from_bytes(pair[moving], "little")

    return None


def _candidates(data: bytes, pairs: list[Pair]) -> list[Pair]:
    """Unsatisfied equal-width pairs neither operand of which sits in *data*."""
    out: list[Pair] = []
    for a, b in pairs:
        if a == b or len(a) != len(b) or len(a) not in FIELD_WIDTHS:
            continue

        # Redqueen's job: an operand that is already in the input.
        if len(a) >= MIN_OPERAND and (a in data or b in data):
            continue

        out.append((a, b))

    return out[:MAX_PAIRS]


def _runs(offsets: list[int]) -> list[tuple[int, int]]:
    """``(start, length)`` of each run of consecutive offsets."""
    runs: list[tuple[int, int]] = []
    for off in offsets:
        if runs and runs[-1][0] + runs[-1][1] == off:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((off, 1))

    return runs


def _field_width(length: int) -> int | None:
    """Smallest operand width holding a run of *length* bytes."""
    return next((w for w in FIELD_WIDTHS if w >= length), None)


def _patch(data: bytes, win: _Window, x: int, order: Order) -> bytes:
    field = x.to_bytes(win.width, order.value)
    return data[: win.start] + field + data[win.start + win.width :]


class AffineMod:
    """Modular-compare solver over one target's cmplog probe."""

    def __init__(self, probe: Probe, max_execs: int = MAX_SEARCH_EXECS):
        self._probe = probe
        self._max_execs = max_execs
        self._queue: deque[bytes] = deque(maxlen=QUEUE_MAX)
        self._outcomes: Counter[str] = Counter()
        self._execs = 0
        self._left = 0

    def _run(self, data: bytes) -> list[Pair] | None:
        """One budgeted probe; None when the search budget is spent."""
        if self._left <= 0:
            return None

        self._left -= 1
        self._execs += 1
        return self._probe(data)

    def _screen(self, data: bytes, base: list[Pair], cands: list[Pair]) -> list[_Window]:
        """Windows over the bytes that move a candidate pair's operand."""
        moved: dict[tuple[Pair, Side], list[int]] = {}
        base_vals = {}
        for pair in cands:
            for side in Side:
                const = pair[1 - side.value]
                base_vals[(pair, side)] = _read(base, const, side)
                moved[(pair, side)] = []

        for i in range(min(len(data), SCREEN_BYTES)):
            mutant = data[:i] + bytes([(data[i] + 1) & 0xFF]) + data[i + 1 :]
            pairs = self._run(mutant)
            if pairs is None:
                break

            for key, offs in moved.items():
                pair, side = key
                v = _read(pairs, pair[1 - side.value], side)
                if v is not None and v != base_vals[key]:
                    offs.append(i)

        return self._windows(data, moved)

    def _windows(self, data: bytes, moved: dict[tuple[Pair, Side], list[int]]) -> list[_Window]:
        wins: list[_Window] = []
        for (pair, side), offs in moved.items():
            for start, length in _runs(offs):
                width = _field_width(length)
                if width is None or width > len(data):
                    continue

                # Keep the run inside the window at the end of the input.
                start = min(start, len(data) - width)
                wins.append(_Window(start, width, pair[1 - side.value], side))

        return wins[:MAX_WINDOWS]

    def _fit(self, data: bytes, win: _Window, order: Order) -> core.Fit:
        def observe(x: int) -> int | None:
            pairs = self._run(_patch(data, win, x, order))
            return _read(pairs, win.const, win.side)

        field = data[win.start : win.start + win.width]
        x0 = int.from_bytes(field, order.value)
        return core.fit(observe, x0, 8 * win.width, max_execs=min(core.MAX_FIT_EXECS, self._left))

    def _solve_window(self, data: bytes, win: _Window) -> None:
        orders = (Order.LITTLE,) if win.width == 1 else tuple(Order)
        for order in orders:
            result = self._fit(data, win, order)
            self._outcomes[result.outcome.value] += 1
            if result.model is None:
                continue

            self._queue_solutions(data, win, order, result.model)
            return

    def _queue_solutions(
        self, data: bytes, win: _Window, order: Order, model: core.AffineMod | core.Ramp
    ) -> None:
        target = int.from_bytes(win.const, "little")
        x0 = int.from_bytes(data[win.start : win.start + win.width], order.value)
        sols = [x for x in core.solve(model, target, 8 * win.width, near=x0) if x != x0]

        for n, x in enumerate(sols):
            cand = _patch(data, win, x, order)
            self._queue.append(cand)
            # One verifying probe per window: does the compare now hold?
            if n == 0 and _read(self._run(cand), win.const, win.side) == target:
                self._outcomes["solved"] += 1

    def search(self, data: bytes) -> None:
        """Search *data*; queue every field value that satisfies a modular compare."""
        self._left = self._max_execs
        base = self._run(data)
        if not base:
            return

        cands = _candidates(data, base)
        if not cands:
            return

        for win in self._screen(data, base, cands):
            self._solve_window(data, win)

    def pop(self) -> bytes | None:
        return self._queue.popleft() if self._queue else None

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict[str, int]:
        return {"execs": self._execs, **self._outcomes}
