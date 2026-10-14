"""--affine-mod: solve ``(a*x + b) % m == t`` compares; queue what settles.

On each seed's first round the input is probed under cmplog. A pair whose
operands are not in the input (Redqueen has nothing to patch) may hide a
modular map of some field. Locate the field, fit the map
(``core.affine_mod``), and queue the field values that satisfy the compare;
``OperatorEngine.mutate`` hands one per round to the normal pipeline, off the
bandit tournament (same path as ``--i2s-fixpoint``), so coverage decides.

    pair (v, t) ──screen──▶ bytes that move v ──▶ window (offset, width)
        ──fit (LE, then BE)──▶ v = (a*x + b) % m ──solve──▶ x with v == t

--derived-cmp (``Families.DERIVED``) adds the ``core.derived_fit`` families
on the same screen, cheapest first::

    bytes that move one operand byte each ──▶ byte map   (strcmp(rot13(x)))
    run of bytes moving an integer v      ──▶ additive   (sum, checksum)
    window, modular fit failed            ──▶ GF(2)      (parity, syndrome)
"""

from collections import Counter, deque
from enum import Enum

from fuzzer_tool.core import affine_mod as core
from fuzzer_tool.core import derived_fit
from fuzzer_tool.core.i2s_fixpoint import MIN_OPERAND, Pair, Probe

# Pending candidates; oldest dropped first.
QUEUE_MAX = 64

# Bytes screened for influence on an operand: one exec each.
SCREEN_BYTES = 32
DERIVED_SCREEN_BYTES = 64

# Pairs screened / windows fitted per search. Screening a pair is free (each
# screen exec reads every pair), so the pair cap only bounds Python work;
# the shim logs several constant pairs first, which a cap of 4 let crowd out
# the target's own compares. Each window costs up to two fits.
MAX_PAIRS = 64
MAX_WINDOWS = 4

# Longest string operand the byte map takes (strcmp/memcmp records).
MAX_STRING_OPERAND = 32

# Field widths in bytes, as a compare operand can be.
FIELD_WIDTHS = (1, 2, 4, 8)

# Probe budget per search (screen + fits + one verify per window).
MAX_SEARCH_EXECS = 400


class Order(Enum):
    LITTLE = "little"
    BIG = "big"


class Families(Enum):
    """Map families one search fits."""

    MODULAR = "modular"  # --affine-mod: (a*x + b) % m only
    DERIVED = "derived"  # --derived-cmp: + byte map, additive, GF(2)


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


# (pair, side) -> {screened offset: moving operand after +1 there}
Key = tuple[Pair, Side]
Moved = dict[Key, dict[int, bytes]]


def _signed(v: int, bits: int) -> int:
    return v - (1 << bits) if v >> (bits - 1) else v


def _weight(x: int, delta: int, bits: int) -> int | None:
    """Per-unit weight of a byte from its +1 screen delta.

    A byte at 0xFF wrapped to 0: the delta is -255 * w.
    """
    if x < derived_fit.BYTE_MAX:
        return _signed(delta, bits)

    back = _signed(-delta % (1 << bits), bits)
    return back // derived_fit.BYTE_MAX if back % derived_fit.BYTE_MAX == 0 else None


def _links(base: bytes, after: dict[int, bytes]) -> dict[int, int] | None:
    """Operand position -> the one input offset moving it; None if any overlap."""
    links: dict[int, int] = {}
    for off, raw in after.items():
        changed = [j for j in range(len(base)) if raw[j] != base[j]]
        if len(changed) != 1 or changed[0] in links:
            return None
        links[changed[0]] = off

    return links


def _read_raw(pairs: list[Pair] | None, const: bytes, side: Side) -> bytes | None:
    """The moving operand of the first pair carrying *const*; None if absent."""
    if pairs is None:
        return None

    moving = side.value
    for pair in pairs:
        if pair[1 - moving] == const and len(pair[moving]) == len(const):
            return pair[moving]

    return None


def _read(pairs: list[Pair] | None, const: bytes, side: Side) -> int | None:
    raw = _read_raw(pairs, const, side)
    return None if raw is None else int.from_bytes(raw, "little")


def _int_width(n: int) -> bool:
    return n in FIELD_WIDTHS


def _any_width(n: int) -> bool:
    return n in FIELD_WIDTHS or MIN_OPERAND <= n <= MAX_STRING_OPERAND


def _candidates(data: bytes, pairs: list[Pair], width_ok) -> list[Pair]:
    """Unsatisfied equal-width pairs neither operand of which sits in *data*."""
    out: list[Pair] = []
    for a, b in pairs:
        if a == b or len(a) != len(b) or not width_ok(len(a)):
            continue

        # Redqueen's job: an integer operand that is already in the input. A
        # string operand stays: f(x) may equal x on this seed (rot13 of
        # punctuation) and still differ once Redqueen writes the constant.
        in_data = a in data or b in data
        if len(a) >= MIN_OPERAND and in_data and _int_width(len(a)):
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


def _put(data: bytes, assign: dict[int, int]) -> bytes:
    out = bytearray(data)
    for off, value in assign.items():
        out[off] = value
    return bytes(out)


class AffineMod:
    """Derived-compare solver over one target's cmplog probe."""

    def __init__(
        self, probe: Probe, max_execs: int = MAX_SEARCH_EXECS, families: Families = Families.MODULAR
    ):
        self._probe = probe
        self._max_execs = max_execs
        self.families = families
        self._queue: deque[bytes] = deque(maxlen=QUEUE_MAX)
        self._outcomes: Counter[str] = Counter()
        self._execs = 0
        self._left = 0

        # Ifs up: bind the family-dependent settings once.
        derived = families is Families.DERIVED
        self._screen_bytes = DERIVED_SCREEN_BYTES if derived else SCREEN_BYTES
        self._width_ok = _any_width if derived else _int_width
        self._derived = derived

    def _run(self, data: bytes) -> list[Pair] | None:
        """One budgeted probe; None when the search budget is spent."""
        if self._left <= 0:
            return None

        self._left -= 1
        self._execs += 1
        return self._probe(data)

    def _screen(self, data: bytes, base: list[Pair], cands: list[Pair]) -> Moved:
        """Per pair side: the operand after +1 at each byte that moves it."""
        keys = [(pair, side) for pair in cands for side in Side]
        moved: Moved = {key: {} for key in keys}
        # Read back, not pair[side]: two pairs may share the constant.
        base_raw = {key: _read_raw(base, key[0][1 - key[1].value], key[1]) for key in keys}

        for i in range(min(len(data), self._screen_bytes)):
            mutant = data[:i] + bytes([(data[i] + 1) & 0xFF]) + data[i + 1 :]
            pairs = self._run(mutant)
            if pairs is None:
                break

            for key in keys:
                pair, side = key
                raw = _read_raw(pairs, pair[1 - side.value], side)
                if raw is not None and raw != base_raw[key]:
                    moved[key][i] = raw

        return {key: after for key, after in moved.items() if after}

    def _windows(self, data: bytes, moved: Moved) -> list[_Window]:
        wins: list[_Window] = []
        for (pair, side), after in moved.items():
            if not _int_width(len(pair[side.value])):
                continue

            for start, length in _runs(sorted(after)):
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

        if self._derived:
            self._solve_gf2(data, win)

    def _verify(self, cand: bytes, const: bytes, side: Side) -> None:
        """Queue *cand*; one probe checks the compare now holds."""
        self._queue.append(cand)
        pairs = self._run(cand)
        raw = _read_raw(pairs, const, side)
        if raw == const:
            self._outcomes["solved"] += 1
        elif raw is None and pairs is not None:
            # Satisfied string compares log nothing; coverage decides.
            self._outcomes["vanished"] += 1

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

    # ── --derived-cmp families ───────────────────────────────────────

    def _solve_gf2(self, data: bytes, win: _Window) -> None:
        # Byte order only permutes the bits of x: one order suffices.
        def read_gf2(x: int) -> int | None:
            return _read(self._run(_patch(data, win, x, Order.LITTLE)), win.const, win.side)

        x0 = int.from_bytes(data[win.start : win.start + win.width], "little")
        model = derived_fit.fit_gf2(read_gf2, x0, 8 * win.width)
        if model is None:
            return

        self._outcomes["gf2"] += 1
        x = derived_fit.solve_gf2(model, int.from_bytes(win.const, "little"))
        if x is not None and x != x0:
            self._verify(_patch(data, win, x, Order.LITTLE), win.const, win.side)

    def _solve_bytemap(self, data: bytes, key: Key, base: bytes, after: dict[int, bytes]) -> bool:
        """strcmp(f(x), "...") with one input byte per operand byte."""
        pair, side = key
        const = pair[1 - side.value]
        links = _links(base, after)
        if links is None:
            return False

        # A position no screened byte moves must already match.
        if any(base[j] != const[j] for j in range(len(base)) if j not in links):
            return False

        def read_bytes(assign: dict[int, int]) -> bytes | None:
            # A string compare that holds logs no record: a run with pairs but
            # without this one reads as satisfied (_verify re-checks it).
            pairs = self._run(_put(data, assign))
            raw = _read_raw(pairs, const, side)
            return const if raw is None and pairs is not None else raw

        xs = {off: data[off] for off in links.values()}
        assign = derived_fit.invert_bytes(read_bytes, links, xs, base, const, self._left)
        if assign is None:
            return False

        self._outcomes["bytemap"] += 1
        self._verify(_put(data, assign), const, side)
        return True

    def _solve_additive(self, data: bytes, key: Key, base: bytes, after: dict[int, bytes]) -> bool:
        """v = v0 + sum w_i * dx_i over the screened bytes (sums, checksums)."""
        pair, side = key
        const = pair[1 - side.value]
        bits = 8 * len(base)
        v0 = int.from_bytes(base, "little")

        offs, weights = [], []
        for off, raw in sorted(after.items()):
            delta = (int.from_bytes(raw, "little") - v0) % (1 << bits)
            w = _weight(data[off], delta, bits)
            if w is not None:
                offs.append(off)
                weights.append(w)

        target = int.from_bytes(const, "little")
        xs = [data[off] for off in offs]
        out = derived_fit.solve_additive(xs, weights, v0, target, bits)
        if out is None:
            return False

        self._outcomes["additive"] += 1
        self._verify(_put(data, dict(zip(offs, out, strict=True))), const, side)
        return True

    def _solve_derived(self, data: bytes, base_pairs: list[Pair], moved: Moved) -> None:
        # Cheapest first: additive needs 1 exec, the byte map one per guess round.
        for key, after in moved.items():
            if self._left <= 0:
                return

            pair, side = key
            base = _read_raw(base_pairs, pair[1 - side.value], side)
            if base is None:
                continue

            many = len(after) > 1
            if _int_width(len(base)) and many and self._solve_additive(data, key, base, after):
                continue
            if len(base) >= MIN_OPERAND:
                self._solve_bytemap(data, key, base, after)

    def search(self, data: bytes) -> None:
        """Search *data*; queue every field value that satisfies a derived compare."""
        self._left = self._max_execs
        base = self._run(data)
        if not base:
            return

        cands = _candidates(data, base, self._width_ok)
        if not cands:
            return

        moved = self._screen(data, base, cands)
        if self._derived:
            self._solve_derived(data, base, moved)

        for win in self._windows(data, moved):
            self._solve_window(data, win)

    def pop(self) -> bytes | None:
        return self._queue.popleft() if self._queue else None

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> dict[str, int]:
        return {"execs": self._execs, **self._outcomes}
