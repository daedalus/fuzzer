"""Naive reference of the deterministic stage schedule.

Copies the seed per mutant and dedups with sets of whole mutants, so it
shares no table, predicate or key encoding with
``services/operators._deterministic_mutation_stream``. Spec:

    bitflip 1/1 | byteflip 8/8 | arith8 | arith16 | arith32 | int8 | int16 | int32

- arith8: +j, -j for j in 1..ARITH_MAX; drops what bit/byte flips made.
- arith16/32: LE then BE, + then -, j ascending; only changes that reach
  past the low half of the word (AFL: carry/borrow), else the narrower
  pass already made them.
- int8: INTERESTING_UNSIGNED_8; drops seed copies and flip/arith8 repeats.
- int16/32: AFL's widened tables (INTERESTING_8 + 16 [+ 32]), LE then BE.
- wide passes drop seed copies, any single-byte change the 8-bit passes
  cover, and any multi-byte mutant already yielded.
"""

from __future__ import annotations

from itertools import islice

from fuzzer_tool.core.mutations import (
    ARITH_MAX,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    INTERESTING_UNSIGNED_8,
)

ENDIANS = ("little", "big")
WIDE_VALUES = {
    2: INTERESTING_8 + INTERESTING_16,
    4: INTERESTING_8 + INTERESTING_16 + INTERESTING_32,
}


# A mutant is its minimal diff (first changed offset, changed span); equal
# diffs are equal mutants. Full buffers are built only for the output.
SEED = ("seed",)


def _put(data: bytes, pos: int, chunk: bytes):
    diffs = [pos + k for k in range(len(chunk)) if chunk[k] != data[pos + k]]
    if not diffs:
        return SEED
    lo, hi = diffs[0], diffs[-1]
    return (lo, bytes(chunk[lo - pos : hi - pos + 1]))


def _apply(data: bytes, mutant) -> bytes:
    lo, span = mutant
    return data[:lo] + span + data[lo + len(span) :]


def _arith8(data: bytes, i: int):
    for j in range(1, ARITH_MAX + 1):
        for sign in (1, -1):
            yield _put(data, i, bytes([(data[i] + sign * j) % 256]))


def _arith_wide(data: bytes, i: int, width: int):
    half = width // 2
    for endian in ENDIANS:
        v = int.from_bytes(data[i : i + width], endian)
        for sign in (1, -1):
            for j in range(1, ARITH_MAX + 1):
                new = ((v + sign * j) % (1 << (8 * width))).to_bytes(width, endian)
                touched = [k for k in range(width) if new[k] != data[i + k]]
                high = [k for k in touched if (k >= half if endian == "little" else k < half)]
                if high:
                    yield _put(data, i, new)


def _interest8(data: bytes, i: int):
    for v in INTERESTING_UNSIGNED_8:
        yield _put(data, i, bytes([v & 0xFF]))


def _interest_wide(data: bytes, i: int, width: int):
    mask = (1 << (8 * width)) - 1
    for v in WIDE_VALUES[width]:
        for endian in ENDIANS:
            yield _put(data, i, (v & mask).to_bytes(width, endian))


def _bitflip(data: bytes, i: int):
    for bit in range(8):
        yield _put(data, i, bytes([data[i] ^ (1 << bit)]))


def _byteflip(data: bytes, i: int):
    yield _put(data, i, bytes([data[i] ^ 0xFF]))


# (width, candidate generator) in schedule order.
PASSES = (
    (1, _bitflip),
    (1, _byteflip),
    (1, _arith8),
    (2, lambda d, i: _arith_wide(d, i, 2)),
    (4, lambda d, i: _arith_wide(d, i, 4)),
    (1, _interest8),
    (2, lambda d, i: _interest_wide(d, i, 2)),
    (4, lambda d, i: _interest_wide(d, i, 4)),
)


def _distinct(width: int) -> int:
    return len({v & ((1 << (8 * width)) - 1) for v in WIDE_VALUES[width]})


# Upper-bound candidates per site, in PASSES order.
PER_SITE = (
    8,
    1,
    2 * ARITH_MAX,
    4 * ARITH_MAX,
    4 * ARITH_MAX,
    len(INTERESTING_UNSIGNED_8),
    2 * _distinct(2),
    2 * _distinct(4),
)
PER_BYTE = sum(PER_SITE)


def _sites(n: int, width: int, live: set[int] | None) -> list[int]:
    sites = range(n - width + 1)
    if live is None:
        return list(sites)
    return [i for i in sites if any(k in live for k in range(i, i + width))]


def _split(costs: list[int], budget: int) -> list[int]:
    """Proportional quotas, remainder to the largest under-allocation."""
    total = sum(costs)
    if total <= budget:
        return list(costs)
    quotas = [int(budget * c / total) for c in costs]
    short = budget - sum(quotas)
    for k in sorted(range(len(costs)), key=lambda k: costs[k] - quotas[k], reverse=True):
        add = min(short, costs[k] - quotas[k])
        quotas[k] += max(0, add)
        short -= max(0, add)
    return quotas


def _quotas(n: int, max_mutations: int, live: set[int] | None) -> list[int]:
    """Per-pass budget; with *live*, the passes after byteflip are re-split."""
    costs = [len(_sites(n, w, None)) * c for (w, _g), c in zip(PASSES, PER_SITE, strict=True)]
    quotas = _split(costs, max_mutations)
    if live is None:
        return quotas

    gated = [
        len(_sites(n, w, live)) * c for (w, _g), c in zip(PASSES[2:], PER_SITE[2:], strict=True)
    ]
    return quotas[:2] + _split(gated, max_mutations - sum(quotas[:2]))


def _skip_sets(data: bytes) -> dict[int, set]:
    """Pass index -> mutants it must not resend (wide passes share index -1)."""
    n = len(data)
    flips = {m for i in range(n) for g in (_bitflip, _byteflip) for m in g(data, i)}
    arith8 = {m for i in range(n) for m in _arith8(data, i)}
    before_int8 = flips | arith8 | {SEED}
    covered8 = before_int8 | {m for i in range(n) for m in _interest8(data, i)}
    return {2: flips, 5: before_int8, -1: covered8}


def _fresh(candidates, banned: set, multi: set):
    """Candidates in order, minus banned, earlier wide yields and repeats."""
    local: set = set()
    for m in candidates:
        if m in banned or m in multi or m in local:
            continue
        local.add(m)
        yield m


def reference_stream(data: bytes, max_mutations: int = 10**9, live: set[int] | None = None):
    """Yield the schedule. *live* gates passes after byteflip (no cap)."""
    n = len(data)
    if n == 0:
        return

    quotas = _quotas(n, max_mutations, live)
    skip = _skip_sets(data)
    multi: set = set()  # every wide-pass yield so far
    out = []
    spare = 0
    for k, (width, gen) in enumerate(PASSES):
        banned = skip[-1] if width > 1 else skip.get(k, set())
        candidates = (m for i in _sites(n, width, live if k >= 2 else None) for m in gen(data, i))
        taken = list(islice(_fresh(candidates, banned, multi), quotas[k] + spare))
        if width > 1:
            multi.update(taken)
        out += taken
        spare = quotas[k] + spare - len(taken) if k >= 2 else 0

    for m in out:
        yield _apply(data, m)
