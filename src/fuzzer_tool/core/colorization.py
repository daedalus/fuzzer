"""Colorization: prepare inputs for CmpLog comparison tracing.

Ports AFL++'s colorization algorithm that diversifies an input's bytes
while preserving its execution path. This ensures CmpLog sees diverse
comparison operands when analyzing the target's comparison operations.

Algorithm:
1. Create a "changed" copy with all bytes replaced (random or type-aware)
2. Binary-search over ranges: replace a range in the original with changed
3. If execution path stays the same → the range is "safe" to diversify
4. If execution path changes → split the range and try smaller pieces
5. Merge adjacent safe ranges into tainted regions

The tainted regions are returned for CmpLog to use when generating
diverse comparison values.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import cache

import numpy as np

from fuzzer_tool.core import group_testing
from fuzzer_tool.core.rand_pool import RandPool

log = logging.getLogger(__name__)

# Offset that would map a byte back onto itself under the shift below, so
# it is the one draw the replacement has to reject.
_SELF_OFFSET = 0xFF


class ColorMode(Enum):
    """How the path-preserving byte set is searched for."""

    BISECT = "bisect"  # AFL++ range bisection: sequential, fewest execs
    POOLED = "pooled"  # non-adaptive group testing: one parallel round


# Pool inclusion is p = 1/(_POOL_D + 1) = 1/4 = two random bits == 0, so a
# byte of entropy decides 4 pool slots with no modulo bias. Soundness of the
# result does not depend on this; only how many bytes get proven dead does.
_POOL_D = 3
_POOL_MASK = 0b11
# Baseline run + final verification run, on top of the pool runs.
_FIXED_EXECS = 2


def _diverse_copy(data: bytes, rng: RandPool) -> bytearray:
    """Return a copy in which every byte differs from its original.

    Drawing until the value differs is a shifted draw done once: uniform on
    {0..255} minus ``x`` is exactly ``(x + 1 + u) mod 256`` for ``u`` uniform
    on {0..254}. That removes the retry loop, and the shift is then plain
    arithmetic over the whole buffer instead of a per-byte Python round trip.

    Entropy still comes from the caller's pool -- numpy does the arithmetic
    only -- so ``--seed`` keeps determining the result.
    """
    n = len(data)
    if n == 0:
        return bytearray()

    # 255 offsets do not tile a 256-value byte, so the one offset that is a
    # no-op is rejected. Expected redraws are n/256; `find` scans in C.
    offsets = bytearray(rng.randbytes(n))
    pos = offsets.find(_SELF_OFFSET)
    while pos != -1:
        drawn = rng.randbytes(1)[0]
        offsets[pos] = drawn
        if drawn != _SELF_OFFSET:
            pos = offsets.find(_SELF_OFFSET, pos + 1)

    original = np.frombuffer(data, dtype=np.uint8)
    shifted = original + np.uint8(1) + np.frombuffer(bytes(offsets), dtype=np.uint8)
    return bytearray(shifted.tobytes())


def _bisect_safe(data, changed, exec_fn, original_checksum, max_execs):
    """Binary search: largest range first, split on path change."""
    length = len(data)
    exec_count = 1
    ranges: list[list[int]] = [[0, length - 1]]
    safe_ranges: list[list[int]] = []

    while ranges and exec_count < max_execs:
        ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
        start, end = ranges.pop(0)
        size = end - start + 1

        test = bytearray(data)
        test[start : end + 1] = changed[start : end + 1]

        cksum = exec_fn(bytes(test))
        exec_count += 1

        if cksum == original_checksum:
            safe_ranges.append([start, end])
        elif size > 1:
            mid = start + size // 2
            ranges.append([start, mid - 1])
            ranges.append([mid, end])

    return safe_ranges, exec_count


def _pooled_safe(data, changed, exec_fn, original_checksum, max_execs, rng):
    """Group testing: a byte in any path-preserving pool is path-irrelevant.

    Pools are fixed up front (independent executions, so they could run in
    parallel; the fuzzer still runs them serially). Replace one random pool
    at a time in the original; if the path holds, every byte in it is dead
    (COMP, ``group_testing.comp``: items in no negative pool are the only
    candidates for "matters"). Dead = union of surviving pools.

        pool 1: x . x . . x      path same  -> bytes 0,2,5 dead
        pool 2: . x . x . .      path moved -> no conclusion
        pool 3: . . . . x x      path same  -> bytes 4,5 dead

    Bytes in no surviving pool stay live (conservative), so a short budget
    only under-taints. The union is then executed once: interacting bytes
    that are individually dead but jointly live would break the path, in
    which case nothing is claimed.
    """
    length = len(data)
    budget = max_execs - _FIXED_EXECS
    tests = min(group_testing.tests_needed(length, _POOL_D), budget)
    if tests < 1:
        return [], 1

    orig = np.frombuffer(data, dtype=np.uint8)
    diverse = np.frombuffer(bytes(changed), dtype=np.uint8)
    pools, outcomes = [], []
    for _ in range(tests):
        bits = np.frombuffer(rng.randbytes(length), dtype=np.uint8) & _POOL_MASK
        idx = np.flatnonzero(bits == 0)
        test = orig.copy()
        test[idx] = diverse[idx]

        pools.append(frozenset(idx.tolist()))
        outcomes.append(exec_fn(test.tobytes()) != original_checksum)

    exec_count = 1 + tests
    live = group_testing.comp(length, pools, outcomes)
    dead = sorted(set(range(length)) - live)
    if not dead:
        return [], exec_count

    verify = orig.copy()
    verify[dead] = diverse[dead]
    exec_count += 1
    if exec_fn(verify.tobytes()) != original_checksum:
        return [], exec_count

    return [[i, i] for i in dead], exec_count


@dataclass
class TaintRegion:
    """A contiguous range of bytes that can be safely diversified."""

    start: int
    end: int  # inclusive


@dataclass
class ColorizationResult:
    """Result of colorizing an input for CmpLog."""

    # The colorized input (bytes where safe ranges have been diversified)
    colorized: bytes
    # Taint regions (contiguous ranges that can be mutated freely)
    taints: list[TaintRegion] = field(default_factory=list)
    # Original execution checksum (for verification)
    original_checksum: int = 0
    # Number of executions used
    exec_count: int = 0


def colorize(
    data: bytes,
    exec_fn,
    use_type_aware: bool = True,
    max_execs: int = 0,
    *,
    rng: RandPool,
    mode: ColorMode = ColorMode.BISECT,
) -> ColorizationResult:
    """Colorize an input for CmpLog comparison tracing.

    Replaces bytes in the input with diverse values while preserving
    the execution path. Returns the colorized input and taint regions.

    Args:
        data: Original input to colorize.
        exec_fn: Callable(bytes) -> int, returns execution path checksum.
            Should return the same checksum for inputs that take the same path.
        use_type_aware: If True, use type-aware replacement (preserves character
            classes). If False, use random replacement.
        max_execs: Maximum executions (0 = unlimited, use 2 * len(data)).
        rng: The pool every replacement byte is drawn from. Required and
            keyword-only: both branches below draw, and a default would put
            them back on the stdlib global (Hard Rule 16).
        mode: ``BISECT`` (default) or ``POOLED`` (see ``_pooled_safe``).

    Returns:
        ColorizationResult with the colorized input and taint regions.
    """
    if not data:
        return ColorizationResult(colorized=data)

    length = len(data)
    if max_execs <= 0:
        max_execs = length * 2

    # Get baseline checksum
    original_checksum = exec_fn(data)
    exec_count = 1

    # Build the fully-changed copy; `data` itself is the baseline, so no
    # separate backup is needed.
    if use_type_aware:
        from fuzzer_tool.core.mutations import type_replace_byte

        changed = bytearray(type_replace_byte(b, rng) for b in data)
    else:
        changed = _diverse_copy(data, rng)

    if mode is ColorMode.POOLED:
        safe_ranges, exec_count = _pooled_safe(
            data, changed, exec_fn, original_checksum, max_execs, rng
        )
    else:
        safe_ranges, exec_count = _bisect_safe(data, changed, exec_fn, original_checksum, max_execs)

    # Build colorized output: apply safe ranges
    colorized = bytearray(data)
    for start, end in safe_ranges:
        colorized[start : end + 1] = changed[start : end + 1]

    # Merge adjacent safe ranges into taint regions
    taints = _merge_ranges(safe_ranges)

    log.debug(
        "Colorization: %d/%d ranges safe, %d taints, %d execs",
        len(safe_ranges),
        length,
        len(taints),
        exec_count,
    )

    return ColorizationResult(
        colorized=bytes(colorized),
        taints=taints,
        original_checksum=original_checksum,
        exec_count=exec_count,
    )


@cache
def _class_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-byte ``type_replace_byte`` behaviour as lookup tables.

    ``start``/``size`` describe the byte's class (size 0 = no draw);
    ``fixed`` is the deterministic replacement for swap-map and classless
    bytes. Lazy: ``core.mutations`` is imported lazily here as in ``colorize``.
    """
    from fuzzer_tool.core.mutations import type_replace_byte
    from fuzzer_tool.core.mutations.generic import _SWAP_MAP, _in_class

    start = np.zeros(256, dtype=np.int32)
    size = np.zeros(256, dtype=np.int32)
    fixed = np.zeros(256, dtype=np.uint8)
    for b in range(256):
        cls = None if b in _SWAP_MAP else _in_class(b)
        if cls is None:
            fixed[b] = type_replace_byte(b, None)
            continue
        start[b], size[b] = cls[0], cls[1] - cls[0]
    return start, size, fixed


def _taint_index(length: int, taints) -> np.ndarray:
    """Sorted unique byte offsets covered by *taints*, clipped to *length*."""
    mask = np.zeros(length, dtype=bool)
    for t in taints:
        mask[max(t.start, 0) : min(t.end, length - 1) + 1] = True
    return np.flatnonzero(mask)


def boost(data: bytes, taints, rng: RandPool) -> bytes:
    """Path-equivalent sibling of *data* (Revizor input boosting).

    Redraws every byte inside *taints* from its ``type_replace_byte`` class
    -- the replacement colorization proved path-preserving -- and keeps the
    rest. Revizor keeps the bytes the trace depends on and randomizes the
    others; colorization's taints are exactly those others.

    Vectorized per-byte ``type_replace_byte``: a class byte ``b`` becomes
    ``start + (b - start + 1 + u) mod (size + 1)``, ``u`` uniform on
    ``[0, size)``, i.e. uniform over the class minus ``b`` (as in
    ``_diverse_copy``). ``u`` is a 16-bit multiply-shift draw, so its bias
    is at most ``size / 2**16``.

        data  : F U Z Z _ a b c        taints = [4..7]
        boost : F U Z Z \\t d f a       (header kept, tail redrawn per class)
    """
    idx = _taint_index(len(data), taints)
    if idx.size == 0:
        return bytes(data)

    start, size, fixed = _class_tables()
    out = np.frombuffer(data, dtype=np.uint8).copy()
    vals = out[idx].astype(np.int32)

    # Classless/swap bytes: deterministic, no draw.
    out[idx] = fixed[vals]

    # Class bytes: one 16-bit draw each, shifted off the original value.
    drawn = np.flatnonzero(size[vals] > 0)
    if drawn.size:
        b, lo, sz = vals[drawn], start[vals[drawn]], size[vals[drawn]]
        r = np.frombuffer(rng.randbytes(2 * drawn.size), dtype="<u2").astype(np.int32)
        u = (r * sz) >> 16
        out[idx[drawn]] = lo + (b - lo + 1 + u) % (sz + 1)

    return out.tobytes()


def _merge_ranges(ranges: list[list[int]]) -> list[TaintRegion]:
    """Merge overlapping/adjacent ranges into contiguous taint regions."""
    if not ranges:
        return []

    # Sort by start
    sorted_ranges = sorted(ranges, key=lambda r: r[0])

    merged = [list(sorted_ranges[0])]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + 1:
            # Overlapping or adjacent — merge
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [TaintRegion(start=s, end=e) for s, e in merged]
