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
import random
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


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

    # Create backup and changed copies
    backup = bytearray(data)

    if use_type_aware:
        from fuzzer_tool.core.mutations import type_replace_byte

        changed = bytearray(type_replace_byte(b) for b in data)
    else:
        changed = bytearray(length)
        for i in range(length):
            c = random.randint(0, 255)
            while c == data[i]:
                c = random.randint(0, 255)
            changed[i] = c

    # Initialize with one range covering the entire input
    ranges: list[list[int]] = [[0, length - 1]]  # [start, end] inclusive
    safe_ranges: list[list[int]] = []  # ranges that can be diversified

    # Binary search over ranges
    while ranges and exec_count < max_execs:
        # Pick the largest range
        ranges.sort(key=lambda r: r[1] - r[0], reverse=True)
        rng = ranges.pop(0)

        start, end = rng
        size = end - start + 1

        # Replace this range in the original with changed values
        test = bytearray(data)
        test[start : end + 1] = changed[start : end + 1]

        cksum = exec_fn(bytes(test))
        exec_count += 1

        if cksum == original_checksum:
            # Path preserved — this range is safe to diversify
            safe_ranges.append([start, end])
        else:
            # Path changed — split and try smaller pieces
            if size > 1:
                mid = start + size // 2
                ranges.append([start, mid - 1])
                ranges.append([mid, end])

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
