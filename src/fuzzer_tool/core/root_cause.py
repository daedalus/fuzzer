"""Root-cause byte diff: isolate the minimal edit, relative to a known-good
baseline input, that is responsible for a crash.

This answers a different question than crash minimization (``tmin``/
``minimize_bytes``). Shrinking the crashing input alone still leaves every
surviving byte looking equally suspicious. Given a non-crashing baseline and
the crashing input, this module:

1. Levenshtein-aligns the two into an edit script (insert/replace/delete
   ops transforming baseline -> crash) via
   :func:`fuzzer_tool.core.similarity.levenshtein_align`.
2. Delta-debugs (Zeller's ddmin) over the *edits themselves*, not over byte
   offsets in the crash. Re-executing the target against baseline+subset
   candidates means every edit kept in the final minimal set is verified
   causal -- applying it (and only it, plus whatever the search couldn't
   remove) reproduces the same crash signature. Edits that are cosmetic
   (padding, unrelated field changes) get dropped even if they happen to
   sit right next to the real cause.
"""

from __future__ import annotations

from collections.abc import Callable

from fuzzer_tool.core.similarity import levenshtein_align

# ("match", pos, b"") | ("replace", pos, byte) | ("insert", pos, byte) | ("delete", pos, b"")
EditOp = tuple[str, int, bytes]

InterestingFn = Callable[[bytes], bool]


def build_edit_script(base: bytes, crash: bytes) -> list[EditOp]:
    """Return the Levenshtein edit script that transforms *base* into *crash*."""
    return levenshtein_align(base, crash)


def edit_indices(script: list[EditOp]) -> list[int]:
    """Indices of the non-``match`` ops in *script* -- the only ones that
    can possibly matter, since ``match`` is always identity."""
    return [i for i, (op, _pos, _data) in enumerate(script) if op != "match"]


def apply_edit_subset(base: bytes, script: list[EditOp], active: set[int]) -> bytes:
    """Reconstruct bytes from *base*, applying only the script ops whose
    index is in *active*. Ops not in *active* are identity: a ``replace``
    or ``delete`` keeps the original base byte, an ``insert`` contributes
    nothing. ``match`` ops are always identity regardless of *active*.
    """
    out = bytearray()
    for i, (op, pos, data) in enumerate(script):
        if op == "match":
            out.append(base[pos])
        elif op == "replace":
            out.append(data[0] if i in active else base[pos])
        elif op == "insert":
            if i in active:
                out.extend(data)
        elif op == "delete":
            if i not in active:
                out.append(base[pos])
    return bytes(out)


def ddmin_edits(
    base: bytes,
    script: list[EditOp],
    interesting_fn: InterestingFn,
    max_stages: int = 200,
) -> tuple[bytes, list[int]]:
    """Delta-debug the edit script down to a minimal causal subset.

    Zeller's ddmin, generalized from byte-offset removal to edit-index
    removal: repeatedly splits the active edit set into ``n`` chunks and
    tests whether a single chunk (or its complement) alone, applied to
    *base*, still reproduces the crash (``interesting_fn`` returns True).
    Granularity coarsens back to 2-way splits on any successful reduction
    and doubles on failure, same as the classic algorithm.

    Args:
        base: Non-crashing baseline input.
        script: Edit script from :func:`build_edit_script` (base -> crash).
        interesting_fn: ``bytes -> bool``, True if the candidate reproduces
            the target crash (same signature as the original).
        max_stages: Cap on the number of ``interesting_fn`` calls (target
            executions are expensive; this bounds the search).

    Returns:
        ``(minimal_bytes, minimal_indices)`` -- the reconstructed candidate
        and the sorted list of edit-script indices kept. If no single edit
        was independently sufficient to reproduce the crash, the full edit
        set (or whatever ddmin got down to) is returned.
    """
    changes = edit_indices(script)
    if not changes:
        return base, []

    calls = [0]

    def _test(subset: list[int]) -> bool:
        if calls[0] >= max_stages:
            return False
        calls[0] += 1
        if not subset:
            return False
        return interesting_fn(apply_edit_subset(base, script, set(subset)))

    current = list(changes)
    n = 2
    while len(current) >= 2 and calls[0] < max_stages:
        chunk_size = max(1, len(current) // n)
        chunks = [current[i : i + chunk_size] for i in range(0, len(current), chunk_size)]
        reduced = False

        for chunk in chunks:
            if _test(chunk):
                current = chunk
                n = 2
                reduced = True
                break
            complement = [c for c in current if c not in chunk]
            if complement and _test(complement):
                current = complement
                n = max(n - 1, 2)
                reduced = True
                break
            if calls[0] >= max_stages:
                break

        if not reduced:
            if n >= len(current):
                break
            n = min(n * 2, len(current))

    return apply_edit_subset(base, script, set(current)), sorted(current)


def _byte_repr(b: int) -> str:
    return chr(b) if 32 <= b < 127 else f"\\x{b:02x}"


def format_root_cause_report(
    base: bytes,
    crash: bytes,
    script: list[EditOp],
    minimal_indices: list[int],
) -> str:
    """Human-readable report of the isolated root-cause edits."""
    total = len(edit_indices(script))
    lines = [
        "=== root-cause byte diff ===",
        f"baseline: {len(base)} bytes",
        f"crash:    {len(crash)} bytes",
        f"total edits (baseline -> crash): {total}",
        f"root-cause edits (minimal, execution-confirmed): {len(minimal_indices)}",
        "",
    ]
    if not minimal_indices:
        lines.append(
            "(no subset of the diff reproduced the crash alone -- baseline and "
            "crash may differ by more than this input pair can isolate; "
            "try a closer --baseline)"
        )
        return "\n".join(lines)

    for i in sorted(minimal_indices):
        op, pos, data = script[i]
        if op == "replace":
            old = base[pos]
            new = data[0]
            lines.append(
                f"  offset 0x{pos:04x}: REPLACE {old:#04x} ('{_byte_repr(old)}') "
                f"-> {new:#04x} ('{_byte_repr(new)}')"
            )
        elif op == "insert":
            ascii_repr = "".join(_byte_repr(b) for b in data)
            lines.append(f"  offset 0x{pos:04x}: INSERT {data.hex()} (\"{ascii_repr}\")")
        elif op == "delete":
            old = base[pos]
            lines.append(f"  offset 0x{pos:04x}: DELETE {old:#04x} ('{_byte_repr(old)}')")
    return "\n".join(lines)
