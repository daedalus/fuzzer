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

Scrambled-field descrambling (optional)
----------------------------------------
Everything above operates on raw file bytes -- the domain the target
actually reads off disk. If a target unscrambles a fixed-width field with
a linear (XOR-bitmask) transform before using it (a flags word, a packed
checksum, any bit-packed obfuscation layer), the isolated root-cause bytes
in *that* domain are still meaningful to ddmin (they correctly identify
*which file bytes* are causal) but not directly meaningful to a human: the
raw scrambled bits don't correspond one-to-one to the logical field values
a human reasons about. :func:`invert_scrambled_field` optionally maps an
isolated field's raw value back through a previously-recovered
:class:`~fuzzer_tool.core.xor_map_solver.XorBitmaskModel`'s inverse (via
:mod:`fuzzer_tool.core.gf2_common`'s bitmask-vector layer), so
:func:`format_root_cause_report` can report the original-domain field
value alongside the raw one when a caller supplies a model for that field.
This only applies to a *square* model (``out_bits == in_bits``, i.e. a
fixed-width field scrambled onto itself) -- the general variable-length
CRC-style checksum a target might also recover has no meaningful inverse
and isn't in scope here.
"""

from __future__ import annotations

from collections.abc import Callable

from fuzzer_tool.core.gf2_common import (
    bitmask_from_indices,
    invert_bitmask_map,
    verified_apply_inverse,
)
from fuzzer_tool.core.similarity import levenshtein_align
from fuzzer_tool.core.xor_map_solver import XorBitmaskModel

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
        elif op == "insert" and i in active:
            out.extend(data)
        elif op == "delete" and i not in active:
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


def invert_scrambled_field(model: XorBitmaskModel, value: int) -> int | None:
    """Recover the original-domain value of a scrambled field.

    Only defined for a **square** *model* (``out_bits == in_bits``): every
    input-bit index referenced across ``model.masks`` must be
    ``< model.out_bits``, i.e. the field is scrambled onto itself rather
    than being a variable-length checksum over a wider input (the general
    CRC-style case, where "inverting" isn't meaningful). Returns ``None``
    for a non-square model, a singular one, or when the per-query
    round-trip guard in :func:`fuzzer_tool.core.gf2_common.verified_apply_inverse`
    rejects the candidate -- a model recovered from partial evidence can be
    structurally full-rank while still being wrong for a specific value, so
    the raw pseudo-inverse is never trusted blind.
    """
    n_bits = model.out_bits
    forward = []
    for indices in model.masks:
        if any(i >= n_bits for i in indices):
            return None  # not square: references bits outside the output width
        forward.append(bitmask_from_indices(indices))
    inverse = invert_bitmask_map(forward, n_bits)
    if inverse is None:
        return None
    return verified_apply_inverse(forward, inverse, value)


def field_value(data: bytes, offset: int, nbytes: int) -> int:
    """Read an ``nbytes``-wide field at *offset* as an integer.

    Little-endian, matching :func:`fuzzer_tool.core.xor_map_solver.compute_xor_checksum`'s
    bit convention (input bit ``i`` is byte ``i // 8`` bit ``i % 8``,
    LSB-first per byte) -- so a field read this way lines up with the bit
    indices an :class:`XorBitmaskModel` recovered over that field expects.
    """
    return int.from_bytes(data[offset : offset + nbytes], "little")


def describe_scrambled_field(
    base: bytes,
    minimal_bytes: bytes,
    offset: int,
    model: XorBitmaskModel,
) -> str | None:
    """Report a root-cause field's value in both the raw (scrambled) and
    recovered original domain, for a field at *offset* covered by *model*.

    Compares the baseline's field value against the ddmin-minimized
    candidate's field value at the same offset -- the raw scrambled diff
    ddmin already isolated, plus (when invertible) what that diff means in
    the field's original, pre-scramble domain. Returns ``None`` -- rather
    than a guessed or partial answer -- if the field isn't covered by
    *minimal_bytes*, or if either value fails to invert (see
    :func:`invert_scrambled_field`).
    """
    nbytes = model.out_bits // 8
    if offset + nbytes > len(base) or offset + nbytes > len(minimal_bytes):
        return None
    base_raw = field_value(base, offset, nbytes)
    minimal_raw = field_value(minimal_bytes, offset, nbytes)
    base_orig = invert_scrambled_field(model, base_raw)
    minimal_orig = invert_scrambled_field(model, minimal_raw)
    if base_orig is None or minimal_orig is None:
        return None
    lines = [
        f"  scrambled field @ 0x{offset:04x} ({nbytes} bytes, {model.out_bits}-bit model):",
        f"    raw:      {base_raw:#0{2 + nbytes * 2}x} -> {minimal_raw:#0{2 + nbytes * 2}x}",
        f"    original: {base_orig:#0{2 + nbytes * 2}x} -> {minimal_orig:#0{2 + nbytes * 2}x}",
    ]
    return "\n".join(lines)


def format_root_cause_report(
    base: bytes,
    crash: bytes,
    script: list[EditOp],
    minimal_indices: list[int],
    scrambled_field: tuple[int, XorBitmaskModel] | None = None,
    minimal_bytes: bytes | None = None,
) -> str:
    """Human-readable report of the isolated root-cause edits.

    *scrambled_field*, if given, is ``(offset, model)`` for a field
    previously recovered by :mod:`fuzzer_tool.core.xor_map_solver`; when
    present (and *minimal_bytes* -- the ddmin-reconstructed candidate --
    is also given), the report appends the field's descrambled diff via
    :func:`describe_scrambled_field`, silently omitted if it doesn't apply
    (model not square, field outside the diffed range, etc.) rather than
    guessing.
    """
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
            lines.append(f'  offset 0x{pos:04x}: INSERT {data.hex()} ("{ascii_repr}")')
        elif op == "delete":
            old = base[pos]
            lines.append(f"  offset 0x{pos:04x}: DELETE {old:#04x} ('{_byte_repr(old)}')")

    if scrambled_field is not None and minimal_bytes is not None:
        offset, model = scrambled_field
        field_desc = describe_scrambled_field(base, minimal_bytes, offset, model)
        if field_desc is not None:
            lines.append("")
            lines.append(field_desc)

    return "\n".join(lines)
