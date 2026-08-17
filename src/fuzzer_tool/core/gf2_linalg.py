"""GF(2) linear-map algebra over bitmask-encoded maps.

Complements :mod:`fuzzer_tool.core.xor_map_solver`, which *recovers* an
unknown ``f: {0,1}^n -> {0,1}^n`` XOR-bitmask map from observed pairs. This
module operates on an already-solved map: inverting it, composing two of
them, and applying one to a value.

Map representation
-------------------
A map is ``masks: list[int]`` of length ``n_bits``, where ``masks[j]`` is
the bitmask of *input* bit positions that XOR together to produce *output*
bit ``j``:

    output_bit_j = XOR_{i where (masks[j] >> i) & 1} input_bit_i

This is the same map produced by
:func:`fuzzer_tool.core.xor_map_solver.recover_xor_model` (there stored as
per-bit index tuples in :class:`~fuzzer_tool.core.xor_map_solver.XorBitmaskModel.masks`
-- use :func:`bitmask_from_indices` / :func:`indices_from_bitmask` to
convert between the two encodings).

Ported from `xoreaxeaxeax/skitter-creek-bath-salts`'s
``analysis/unspaghettify.py`` (``invert_xor_map``, ``forward_transform``,
``inverse_transform``, ``compose_xor_maps``), refined against the C twin of
the same algorithm in ``userspace/alias_map.h`` (``apply_xor_map``,
``compute_inverse``). No Z3 dependency -- this is pure bitmask/Gaussian-
elimination arithmetic, unlike :mod:`xor_map_solver`.

Per-query verified inverse
---------------------------
:func:`invert_bitmask_map` only catches *total* singularity: it returns
``None`` once, at construction, if the matrix has no inverse at all. It
cannot catch a subtler failure the C source guards against: a caller might
hand this module a ``masks`` list that is not actually the true forward
map -- e.g. a partially-recovered map where some rows are placeholders
rather than confirmed evidence -- and that list can still be *structurally*
invertible (full rank) while being *semantically* wrong for some inputs.
:func:`verified_apply_inverse` is the guarded entry point: it forward-
reapplies the candidate inverse and rejects per query, on mismatch,
instead of trusting the pseudo-inverse blind. Callers recovering the
original domain of a solved map (e.g. mapping a minimized diff back
through the inverse) should call :func:`verified_apply_inverse`, not
:func:`apply_bitmask_map` on raw :func:`invert_bitmask_map` output.

This module is a leaf utility: nothing here is wired into a hot path, and
none of it has a solver timeout to worry about -- Gaussian elimination
over an ``n_bits x n_bits`` matrix is ``O(n_bits^3)`` bit-ops, trivial at
the field widths (<=64 bits) this tool ever deals with.
"""

from __future__ import annotations


def bitmask_from_indices(indices: tuple[int, ...] | list[int]) -> int:
    """Encode a list of input-bit indices as a single bitmask int.

    Inverse of :func:`indices_from_bitmask`. Convenience for converting
    :class:`fuzzer_tool.core.xor_map_solver.XorBitmaskModel.masks` (index
    tuples) into this module's bitmask-int representation.
    """
    mask = 0
    for i in indices:
        mask |= 1 << i
    return mask


def indices_from_bitmask(mask: int) -> tuple[int, ...]:
    """Decode a bitmask int back into a sorted tuple of set bit indices."""
    indices = []
    i = 0
    m = mask
    while m:
        if m & 1:
            indices.append(i)
        m >>= 1
        i += 1
    return tuple(indices)


def apply_bitmask_map(masks: list[int], value: int) -> int:
    """Apply a GF(2) bitmask map to *value*.

    Port of `unspaghettify.py`'s ``forward_transform``/``inverse_transform``,
    unified into one function since forward vs. inverse is purely a
    property of which ``masks`` list is passed in -- the application logic
    is identical either way.

    Output bit ``j`` is the XOR (parity) of the input bits of *value*
    selected by ``masks[j]``.
    """
    result = 0
    for j, bits in enumerate(masks):
        if (bits & value).bit_count() & 1:
            result |= 1 << j
    return result


def invert_bitmask_map(masks: list[int], n_bits: int) -> list[int] | None:
    """Gauss-Jordan pseudo-inverse of a square GF(2) bitmask map.

    Port of `unspaghettify.py`'s ``invert_xor_map``. ``masks`` must have
    exactly ``n_bits`` rows, each a bitmask over ``n_bits`` input
    positions (this module only inverts square maps -- a solved
    :class:`XorBitmaskModel` from :mod:`xor_map_solver` is always square,
    ``out_bits == in_bits``, since it's recovered over a fixed-width
    field).

    Returns ``None`` if the map is singular (not full rank) -- this must
    be handled by callers, not asserted away: a solved XOR map built from
    partial evidence is not guaranteed invertible, and a checksum/flags
    field genuinely can drop input-bit information (e.g. a field narrower
    than the data it covers).

    Does *not* by itself guarantee every individual value round-trips
    correctly if ``masks`` was not actually the true forward map (see the
    module docstring's per-query note) -- callers applying the result to
    recover original-domain values should go through
    :func:`verified_apply_inverse`, not this function's output directly.
    """
    if len(masks) != n_bits:
        raise ValueError(f"expected {n_bits} rows, got {len(masks)}")
    full = (1 << n_bits) - 1
    for row in masks:
        if row & ~full:
            raise ValueError(f"mask {row:#x} has bits outside the {n_bits}-bit field")

    # Augmented Gauss-Jordan: [M | I] -> [I | M^-1]. Each row carries the
    # M-side bits and the identity-side bits packed as separate ints so
    # elimination is a single XOR per row per pivot column.
    m_side = list(masks)
    i_side = [1 << j for j in range(n_bits)]

    for col in range(n_bits):
        pivot = None
        for r in range(col, n_bits):
            if (m_side[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            return None  # column has no nonzero row at or below `col` -- singular
        if pivot != col:
            m_side[col], m_side[pivot] = m_side[pivot], m_side[col]
            i_side[col], i_side[pivot] = i_side[pivot], i_side[col]
        for r in range(n_bits):
            if r != col and (m_side[r] >> col) & 1:
                m_side[r] ^= m_side[col]
                i_side[r] ^= i_side[col]

    return i_side


def verified_apply_inverse(masks_fwd: list[int], masks_inv: list[int], value: int) -> int | None:
    """Apply *masks_inv* to *value*, rejecting the result unless it
    round-trips through the true forward map *masks_fwd*.

    Ported from `alias_map.h`'s ``target_to_alias``, which never trusts
    ``compute_inverse``'s output directly:

        adj_alias = apply_xor_map(map->inverse, map->bits, adj_target);
        if (apply_xor_map(map->forward, map->bits, adj_alias) != adj_target)
            return -1;

    Why this is needed even when :func:`invert_bitmask_map` succeeded:
    construction-time singularity is a *global* rank check on whatever
    ``masks`` was handed to it. If that ``masks`` was itself only a
    partial or placeholder reconstruction of the true map (per the C
    source's own comment: "pa[col] unrecoverable; row left as identity,
    zeroed below"), the resulting pseudo-inverse can be structurally
    full-rank -- ``invert_bitmask_map`` returns a real answer, not
    ``None`` -- while still producing a wrong value for the specific
    inputs that depended on the placeholder rows. A one-time rank check at
    construction cannot see this, because the matrix isn't fully
    singular, just partially so; only re-deriving the candidate's forward
    image and comparing it against the value actually being inverted
    catches it, per query.

    Returns the recovered value, or ``None`` if the round-trip check
    fails (candidate is not trustworthy for this specific *value*).
    """
    candidate = apply_bitmask_map(masks_inv, value)
    if apply_bitmask_map(masks_fwd, candidate) != value:
        return None
    return candidate


def compose_bitmask_maps(inner: list[int], outer: list[int]) -> list[int]:
    """Compose two GF(2) bitmask maps: ``result(x) == outer(inner(x))``.

    Port of `unspaghettify.py`'s ``compose_xor_maps``. ``outer``'s bit
    positions are interpreted over ``inner``'s *output* domain, so
    ``len(inner)`` must equal the bit width ``outer``'s masks are defined
    over.

    Each output row is the XOR of the ``inner`` rows selected by the
    corresponding ``outer`` row -- linear composition, no per-value
    evaluation needed.
    """
    n_mid = len(inner)
    composed = []
    for bits in outer:
        row = 0
        selected = bits
        k = 0
        while selected:
            if selected & 1:
                if k >= n_mid:
                    raise ValueError(
                        f"outer row references bit {k} but inner only has {n_mid} rows"
                    )
                row ^= inner[k]
            selected >>= 1
            k += 1
        composed.append(row)
    return composed
