"""Shared GF(2) arithmetic: polynomial/field layer, and bitmask-vector layer.

This module is the single source of truth for two distinct GF(2) math
layers, kept in one file because both are "GF(2) utilities with no
dependents beyond a couple of call sites" and splitting them cost more
in cross-file navigation than it bought in separation:

**Polynomial / field layer** (original contents of this module):

- GF(2) polynomial operations on ``int``-encoded polynomials (bit k =
  coefficient of ``x^k``).
- Rabin irreducibility testing and irreducible-polynomial search.
- The :class:`GF2n` field class, which wraps the polynomial layer into a
  full ``GF(2^q)`` field with ``add``/``mul``/``pow``/``inv`` methods and a
  primitive-element search.
- A generic primitive-root finder used by :class:`GF2n` and any future
  GF(2^n) consumer.

**Bitmask-vector layer** (merged in from the former ``gf2_linalg.py``,
per `docs/handover/handover_skittercreek_tailslayer_port.md` item 2): GF(2)
linear-map algebra over bitmask-encoded maps, operating on an
*already-solved* ``f: {0,1}^n -> {0,1}^n`` XOR-bitmask map -- inverting
it, composing two of them, applying one to a value. This is a different
mathematical object from the polynomial layer above (vector-space maps
over ``GF(2)^n``, not the polynomial ring ``GF(2)[x]``); the two layers
don't call into each other. See that section's own docstring further
down for the map representation, provenance, and the per-query
verified-inverse rationale.

Callers in this repo
--------------------
- :mod:`fuzzer_tool.core.berlekamp_massey` — CRC polynomial recovery via
  ``poly_gcd`` (polynomial layer).
- :mod:`fuzzer_tool.core.cook_mertz` — Cook-Mertz MLE interpolation field
  layer, which imports :class:`GF2n` from here (polynomial layer).
- :func:`compose_linear_runs` (bitmask-vector layer) is the first caller
  of :func:`compose_bitmask_maps`, added for item 3 of the handover doc:
  folding maximal runs of XOR-linear operators (the "bitflip family",
  see :data:`fuzzer_tool.core.operator_categories.XOR_LINEAR_OPS`) in a
  mutation chain into one composed map. Its own intended consumer
  (:mod:`fuzzer_tool.core.root_cause`, for mapping a minimized
  transformed-domain diff back through a recovered linear map's inverse)
  is still separate, later work gated on that need materializing -- not
  done speculatively here.
"""

from __future__ import annotations

import random


def poly_deg(p: int) -> int:
    """Degree of a GF(2) polynomial encoded as a Python ``int``."""
    return p.bit_length() - 1 if p else -1


def poly_mul(a: int, b: int) -> int:
    """Carryless (GF(2)) polynomial multiplication."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def poly_divmod(a: int, b: int):
    """Return ``(quotient, remainder)`` of GF(2)-polynomial division ``a / b``."""
    if b == 0:
        raise ZeroDivisionError
    db = poly_deg(b)
    r = a
    q = 0
    while poly_deg(r) >= db:
        shift = poly_deg(r) - db
        r ^= b << shift
        q ^= 1 << shift
    return q, r


def poly_mod(a: int, b: int) -> int:
    """Polynomial remainder of ``a`` divided by ``b`` over GF(2)."""
    return poly_divmod(a, b)[1]


def poly_gcd(a: int, b: int) -> int:
    """GCD of two polynomials over GF(2)."""
    while b:
        a, b = b, poly_mod(a, b)
    return a


def poly_powmod(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation ``base**exp mod mod`` over GF(2)."""
    result = 1
    base = poly_mod(base, mod)
    while exp:
        if exp & 1:
            result = poly_mod(poly_mul(result, base), mod)
        base = poly_mod(poly_mul(base, base), mod)
        exp >>= 1
    return result


def _prime_factors(n: int) -> set[int]:
    """Return the distinct prime factors of ``n`` via trial division."""
    fs: set[int] = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def is_irreducible(poly: int, q: int) -> bool:
    """Rabin's irreducibility test for a degree-``q`` GF(2) polynomial."""
    x = 2  # the polynomial "x"
    x_mod = poly_mod(x, poly)
    # x^(2^q) mod poly must equal x mod poly
    t = x_mod
    for _ in range(q):
        t = poly_mod(poly_mul(t, t), poly)
    if t != x_mod:
        return False
    # for each prime p | q, gcd(x^(2^(q/p)) - x, poly) must be 1
    for p in _prime_factors(q):
        e = q // p
        t = x_mod
        for _ in range(e):
            t = poly_mod(poly_mul(t, t), poly)
        g = poly_gcd(t ^ x_mod, poly)
        if poly_deg(g) > 0:
            return False
    return True


def find_irreducible(q: int) -> int:
    """Return the first irreducible polynomial of degree ``q`` over GF(2)."""
    high = 1 << q
    for const_term in (1,):  # constant term must be 1 (else divisible by x)
        cand = high | const_term
        while cand < (1 << (q + 1)):
            if is_irreducible(cand, q):
                return cand
            cand += 2  # keep constant term = 1 (odd)
    raise RuntimeError(f"no irreducible polynomial of degree {q} found")


def find_primitive_root(order: int, is_primitive, rng) -> int:
    """Find a primitive root of a finite group by random search.

    Args:
        order: Size of the multiplicative group (e.g. ``2**q - 1``).
        is_primitive: Callable ``is_primitive(candidate) -> bool``.
        rng: Random source with ``randrange(lo, hi)``.  Accepted types:
            ``random.Random``, :class:`fuzzer_tool.core.rand_pool.RandPool`,
            or any duck-typed object with the same method.

    Returns:
        A primitive element in ``[1, order)``.
    """
    if order <= 1:
        return 1
    while True:
        a = rng.randrange(1, order)
        if is_primitive(a):
            return a


class GF2n:
    """The field GF(2^q) = GF(2)[x] / (modulus), modulus an irreducible degree-q poly."""

    def __init__(self, q: int, modulus: int | None = None, seed: int = 0) -> None:
        self.q = q
        self.order = 1 << q  # |F|
        self.m = self.order - 1  # |F*|, exponent of the multiplicative group
        self.mod = modulus if modulus is not None else find_irreducible(q)
        assert is_irreducible(self.mod, q), "supplied modulus is not irreducible"
        self._rng = random.Random(seed)
        self.gen = self._find_primitive()

    # ---- field ops ----
    def add(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        return poly_mod(poly_mul(a, b), self.mod)

    def pow(self, a: int, e: int) -> int:
        e = e % self.m if a != 0 else e
        return poly_powmod(a, e, self.mod)

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("no inverse of 0")
        return self.pow(a, self.m - 1)

    # ---- primitive element ----
    def _find_primitive(self) -> int:
        if self.m == 1:
            return 1
        factors = _prime_factors(self.m)

        def _is_primitive(a: int) -> bool:
            return all(self.pow(a, self.m // p) != 1 for p in factors)

        return find_primitive_root(self.m, _is_primitive, self._rng)

    def omega_powers(self) -> dict[int, int]:
        """Return dict ``i -> omega^i`` for ``i = 0..m``."""
        powers: dict[int, int] = {0: 1}
        cur = 1
        for i in range(1, self.m + 1):
            cur = self.mul(cur, self.gen)
            powers[i] = cur
        assert powers[self.m] == 1
        return powers


# ── Bitmask-vector layer (formerly gf2_linalg.py) ────────────────────────
#
# GF(2) linear-map algebra over bitmask-encoded maps. Complements
# :mod:`fuzzer_tool.core.xor_map_solver`, which *recovers* an unknown
# ``f: {0,1}^n -> {0,1}^n`` XOR-bitmask map from observed pairs. This
# section operates on an already-solved map: inverting it, composing two
# of them, and applying one to a value.
#
# Map representation
# -------------------
# A map is ``masks: list[int]`` of length ``n_bits``, where ``masks[j]`` is
# the bitmask of *input* bit positions that XOR together to produce *output*
# bit ``j``:
#
#     output_bit_j = XOR_{i where (masks[j] >> i) & 1} input_bit_i
#
# This is the same map produced by
# :func:`fuzzer_tool.core.xor_map_solver.recover_xor_model` (there stored as
# per-bit index tuples in :class:`~fuzzer_tool.core.xor_map_solver.XorBitmaskModel.masks`
# -- use :func:`bitmask_from_indices` / :func:`indices_from_bitmask` to
# convert between the two encodings).
#
# Ported from `xoreaxeaxeax/skitter-creek-bath-salts`'s
# ``analysis/unspaghettify.py`` (``invert_xor_map``, ``forward_transform``,
# ``inverse_transform``, ``compose_xor_maps``), refined against the C twin of
# the same algorithm in ``userspace/alias_map.h`` (``apply_xor_map``,
# ``compute_inverse``). Pure bitmask/Gaussian-elimination arithmetic, unlike
# :mod:`xor_map_solver`'s z3-backed recovery path.
#
# Per-query verified inverse
# ---------------------------
# :func:`invert_bitmask_map` only catches *total* singularity: it returns
# ``None`` once, at construction, if the matrix has no inverse at all. It
# cannot catch a subtler failure the C source guards against: a caller might
# hand this module a ``masks`` list that is not actually the true forward
# map -- e.g. a partially-recovered map where some rows are placeholders
# rather than confirmed evidence -- and that list can still be *structurally*
# invertible (full rank) while being *semantically* wrong for some inputs.
# :func:`verified_apply_inverse` is the guarded entry point: it forward-
# reapplies the candidate inverse and rejects per query, on mismatch,
# instead of trusting the pseudo-inverse blind. Callers recovering the
# original domain of a solved map (e.g. mapping a minimized diff back
# through the inverse) should call :func:`verified_apply_inverse`, not
# :func:`apply_bitmask_map` on raw :func:`invert_bitmask_map` output.
#
# This section is a leaf utility: nothing here is wired into a hot path, and
# none of it has a solver timeout to worry about -- Gaussian elimination
# over an ``n_bits x n_bits`` matrix is ``O(n_bits^3)`` bit-ops, trivial at
# the field widths (<=64 bits) this tool ever deals with.


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
    positions (this function only inverts square maps -- a solved
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
    section docstring's per-query note) -- callers applying the result to
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


def compose_linear_runs(
    chain: list[tuple[str, list[int]]],
) -> list[tuple[str, list[int]]]:
    """Compress maximal runs of XOR-linear operators in a mutation chain.

    Per ``docs/handover/handover_skittercreek_tailslayer_port.md`` item 3
    (the scoped version, since the general "compose the whole lineage"
    feature was rejected there): ``lineage.py``'s mutation chains are
    heterogeneous -- havoc byte flips, splices, dictionary insertions,
    structural mutations -- and only the ones drawn from
    :data:`fuzzer_tool.core.operator_categories.XOR_LINEAR_OPS` (the
    "bitflip family") are fixed-width, non-shifting XOR-linear maps that
    :func:`compose_bitmask_maps` can validly fold together. This function
    is the first caller of :func:`compose_bitmask_maps` in the tree; see
    that module's docstring for why it landed ahead of one.

    ``chain`` is an ordered sequence of ``(op_name, masks)`` steps applied
    in that order -- ``masks`` a bitmask map (see the module docstring for
    the representation) already known for that step, all over the same bit
    width. This function does not derive ``masks`` itself: a caller
    driving actual mutation replay is the one that knows, at the time each
    step runs, what its concrete XOR mask was.

    Returns a new chain where every maximal run of consecutive
    XOR-linear-only steps is replaced by a single composed step (name
    ``"+".join`` of the run's op names, in order), and every non-linear
    step is passed through unchanged, in its original position. Composing
    a run of length 1 is a no-op (returned unchanged, not renamed) so
    single isolated linear steps round-trip identically instead of
    growing a spurious composite name.

    Non-linear steps are never merged across, into, or with each other --
    only consecutive linear steps compose; a run is broken the moment a
    non-linear op appears, even if a linear op follows it.
    """
    from fuzzer_tool.core.operator_categories import is_xor_linear

    result: list[tuple[str, list[int]]] = []
    run_names: list[str] = []
    run_masks: list[int] | None = None

    def _flush() -> None:
        nonlocal run_names, run_masks
        if run_masks is None:
            return
        if len(run_names) == 1:
            result.append((run_names[0], run_masks))
        else:
            result.append(("+".join(run_names), run_masks))
        run_names = []
        run_masks = None

    for name, masks in chain:
        if is_xor_linear(name):
            if run_masks is None:
                run_masks = masks
            else:
                # Sequential application: apply the run so far, then this
                # step -- result(x) == masks(run_so_far(x)).
                run_masks = compose_bitmask_maps(run_masks, masks)
            run_names.append(name)
        else:
            _flush()
            result.append((name, masks))
    _flush()
    return result


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
