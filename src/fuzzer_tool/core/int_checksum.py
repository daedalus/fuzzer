"""Integer-modulus checksum computation (Adler-32, Fletcher, weighted sums).

Parallel to :mod:`fuzzer_tool.core.crc32`, not an extension of it.  ``crc32.py``
dispatches on a GF(2) model ``(poly, width, init, final_xor, reflect_in,
reflect_out)``; that tuple has no slot for a modulus or an integer multiplier,
and overloading its fields with a second, incompatible meaning would be worse
than a small parallel module.

Two model families are supported, matching the two shapes that show up in real
formats:

- ``weighted_sum``: ``(sum(data[j] * multiplier**j) + init_a) mod modulus`` —
  the bespoke ``sum(data[i] * k^i) mod N`` schemes found in network protocols
  and custom binary logs.  ``multiplier=1`` degenerates to a plain byte sum.
- ``fletcher``: two running sums mod ``modulus``, packed into one output.
  Covers Fletcher-16/32 *and* Adler-32 — Adler is the same two-running-sums
  shape with a prime modulus and ``init_a=1``.

Both families are **integer-linear mod N** (true Z/NZ arithmetic), which is
exactly what the GF(2) machinery in ``berlekamp_massey.py`` cannot represent.

The active model is set by :class:`~fuzzer_tool.core.checksum_learner.ChecksumLearner`
when integer-checksum recovery verifies, and read from mutation operators —
hence the same lock pattern ``crc32.py`` uses.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import numpy as np

# Above this word count the O(n^2)-magnitude index-weighted sum can exceed
# int64 (n^2/2 * 65535 for 2-byte words), so the numpy path is unsafe and the
# exact Python fallback is used instead.  2**21 words keeps the worst case at
# ~1.4e17, two orders of magnitude below the int64 ceiling.
_NUMPY_SAFE_WORDS = 1 << 21

KIND_WEIGHTED_SUM = "weighted_sum"
KIND_FLETCHER = "fletcher"


class IntModel(NamedTuple):
    """A recovered integer-modulus checksum model.

    Attributes:
        kind: Either ``weighted_sum`` or ``fletcher``.
        modulus: The modulus ``N`` of the ``Z/NZ`` arithmetic.
        multiplier: Base ``k`` for ``weighted_sum``; unused (``1``) for ``fletcher``.
        init_a: Initial value of the (only, or first) running sum.
        init_b: Initial value of the second running sum; ``fletcher`` only.
        word_bytes: Bytes per accumulated word — ``1`` for Adler-32 and
            Fletcher-16, ``2`` for Fletcher-32.
        out_bits: Total output width in bits.  For ``fletcher`` the two halves
            are packed as ``(b << out_bits // 2) | a``.
        big_endian: Word byte order when ``word_bytes == 2``.
    """

    kind: str
    modulus: int
    multiplier: int = 1
    init_a: int = 0
    init_b: int = 0
    word_bytes: int = 1
    out_bits: int = 32
    big_endian: bool = False

    @property
    def nbytes(self) -> int:
        """Width of the packed checksum field in bytes."""
        return self.out_bits // 8


# Well-known models, tried as a cheap pre-check before any general recovery.
# Mirrors smt_solver.py's ``_COMMON_DIVISORS`` heuristic-first philosophy: try
# the cheap thing before the general solver.
ADLER32 = IntModel(KIND_FLETCHER, 65521, init_a=1, word_bytes=1, out_bits=32)
FLETCHER16 = IntModel(KIND_FLETCHER, 255, word_bytes=1, out_bits=16)
FLETCHER32_LE = IntModel(KIND_FLETCHER, 65535, word_bytes=2, out_bits=32, big_endian=False)
FLETCHER32_BE = IntModel(KIND_FLETCHER, 65535, word_bytes=2, out_bits=32, big_endian=True)
SUM16 = IntModel(KIND_WEIGHTED_SUM, 1 << 16, out_bits=16)
SUM32 = IntModel(KIND_WEIGHTED_SUM, 1 << 32, out_bits=32)

COMMON_MODELS: tuple[IntModel, ...] = (
    ADLER32,
    FLETCHER16,
    FLETCHER32_LE,
    FLETCHER32_BE,
    SUM16,
    SUM32,
)

# Module-level active model.  ``None`` means "no integer model recovered".
# Protected by a lock because recovery runs in the main fuzz loop while
# mutations may read it from worker threads — same reasoning as crc32.py.
_active_int_model: IntModel | None = None
_int_lock = threading.Lock()


def set_active_int_model(model: IntModel | None) -> None:
    """Set the active integer-checksum model.

    Call this from the fuzzer when integer recovery verifies or state is
    restored from ``state.json``.
    """
    global _active_int_model
    with _int_lock:
        _active_int_model = model


def get_active_int_model() -> IntModel | None:
    """Return the active integer model, or ``None`` when none is recovered."""
    with _int_lock:
        return _active_int_model


def clear_active_int_model() -> None:
    """Drop the active integer model (used by tests and state resets)."""
    set_active_int_model(None)


# ── word extraction ────────────────────────────────────────────────────


def to_words(data: bytes, word_bytes: int, big_endian: bool) -> np.ndarray:
    """Split *data* into an array of unsigned words.

    A trailing odd byte is zero-padded when ``word_bytes == 2``, matching the
    usual Fletcher-32 convention.
    """
    if word_bytes == 1:
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    if len(data) % 2:
        data = data + b"\x00"
    dtype = np.dtype(">u2") if big_endian else np.dtype("<u2")
    return np.frombuffer(data, dtype=dtype).astype(np.int64)


def sums(data: bytes, word_bytes: int, big_endian: bool) -> tuple[int, int, int]:
    """Return ``(n, S1, S2)`` for the two-running-sum recurrence.

    ``S1 = sum(w)`` and ``S2 = sum((n - t) * w[t])`` are the exact,
    *unreduced* integers the recurrence needs:

    ``a_n = init_a + S1``  and  ``b_n = init_b + n * init_a + S2``  (mod N)

    Both are closed forms of the running loop — reduction mod ``N`` is a ring
    homomorphism, so reducing at each step or only at the end gives the same
    answer.  Recovery needs the unreduced values, so they are returned raw.
    """
    words = to_words(data, word_bytes, big_endian)
    n = int(words.size)
    if n == 0:
        return 0, 0, 0
    s1 = int(words.sum())
    if n <= _NUMPY_SAFE_WORDS:
        # Vectorized index-weighted sum; safe from int64 overflow below the
        # _NUMPY_SAFE_WORDS ceiling checked above.
        weights = np.arange(n, 0, -1, dtype=np.int64)
        s2 = int(weights @ words)
    else:
        s2 = sum((n - t) * int(w) for t, w in enumerate(words))
    return n, s1, s2


def weighted_raw(data: bytes, multiplier: int) -> int:
    """Return the exact, unreduced ``sum(data[j] * multiplier**j)``.

    Uses Horner's method over the reversed byte string, so the intermediate
    magnitude grows once rather than recomputing ``multiplier**j`` per byte.
    For ``multiplier == 1`` this is just the byte sum and stays cheap for any
    input length; for larger multipliers the result has ``O(len(data))``
    limbs, which is why recovery caps the data length it feeds this.
    """
    if multiplier == 1:
        return sum(data)
    acc = 0
    for byte in reversed(data):
        acc = acc * multiplier + byte
    return acc


# ── evaluation ─────────────────────────────────────────────────────────


def eval_model(model: IntModel, data: bytes) -> int:
    """Compute the checksum of *data* under *model*."""
    if model.kind == KIND_FLETCHER:
        n, s1, s2 = sums(data, model.word_bytes, model.big_endian)
        mod = model.modulus
        a = (model.init_a + s1) % mod
        b = (model.init_b + n * model.init_a + s2) % mod
        return (b << (model.out_bits // 2)) | a
    if model.kind == KIND_WEIGHTED_SUM:
        return (weighted_raw(data, model.multiplier) + model.init_a) % model.modulus
    raise ValueError(f"unknown integer checksum kind: {model.kind!r}")


def compute_int_checksum(data: bytes, model: IntModel | None = None) -> int | None:
    """Compute an integer checksum over *data* using the active model.

    Args:
        data: Input bytes.
        model: Optional explicit model.  When ``None`` the module's active
            model is used.

    Returns:
        The checksum, or ``None`` when no integer model has been recovered.
        Callers must handle ``None`` — there is deliberately no fallback to a
        guessed model (see the non-negotiables in the handover: never silently
        substitute an unverified model).
    """
    active = model if model is not None else get_active_int_model()
    if active is None:
        return None
    return eval_model(active, data)


# ── serialization ──────────────────────────────────────────────────────


def model_to_dict(model: IntModel | None) -> dict[str, object] | None:
    """Serialize *model* for ``state.json`` persistence."""
    return None if model is None else model._asdict()


def model_from_dict(data: dict[str, object] | None) -> IntModel | None:
    """Rebuild a model from :func:`model_to_dict` output.

    Returns ``None`` for missing or malformed input rather than raising — a
    corrupt state file must not take down the fuzz run.
    """
    if not data:
        return None
    try:
        model = IntModel(**{k: data[k] for k in IntModel._fields if k in data})  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # NamedTuple annotations are not enforced at runtime, so a state file
    # carrying a string where an int belongs would otherwise build a model
    # that only explodes later, inside a mutation.
    if not isinstance(model.kind, str) or model.kind not in (KIND_FLETCHER, KIND_WEIGHTED_SUM):
        return None
    ints = (model.modulus, model.multiplier, model.init_a, model.init_b, model.word_bytes)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in ints):
        return None
    if model.modulus < 2 or model.multiplier < 1 or model.word_bytes not in (1, 2):
        return None
    if not isinstance(model.out_bits, int) or model.out_bits not in (16, 32, 64):
        return None
    return model
