"""LTE turbo-code repair and bit-error injection (``--turbo-k``, ``--turbo-qpp``).

A target behind a turbo decoder (LTE PHY stacks, SDR receivers, 3GPP test
harnesses) feeds plain bit flips to its iterative decoder, which either
"corrects" them or rejects the block, so the code behind it is never reached.
This operator treats the buffer as consecutive rate-1/3 PCCC codewords
(TS 36.212 5.1.3.2) and, on one codeword, does one of:

* repair: flip 1..4 systematic bits, then re-encode parity and tail;
* correctable: re-encode, then flip 1..3 bits anywhere: the iterative
  (BCJR / max-log-MAP) decoder must converge back to the codeword;
* uncorrectable: re-encode, then flip a burst of 4..16 systematic bits plus
  the same information bits' parity in both encoders (3 x burst bits): the
  decoder fails or converges elsewhere, exercising the CRC-reject path;
* tail break: re-encode, then flip 1..12 termination bits, so the trellis no
  longer ends in state 0: exercises termination handling.

Encoder (two identical RSCs, transfer function [1, g1/g0])::

    c_k ──┬────────────────────────────────────────▶ x_k  (systematic)
          │      ┌────────── g0 = 1+D^2+D^3 ◀────┐
          └─▶(+)─┴─a─▶[D]─s1─▶[D]─s2─▶[D]─s3─────┘
                  └──── g1 = 1+D+D^3 ───────────▶ z_k  (parity 1)
    c'_i = c_PI(i), PI(i) = (f1*i + f2*i^2) mod K ─▶ RSC #2 ─▶ z'_k (parity 2)

Termination: 3 steps per encoder with input = feedback (a = 0), giving
x_K..K+2, z_K..K+2 (RSC #1) and x'_K..K+2, z'_K..K+2 (RSC #2).

Codeword layout (N = 3K + 12 bits, contiguous, MSB-first within a byte;
codeword i at bits [i*N, (i+1)*N), a trailing partial codeword is untouched)::

    0        K        2K       3K     +3     +6     +9     +12
    | x  K   | z  K   | z' K   | x_K.. | z_K.. | x'_K..| z'_K..|

RNG draws (``randint`` only): codeword, mode, then per mode
REPAIR count + count picks; CORRECTABLE count + count picks; UNCORRECTABLE
burst start + burst length; TAIL_BREAK count + count picks. Picks are a
partial Fisher-Yates (``randint(i, n-1)``). If the result equals the input,
systematic bit 0 is flipped (no draw) so the operator always changes data.
Stdlib + numpy only (Hard Rule 51).
"""

from __future__ import annotations

import functools
from enum import IntEnum

import numpy as np

#: LTE turbo block sizes (36.212 Table 5.1.3-3).
MIN_K, MAX_K = 40, 6144
#: First LTE table row.
DEFAULT_K, DEFAULT_F1, DEFAULT_F2 = 40, 3, 10
#: 3 tail steps x (systematic, parity) x 2 encoders.
TAIL_BITS = 12
_TAIL_STEPS = 3
_STREAMS = 3  # rate 1/3

_MAX_DATA_EDITS = 4
_MAX_CORRECTABLE = 3
_MIN_BURST, _MAX_BURST = 4, 16


class Mode(IntEnum):
    """turbo_fix edit; values are the RNG draw."""

    REPAIR = 0
    CORRECTABLE = 1
    UNCORRECTABLE = 2
    TAIL_BREAK = 3


# 1/g0 = q / (1 + D^7) with q = (1+D^7)/g0 = 1 + D^2 + D^3 + D^4 (g0 is primitive).
_PERIOD = 7
_Q_TAPS = (2, 3, 4)


@functools.cache
def _qpp(k: int, f1: int, f2: int) -> tuple[np.ndarray, np.ndarray]:
    """(PI, PI^-1) for the QPP interleaver; ValueError unless a permutation."""
    if not MIN_K <= k <= MAX_K:
        raise ValueError(f"need {MIN_K} <= k <= {MAX_K}, got k={k}")

    i = np.arange(k, dtype=np.int64)
    pi = (f1 % k * i + f2 % k * i * i) % k
    if np.any(np.bincount(pi, minlength=k) != 1):
        raise ValueError(f"QPP f1={f1} f2={f2} is not a permutation of 0..{k - 1}")

    inv = np.empty(k, dtype=np.int64)
    inv[pi] = i
    pi.flags.writeable = inv.flags.writeable = False
    return pi, inv


def check_params(k: int, f1: int, f2: int) -> None:
    """ValueError unless 40 <= k <= 6144 and (f1*i + f2*i^2) mod k permutes 0..k-1."""
    _qpp(k, f1, f2)


def parse_qpp(text: str) -> tuple[int, int]:
    """Parse ``"F1,F2"`` (decimal, non-negative); ValueError on failure."""
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"QPP must be F1,F2, got {text!r}")

    try:
        f1, f2 = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"QPP coefficient is not a number: {exc}") from exc
    if f1 < 0 or f2 < 0:
        raise ValueError(f"QPP coefficients must be >= 0, got {f1},{f2}")
    return f1, f2


def _feedback(c: np.ndarray) -> np.ndarray:
    """Register input a_k = c_k ^ a_{k-2} ^ a_{k-3}, without a per-bit loop.

    a = c * q / (1 + D^7): u = c * q is three shifted XORs, and dividing by
    1 + D^7 (a_k = u_k ^ a_{k-7}) is a prefix XOR down each column of u laid
    out in rows of 7.
    """
    k = len(c)
    rows = -(-k // _PERIOD)
    u = np.zeros(rows * _PERIOD, dtype=np.uint8)
    u[:k] = c
    for tap in _Q_TAPS:
        u[tap:k] ^= c[:-tap]
    return np.bitwise_xor.accumulate(u.reshape(rows, _PERIOD), axis=0).ravel()[:k]


def _rsc(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One constituent encoder: (parity z, tail x, tail z)."""
    a = _feedback(c)

    # z_k = a_k ^ s1 ^ s3 = a_k ^ a_{k-1} ^ a_{k-3}
    z = a.copy()
    z[1:] ^= a[:-1]
    z[3:] ^= a[:-3]

    # Termination: input = feedback (s2 ^ s3) drives the register to 0.
    s1, s2, s3 = int(a[-1]), int(a[-2]), int(a[-3])
    tx, tz = [], []
    for _ in range(_TAIL_STEPS):
        tx.append(s2 ^ s3)
        tz.append(s1 ^ s3)
        s1, s2, s3 = 0, s1, s2
    return z, np.array(tx, dtype=np.uint8), np.array(tz, dtype=np.uint8)


def _encode(c: np.ndarray, pi: np.ndarray) -> np.ndarray:
    z1, tx1, tz1 = _rsc(c)
    z2, tx2, tz2 = _rsc(c[pi])
    return np.concatenate((c, z1, z2, tx1, tz1, tx2, tz2))


def encode(info_bits, k: int, f1: int, f2: int) -> np.ndarray:
    """Return the N = 3k+12 codeword bits (uint8 0/1) of *k* info bits."""
    pi, _ = _qpp(k, f1, f2)
    c = np.asarray(info_bits, dtype=np.uint8) & 1
    if c.shape != (k,):
        raise ValueError(f"need {k} info bits, got shape {c.shape}")
    return _encode(c, pi)


def _pick(rng, n: int, count: int) -> list[int]:
    """*count* distinct indices in range(n) (sparse partial Fisher-Yates)."""
    swapped: dict[int, int] = {}
    out = []
    for i in range(count):
        j = rng.randint(i, n - 1)
        out.append(swapped.get(j, j))
        swapped[j] = swapped.get(i, i)
    return out


def _correctable(rng, k: int, _inv: np.ndarray) -> list[int]:
    """1..3 bits anywhere in the codeword."""
    return _pick(rng, _STREAMS * k + TAIL_BITS, rng.randint(1, _MAX_CORRECTABLE))


def _burst(rng, k: int, inv: np.ndarray) -> np.ndarray:
    """A systematic burst plus its parity in both encoders (wraps mod k).

    Info bit j drives z_j in RSC #1 and z'_{PI^-1(j)} in RSC #2.
    """
    start = rng.randint(0, k - 1)
    sys = (start + np.arange(rng.randint(_MIN_BURST, _MAX_BURST))) % k
    return np.concatenate((sys, k + sys, 2 * k + inv[sys]))


def _tail(rng, k: int, _inv: np.ndarray) -> list[int]:
    """1..12 termination bits."""
    base = _STREAMS * k
    return [base + p for p in _pick(rng, TAIL_BITS, rng.randint(1, TAIL_BITS))]


_INJECT = {Mode.CORRECTABLE: _correctable, Mode.UNCORRECTABLE: _burst, Mode.TAIL_BREAK: _tail}


def turbo_fix(data: bytes, rng, k: int, f1: int, f2: int) -> bytes | None:
    """Mutate one turbo codeword of *data*; None if no full codeword fits.

    Args:
        data: Bitstream of consecutive N = 3k+12 bit codewords (layout above).
        rng: ``RandPool`` (or ScriptedRng in tests).
        k: Information block size, 40..6144.
        f1: QPP linear coefficient.
        f2: QPP quadratic coefficient.

    Raises:
        ValueError: Invalid (k, f1, f2); never for any *data*.
    """
    pi, inv = _qpp(k, f1, f2)
    n = _STREAMS * k + TAIL_BITS
    count = len(data) * 8 // n
    if not count:
        return None

    # Unpack only the bytes the chosen codeword touches.
    lo = rng.randint(0, count - 1) * n
    b0, b1 = lo // 8, (lo + n + 7) // 8
    off = lo - 8 * b0
    window = np.unpackbits(np.frombuffer(data, dtype=np.uint8, count=b1 - b0, offset=b0))
    cw = window[off : off + n].copy()

    mode = Mode(rng.randint(0, len(Mode) - 1))
    if mode is Mode.REPAIR:
        cw[_pick(rng, k, rng.randint(1, _MAX_DATA_EDITS))] ^= 1
    cw = _encode(cw[:k], pi)
    if mode is not Mode.REPAIR:
        cw[_INJECT[mode](rng, k, inv)] ^= 1

    # A non-codeword input can coincide with the edit: force a change.
    if np.array_equal(cw, window[off : off + n]):
        cw[0] ^= 1
    window[off : off + n] = cw
    return data[:b0] + np.packbits(window).tobytes() + data[b1:]
