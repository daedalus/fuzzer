"""Quick, cheap sanity check for the fuzzer's PRNG stream at startup.

Not a cryptographic test suite -- fuzzing does not need one (see the
modulo-bias note in ``rand_pool.py``). This exists to catch a *broken* RNG
setup before a campaign burns hours on it: a seed wired to a constant, a
bit-generator that silently degenerates, a pool-refill bug that makes the
stream sticky, etc.

Rather than reimplementing statistical tests, this reuses the checks already in
``core/randomness.py`` (calibrated there against ``os.urandom`` -- see
``test_randomness.py``):

  * ``monobit``       -- #1-bits vs #0-bits imbalance
  * ``byte_chisq``    -- byte-histogram uniformity
  * ``runs_test``     -- bit oscillation rate (catches RLE/padding-like output)
  * ``shannon_entropy`` -- bits/byte uniformity (catches low-entropy streams
    that pass the above by accident)

plus ``repeat_test`` on the raw draws, which is the one failure mode the
bit/byte-level tests structurally can't see: a stream that repeats its
previous value more often than chance (e.g. a pool-refill boundary bug or
an EMA-style feedback loop feeding the RNG), since marginal byte frequencies
stay uniform under stickiness.

``entropy`` is a G-test on the same histogram as ``byte_chisq`` (corr
~0.99), so it is reported but left out of the ``fishers_method``
combination: counting it twice flagged ~3.7% of healthy streams at the 1%
threshold. This never raises:
a suspect RNG is a warning, not a reason to abort a run that was otherwise
ready to go.

The battery sees only the low byte of one refill, so it passes a pool that
replays every refill (p=0.84), has its top 24 bits stuck (p=0.78), copies
words at lag 3 (p=0.89), or ignores ``--seed`` (statistics cannot see a
constant seed). ``structure_check`` covers those with exact 32-bit word
comparisons on cloned streams; see its docstring.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from fuzzer_tool.core.rand_pool import _NO_SEED_FALLBACK, _POOL_ENTRIES, RandPool
from fuzzer_tool.core.randomness import (
    byte_chisq,
    fishers_method,
    monobit,
    repeat_test,
    runs_test,
    shannon_entropy_test,
)

_DEFAULT_N_BYTES = 4096
_P_THRESHOLD = 0.01  # standard NIST SP 800-22 significance level
_REDUNDANT = frozenset({"entropy"})  # same histogram as byte_chisq

# structure_check: exact 32-bit word checks on a cloned stream.
_WINDOW = 1 << 16  # words drawn per stream: 16 refills
_WORD_SCALE = 4294967296.0  # 2**32: random() = word / 2**32, exact
_REPLAY_PERIOD = _POOL_ENTRIES  # one refill
_MAX_LAG = 64
_LANES = 4  # bytes per 32-bit word
# A healthy stream repeats a word with p = 2**-32 per pair: over _WINDOW
# words at 64 lags that is ~1e-3 expected repeats, so P(> 1) ~ 5e-7.
_MAX_EXACT_MATCHES = 1
# Per-lane chisq threshold: gross failures only (stuck bits give p = 0);
# 4 lanes add ~4e-4 false alarms on top of the battery's 1%.
_LANE_P = 1e-4


@dataclass
class RngHealthResult:
    n_bytes: int
    combined_p: float
    pvalues: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.combined_p >= _P_THRESHOLD and not self.warnings

    def summary(self) -> str:
        if self.ok:
            detail = ", ".join(f"{name}={p:.3f}" for name, p in self.pvalues.items())
            return f"OK (n={self.n_bytes}B, {detail})"
        reasons = list(self.warnings)
        if self.combined_p < _P_THRESHOLD:
            detail = ", ".join(f"{name}={p:.4f}" for name, p in self.pvalues.items())
            reasons.append(f"combined p={self.combined_p:.4f} ({detail})")
        return "SUSPECT: " + "; ".join(reasons)


def quick_health_check(rng, n_bytes: int = _DEFAULT_N_BYTES) -> RngHealthResult:
    """Run a quick sanity check on *rng*'s output stream.

    Draws ``n_bytes`` bytes (default 4096 -- one ``RandPool`` refill's
    worth, negligible next to a fuzzing campaign) and runs monobit,
    byte_chisq, runs_test, shannon_entropy and repeat_test, then combines
    all but entropy (redundant with byte_chisq) with Fisher's method. Also flags an outright-constant
    stream, which a single small sample could in principle slip past the
    statistical tests.

    *rng* only needs a ``randint_list(a, b, count)`` method -- the public
    ``RandPool`` API -- so this does not reach into pool internals and
    works with any drop-in replacement (e.g. ``tests/support/scripted_rng.py``).
    """
    values = rng.randint_list(0, 255, n_bytes)
    data = bytes(values)

    warnings: list[str] = []
    distinct = len(set(values))
    if distinct <= 1:
        warnings.append("RNG stream is constant (a single repeated byte value)")

    pvalues = {
        "monobit": monobit(data),
        "byte_chisq": byte_chisq(data),
        "runs": runs_test(data),
        "entropy": shannon_entropy_test(data),
        "repeat": repeat_test(values, alphabet=256),
    }
    combined_p = fishers_method([p for name, p in pvalues.items() if name not in _REDUNDANT])

    return RngHealthResult(
        n_bytes=len(data),
        combined_p=combined_p,
        pvalues=pvalues,
        warnings=warnings,
    )


def structure_check(pool, seed: int | None = None, default_pool=None) -> list[str]:
    """Exact stream checks the byte battery cannot see; returns warnings.

    ``quick_health_check`` tests the low byte of one refill. This draws
    ``_WINDOW`` raw 32-bit words from a deep copy of each pool, so the live
    streams are untouched, and checks:

      * replay  -- word i == word i + one refill (a pool replaying its batch)
      * lag     -- word i == word i - k, k = 1.._MAX_LAG
      * lanes   -- byte_chisq per byte lane (stuck high bits; the battery
        only sees lane 0)
      * seed    -- with a campaign *seed*, the stream is not the unseeded
        fallback (a seed wired to a constant)
      * default -- *default_pool* does not share the stream (fallback draws
        replaying the hotpath, the pre-2026-11-08 bug)

    Stream sharing compares overlapping word pairs (64-bit keys), so it is
    offset-agnostic within the window. Never raises on a suspect stream;
    *pool* needs ``random_list`` and must deep-copy (``RandPool`` does).
    """
    words = _words(pool)
    warnings = [*_replay(words), *_lag(words), *_lanes(words)]
    keys = np.sort(_pair_keys(words))  # sorted once, probed by both checks

    # An explicit seed equal to the fallback legitimately matches it.
    seeded = seed is not None and seed != _NO_SEED_FALLBACK
    if seeded and _shared(keys, _words(RandPool(seed=None))):
        warnings.append(f"seed {seed} not applied: stream matches the unseeded fallback")

    if default_pool is not None and _shared(keys, _words(default_pool)):
        warnings.append("default pool shares the main pool's stream")
    return warnings


def _words(pool) -> np.ndarray:
    """Next ``_WINDOW`` raw words of a clone of *pool*, as uint64."""
    floats = np.asarray(copy.deepcopy(pool).random_list(_WINDOW))
    return (floats * _WORD_SCALE).astype(np.uint64)


def _replay(words: np.ndarray) -> list[str]:
    matches = int(np.count_nonzero(words[_REPLAY_PERIOD:] == words[:-_REPLAY_PERIOD]))
    if matches <= _MAX_EXACT_MATCHES:
        return []
    return [f"stream replays every {_REPLAY_PERIOD} draws ({matches} repeats)"]


def _lag(words: np.ndarray) -> list[str]:
    hits = np.array([np.count_nonzero(words[k:] == words[:-k]) for k in range(1, _MAX_LAG + 1)])
    if hits.sum() <= _MAX_EXACT_MATCHES:
        return []
    k = int(hits.argmax()) + 1
    return [f"words repeat at lag {k} ({int(hits[k - 1])} of {len(words) - k})"]


def _lanes(words: np.ndarray) -> list[str]:
    # Column j = byte j of each word, j = 0 the low byte.
    shifts = np.arange(_LANES, dtype=np.uint64) * np.uint64(8)
    lanes = ((words[:, None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)
    return [
        f"byte lane {j} non-uniform (chisq p < {_LANE_P})"
        for j in range(_LANES)
        if byte_chisq(lanes[:, j].tobytes()) < _LANE_P
    ]


def _pair_keys(words: np.ndarray) -> np.ndarray:
    """Overlapping word pairs as 64-bit keys: (w[i] << 32) | w[i + 1]."""
    return (words[:-1] << np.uint64(32)) | words[1:]


def _shared(sorted_keys: np.ndarray, other: np.ndarray) -> bool:
    """True when *other* shares word pairs with *sorted_keys*: the same stream.

    searchsorted over one sorted side: 1.8 ms vs 20 ms for intersect1d at
    2**16 words, same counts.
    """
    probe = _pair_keys(other)
    idx = np.minimum(np.searchsorted(sorted_keys, probe), sorted_keys.size - 1)
    return int(np.count_nonzero(sorted_keys[idx] == probe)) > _MAX_EXACT_MATCHES
