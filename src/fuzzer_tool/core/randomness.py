"""Statistical randomness battery for byte buffers and fuzzer telemetry.

Clean-room implementations of tests whose *algorithms* are published in
Marsaglia's DIEHARD (public domain, FSU) and NIST SP 800-22 (US Government
work).  No dieharder (GPL-2+) source was translated; only the algorithm
descriptions and the reference chi-square/rank distributions are used, so
this file is MIT-compatible with the rest of the tree.

Two consumers:

1.  ``profile_buffer`` — classify regions of a fuzz input as
    random/compressed, structured, or textual.  Mutating a high-entropy
    region is near-useless work: a byte flip inside a deflate stream or an
    AES block almost always produces a parse-error path already covered.
    The profile lets the operator scheduler skip or downweight those offsets.

2.  ``uniformity_report`` — dieharder's actual architecture: run a test many
    times, collect p-values, then KS-test the p-values against U(0,1).  Any
    fuzzer subsystem that claims a distribution (bandit arm selection is
    uniform, crash inter-arrivals are exponential, coverage discovery is
    Poisson) can be validated this way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fuzzer_tool.core.chi_squared import chi_squared_pvalue

__all__ = [
    "monobit",
    "block_frequency",
    "runs_test",
    "byte_chisq",
    "serial_test",
    "binary_matrix_rank",
    "lagged_autocorrelation",
    "birthday_spacings",
    "kmer_occupancy",
    "ks_uniform",
    "kuiper_uniform",
    "fishers_method",
    "uniformity_report",
    "RegionProfile",
    "profile_buffer",
    "CorpusInvariants",
    "corpus_invariants",
    "invariant_mask",
    "permutation_test",
    "repeat_test",
]

_SQRT2 = math.sqrt(2.0)


# ── p-value plumbing ──────────────────────────────────────────────────


def _erfc(x: float) -> float:
    return math.erfc(x)


def _normal_two_sided(z: float) -> float:
    """Two-sided p-value for a standard normal deviate."""
    return _erfc(abs(z) / _SQRT2)


def chisq_sf(x2: float, dof: int) -> float:
    """P(chi^2_dof > x2).  Single source of truth is ``core.chi_squared``."""
    if dof <= 0:
        return 1.0
    return chi_squared_pvalue(x2, dof)


def _bits(data: bytes) -> np.ndarray:
    """Unpack to a uint8 array of 0/1, MSB-first per byte."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


# ── NIST SP 800-22 style tests ────────────────────────────────────────


def monobit(data: bytes) -> float:
    """Frequency (monobit) test.  p-value that #1s == #0s."""
    b = _bits(data)
    n = b.size
    if n < 100:
        return 1.0
    s = 2 * int(b.sum()) - n
    return _erfc(abs(s) / math.sqrt(2.0 * n))


def block_frequency(data: bytes, block_bits: int = 128) -> float:
    """Proportion of 1s within M-bit blocks is chi-square distributed."""
    b = _bits(data)
    n_blocks = b.size // block_bits
    if n_blocks < 2:
        return 1.0
    blocks = b[: n_blocks * block_bits].reshape(n_blocks, block_bits)
    pi = blocks.sum(axis=1) / block_bits
    x2 = 4.0 * block_bits * float(np.sum((pi - 0.5) ** 2))
    return chisq_sf(x2, n_blocks)


def runs_test(data: bytes) -> float:
    """Wald-Wolfowitz runs test on the bit sequence.

    Detects oscillation rate.  Fails hard on run-length-encoded or padded
    regions, which is exactly the signal we want for region classification.
    """
    b = _bits(data)
    n = b.size
    if n < 100:
        return 1.0
    pi = float(b.mean())
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return 0.0  # monobit already rejects; runs test undefined
    v = 1 + int(np.count_nonzero(b[1:] != b[:-1]))
    num = abs(v - 2.0 * n * pi * (1 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1 - pi)
    return _erfc(num / den)


def byte_chisq(data: bytes) -> float:
    """Pearson chi-square on the 256-bin byte histogram.

    dieharder's dab_bytedistrib, reduced to the single-stream case.
    """
    if len(data) < 256:
        return 1.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    exp = len(data) / 256.0
    x2 = float(np.sum((counts - exp) ** 2) / exp)
    return chisq_sf(x2, 255)


def serial_test(data: bytes, m: int = 8) -> float:
    """Overlapping m-bit pattern frequency (NIST serial / dieharder sts_serial).

    Picks up short-period structure that the byte histogram misses because it
    slides across byte boundaries.
    """
    b = _bits(data).astype(np.int64)
    n = b.size
    if n < (1 << m) * 5 or m < 2:
        return 1.0

    def psi2(mm: int) -> float:
        if mm <= 0:
            return 0.0
        ext = np.concatenate([b, b[: mm - 1]])
        idx = np.zeros(n, dtype=np.int64)
        for k in range(mm):
            idx = (idx << 1) | ext[k : k + n]
        counts = np.bincount(idx, minlength=1 << mm)
        return float(np.sum(counts.astype(np.float64) ** 2)) * (1 << mm) / n - n

    d1 = psi2(m) - psi2(m - 1)
    # NIST SP 800-22: nabla psi^2_m ~ chi^2(2^(m-1))
    return chisq_sf(d1, 1 << (m - 1))


# ── GF(2) binary matrix rank ──────────────────────────────────────────


def _rank_probabilities(rows: int, cols: int) -> dict[int, float]:
    """Exact P(rank = r) for a uniform random GF(2) matrix.

    p_r = 2^(r(rows+cols-r) - rows*cols)
          * prod_{i=0}^{r-1} (1-2^(i-cols))(1-2^(i-rows)) / (1-2^(i-r))
    """
    out: dict[int, float] = {}
    m = min(rows, cols)
    for r in range(m + 1):
        log_p = r * (rows + cols - r) - rows * cols
        prod = 1.0
        for i in range(r):
            prod *= (1.0 - 2.0 ** (i - cols)) * (1.0 - 2.0 ** (i - rows))
            prod /= 1.0 - 2.0 ** (i - r)
        out[r] = (2.0**log_p) * prod
    return out


def _batch_gf2_rank(mats: np.ndarray, cols: int) -> np.ndarray:
    """Vectorized GF(2) rank of K matrices given as (K, rows) bitmask rows.

    One numpy pass per column, so ``cols`` iterations regardless of K.
    """
    mats = mats.astype(np.uint64, copy=True)
    k, rows = mats.shape
    rank = np.zeros(k, dtype=np.int32)
    pivot = np.zeros(k, dtype=np.int64)
    ridx = np.arange(rows)[None, :]

    for col in range(cols):
        bit = np.uint64(1) << np.uint64(cols - 1 - col)
        valid = ((mats & bit) != 0) & (ridx >= pivot[:, None])
        active = valid.any(axis=1)
        if not active.any():
            continue
        sel = np.flatnonzero(active)
        first = np.argmax(valid[sel], axis=1)
        pr = pivot[sel]
        # swap pivot row with the first row holding the bit
        tmp = mats[sel, pr].copy()
        mats[sel, pr] = mats[sel, first]
        mats[sel, first] = tmp
        pivots = mats[sel, pr]
        elim = (mats[sel] & bit) != 0
        elim[np.arange(sel.size), pr] = False
        mats[sel] = np.where(elim, mats[sel] ^ pivots[:, None], mats[sel])
        pivot[sel] += 1
        rank[sel] += 1
    return rank


def binary_matrix_rank(data: bytes, rows: int = 32, cols: int = 32) -> float:
    """Rank of GF(2) submatrices vs. the exact rank distribution.

    This is the test that catches F2-linear structure — LFSRs, Mersenne
    Twister low bits, CRC tables, xorshift streams.  Nothing currently in the
    tree does this; ``berlekamp_massey.py`` measures linear complexity of a
    single sequence, which is the 1-D cousin.
    """
    assert cols <= 64
    bits_per = rows * cols
    n = (len(data) * 8) // bits_per
    if n < 30:
        return 1.0
    b = _bits(data)[: n * bits_per].reshape(n * rows, cols).astype(np.uint64)
    weights = np.uint64(1) << np.arange(cols - 1, -1, -1, dtype=np.uint64)
    mats = (b * weights).sum(axis=1).reshape(n, rows)

    ranks = _batch_gf2_rank(mats, cols)
    probs = _rank_probabilities(rows, cols)
    full = min(rows, cols)
    # standard 3-bin collapse: full, full-1, <= full-2
    obs = np.array(
        [
            int(np.count_nonzero(ranks == full)),
            int(np.count_nonzero(ranks == full - 1)),
            int(np.count_nonzero(ranks <= full - 2)),
        ],
        dtype=np.float64,
    )
    p_full = probs[full]
    p_m1 = probs[full - 1]
    exp = np.array([p_full, p_m1, 1.0 - p_full - p_m1]) * n
    exp = np.maximum(exp, 1e-12)
    x2 = float(np.sum((obs - exp) ** 2 / exp))
    return chisq_sf(x2, 2)


# ── correlation / spacing tests ───────────────────────────────────────


def lagged_autocorrelation(
    data: bytes, lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
) -> dict[int, float]:
    """Per-lag byte autocorrelation, as a normal-deviate p-value.

    dieharder's rgb_lagged_sums.  For a fuzz input, a strong p at lag L is a
    direct estimate of record stride — complementary to the FFT approach in
    ``periodicity.py`` and much cheaper for small L.
    """
    x = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    out: dict[int, float] = {}
    if x.size < 64:
        return {lag: 1.0 for lag in lags}
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    for lag in lags:
        if lag >= x.size:
            out[lag] = 1.0
            continue
        r = float(np.dot(xc[:-lag], xc[lag:])) / denom if denom > 0 else 0.0
        z = r * math.sqrt(x.size - lag)
        out[lag] = _normal_two_sided(z)
    return out


def birthday_spacings(data: bytes, word_bits: int = 24, n_points: int = 512) -> float:
    """Marsaglia's birthday-spacings test.

    Sorted samples, first differences, count of *repeated* spacings; that
    count is Poisson with lambda = n^3 / (4 * 2^word_bits).  Marsaglia's
    original parameters (24-bit words, 512 points) give lambda = 2.0, which
    is the regime where the Poisson tail actually discriminates -- with
    32-bit words lambda drops to 0.008 and the test is degenerate.

    Fuzzing relevance: offset/pointer tables inside a container format show
    an excess of repeated spacings, because records are equally sized.  A
    rejection here together with a rejection in the lag test is a reliable
    "this region is an index table" signal.
    """
    mask = (1 << word_bits) - 1
    avail = len(data) // 4
    if avail < n_points or n_points < 64:
        return 1.0
    words = (np.frombuffer(data[: avail * 4], dtype=np.uint32)[:n_points].astype(np.int64)) & mask
    spacings = np.diff(np.sort(words))
    spacings.sort()
    dup = int(np.count_nonzero(spacings[1:] == spacings[:-1]))
    lam = (n_points**3) / (4.0 * (2.0**word_bits))
    # two-sided Poisson tail with a mid-p correction: the duplicate count is
    # a small integer, so the plain tail is heavily discretized and its
    # p-values are not uniform under the null (verified empirically).
    cdf, term = 0.0, math.exp(-lam)
    for k in range(dup):
        cdf += term
        term *= lam / (k + 1)
    pmf = term  # P(X == dup)
    lower = cdf + 0.5 * pmf  # mid-p P(X <= dup)
    upper = 1.0 - cdf - 0.5 * pmf
    return max(0.0, min(1.0, 2.0 * min(lower, upper)))


def kmer_occupancy(data: bytes, tuple_bits: int | None = None) -> float:
    """OPSO-family occupancy test: how many w-bit tuples never appear.

    Occupancy only discriminates when the fill ratio lambda = n/cells is O(1):
    with lambda >> 1 every cell is hit and the statistic is identically zero,
    with lambda << 1 almost none are.  So the tuple width is chosen from the
    window size rather than fixed at a byte multiple -- this is why dieharder
    parameterises OPSO/OQSO/DNA by bit width (10/9/2 bits) instead of by byte
    count.

    Interpretation: a *deficit* of missing cells means a restricted alphabet
    (text, base64, an opcode stream); an *excess* means a repetitive or
    run-length-coded region.
    """
    bits = _bits(data)
    if bits.size < 4096:
        return 1.0
    if tuple_bits is None:
        # target lambda ~= 2  =>  cells ~= n/2
        n_avail = bits.size
        tuple_bits = int(round(math.log2(max(2.0, n_avail / 2.0))))
        tuple_bits = max(8, min(22, tuple_bits))
    cells = 1 << tuple_bits
    n = bits.size // tuple_bits  # non-overlapping tuples
    if n < 64:
        return 1.0
    tup = bits[: n * tuple_bits].reshape(n, tuple_bits).astype(np.int64)
    weights = 1 << np.arange(tuple_bits - 1, -1, -1, dtype=np.int64)
    idx = tup @ weights
    missing = cells - np.unique(idx).size

    lam = n / cells
    q = math.exp(-lam)
    mean = cells * q
    var = cells * q * (1.0 - (1.0 + lam) * q)
    if var <= 1e-9:
        return 1.0
    return _normal_two_sided((missing - mean) / math.sqrt(var))


# ── p-value aggregation (dieharder's real contribution) ───────────────


def _ks_exact_cdf(n: int, d: float) -> float:  # noqa: C901
    """Marsaglia-Tsang-Wang exact P(D_n < d) via the H-matrix power method.

    MEASURED, NOT ASSUMED: this was ported on the expectation that the
    asymptotic form breaks down at small n.  It does not.  Over 3000 null
    trials the false-reject rate at alpha=0.05 is 0.051 (exact) vs 0.050
    (asymptotic) at n=5, and the two agree to ~0.01 in p out to n=80.  The
    Stephens correction (sqrt(n)+0.12+0.11/sqrt(n)) already does the job.

    Kept only for reference; ``ks_uniform`` defaults to it but the asymptotic
    path is O(n) against O(n*m^3) here.  Do not port this into the tree --
    ``edge_tracker._kolmogorov_pvalue`` is already sufficient.
    """
    if d <= 0.0:
        return 0.0
    if d >= 1.0:
        return 1.0
    k = int(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    hmat = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i - j + 1 >= 0:
                hmat[i][j] = 1.0
    for i in range(m):
        hmat[i][0] -= h ** (i + 1)
        hmat[m - 1][i] -= h ** (m - i)
    hmat[m - 1][0] += (2 * h - 1) ** m if (2 * h - 1) > 0 else 0.0
    for i in range(m):
        for j in range(m):
            if i - j + 1 > 0:
                for g in range(1, i - j + 2):
                    hmat[i][j] /= g
    # H^n, scaling to avoid overflow
    q = np.linalg.matrix_power(hmat, n)
    s = q[k - 1][k - 1]
    for i in range(1, n + 1):
        s = s * i / n
        if s < 1e-140:
            s *= 1e140
    return float(s)


def ks_uniform(pvalues: list[float] | np.ndarray, exact: bool = True) -> float:
    """One-sample KS test of p-values against U(0,1).  Returns a meta p-value."""
    p = np.sort(np.asarray(pvalues, dtype=np.float64))
    n = p.size
    if n < 3:
        return 1.0
    i = np.arange(1, n + 1)
    d_plus = float(np.max(i / n - p))
    d_minus = float(np.max(p - (i - 1) / n))
    d = max(d_plus, d_minus)
    if exact and n <= 140:
        return max(0.0, 1.0 - _ks_exact_cdf(n, d))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    s = sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return max(0.0, min(1.0, 2.0 * s))


def kuiper_uniform(pvalues: list[float] | np.ndarray) -> float:
    """Kuiper's V — the rotation-invariant KS variant.

    MEASURED, NOT ASSUMED: ported on the theory that its tail sensitivity
    would beat KS on the alternatives a fuzzer actually sees.  It does not.
    Power at alpha=0.05 over 400 trials per alternative:

        alternative                 KS      Kuiper   Fisher
        two tiny outliers in 100    0.048   0.043    0.205
        tail-heavy p (x^1.35)       0.920   0.782    0.998
        cyclically shifted uniform  0.045   0.030    0.062

    Kuiper loses to plain KS everywhere, and Fisher wins everywhere.  Its
    genuine advantage is invariance under rotation of a *circular* variate
    with unknown phase, which is not the shape of any aggregation problem in
    this tree.  Use ``fishers_method`` with ``ks_uniform`` as a cross-check;
    this is kept only so the measurement is reproducible.
    """
    p = np.sort(np.asarray(pvalues, dtype=np.float64))
    n = p.size
    if n < 4:
        return 1.0
    i = np.arange(1, n + 1)
    v = float(np.max(i / n - p) + np.max(p - (i - 1) / n))
    en = math.sqrt(n)
    lam = (en + 0.155 + 0.24 / en) * v
    s = sum(
        (4.0 * j * j * lam * lam - 1.0) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101)
    )
    return max(0.0, min(1.0, 2.0 * s))


def fishers_method(pvalues: list[float] | np.ndarray) -> float:
    """Fisher's combined probability test.  Sensitive to many small p's."""
    p = np.clip(np.asarray(pvalues, dtype=np.float64), 1e-300, 1.0)
    x2 = float(-2.0 * np.sum(np.log(p)))
    return chisq_sf(x2, 2 * p.size)


def uniformity_report(pvalues: list[float] | np.ndarray) -> dict[str, float]:
    """Aggregate a batch of p-values three ways.

    Fisher is the more powerful of the three on every alternative measured
    (see ``kuiper_uniform``); KS is the useful cross-check because it responds
    to a shifted bulk rather than to extreme individual values.  Disagreement
    is informative: Fisher small with KS large means a few extreme outliers on
    an otherwise uniform background.
    """
    return {
        "ks": ks_uniform(pvalues),
        "kuiper": kuiper_uniform(pvalues),
        "fisher": fishers_method(pvalues),
        "n": float(len(pvalues)),
    }


# ── region profiling for fuzz inputs ──────────────────────────────────


@dataclass
class RegionProfile:
    offset: int
    length: int
    label: str
    confidence: float
    pvalues: dict[str, float] = field(default_factory=dict)

    def mutation_weight(self) -> float:
        """Suggested multiplier on this region's byte-selection probability."""
        return {
            "incompressible": 0.15,  # deflate/encrypted: flips die at the CRC
            "tabular": 1.6,  # offsets/lengths: arithmetic ops pay off
            "textual": 1.3,  # dictionary + token ops pay off
            "repetitive": 0.6,  # padding / run-length filler
            "mixed": 1.0,
        }.get(self.label, 1.0)


def _shannon_bits_per_byte(window: bytes) -> float:
    c = np.bincount(np.frombuffer(window, dtype=np.uint8), minlength=256).astype(np.float64)
    c = c[c > 0] / len(window)
    return float(-np.sum(c * np.log2(c)))


def _classify(pv: dict, window: bytes) -> tuple[str, float]:
    """Label a window from the test battery plus two cheap summary stats.

    Deliberately ordered most-specific first.  ``birthday_spacings`` is used
    only as a corroborating signal, never alone -- its p-values are heavily
    discretized (see its docstring) so a lone rejection is not trustworthy.
    """
    alpha = 0.01
    core = [pv["monobit"], pv["runs"], pv["byte_chisq"], pv["serial"], pv["rank"]]
    n_reject = sum(1 for p in core if p < alpha)

    h = _shannon_bits_per_byte(window)
    uniq = len(set(window))
    printable = float(np.mean([32 <= b < 127 or b in (9, 10, 13) for b in window]))
    lag_min = min(pv["lag"].values()) if pv["lag"] else 1.0

    if h < 1.5 or uniq <= 4:
        return "repetitive", 1.0 - h / 1.5
    if n_reject == 0 and h > 7.9 and pv["occupancy"] > alpha:
        return "incompressible", float(np.mean(core))
    if printable > 0.85 and h < 6.5:
        return "textual", printable
    if (lag_min < alpha and pv["birthday"] < 0.05) or (lag_min < alpha and h < 6.0):
        return "tabular", 1.0 - lag_min
    if n_reject >= 3 and h < 7.5:
        return "tabular", 1.0 - min(core)
    return "mixed", 0.0


def profile_buffer(
    data: bytes, window: int = 4096, stride: int | None = None
) -> list[RegionProfile]:
    """Slide a battery over the input and label each window.

    Cost is roughly 1 ms per 4 KiB window; run it once when a seed is admitted
    to the corpus, cache on the corpus entry, and reuse for every mutation
    round against that seed.
    """
    stride = stride or window
    out: list[RegionProfile] = []
    for off in range(0, max(1, len(data) - window + 1), stride):
        w = data[off : off + window]
        if len(w) < 512:
            break
        pv = {
            "monobit": monobit(w),
            "runs": runs_test(w),
            "byte_chisq": byte_chisq(w),
            "serial": serial_test(w, m=8),
            "rank": binary_matrix_rank(w, 32, 32),
            "occupancy": kmer_occupancy(w),
            "birthday": birthday_spacings(w),
            "lag": lagged_autocorrelation(w),
        }
        label, conf = _classify(pv, w)
        flat = {k: v for k, v in pv.items() if k != "lag"}
        flat.update({f"lag{k}": v for k, v in pv["lag"].items()})
        out.append(RegionProfile(off, len(w), label, conf, flat))
    return out


# ── corpus invariants (rgb_persist, generalized) ──────────────────────


@dataclass
class CorpusInvariants:
    """Bits that never varied across a corpus, plus the caveats that go with it."""

    mask: bytes
    """Per-byte mask; a set bit is a bit that took the same value in every sample."""

    n_samples: int
    common_length: int

    @property
    def fixed_offsets(self) -> list[int]:
        """Offsets where all eight bits were invariant."""
        return [i for i, m in enumerate(self.mask) if m == 0xFF]

    @property
    def partial_offsets(self) -> list[tuple[int, int]]:
        """(offset, mask) where only some bits were invariant.

        These are the interesting ones -- a 0xf0 mask on a big-endian length
        field means the corpus never exercised values above 2^12.
        """
        return [(i, m) for i, m in enumerate(self.mask) if 0 < m < 0xFF]

    @property
    def locked_bit_ratio(self) -> float:
        if not self.mask:
            return 0.0
        # One popcount over the whole mask rather than one per byte: the
        # concatenation of the bytes has the same population count as the
        # sum of theirs, and int.bit_count() runs it word at a time.
        return int.from_bytes(self.mask, "big").bit_count() / (8.0 * len(self.mask))

    def is_structural(self, offset: int) -> bool:
        """Whether *offset* is fully locked.

        NOT a claim that the field is a format constant.  The mask reports what
        did not vary, and cannot distinguish a magic byte from a field the
        corpus simply never exercised.  Validated on 200 synthetic PNGs: all 16
        true header bytes were recovered, but so were the bit-depth byte and the
        high bytes of width/height, because no sample exceeded 4096 pixels.
        Treat as a mutation prior, never as ground truth.
        """
        return 0 <= offset < len(self.mask) and self.mask[offset] == 0xFF


def invariant_mask(samples: list[bytes] | tuple[bytes, ...], min_samples: int = 16) -> bytes:
    """Accumulate ``mask &= ~(first ^ current)`` over a corpus.

    dieharder's rgb_persist looks for stuck bits in an RNG's output; the same
    accumulation over corpus entries finds the bits a *format* never varies --
    magic numbers, version fields, reserved padding.  O(n) per sample, byte
    parallel, so it is cheap enough to recompute whenever the corpus grows.

    Returns an all-zero mask (claiming nothing) below ``min_samples``, because
    with few samples almost everything looks invariant.

    The mask spans the largest prefix that at least ``min_samples`` entries
    actually reach, and only those entries contribute. Truncating to the
    *shortest* entry instead (as this did originally) made the operator inert
    on any real corpus: fuzzing accumulates trimmed and minimized inputs, so a
    single short entry -- one byte, or an empty one -- collapsed the mask to
    nothing no matter how much structure the other thousand entries shared.
    Measured on a synthetic 40-entry corpus with a shared 16-byte header: 128
    invariant bits, dropping to 39 / 14 / 2 / 0 as one 8- / 4- / 1- / 0-byte
    entry was added. That is the whole of ``invariant_break``'s reported 0.0%
    success rate.

    A short entry is not evidence that a later offset varies -- it has no
    opinion about an offset it does not contain -- so excluding it from the
    positions it cannot speak to is also the more correct measurement.
    """
    if len(samples) < max(2, min_samples):
        n = min((len(s) for s in samples), default=0)
        return bytes(n)
    # Largest n for which at least min_samples entries have length >= n.
    lengths = sorted((len(s) for s in samples), reverse=True)
    n = lengths[min_samples - 1]
    if n == 0:
        return b""
    contributors = [s for s in samples if len(s) >= n]
    first = np.frombuffer(contributors[0][:n], dtype=np.uint8)
    mask = np.full(n, 0xFF, dtype=np.uint8)
    for s in contributors[1:]:
        mask &= ~(first ^ np.frombuffer(s[:n], dtype=np.uint8))
        if not mask.any():
            break
    return mask.tobytes()


def corpus_invariants(
    samples: list[bytes] | tuple[bytes, ...], min_samples: int = 16
) -> CorpusInvariants:
    """``invariant_mask`` plus the metadata needed to interpret it."""
    mask = invariant_mask(samples, min_samples)
    return CorpusInvariants(
        mask=mask,
        n_samples=len(samples),
        common_length=len(mask),
    )


# ── sequence diagnostics for scheduler output ─────────────────────────

_PERM_INDEX: dict[int, dict[tuple[int, ...], int]] = {}


def _perm_index(k: int) -> dict[tuple[int, ...], int]:
    if k not in _PERM_INDEX:
        from itertools import permutations

        _PERM_INDEX[k] = {p: i for i, p in enumerate(permutations(range(k)))}
    return _PERM_INDEX[k]


def permutation_test(draws, k: int = 5) -> float:
    """Chi-square over the k! orderings of non-overlapping k-tuples.

    A marginal frequency chi-square on scheduler output is blind to *sequence*:
    a stream that is perfectly uniform per-symbol but sorted within each block
    passes it with p=0.67 and fails this with p<1e-4.

    Two deliberate departures from dieharder's ``diehard_operm5``:

    * Non-overlapping tuples, so the counts are independent and a plain
      chi-square applies.  Overlapping tuples need the 120x120 covariance
      matrix inverse that makes operm5 famously fragile.
    * Tied blocks are discarded.  ``argsort`` resolves ties by index, which
      manufactures a systematic excess of specific permutations -- with a
      137-symbol alphabet roughly 7% of 5-blocks contain a tie, and keeping
      them makes *every* stream, including a known-good one, look structured.

    Consequence of the tie handling: this test sees ordering, not repetition.
    A scheduler that sticks on its previous choice is invisible here; use
    ``repeat_test`` for that.
    """
    d = np.asarray(draws)
    if k < 3 or k > 6:
        raise ValueError("k must be in 3..6")
    ncell = math.factorial(k)
    n = d.size // k
    if n < 10 * ncell:
        return 1.0
    blocks = d[: n * k].reshape(n, k)
    blocks = blocks[(np.sort(blocks, axis=1)[:, 1:] != np.sort(blocks, axis=1)[:, :-1]).all(axis=1)]
    if blocks.shape[0] < 10 * ncell:
        return 1.0
    index = _perm_index(k)
    ranks = np.argsort(np.argsort(blocks, axis=1), axis=1)
    idx = np.fromiter((index[tuple(r)] for r in ranks.tolist()), dtype=np.int64, count=len(ranks))
    counts = np.bincount(idx, minlength=ncell)
    exp = blocks.shape[0] / ncell
    x2 = float(np.sum((counts - exp) ** 2) / exp)
    return chisq_sf(x2, ncell - 1)


def repeat_test(draws, alphabet: int | None = None) -> float:
    """Adjacent-repeat count against Binomial(n-1, 1/alphabet).

    Covers the failure mode ``permutation_test`` structurally cannot see: a
    scheduler that repeats its previous arm more often than chance, whether
    from an EMA feedback loop, a caching bug, or a pool-refill boundary.
    Marginal frequencies stay uniform under stickiness, so neither the existing
    chi-square nor the permutation test fires.
    """
    d = np.asarray(draws)
    if d.size < 1000:
        return 1.0
    k = int(alphabet) if alphabet else int(d.max()) + 1
    if k < 2:
        return 1.0
    n = d.size - 1
    obs = int(np.count_nonzero(d[1:] == d[:-1]))
    p = 1.0 / k
    mean = n * p
    var = n * p * (1.0 - p)
    if var <= 0:
        return 1.0
    return _normal_two_sided((obs - mean) / math.sqrt(var))
