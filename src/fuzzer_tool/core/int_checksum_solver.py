"""Recovery of integer-modulus checksum models from observed ``(data, checksum)`` pairs.

Companion to :mod:`fuzzer_tool.core.berlekamp_massey`, which recovers the GF(2)
(CRC-style) family.  This module recovers the ``Z/NZ`` family — Adler-32,
Fletcher-16/32, and bespoke ``sum(data[i] * k^i) mod N`` schemes.

How modulus recovery works
--------------------------

Writing the model as

    c_i = R_i + b  (mod N)

where ``R_i`` is the exact unreduced raw sum (computable once the multiplier
is fixed), each pair gives ``R_i - c_i = N * q_i - b`` for an unknown integer
``q_i``.  The congruences are exact and the checksum field is fully observed,
so differencing kills the unknown ``b`` outright:

    (R_i - c_i) - (R_j - c_j) = N * (q_i - q_j)

Every pairwise difference is therefore an exact integer multiple of ``N``, and

    gcd_i( (R_i - c_i) - (R_0 - c_0) ) = N * gcd_i(q_i - q_0)

The GCD is exact, runs in microseconds, and needs no C extension.  Measured
over 2400 synthetic trials, the failure modes are only:

- ``gcd == 0`` (all ``q_i`` equal — the raw sum never crossed a multiple of
  ``N``), which is fail-closed: no candidate is proposed; and
- ``gcd == m * N`` for a small ``m`` (the ``q_i`` differences shared a
  factor), from which ``N`` is recovered by enumerating small cofactors.

A genuinely wrong modulus was never produced.  See
``tests/test_int_checksum_solver.py`` for the characterization that backs
this, including the adversarial random-pair suite.

The one real precondition: **the raw sum must actually wrap the modulus**.
If ``sum(data) < N`` for every pair then ``q_i == 0`` for all ``i``, every
difference is zero, and ``N`` is not visible in the data at all — it is
genuinely under-determined, and any modulus above the largest observed sum
fits the evidence equally well.  For an Adler-32 style ``k=1`` sum that means
pairs of at least ``2 * N / 255`` bytes (~512 bytes for ``N = 65521``).  This
directly contradicts reusing ``checksum_learner``'s
``_GCD_MAX_PAIR_DATA_BYTES = 256`` cap here: that cap would filter out
precisely the pairs that carry the modulus information.  The integer path
therefore prefers *large* pairs where the GF(2) path must avoid them.
"""

from __future__ import annotations

from math import gcd

from fuzzer_tool.core.int_checksum import (
    COMMON_MODELS,
    KIND_FLETCHER,
    KIND_WEIGHTED_SUM,
    IntModel,
    eval_model,
    sums,
    weighted_raw,
)

# Cost bounds.  Every loop below is bounded: recovery runs inside fuzz_one()
# and a prior unbounded GCD path already cost this project a 30+ second stall
# (see the
# comments at checksum_learner.py:35-63).

# Pairs the general search reduces over.  The GCD collapses to the exact
# modulus at ~12 pairs and to a small multiple of it well before that, so
# more pairs buy nothing and only cost bigint work.
INT_PAIRS_MAX = 32

# weighted_raw() with multiplier > 1 produces an integer with O(len(data))
# limbs, and the GCD reduction cost scales with that bit length.  multiplier
# == 1 is just a byte sum (one small int regardless of length) and is exempt
# — which matters, because that is the case that *needs* long pairs to wrap.
_MAX_WEIGHTED_DATA_BYTES = 64

# Multipliers tried for the weighted_sum family: 1 (plain sum) plus the small
# odd constants conventionally used as string/rolling-hash bases.
_MULTIPLIER_CANDIDATES = (1, 2, 3, 17, 31, 33, 37, 101, 131, 251, 257)

# When the GCD lands on m*N rather than N, m is small.  Enumerate cofactors up
# to this bound and take g // m as a modulus candidate.
_MAX_COFACTOR = 64

# Index triples examined by the consensus path when the single chain fails.
# Bounded like every other loop here: this runs inside fuzz_one().
_CONSENSUS_TRIALS = 24

# Hard ceiling on candidate moduli tested per configuration.
_MAX_MODULUS_CANDIDATES = 12

# A modulus below this is not a plausible checksum modulus and inflates the
# chance of two pairs agreeing by coincidence.
_MIN_MODULUS = 16

# Checksums wider than this are not what this path is for.
_MAX_MODULUS = 1 << 64

# Verification threshold.  The handover mandates >= 2 distinct pairs as a hard
# floor; 4 is stricter and costs nothing when pairs are plentiful, and the
# adversarial suite measures the false-positive rate at both.
MIN_VERIFY_MATCHES = 4
MIN_VERIFY_FLOOR = 2

# (out_bits, word_bytes, big_endian) configurations swept for the fletcher
# family when the modulus is unknown.
_FLETCHER_CONFIGS = (
    (32, 1, False),
    (16, 1, False),
    (32, 2, False),
    (32, 2, True),
)

Pair = tuple[bytes, int]


def _field_width(modulus: int) -> int:
    """Smallest supported checksum field width (bits) that holds *modulus*.

    Restricted to the widths ``int_checksum.model_from_dict`` accepts, so a
    recovered model always survives a ``state.json`` round trip.
    """
    needed = (modulus - 1).bit_length()
    for width in (16, 32, 64):
        if needed <= width:
            return width
    return 64


def verify_model(model: IntModel, pairs: list[Pair], min_matches: int = MIN_VERIFY_MATCHES) -> bool:
    """True when *model* reproduces the checksums of enough distinct pairs.

    Mirrors ``ChecksumLearner._verify``'s contract: a candidate is only ever
    activated when it reproduces real observed pairs.  Two extra guards on top
    of the raw count:

    - the matched pairs must carry at least two *distinct* checksum values, so
      a degenerate model that maps everything to one constant cannot pass by
      matching a run of identical observations;
    - ``min_matches`` is clamped to the number of pairs available but never
      below :data:`MIN_VERIFY_FLOOR`.
    """
    required = max(MIN_VERIFY_FLOOR, min(min_matches, len(pairs)))
    matched: set[int] = set()
    matches = 0
    for data, checksum in pairs:
        try:
            if eval_model(model, data) == checksum:
                matches += 1
                matched.add(checksum)
        except (ValueError, OverflowError):
            return False
    return matches >= required and len(matched) >= 2


def _score_candidate(residual: list[int], modulus: int) -> int:
    """Size of the largest congruence class of *residual* mod *modulus*.

    Every good pair satisfies ``r_i = N * q_i - b``, hence ``r_i = -b
    (mod N)`` — so the true modulus collapses *all* good residuals into one
    class.  A corrupt pair joins that class only by chance (probability
    ``1/N``), which makes class size a sharp, cheap outlier-tolerant score.
    """
    classes: dict[int, int] = {}
    best = 0
    for value in residual:
        key = value % modulus
        count = classes[key] = classes.get(key, 0) + 1
        if count > best:
            best = count
    return best


def _consensus_candidates(residual: list[int], min_modulus: int, max_modulus: int) -> list[int]:
    """Outlier-tolerant modulus candidates, for pair sets containing garbage.

    The single-chain GCD in :func:`_modulus_candidates` is all-or-nothing: one
    corrupt pair drags the GCD to 1 and recovery fails outright.  That is not
    a hypothetical — ``ChecksumLearner.extract_cmplog_pairs`` is explicitly
    heuristic ("scans for 4-byte operands that appear near data"), so bad
    pairs are expected in the pool.

    Instead of one chain over all residuals, take GCDs over small index
    triples (most of which avoid any given bad pair) and score each candidate
    by congruence-class size.  Measured on 200 trials per configuration: with
    1-3 corrupt pairs out of 12-24, the single chain recovers the modulus
    0/200 times and this recovers it 199-200/200, with no regression on clean
    pair sets.

    Triples are chosen by a fixed deterministic pattern, not sampled, so
    recovery is reproducible across runs — the same reason the rest of the
    fuzzer mixes seeds deterministically.
    """
    m = len(residual)
    if m < 4:
        return []
    candidates: set[int] = set()
    stride = max(1, m // 3)
    trials = 0
    for i in range(m):
        if trials >= _CONSENSUS_TRIALS:
            break
        for j, k in ((i + 1, i + 2), (i + stride, i + 2 * stride)):
            if k >= m:
                continue
            trials += 1
            g = gcd(abs(residual[i] - residual[j]), abs(residual[j] - residual[k]))
            if not g:
                continue
            for cofactor in range(1, _MAX_COFACTOR + 1):
                if g % cofactor == 0 and min_modulus <= g // cofactor <= max_modulus:
                    candidates.add(g // cofactor)

    required = max(MIN_VERIFY_MATCHES, (m + 1) // 2)
    scored = [(_score_candidate(residual, c), c) for c in candidates]
    ranked = sorted((s for s in scored if s[0] >= required), reverse=True)
    return [c for _score, c in ranked[:_MAX_MODULUS_CANDIDATES]]


def _modulus_candidates(residual: list[int], min_modulus: int, max_modulus: int) -> list[int]:
    """Candidate moduli from the GCD of pairwise differences of *residual*.

    *residual* holds ``R_i - c_i`` per pair; each pairwise difference is an
    exact multiple of the true modulus, so the GCD is ``N * gcd(q_i - q_0)``.
    Returns the GCD and its small-cofactor quotients, largest first.
    """
    if len(residual) < 2:
        return []
    base = residual[0]
    g = 0
    for value in residual[1:]:
        g = gcd(g, abs(value - base))
        if g == 1:
            break  # chain poisoned (wrong hypothesis, or a corrupt pair)
    if g <= 1:
        # Either the hypothesis is wrong or one pair is garbage; the chain
        # cannot tell those apart, so let consensus arbitrate.
        return _consensus_candidates(residual, min_modulus, max_modulus)
    candidates: list[int] = []
    for cofactor in range(1, _MAX_COFACTOR + 1):
        if g % cofactor:
            continue
        candidate = g // cofactor
        if min_modulus <= candidate <= max_modulus:
            candidates.append(candidate)
        if len(candidates) >= _MAX_MODULUS_CANDIDATES:
            break
    if not candidates:
        return _consensus_candidates(residual, min_modulus, max_modulus)
    return candidates


def _modal_value(values) -> int:
    """Most common value in *values*.

    Used to pin down ``init`` offsets: for every honest pair the offset is the
    same constant, so the mode is it — whereas anchoring on one arbitrary pair
    silently adopts that pair's corruption.
    """
    counts: dict[int, int] = {}
    best = (0, 0)
    for value in values:
        count = counts[value] = counts.get(value, 0) + 1
        if count > best[0]:
            best = (count, value)
    return best[1]


def _recover_weighted(pairs: list[Pair], min_matches: int) -> IntModel | None:
    """Recover a ``sum(data[j] * k**j) + init mod N`` model."""
    for multiplier in _MULTIPLIER_CANDIDATES:
        if multiplier == 1:
            usable = pairs
        else:
            usable = [p for p in pairs if len(p[0]) <= _MAX_WEIGHTED_DATA_BYTES]
        if len(usable) < 2:
            continue
        required = max(MIN_VERIFY_FLOOR, min(min_matches, len(usable)))
        raws = [weighted_raw(data, multiplier) for data, _ in usable]
        residual = [raw - checksum for raw, (_, checksum) in zip(raws, usable, strict=True)]
        for modulus in _modulus_candidates(residual, _MIN_MODULUS, _MAX_MODULUS):
            # Outlier-tolerant version of "the modulus exceeds every observed
            # checksum": require only that enough pairs fit under it.
            if sum(c < modulus for _, c in usable) < required:
                continue
            init = _modal_value(
                (checksum - raw) % modulus for raw, (_, checksum) in zip(raws, usable, strict=True)
            )
            model = IntModel(
                KIND_WEIGHTED_SUM,
                modulus,
                multiplier=multiplier,
                init_a=init,
                out_bits=_field_width(modulus),
            )
            if verify_model(model, pairs, min_matches):
                return model
    return None


def _recover_fletcher(pairs: list[Pair], min_matches: int) -> IntModel | None:
    """Recover a two-running-sum (Adler/Fletcher) model with unknown modulus.

    With ``a_n = init_a + S1`` and ``b_n = init_b + n * init_a + S2`` (mod N),
    the low half of the packed checksum gives ``a_n`` directly, so the same
    GCD-of-differences argument recovers ``N`` from the ``a`` side alone.
    ``init_a`` and ``init_b`` then fall out of a single pair by substitution.
    """
    for out_bits, word_bytes, big_endian in _FLETCHER_CONFIGS:
        half = out_bits // 2
        mask = (1 << half) - 1
        usable = [p for p in pairs if p[1] >> out_bits == 0]
        if len(usable) < 2:
            continue
        decomposed = [
            (*sums(data, word_bytes, big_endian), checksum & mask, checksum >> half)
            for data, checksum in usable
        ]
        residual = [s1 - a for _n, s1, _s2, a, _b in decomposed]
        halves = [(a, b) for *_x, a, b in decomposed]
        # A checksum is a residue, so the modulus must exceed every observed
        # half -- but only for *honest* pairs. Taking max() here lets one
        # corrupt pair inflate the bound past the true modulus and filter it
        # out entirely (caught by test_recovers_despite_corrupt_pairs). Only
        # require that enough pairs fit, matching the verification threshold.
        required = max(MIN_VERIFY_FLOOR, min(min_matches, len(usable)))
        for modulus in _modulus_candidates(residual, _MIN_MODULUS, 1 << half):
            if sum(a < modulus and b < modulus for a, b in halves) < required:
                continue
            # Anchoring the inits on pair 0 breaks if pair 0 is the corrupt
            # one, so take the value the majority of pairs agree on: for
            # honest pairs (a - S1) mod N is the same constant.
            init_a = _modal_value((a - s1) % modulus for _n, s1, _s2, a, _b in decomposed)
            init_b = _modal_value(
                (b - n * init_a - s2) % modulus for n, _s1, s2, _a, b in decomposed
            )
            model = IntModel(
                KIND_FLETCHER,
                modulus,
                init_a=init_a,
                init_b=init_b,
                word_bytes=word_bytes,
                out_bits=out_bits,
                big_endian=big_endian,
            )
            if verify_model(model, pairs, min_matches):
                return model
    return None


def recover_int_model(
    pairs: list[Pair],
    min_matches: int = MIN_VERIFY_MATCHES,
    max_pairs: int = INT_PAIRS_MAX,
) -> IntModel | None:
    """Recover an integer-modulus checksum model from observed pairs.

    Tries, in order: the well-known models (Adler-32, Fletcher-16/32, plain
    sums) as a cheap pre-check, then general modulus recovery for the fletcher
    family, then for the weighted-sum family.

    Args:
        pairs: Observed ``(data, checksum)`` pairs.  Should already be
            deduplicated by the caller.
        min_matches: Distinct pairs a candidate must reproduce before it is
            returned.  Clamped to at least :data:`MIN_VERIFY_FLOOR`.
        max_pairs: Cap on pairs fed to the general search.

    Returns:
        A verified :class:`~fuzzer_tool.core.int_checksum.IntModel`, or
        ``None``.  Never returns an unverified model.
    """
    if len(pairs) < 2:
        return None

    # Empty-data pairs carry no modulus information (every raw sum is 0) and
    # only dilute the GCD.  gzip extraction produces these by design.
    usable = [p for p in pairs if p[0]]
    if len(usable) < 2:
        return None

    for model in COMMON_MODELS:
        if verify_model(model, usable, min_matches):
            return model

    # Prefer the longest pairs: the modulus is only visible once the raw sum
    # wraps it, so long pairs carry the signal short ones cannot.
    ranked = sorted(usable, key=lambda p: len(p[0]), reverse=True)[:max_pairs]

    model = _recover_fletcher(ranked, min_matches)
    if model is not None:
        return model
    return _recover_weighted(ranked, min_matches)
