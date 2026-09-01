"""Quantum-inspired Evolutionary Algorithm (QEA) encoding.

Implements an alternative individual representation where each bit is
represented as a qubit-like probability amplitude pair (α, β) with
α² + β² = 1, meaning "this bit is P(0)=α² likely to be 0" rather than
a committed value. Amplitudes are incrementally updated via rotation
gates after each evaluation.

Amplitudes stored as ``numpy.ndarray[numpy.float64]`` for compact memory
(2× less than ``list[float]``) and vectorized operations (19× faster
rotation gate). Falls back gracefully with a clear error if numpy is
unavailable.

This is structurally different from the committed-value GA (core/ga.py):
- GA: crossover/mutation directly manipulates committed bytes
- CEM: refits a parametric distribution over the whole population in batches
- QEA: maintains continuous uncertainty per bit, updated incrementally
  after every evaluation, preserving diversity longer

Reference: Han & Kim, "Quantum-inspired evolutionary algorithm for a
class of combinatorial optimization", IEEE Trans. Evol. Comp. 2002.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fuzzer_tool.core.mutations import crossover

log = logging.getLogger(__name__)

# Amplitude arrays store 8 float64 values per input byte (unpackbits ->
# 8 bits/byte, one 8-byte float per bit): a 64x memory amplification with
# no cap elsewhere in this module. An oversized input reaching QEA (e.g. a
# mutator bug that grows a seed past the fuzzer's max_len — see
# operators.py _op_fuse_this) turns directly into a multi-GB allocation:
# an 186.9 MB seed becomes ~12 GB of float64, and OOMs the fuzzer even
# though the input itself fit comfortably in memory. QEA's amplitude
# representation only needs a bounded prefix to drive rotation-gate
# feedback; cap what it converts rather than trusting every caller
# upstream to already be bounded.
QEA_MAX_INPUT_BYTES = 65536  # amplitude array capped at 65536*8*8B = 4 MiB


def _qea_cap(data: bytes) -> bytes:
    """Truncate to the prefix QEA will actually represent as amplitudes.

    Applied consistently everywhere data enters the QEA population so an
    individual's amplitudes and its best_collapsed bytes never disagree
    in length (num_bytes == len(amplitudes) // 8 relies on this).
    """
    return data[:QEA_MAX_INPUT_BYTES] if len(data) > QEA_MAX_INPUT_BYTES else data


if TYPE_CHECKING:
    from fuzzer_tool.core.edge_tracker import EdgeTracker
    from fuzzer_tool.core.ga import FitnessFunction, Speciation


# ── Constants ──────────────────────────────────────────────────────────

# Clamp limits for amplitude α — ensures minimum uncertainty and
# prevents complete convergence/stagnation.
ALPHA_MIN = 0.01
ALPHA_MAX = 0.99

# Default uniform α: P(0) = 0.5, P(1) = 0.5
ALPHA_UNIFORM = 1.0 / math.sqrt(2)  # 0.7071...

# Default strong bias α: P(matching bit) ≈ 0.81
ALPHA_STRONG = 0.9

# Bits per byte
BITS_PER_BYTE = 8


# ── Helper: bytes ↔ bit list conversions (used externally by tests) ────


def _bytes_to_bits(data: bytes) -> list[int]:
    """Decompose bytes into a list of bits (MSB first per byte)."""
    bits: list[int] = []
    for byte in data:
        for shift in range(BITS_PER_BYTE - 1, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    """Assemble a list of bits back into bytes (MSB first per byte)."""
    result = bytearray()
    for i in range(0, len(bits), BITS_PER_BYTE):
        chunk = bits[i : i + BITS_PER_BYTE]
        byte = 0
        for bit in chunk:
            byte = (byte << 1) | (bit & 1)
        if len(chunk) < BITS_PER_BYTE:
            byte <<= BITS_PER_BYTE - len(chunk)
        result.append(byte)
    return bytes(result)


# ── Individual ─────────────────────────────────────────────────────────


@dataclass
class QEAIndividual:
    """A QEA individual: qubit-amplitude representation per bit.

    Each bit position i has an amplitude α_i where P(bit=0) = α_i² and
    P(bit=1) = 1 - α_i². The individual only collapses to concrete bytes
    at evaluation time via sampling from these amplitudes.

    Attributes:
        amplitudes: Per-bit α values as ndarray, length = 8 * num_bytes.
        fitness: Current fitness score.
        edge_count: Number of unique edges covered by collapsed output.
        species_id: Species assignment for speciation.
        generation: Generation this individual was created in.
        best_collapsed: Best concrete byte string found so far.
        best_fitness: Fitness of the best collapsed output.
        seed_key: SHA-256 hash prefix of best_collapsed.
        crash: Whether this individual triggered a crash.
        coupling: Optional per-byte intra-byte coupling tensor, shape
            (num_bytes, 8, 8). None (the default) means this individual
            carries no learned bit-pair correlation and behaves exactly
            as it did before this field existed -- collapse()/rotation_gate()
            never look at it. Only populated when QEALifecycle is
            constructed with use_correlation=True; see
            collapse_correlated()/update_couplings() below.
    """

    amplitudes: np.ndarray | list[float]
    fitness: float = 0.0
    edge_count: int = 0
    novelty_score: float = 0.0
    diversity_score: float = 0.0
    freshness_score: float = 0.0
    mutation_potential: float = 0.0
    species_id: int = -1
    generation: int = 0
    best_collapsed: bytes = b""
    best_fitness: float = 0.0
    seed_key: str = ""
    crash: bool = False
    coupling: np.ndarray | list | None = None

    def __post_init__(self):
        if not self.seed_key and self.best_collapsed:
            self.seed_key = hashlib.sha256(self.best_collapsed).hexdigest()[:16]
        # Ensure amplitudes are always ndarray
        if isinstance(self.amplitudes, list):
            self.amplitudes = np.array(self.amplitudes, dtype=np.float64)
        if self.coupling is not None and not isinstance(self.coupling, np.ndarray):
            self.coupling = np.array(self.coupling, dtype=np.float64)

    @property
    def num_bytes(self) -> int:
        """Number of bytes implied by the amplitude length."""
        return max(1, len(self.amplitudes) // BITS_PER_BYTE)

    def to_dict(self) -> dict:
        return {
            # Round amplitudes (probabilities) to 6dp: full f64 precision has
            # no signal and roughly triples the on-disk JSON (200 populations x
            # len(amplitudes) entries, saved every shutdown).
            "amplitudes": self.amplitudes.round(6).tolist(),
            "fitness": self.fitness,
            "edge_count": self.edge_count,
            "novelty_score": self.novelty_score,
            "diversity_score": self.diversity_score,
            "freshness_score": self.freshness_score,
            "mutation_potential": self.mutation_potential,
            "species_id": self.species_id,
            "generation": self.generation,
            "best_collapsed": self.best_collapsed.hex(),
            "best_fitness": self.best_fitness,
            "seed_key": self.seed_key,
            "crash": self.crash,
            # Same rounding rationale as amplitudes above. Omitted (null)
            # for the common case (use_correlation=False) rather than
            # serializing a same-shaped zero tensor for every individual.
            "coupling": self.coupling.round(6).tolist() if self.coupling is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QEAIndividual:
        bc = d.get("best_collapsed", "")
        coupling = d.get("coupling")
        return cls(
            amplitudes=np.array(d.get("amplitudes", []), dtype=np.float64),
            fitness=d.get("fitness", 0.0),
            edge_count=d.get("edge_count", 0),
            novelty_score=d.get("novelty_score", 0.0),
            diversity_score=d.get("diversity_score", 0.0),
            freshness_score=d.get("freshness_score", 0.0),
            mutation_potential=d.get("mutation_potential", 0.0),
            species_id=d.get("species_id", -1),
            generation=d.get("generation", 0),
            best_collapsed=bytes.fromhex(bc) if bc else b"",
            best_fitness=d.get("best_fitness", 0.0),
            seed_key=d.get("seed_key", ""),
            crash=d.get("crash", False),
            coupling=np.array(coupling, dtype=np.float64) if coupling is not None else None,
        )


# ── Amplitude vector creation ──────────────────────────────────────────


def _bias_amplitudes_from(data: bytes, *, strong_prob: float = 0.9) -> np.ndarray:
    """Create amplitude array biased toward the given byte values.

    For each bit in ``data``, the amplitude is chosen so the bit collapses
    back to its own value with probability ``strong_prob**2`` — symmetrically
    for zero bits and one bits alike.

    Since P(bit=0) = α², a zero bit takes α = *strong_prob* directly, while a
    one bit needs P(bit=1) = 1 - α² = strong_prob², i.e.
    α = sqrt(1 - strong_prob²). Complementing the *amplitude* instead
    (α = 1 - strong_prob) is wrong: at the 0.9 default it yields
    P(stays 1) = 0.99 against P(stays 0) = 0.81, which biases every collapse
    toward setting bits and compounds through each breed/re-bias cycle.

    Args:
        data: Template byte string to bias toward.
        strong_prob: Amplitude for zero bits; the retention probability for
            every bit is its square (default 0.9 → P(bit keeps its value)
            = 0.81 regardless of whether that value is 0 or 1).

    Returns:
        ndarray of α values, length = 8 * len(data), dtype=float64.
    """
    n_bits = len(data) * 8
    amps = np.full(n_bits, strong_prob, dtype=np.float64)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    # For each 1-bit: P(bit=1) = 1 - α² must equal strong_prob².
    amps[bits == 1] = math.sqrt(max(0.0, 1.0 - strong_prob * strong_prob))
    return amps


def _uniform_amplitudes(n_bits: int, *, alpha: float = ALPHA_UNIFORM) -> np.ndarray:
    """Create uniform amplitude array (all α = 0.7071... = uniform uncertainty).

    Args:
        n_bits: Number of amplitude values to create.
        alpha: Amplitude value for all positions (default ALPHA_UNIFORM).

    Returns:
        ndarray of α values, dtype=float64.
    """
    return np.full(n_bits, alpha, dtype=np.float64)


# ── Collapse: amplitudes → concrete bytes ─────────────────────────────


def collapse(amplitudes: np.ndarray) -> bytes:
    """Sample concrete bytes from qubit amplitudes.

    For each bit position i: bit = 0 with probability α_i², else 1.

    Vectorized: uses ``np.random.random`` and ``np.packbits``.

    Args:
        amplitudes: α values for each bit (ndarray or list, length multiple of 8).

    Returns:
        Collapsed concrete byte string.
    """
    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
    # P(bit=0) = α²: random < α² → bit=0, else bit=1
    bits = (np.random.random(len(amplitudes)) >= amplitudes * amplitudes).astype(np.uint8)
    return bytes(np.packbits(bits).tobytes())


# ── Rotation gate ──────────────────────────────────────────────────────


def rotation_gate(
    amplitudes: np.ndarray,
    collapsed: bytes,
    *,
    improved: bool,
    delta: float = 0.05,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> np.ndarray:
    """Apply QEA rotation gate to update amplitudes based on fitness feedback.

    This is a true rotation on the (α, β) unit circle, matching Han & Kim's
    formulation: α ← cos(Δθ)·α ∓ sin(Δθ)·β, with β = √(1-α²) implicit (see
    module docstring — only the real, non-negative quarter-circle is
    represented, so β is always recoverable from α). ``delta`` is the
    rotation angle Δθ in radians, not a linear step in amplitude space.

    Rotating (α, β) as a pair rather than walking α alone reproduces the
    literature's built-in deceleration near the poles: dα/dθ → 0 as
    θ → 0 (α → 1), so a bit that has become nearly certain resists being
    knocked back into contention by a single ``improved`` flip, the way an
    ``α ± delta`` walk would. Values are still clamped to
    [alpha_min, alpha_max] afterward — that floor/ceiling isn't optional
    even with the trig deceleration, since it's what keeps a bit from
    fully collapsing to certainty (see ALPHA_MIN/ALPHA_MAX docstring) and
    what stops the rotation from walking α past the edge of the
    represented quarter-circle when β is small.

    Vectorized: trig identities applied over the whole array via numpy
    instead of a per-bit Python loop.

    - bit=0, improved=True  → α increases (rotate toward |0⟩, Δθ < 0)
    - bit=0, improved=False → α decreases (rotate toward |1⟩, Δθ > 0)
    - bit=1, improved=True  → α decreases (rotate toward |1⟩, Δθ > 0)
    - bit=1, improved=False → α increases (rotate toward |0⟩, Δθ < 0)

    Args:
        amplitudes: Current α values to update (in-place + return).
        collapsed: Concrete bytes that the amplitudes produced.
        improved: Whether the collapsed outcome was beneficial.
        delta: Rotation angle Δθ in radians (default 0.05 ≈ 2.9°).
        alpha_min: Minimum amplitude clamp (default 0.01).
        alpha_max: Maximum amplitude clamp (default 0.99).

    Returns:
        Updated amplitudes (same ndarray object, modified in place).
    """
    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
    # Unpack collapsed bytes to bit array, pad/truncate to match length
    n_bits = len(amplitudes)
    bits = np.unpackbits(np.frombuffer(collapsed, dtype=np.uint8))
    bits = np.pad(bits, (0, n_bits - len(bits))) if len(bits) < n_bits else bits[:n_bits]

    # bit=0 XOR improved=False → increase α (rotate toward |0⟩, Δθ < 0);
    # the complementary set decreases α (Δθ > 0). Same truth table as the
    # original linear version, just implemented as a rotation direction.
    increase_alpha = (bits == 0) if improved else (bits == 1)

    beta = np.sqrt(np.maximum(0.0, 1.0 - amplitudes * amplitudes))
    cos_d = math.cos(delta)
    sin_d = math.sin(delta)

    # cos(-Δθ)=cos(Δθ) and sin(-Δθ)=-sin(Δθ), so the two rotation
    # directions only differ in the sign in front of the β term.
    rotated_toward_zero = amplitudes * cos_d + beta * sin_d  # Δθ < 0, α increases
    rotated_toward_one = amplitudes * cos_d - beta * sin_d  # Δθ > 0, α decreases

    new_alpha = np.where(increase_alpha, rotated_toward_zero, rotated_toward_one)
    amplitudes[:] = np.clip(new_alpha, alpha_min, alpha_max)
    return amplitudes


# ── Amplitude mutation ─────────────────────────────────────────────────


def mutate_amplitudes(
    amplitudes: np.ndarray,
    *,
    prob: float = 0.02,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> np.ndarray:
    """Randomly perturb amplitudes to maintain diversity.

    Vectorized: uses a boolean mask over the array instead of a per-bit
    Python loop.

    Each bit's α is reset to a random uniform value with probability
    ``prob``. This is QEA's equivalent of GA mutation, preventing
    amplitude stagnation when all values converge to extremes.

    Args:
        amplitudes: Amplitude array to mutate (in-place + return).
        prob: Per-bit mutation probability (default 0.02).
        alpha_min: Minimum amplitude after reset.
        alpha_max: Maximum amplitude after reset.

    Returns:
        Mutated amplitudes (same ndarray, modified in place).
    """
    if not isinstance(amplitudes, np.ndarray):
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
    mask = np.random.random(len(amplitudes)) < prob
    n_mutate = mask.sum()
    if n_mutate > 0:
        amplitudes[mask] = np.random.uniform(alpha_min, alpha_max, size=n_mutate)
    return amplitudes


# ── Intra-byte correlation (partial entanglement) ───────────────────────
#
# The product-state representation above (one independent α per bit) is
# what makes QEA O(n) instead of O(2^n), but it structurally cannot learn
# that e.g. bit 0 and bit 1 of a magic byte tend to be right or wrong
# together — every α update in rotation_gate() depends only on that bit's
# own collapsed value. See docs/handover/handover_qea_hilbert_space_analysis_2026-08-31.md
# for the full analysis this section implements.
#
# This adds a small (8x8) symmetric coupling matrix per byte: a classical
# pairwise (Ising-like) correlation, not a real Hilbert-space entangled
# state -- there is still no joint amplitude vector over 2^8 basis states,
# just a bias term coupling bit i's conditional distribution to bit j's
# current value. Cost is O(n) amplitudes + O(n) coupling entries (fixed
# 64 floats per byte), so it stays linear in n, unlike a true n-qubit
# state. Scoped to within a byte deliberately: cross-byte correlation
# (e.g. a length field several bytes away) is exactly what the structural/
# grammar mutators already handle via field_constraints.py, and extending
# the coupling matrix across byte boundaries would reintroduce the O(n^2)
# (or worse) cost this representation exists to avoid.
#
# This is opt-in (QEALifecycle(use_correlation=True)) and additive: with
# it off, QEAIndividual.coupling stays None and every function below is
# unused, so existing behavior and existing tests are unaffected.

COUPLING_MAX_DEFAULT = 2.0  # clip magnitude for a single J_ij entry
CORRELATION_SWEEPS_DEFAULT = 3  # Gibbs sweeps per collapse_correlated() call


# ── Algorithmic cooling (opt-in rotation-angle decay) ───────────────────
#
# rotation_gate()'s cos(Δθ)/sin(Δθ) rotation already decelerates *per bit*
# as that bit's own amplitude approaches certainty (dα/dθ -> 0 near the
# poles) -- see the Aug 31 Hilbert-space handover, section 4. That is a
# local effect with no memory of generation count. "Algorithmic cooling"
# here is a separate, *global* schedule that shrinks the base angle Δθ
# itself as generations pass, on top of that per-bit deceleration --
# large steps while the population is still unconverged, smaller steps
# as it settles, the way a simulated-annealing temperature schedule
# would.
#
# This fuzzer's QEA run has no fixed generation budget (unlike the
# combinatorial-optimization benchmarks the technique is usually applied
# to), so decaying Δθ monotonically across the whole run and never
# recovering would eventually flatten it to the floor and leave
# mutate_amplitudes()'s fixed 2%-per-bit random reset as the only
# remaining source of adaptation -- working against new coverage that
# shows up late in a long-running session. Anchored instead to
# elite_reset_every: Δθ decays within each reset cycle and snaps back to
# its base value at every elite reset, so cooling and the existing
# incumbent-anchoring fix cooperate (both keyed to the same cycle
# boundary) instead of one undermining the other. With
# elite_reset_every=0 (resets disabled) there is no cycle boundary to
# anchor to, so cooling falls back to decaying against the raw
# generation count -- callers who want indefinite cooling without elite
# resets get that, but should pick a floor they're comfortable settling
# at permanently.
COOLING_DECAY_DEFAULT = 0.98  # per-generation multiplicative decay
COOLING_MIN_ANGLE_DEFAULT = 0.005  # floor Δθ never decays past (radians)


def _zero_coupling(num_bytes: int) -> np.ndarray:
    """An all-zero (uncoupled) coupling tensor for *num_bytes* bytes.

    Shape (num_bytes, 8, 8). Zero coupling reduces collapse_correlated()
    to independent per-bit sampling, i.e. identical to collapse() -- this
    is the correct initial state for a freshly created individual that
    hasn't learned any pairwise structure yet.
    """
    return np.zeros((max(1, num_bytes), 8, 8), dtype=np.float64)


def _alpha_to_field(alpha: np.ndarray) -> np.ndarray:
    """Convert per-bit amplitudes to Ising-style local fields.

    P(bit=0) = α², P(bit=1) = 1 - α². Define state s = +1 for bit=0,
    s = -1 for bit=1; the local field is the log-odds of s=+1:
    h = ln(P(bit=0) / P(bit=1)) = ln(α² / (1 - α²)).

    Clipped away from the α∈{0,1} boundary before the division/log so a
    saturated amplitude (already clamped to [ALPHA_MIN, ALPHA_MAX]
    elsewhere, but this function is also called on raw amplitude arrays)
    can't produce inf/nan.
    """
    a2 = np.clip(np.asarray(alpha, dtype=np.float64) ** 2, 1e-6, 1 - 1e-6)
    return np.log(a2 / (1 - a2))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid, clipped to avoid overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def collapse_correlated(
    amplitudes: np.ndarray,
    coupling: np.ndarray,
    *,
    n_sweeps: int = CORRELATION_SWEEPS_DEFAULT,
) -> bytes:
    """Sample concrete bytes jointly within each byte via Gibbs sampling.

    Unlike collapse(), which draws every bit independently from its own
    α, this draws each byte's 8 bits from an Ising-like model combining
    the per-bit field (from ``amplitudes``) with the byte's pairwise
    coupling matrix, so correlated bit pairs the individual has learned
    (via update_couplings()) bias each other's draw.

    With an all-zero coupling matrix this is equivalent to collapse() up
    to sampling order (each sweep step reduces to independent per-bit
    sampling from the field alone), so a freshly created individual
    behaves identically to the uncorrelated representation until its
    coupling matrix accumulates signal.

    Args:
        amplitudes: α values, length = 8 * num_bytes.
        coupling: Symmetric per-byte coupling tensor, shape
            (num_bytes, 8, 8), zero diagonal.
        n_sweeps: Number of full Gibbs sweeps (one update per bit each)
            to run before reading out the sample. More sweeps mix the
            joint distribution closer to its stationary point; the
            default of 3 is a cheap approximation, not an exact sample.

    Returns:
        Collapsed concrete byte string.
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    n_bits = len(amplitudes)
    if n_bits == 0:
        return b""
    num_bytes = n_bits // BITS_PER_BYTE
    coupling = np.asarray(coupling, dtype=np.float64)

    fields = _alpha_to_field(amplitudes).reshape(num_bytes, BITS_PER_BYTE)

    # Initialize state independently from the field alone (sweep 0 of a
    # zero-coupling model), then refine with coupled Gibbs sweeps.
    #
    # No factor-of-2 here: _alpha_to_field() is defined so that
    # sigmoid(field) == α² directly (that's the whole point of using
    # ln(α²/(1-α²)) rather than the textbook Ising h with an implied
    # H = -h·s/2). The standard Ising "P(s=+1|rest) = sigmoid(2·(h+ΣJs))"
    # doubling is a convention tied to a Hamiltonian written as
    # -h·s - ΣJ·s·s'; that's not the parameterization in use here, and
    # applying it anyway silently shifts every marginal away from α²
    # even at zero coupling (P(bit=0)=0.9² doubled to ~0.94 instead of
    # 0.81) -- caught by test_zero_coupling_matches_marginals.
    p_plus0 = _sigmoid(fields)
    state = np.where(np.random.random((num_bytes, BITS_PER_BYTE)) < p_plus0, 1, -1)

    for _ in range(max(0, n_sweeps)):
        for bit_idx in range(BITS_PER_BYTE):
            # sum_j J[bit_idx, j] * s_j, diagonal is zero so no self-term
            # to subtract, but coupling isn't guaranteed zero-diagonal by
            # callers -- guard explicitly rather than trust that invariant.
            j_row = coupling[:, bit_idx, :].copy()
            j_row[:, bit_idx] = 0.0
            local_field = fields[:, bit_idx] + np.einsum("bj,bj->b", j_row, state)
            p_plus = _sigmoid(local_field)
            draw = np.random.random(num_bytes) < p_plus
            state[:, bit_idx] = np.where(draw, 1, -1)

    bits = (state == -1).astype(np.uint8).reshape(-1)  # s=+1 -> bit 0, s=-1 -> bit 1
    return bytes(np.packbits(bits).tobytes())


def update_couplings(
    coupling: np.ndarray,
    collapsed: bytes,
    *,
    improved: bool,
    delta: float = 0.02,
    coupling_max: float = COUPLING_MAX_DEFAULT,
) -> np.ndarray:
    """Hebbian-style update of the per-byte coupling matrix.

    For each byte, with state s_i = +1 (bit=0) or -1 (bit=1): when the
    collapsed bytes were an improvement, nudge every pair's coupling
    J_ij toward reinforcing the sign relationship the pair actually had
    (ΔJ_ij = +delta * s_i * s_j -- same-sign pairs get a more positive
    J, opposite-sign pairs get a more negative J, both of which raise
    the probability of repeating that same relationship next collapse).
    When it wasn't an improvement, the update is negated, pushing away
    from repeating that specific pairwise combination.

    Args:
        coupling: Coupling tensor to update, shape (num_bytes, 8, 8).
            Modified in place and returned.
        collapsed: Concrete bytes the coupling's individual produced.
        improved: Whether the collapsed outcome was beneficial.
        delta: Per-pair coupling learning rate.
        coupling_max: Clip magnitude for any single J_ij entry, keeping
            the local-field sum in collapse_correlated() from growing
            unbounded over many updates (mirrors ALPHA_MIN/MAX clamping
            amplitudes in rotation_gate()).

    Returns:
        Updated coupling tensor (same ndarray object, modified in place).
    """
    coupling = np.asarray(coupling, dtype=np.float64)
    num_bytes = coupling.shape[0]
    n_bits = num_bytes * BITS_PER_BYTE

    bits = np.unpackbits(np.frombuffer(collapsed, dtype=np.uint8))
    bits = np.pad(bits, (0, n_bits - len(bits))) if len(bits) < n_bits else bits[:n_bits]
    state = np.where(bits.reshape(num_bytes, BITS_PER_BYTE) == 0, 1, -1).astype(np.float64)

    sign = delta if improved else -delta
    outer = np.einsum("bi,bj->bij", state, state)  # s_i * s_j per byte
    idx = np.arange(BITS_PER_BYTE)
    outer[:, idx, idx] = 0.0  # never couple a bit to itself

    coupling += sign * outer
    coupling[:, idx, idx] = 0.0  # re-zero diagonal in case caller passed nonzero
    np.clip(coupling, -coupling_max, coupling_max, out=coupling)
    return coupling


# ── QEALifecycle ───────────────────────────────────────────────────────


class QEALifecycle:
    """QEA lifecycle controller for coverage-guided fuzzing.

    Mirrors the ``GALifecycle`` interface (``pick_seed()``,
    ``on_fuzz_result()``, ``add_to_population()``, ``save()``, ``load()``)
    but uses qubit-amplitude individual representation with rotation gate
    feedback instead of committed-value crossover/mutation.

    Each call to ``pick_seed()`` collapses a selected individual's
    amplitudes to concrete bytes. After the fuzzer evaluates the result,
    ``on_fuzz_result()`` applies the rotation gate to update amplitudes
    based on whether new coverage was found, then triggers generation
    boundaries at a fixed interval.
    """

    def __init__(
        self,
        pop_size: int = 200,
        elite_fraction: float = 0.1,
        generation_size: int = 500,
        rotation_angle: float = 0.05,
        # ^ radians (Δθ in rotation_gate's cos/sin rotation), not a linear
        # amplitude step -- see rotation_gate() docstring.
        mutation_prob: float = 0.02,
        init_alpha: float = ALPHA_UNIFORM,
        strong_bias: float = ALPHA_STRONG,
        tournament_size: int = 3,
        speciation_threshold: float = 0.3,
        elite_reset_every: int = 0,
        use_correlation: bool = False,
        # ^ Opt-in intra-byte coupling (see "Intra-byte correlation" section
        # above collapse_correlated()). Default off: existing behavior,
        # existing tests, and existing saved populations are unaffected.
        correlation_delta: float = 0.02,
        correlation_max: float = COUPLING_MAX_DEFAULT,
        correlation_sweeps: int = CORRELATION_SWEEPS_DEFAULT,
        use_cooling: bool = False,
        # ^ Opt-in global Δθ decay schedule (see "Algorithmic cooling"
        # section above). Default off: existing behavior, existing tests,
        # and existing saved populations are unaffected -- rotation_angle
        # stays constant across generations exactly as before.
        cooling_decay: float = COOLING_DECAY_DEFAULT,
        cooling_min_angle: float = COOLING_MIN_ANGLE_DEFAULT,
        fitness: FitnessFunction | None = None,
    ):
        self.pop_size = pop_size
        self.elite_fraction = elite_fraction
        self.generation_size = generation_size
        self.rotation_angle = rotation_angle
        self.mutation_prob = mutation_prob
        self.init_alpha = init_alpha
        self.strong_bias = strong_bias
        self.tournament_size = tournament_size
        self.speciation_threshold = speciation_threshold
        self.elite_reset_every = elite_reset_every
        self.use_correlation = use_correlation
        self.correlation_delta = correlation_delta
        self.correlation_max = correlation_max
        self.correlation_sweeps = correlation_sweeps
        self.use_cooling = use_cooling
        self.cooling_decay = cooling_decay
        self.cooling_min_angle = cooling_min_angle

        # Lazy import to avoid circular dependency at module level
        from fuzzer_tool.core.ga import FitnessFunction as _FF

        self.fitness_fn = fitness or _FF()

        self.population: list[QEAIndividual] = []
        self.generation = 0
        self.iterations_since_gen = 0
        self._speciation: Speciation | None = None

        # Tracking for rotation gate feedback: the last parent whose
        # collapsed amplitudes produced the seed for this iteration.
        self._last_parent: QEAIndividual | None = None
        self._last_collapsed: bytes = b""

        # Stats
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.species_count = 0

    # ── Initialization ─────────────────────────────────────────────

    def initialize(self, corpus: list[bytes], edge_tracker: EdgeTracker):
        """Seed population from existing corpus."""
        from fuzzer_tool.core.ga import Speciation

        self._speciation = Speciation(edge_tracker, self.speciation_threshold)

        for data in corpus[: self.pop_size]:
            # seed_key is keyed on the corpus's original hash (edge_tracker
            # tracks seeds by their real content), so hash before capping —
            # only the amplitude/best_collapsed representation is bounded.
            seed_key = hashlib.sha256(data).hexdigest()[:16]
            edge_set = edge_tracker.seed_edges.get(seed_key, set())
            capped = _qea_cap(data)
            ind = QEAIndividual(
                amplitudes=_bias_amplitudes_from(capped, strong_prob=self.strong_bias),
                edge_count=len(edge_set),
                generation=0,
                best_collapsed=capped,
                seed_key=seed_key,
                coupling=_zero_coupling(len(capped)) if self.use_correlation else None,
            )
            self.population.append(ind)

        self._evaluate_all(edge_tracker)

    # ── Algorithmic cooling ────────────────────────────────────────

    def _effective_rotation_angle(self) -> float:
        """Current Δθ, decayed by generation if ``use_cooling`` is set.

        With cooling off, returns ``self.rotation_angle`` unchanged --
        identical to every call site before this feature existed.

        With cooling on, decays multiplicatively by
        ``cooling_decay ** g`` where ``g`` is the generation count
        *within the current elite-reset cycle* (``generation %
        elite_reset_every``), floored at ``cooling_min_angle``. Using
        the in-cycle generation rather than the raw count means Δθ
        snaps back to ``rotation_angle`` at every elite reset instead
        of decaying once toward the floor for the life of the run --
        see the "Algorithmic cooling" section above collapse_correlated()
        for why an unrecovered decay is the wrong default for a
        fuzzing run with no fixed generation budget. With
        ``elite_reset_every == 0`` (resets disabled) there's no cycle
        boundary to anchor to, so this falls back to decaying against
        the raw generation count for the rest of the run.
        """
        if not self.use_cooling:
            return self.rotation_angle
        gen_in_cycle = (
            self.generation % self.elite_reset_every
            if self.elite_reset_every > 0
            else self.generation
        )
        decayed = self.rotation_angle * (self.cooling_decay**gen_in_cycle)
        return max(self.cooling_min_angle, decayed)

    # ── Core lifecycle ──────────────────────────────────────────────

    def pick_seed(self) -> bytes:
        """Return a collapsed seed for fuzz_one().

        Selects a parent via tournament, collapses its amplitudes to
        concrete bytes, and records the parent for rotation gate
        feedback on the next ``on_fuzz_result()`` call.

        Returns:
            Collapsed concrete byte string.
        """
        if not self.population:
            return b"\x00" * 64

        parent = self._tournament_select(self.population)
        if self.use_correlation and parent.coupling is not None:
            collapsed_data = collapse_correlated(
                parent.amplitudes, parent.coupling, n_sweeps=self.correlation_sweeps
            )
        else:
            collapsed_data = collapse(parent.amplitudes)

        # Store for rotation gate feedback
        self._last_parent = parent
        self._last_collapsed = collapsed_data

        return collapsed_data

    def on_fuzz_result(
        self,
        data: bytes,
        new_coverage: bool,
        edge_count: int,
        edge_tracker: EdgeTracker,
    ) -> QEAIndividual | None:
        """Called after each fuzz_one iteration.

        Applies rotation gate feedback and potentially adds a new
        individual to the population. Triggers generation evolution
        at the generation boundary.

        Args:
            data: The mutated input that was evaluated.
            new_coverage: Whether it discovered new coverage.
            edge_count: Number of edges covered by this input.
            edge_tracker: Shared edge coverage tracker.

        Returns:
            A new QEAIndividual to add to population if new coverage
            was found, otherwise None.
        """
        self.iterations_since_gen += 1

        # Apply rotation gate feedback using the last-selected parent, then
        # clear it. pick_seed() is only called when QEA wins seed
        # arbitration (one of several strategies), whereas this method runs
        # on every iteration — so without the clear, one individual absorbs
        # rotations driven by results from completely unrelated seeds. Since
        # most iterations find nothing, those spurious rotations are almost
        # all improved=False and drive its amplitudes onto the clamps,
        # destroying exactly the uncertainty QEA exists to preserve.
        if self._last_parent is not None and self._last_collapsed:
            rotation_gate(
                self._last_parent.amplitudes,
                self._last_collapsed,
                improved=new_coverage,
                delta=self._effective_rotation_angle(),
            )
            if self.use_correlation and self._last_parent.coupling is not None:
                update_couplings(
                    self._last_parent.coupling,
                    self._last_collapsed,
                    improved=new_coverage,
                    delta=self.correlation_delta,
                    coupling_max=self.correlation_max,
                )
            self._last_parent = None
            self._last_collapsed = b""

        new_ind: QEAIndividual | None = None
        if new_coverage:
            # Hash the real data for seed_key (matches edge_tracker's
            # keying), but cap what actually gets converted to amplitudes —
            # see QEA_MAX_INPUT_BYTES.
            seed_key = hashlib.sha256(data).hexdigest()[:16]
            capped = _qea_cap(data)
            new_ind = QEAIndividual(
                amplitudes=_bias_amplitudes_from(capped, strong_prob=self.strong_bias),
                edge_count=edge_count,
                generation=self.generation,
                best_collapsed=capped,
                seed_key=seed_key,
                coupling=_zero_coupling(len(capped)) if self.use_correlation else None,
            )
            # Score before returning: add_to_population() admits on fitness,
            # and an unscored individual carries fitness 0.0, which loses to
            # every evaluated member of a full population. Leaving this out
            # silently discards every coverage-finding seed.
            self._score(new_ind, edge_tracker)

        # Generation boundary. Checked regardless of new_coverage — returning
        # early on coverage would mean generations only ever advance during
        # unproductive stretches, starving evolution exactly when the fuzzer
        # is doing well.
        if self.iterations_since_gen >= self.generation_size:
            self._evolve(edge_tracker)
            self.generation += 1
            self.iterations_since_gen = 0

        return new_ind

    def add_to_population(self, ind: QEAIndividual):
        """Add an individual (e.g., new coverage seed) to the population.

        If the population is full, replaces the worst individual if the
        new one has higher fitness.
        """
        if len(self.population) >= self.pop_size:
            worst_idx = min(
                range(len(self.population)),
                key=lambda i: self.population[i].fitness,
            )
            if ind.fitness > self.population[worst_idx].fitness:
                self.population[worst_idx] = ind
        else:
            self.population.append(ind)

    # ── Evolution ───────────────────────────────────────────────────

    def _evolve(self, edge_tracker: EdgeTracker):
        """Run one QEA generation: evaluate, cull, breed."""
        # 1. Assign species
        if self._speciation:
            species_map = self._speciation.assign_species(self.population)
            self.species_count = len(species_map)

        # 2. Compute fitness for all
        self._evaluate_all(edge_tracker)

        # 3. Elitism: keep top fraction.
        #
        # Elites are carried forward verbatim and are also the individuals
        # tournament selection most often picks, so an individual that
        # scored well early can hold a population slot indefinitely and
        # keep supplying parents long after its region is exhausted. That
        # is the incumbent-anchoring shape the p-bit lattice study found
        # was worth breaking: its best arm differed from its worst only in
        # periodically discarding the retained best, and the discard was
        # the whole effect.
        #
        # Every `elite_reset_every` generations, breed the full population
        # instead. Nothing is actually lost -- coverage-finding inputs live
        # in the corpus on disk, not in this population -- so the elites
        # are a search anchor, not a record.
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        reset_gen = (
            self.elite_reset_every > 0
            and self.generation > 0
            and (self.generation + 1) % self.elite_reset_every == 0
        )
        if reset_gen:
            n_elite = 0
            elites: list[QEAIndividual] = []
            log.debug("QEA: elite reset at generation %d", self.generation)
        else:
            n_elite = max(1, int(len(self.population) * self.elite_fraction))
            elites = self.population[:n_elite]

        # 4. Breed new individuals
        n_breed = self.pop_size - n_elite
        offspring: list[QEAIndividual] = []
        for _ in range(n_breed):
            parent_a = self._tournament_select(self.population)
            parent_b = self._tournament_select(self.population)

            # Collapse both parents and crossover the committed bytes. Use
            # each parent's learned coupling when available so a parent's
            # correlated bit pairs (e.g. a magic-byte pattern it converged
            # on) show up in the bytes actually crossed over, not just in
            # its own future collapses.
            if self.use_correlation and parent_a.coupling is not None:
                bytes_a = collapse_correlated(
                    parent_a.amplitudes, parent_a.coupling, n_sweeps=self.correlation_sweeps
                )
            else:
                bytes_a = collapse(parent_a.amplitudes)
            if self.use_correlation and parent_b.coupling is not None:
                bytes_b = collapse_correlated(
                    parent_b.amplitudes, parent_b.coupling, n_sweeps=self.correlation_sweeps
                )
            else:
                bytes_b = collapse(parent_b.amplitudes)

            # Use two-point crossover (from mutations module)
            child_bytes = crossover(bytes_a, bytes_b)

            # Parents are already capped (see QEA_MAX_INPUT_BYTES), so
            # child_bytes can't exceed 2x that — cap anyway rather than
            # relying on that invariant holding as this code evolves.
            child_bytes = _qea_cap(child_bytes)

            # Create child with amplitudes biased toward the crossed bytes.
            # Coupling is deliberately NOT inherited from either parent --
            # crossover already scrambles which bytes came from which
            # parent, so a carried-over coupling matrix would describe
            # correlations for bytes that may no longer sit at those
            # positions. The child starts uncoupled and learns its own,
            # same as _bias_amplitudes_from() already discards the
            # parents' amplitude nuance in favor of a fresh bias.
            child = QEAIndividual(
                amplitudes=_bias_amplitudes_from(child_bytes, strong_prob=self.strong_bias),
                generation=self.generation + 1,
                best_collapsed=child_bytes,
                coupling=_zero_coupling(len(child_bytes)) if self.use_correlation else None,
            )

            # Apply amplitude mutation for diversity
            mutate_amplitudes(child.amplitudes, prob=self.mutation_prob)

            offspring.append(child)

        # 5. Replace population
        self.population = elites + offspring
        self._update_stats()

    # ── Selection ───────────────────────────────────────────────────

    def _tournament_select(self, pool: list[QEAIndividual]) -> QEAIndividual:
        """Tournament selection: pick best of k random individuals."""
        k = min(self.tournament_size, len(pool))
        candidates = random.sample(pool, k)
        return max(candidates, key=lambda i: i.fitness)

    # ── Fitness evaluation ──────────────────────────────────────────

    def _score(self, ind: QEAIndividual, edge_tracker: EdgeTracker) -> None:
        """Score one individual with the same inputs ``_evaluate_all`` uses.

        Factored out so a newly discovered individual is evaluated exactly
        the way population members are, rather than entering admission with
        a default fitness of 0.0.
        """
        total_edges = len(edge_tracker.cumulative_edges) if edge_tracker.cumulative_edges else 1
        w = edge_tracker.compute_wasserstein_weight(ind.seed_key)
        ind.diversity_score = (w - 0.5) / 1.5  # normalize [0.5, 2.0] -> [0, 1]
        self.fitness_fn.score(ind, total_edges, self.generation)

    def _evaluate_all(self, edge_tracker: EdgeTracker):
        """Batch-evaluate diversity scores and compute fitness."""
        n = len(self.population)
        if n == 0:
            return

        total_edges = len(edge_tracker.cumulative_edges) if edge_tracker.cumulative_edges else 1

        # Compute diversity via Wasserstein weights
        for ind in self.population:
            w = edge_tracker.compute_wasserstein_weight(ind.seed_key)
            ind.diversity_score = (w - 0.5) / 1.5  # normalize to [0, 1]

        # Score fitness
        for ind in self.population:
            self.fitness_fn.score(ind, total_edges, self.generation)

    # ── Stats ───────────────────────────────────────────────────────

    def _update_stats(self):
        if not self.population:
            return
        fitnesses = [i.fitness for i in self.population]
        self.best_fitness = max(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)

    # ── Persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize QEA state to a dict (for StateStore pickle)."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "species_count": self.species_count,
            "population": [ind.to_dict() for ind in self.population],
        }

    def from_dict(self, data: dict) -> None:
        """Restore QEA state from a serialized dict."""
        self.generation = data.get("generation", 0)
        self.best_fitness = data.get("best_fitness", 0.0)
        self.avg_fitness = data.get("avg_fitness", 0.0)
        self.species_count = data.get("species_count", 0)
        self.population = [QEAIndividual.from_dict(d) for d in data.get("population", [])]

    def save(self, path: Path):
        """Persist QEA state to disk (legacy interface)."""
        # Compact separators: indent=2 roughly doubled the file for no
        # readability benefit (this is machine state, written every shutdown).
        path.write_text(json.dumps(self.to_dict(), separators=(",", ":")))

    def load(self, path: Path):
        """Restore QEA state from disk (legacy interface)."""
        if not path.exists():
            return
        state = json.loads(path.read_text())
        self.from_dict(state)

    # ── Public helpers (compatibility) ──────────────────────────────

    def select_parent(self) -> QEAIndividual:
        """Alias for _tournament_select over the full population."""
        return self._tournament_select(self.population)
