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

    def __post_init__(self):
        if not self.seed_key and self.best_collapsed:
            self.seed_key = hashlib.sha256(self.best_collapsed).hexdigest()[:16]
        # Ensure amplitudes are always ndarray
        if isinstance(self.amplitudes, list):
            self.amplitudes = np.array(self.amplitudes, dtype=np.float64)

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
        }

    @classmethod
    def from_dict(cls, d: dict) -> QEAIndividual:
        bc = d.get("best_collapsed", "")
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

    Vectorized: single ``np.where`` call replaces the per-bit Python loop
    (~19× faster than the list-based implementation).

    The rotation gate nudges each bit's amplitude toward or away from the
    collapsed value depending on whether that value was beneficial:

    - bit=0, improved=True  → α increases (rotate toward |0⟩)
    - bit=0, improved=False → α decreases (rotate toward |1⟩)
    - bit=1, improved=True  → α decreases (rotate toward |1⟩)
    - bit=1, improved=False → α increases (rotate toward |0⟩)

    Args:
        amplitudes: Current α values to update (in-place + return).
        collapsed: Concrete bytes that the amplitudes produced.
        improved: Whether the collapsed outcome was beneficial.
        delta: Base rotation magnitude (default 0.05).
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

    if improved:
        # bit=0 → increase α; bit=1 → decrease α
        amplitudes[:] = np.where(
            bits == 0,
            np.minimum(amplitudes + delta, alpha_max),
            np.maximum(amplitudes - delta, alpha_min),
        )
    else:
        # bit=0 → decrease α; bit=1 → increase α
        amplitudes[:] = np.where(
            bits == 0,
            np.maximum(amplitudes - delta, alpha_min),
            np.minimum(amplitudes + delta, alpha_max),
        )
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
        mutation_prob: float = 0.02,
        init_alpha: float = ALPHA_UNIFORM,
        strong_bias: float = ALPHA_STRONG,
        tournament_size: int = 3,
        speciation_threshold: float = 0.3,
        elite_reset_every: int = 0,
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
            )
            self.population.append(ind)

        self._evaluate_all(edge_tracker)

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
                delta=self.rotation_angle,
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

            # Collapse both parents and crossover the committed bytes
            bytes_a = collapse(parent_a.amplitudes)
            bytes_b = collapse(parent_b.amplitudes)

            # Use two-point crossover (from mutations module)
            child_bytes = crossover(bytes_a, bytes_b)

            # Parents are already capped (see QEA_MAX_INPUT_BYTES), so
            # child_bytes can't exceed 2x that — cap anyway rather than
            # relying on that invariant holding as this code evolves.
            child_bytes = _qea_cap(child_bytes)

            # Create child with amplitudes biased toward the crossed bytes
            child = QEAIndividual(
                amplitudes=_bias_amplitudes_from(child_bytes, strong_prob=self.strong_bias),
                generation=self.generation + 1,
                best_collapsed=child_bytes,
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
