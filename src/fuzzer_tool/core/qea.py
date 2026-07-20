"""Quantum-inspired Evolutionary Algorithm (QEA) encoding.

Implements an alternative individual representation where each bit is
represented as a qubit-like probability amplitude pair (α, β) with
α² + β² = 1, meaning "this bit is P(0)=α² likely to be 0" rather than
a committed value. Amplitudes are incrementally updated via rotation
gates after each evaluation.

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
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fuzzer_tool.core.mutations import crossover

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


# ── Helper: bytes ↔ bit list conversions ──────────────────────────────


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
        # Pad remaining bits with 0
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
        amplitudes: Per-bit α values, length = 8 * num_bytes.
        fitness: Current fitness score.
        edge_count: Number of unique edges covered by collapsed output.
        species_id: Species assignment for speciation.
        generation: Generation this individual was created in.
        best_collapsed: Best concrete byte string found so far.
        best_fitness: Fitness of the best collapsed output.
        seed_key: SHA-256 hash prefix of best_collapsed.
        crash: Whether this individual triggered a crash.
    """

    amplitudes: list[float]
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

    @property
    def num_bytes(self) -> int:
        """Number of bytes implied by the amplitude length."""
        return max(1, len(self.amplitudes) // BITS_PER_BYTE)

    def to_dict(self) -> dict:
        return {
            "amplitudes": self.amplitudes,
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
            amplitudes=list(d.get("amplitudes", [])),
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


def _bias_amplitudes_from(data: bytes, *, strong_prob: float = 0.9) -> list[float]:
    """Create amplitude vector biased toward the given byte values.

    For each bit in ``data``, sets α = *strong_prob* if the bit is 0
    (strong bias toward collapsing to the same 0), α = 1 - strong_prob
    if the bit is 1 (bias toward matching). This gives a prior that
    favors the seed data while maintaining some uncertainty.

    Args:
        data: Template byte string to bias toward.
        strong_prob: Amplitude value for bits matching 0
            (default 0.9 → P(0) = 0.81, P(1) = 0.19 for zero bits).

    Returns:
        List of α values, length = 8 * len(data).
    """
    bits = _bytes_to_bits(data)
    weak_prob = 1.0 - strong_prob
    return [strong_prob if b == 0 else weak_prob for b in bits]


def _uniform_amplitudes(n_bits: int, *, alpha: float = ALPHA_UNIFORM) -> list[float]:
    """Create uniform amplitude vector (all α = 0.7071... = uniform uncertainty)."""
    return [alpha] * n_bits


# ── Collapse: amplitudes → concrete bytes ─────────────────────────────


def collapse(amplitudes: list[float]) -> bytes:
    """Sample concrete bytes from qubit amplitudes.

    For each bit position i: bit = 0 with probability α_i², else 1.

    Args:
        amplitudes: α values for each bit, length must be multiple of 8.

    Returns:
        Collapsed concrete byte string.
    """
    bits: list[int] = []
    for a in amplitudes:
        # P(bit=0) = α²
        if random.random() < a * a:
            bits.append(0)
        else:
            bits.append(1)
    return _bits_to_bytes(bits)


# ── Rotation gate ──────────────────────────────────────────────────────


def rotation_gate(
    amplitudes: list[float],
    collapsed: bytes,
    *,
    improved: bool,
    delta: float = 0.05,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> list[float]:
    """Apply QEA rotation gate to update amplitudes based on fitness feedback.

    The rotation gate nudges each bit's amplitude toward or away from the
    collapsed value depending on whether that value was beneficial:

    - bit=0, improved=True  → α increases (rotate toward |0⟩)
    - bit=0, improved=False → α decreases (rotate toward |1⟩)
    - bit=1, improved=True  → α decreases (rotate toward |1⟩)
    - bit=1, improved=False → α increases (rotate toward |0⟩)

    The magnitude ``delta`` controls how far each amplitude moves per
    step. Results are clamped to ``[alpha_min, alpha_max]`` to maintain
    minimum uncertainty.

    Args:
        amplitudes: Current α values to update (in-place + return).
        collapsed: Concrete bytes that the amplitudes produced.
        improved: Whether the collapsed outcome was beneficial.
        delta: Base rotation magnitude (default 0.05).
        alpha_min: Minimum amplitude clamp (default 0.01).
        alpha_max: Maximum amplitude clamp (default 0.99).

    Returns:
        Updated amplitudes (same list object, also modified in place).
    """
    collapsed_bits = _bytes_to_bits(collapsed)

    # Extend or truncate collapsed bits to match amplitude length
    n_bits = len(amplitudes)
    while len(collapsed_bits) < n_bits:
        collapsed_bits.append(0)
    collapsed_bits = collapsed_bits[:n_bits]

    for i in range(n_bits):
        bit = collapsed_bits[i]
        if (bit == 0 and improved) or (bit == 1 and not improved):
            # Rotate toward |0⟩ (increase α)
            amplitudes[i] = min(amplitudes[i] + delta, alpha_max)
        else:
            # Rotate toward |1⟩ (decrease α)
            amplitudes[i] = max(amplitudes[i] - delta, alpha_min)

    return amplitudes


# ── Amplitude mutation ─────────────────────────────────────────────────


def mutate_amplitudes(
    amplitudes: list[float],
    *,
    prob: float = 0.02,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> list[float]:
    """Randomly perturb amplitudes to maintain diversity.

    Each bit's α is reset to a random uniform value with probability
    ``prob``. This is QEA's equivalent of GA mutation, preventing
    amplitude stagnation when all values converge to extremes.

    Args:
        amplitudes: Amplitude vector to mutate (in-place + return).
        prob: Per-bit mutation probability (default 0.02).
        alpha_min: Minimum amplitude after reset.
        alpha_max: Maximum amplitude after reset.

    Returns:
        Mutated amplitudes (same list, modified in place).
    """
    for i in range(len(amplitudes)):
        if random.random() < prob:
            amplitudes[i] = random.uniform(alpha_min, alpha_max)
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
            seed_key = hashlib.sha256(data).hexdigest()[:16]
            edge_set = edge_tracker.seed_edges.get(seed_key, set())
            ind = QEAIndividual(
                amplitudes=_bias_amplitudes_from(data, strong_prob=self.strong_bias),
                edge_count=len(edge_set),
                generation=0,
                best_collapsed=data,
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

        # Apply rotation gate feedback using the last-selected parent
        if self._last_parent is not None and self._last_collapsed:
            rotation_gate(
                self._last_parent.amplitudes,
                self._last_collapsed,
                improved=new_coverage,
                delta=self.rotation_angle,
            )

        if new_coverage:
            seed_key = hashlib.sha256(data).hexdigest()[:16]
            ind = QEAIndividual(
                amplitudes=_bias_amplitudes_from(data, strong_prob=self.strong_bias),
                edge_count=edge_count,
                generation=self.generation,
                best_collapsed=data,
                seed_key=seed_key,
            )
            return ind

        # Trigger generation boundary
        if self.iterations_since_gen >= self.generation_size:
            self._evolve(edge_tracker)
            self.generation += 1
            self.iterations_since_gen = 0

        return None

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

        # 3. Elitism: keep top fraction
        n_elite = max(1, int(len(self.population) * self.elite_fraction))
        self.population.sort(key=lambda i: i.fitness, reverse=True)
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

    def save(self, path: Path):
        """Persist QEA state to disk."""
        state = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "species_count": self.species_count,
            "population": [ind.to_dict() for ind in self.population],
        }
        path.write_text(json.dumps(state, indent=2))

    def load(self, path: Path):
        """Restore QEA state from disk."""
        if not path.exists():
            return
        state = json.loads(path.read_text())
        self.generation = state.get("generation", 0)
        self.best_fitness = state.get("best_fitness", 0.0)
        self.avg_fitness = state.get("avg_fitness", 0.0)
        self.species_count = state.get("species_count", 0)
        self.population = [QEAIndividual.from_dict(d) for d in state.get("population", [])]

    # ── Public helpers (compatibility) ──────────────────────────────

    def select_parent(self) -> QEAIndividual:
        """Alias for _tournament_select over the full population."""
        return self._tournament_select(self.population)
