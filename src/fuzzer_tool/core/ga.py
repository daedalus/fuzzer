"""Genetic algorithm lifecycle for coverage-guided fuzzing.

Provides a finite, evolving population with unified fitness scoring,
fitness-proportional parent selection, speciation via MinHash LSH,
and generational replacement with elitism.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .mutations import (
    byte_insert,
    byte_shuffle,
    crossover,
    insert_ascii_num,
    splice,
    type_replace,
)

if TYPE_CHECKING:
    from .edge_tracker import EdgeTracker


@dataclass
class Individual:
    """A single member of the GA population."""

    data: bytes
    fitness: float = 0.0
    edge_count: int = 0
    novelty_score: float = 0.0
    diversity_score: float = 0.0
    freshness_score: float = 0.0
    mutation_potential: float = 0.0
    species_id: int = -1
    generation: int = 0
    crash: bool = False
    seed_key: str = ""

    def __post_init__(self):
        if not self.seed_key:
            self.seed_key = hashlib.sha256(self.data).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "data": self.data.hex(),
            "fitness": self.fitness,
            "edge_count": self.edge_count,
            "species_id": self.species_id,
            "generation": self.generation,
            "crash": self.crash,
            "seed_key": self.seed_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Individual:
        return cls(
            data=bytes.fromhex(d["data"]),
            fitness=d.get("fitness", 0.0),
            edge_count=d.get("edge_count", 0),
            species_id=d.get("species_id", -1),
            generation=d.get("generation", 0),
            crash=d.get("crash", False),
            seed_key=d.get("seed_key", ""),
        )


class FitnessFunction:
    """Unified fitness scorer for GA mode.

    fitness = w_novelty * novelty + w_diversity * diversity
            + w_freshness * freshness + w_mutation * mutation_potential
            + crash_bonus
    """

    def __init__(
        self,
        w_novelty: float = 0.4,
        w_diversity: float = 0.25,
        w_freshness: float = 0.1,
        w_mutation: float = 0.15,
        crash_bonus: float = 10.0,
        fresh_decay: float = 0.95,
    ):
        self.w_novelty = w_novelty
        self.w_diversity = w_diversity
        self.w_freshness = w_freshness
        self.w_mutation = w_mutation
        self.crash_bonus = crash_bonus
        self.fresh_decay = fresh_decay

    def score(
        self,
        ind: Individual,
        total_edges: int,
        current_generation: int,
    ) -> float:
        """Compute unified fitness for an individual."""
        # Novelty: fraction of total unique edges discovered by this seed
        if total_edges > 0:
            ind.novelty_score = min(ind.edge_count / total_edges, 1.0)
        else:
            ind.novelty_score = 0.0

        # Freshness: exponential decay by generation gap
        age = max(1, current_generation - ind.generation)
        ind.freshness_score = self.fresh_decay**age

        # Aggregate
        raw = (
            self.w_novelty * ind.novelty_score
            + self.w_diversity * ind.diversity_score
            + self.w_freshness * ind.freshness_score
            + self.w_mutation * ind.mutation_potential
        )
        if ind.crash:
            raw += self.crash_bonus

        ind.fitness = raw
        return raw


class Speciation:
    """Species manager backed by MinHash LSH buckets from EdgeTracker."""

    def __init__(self, edge_tracker: EdgeTracker, jaccard_threshold: float = 0.3):
        self.edge_tracker = edge_tracker
        self.jaccard_threshold = jaccard_threshold

    def assign_species(self, pop: list[Individual]) -> dict[int, list[Individual]]:
        """Group population into species using MinHash LSH queries."""
        species_map: dict[int, list[Individual]] = {}
        next_species_id = 0

        for ind in pop:
            if ind.crash:
                # Crashes always get their own species to prevent culling
                sid = next_species_id
                next_species_id += 1
                ind.species_id = sid
                species_map.setdefault(sid, []).append(ind)
                continue

            # Query MinHash LSH for similar seeds
            similar = self.edge_tracker.find_similar_seeds(
                ind.seed_key, min_jaccard=self.jaccard_threshold
            )
            if similar:
                # Find the species of the most similar existing seed
                found = False
                for other in pop:
                    if other.seed_key in similar and other.species_id >= 0:
                        ind.species_id = other.species_id
                        species_map.setdefault(other.species_id, []).append(ind)
                        found = True
                        break
                if not found:
                    ind.species_id = next_species_id
                    next_species_id += 1
                    species_map.setdefault(ind.species_id, []).append(ind)
            else:
                # No similar seeds — new species
                ind.species_id = next_species_id
                next_species_id += 1
                species_map.setdefault(ind.species_id, []).append(ind)

        return species_map


# Standalone mutation functions for GA breeding
_GA_MUTATION_OPS = [
    lambda data: _mutate_byte_flip(data),
    lambda data: _mutate_random_bytes(data),
    lambda data: _mutate_block_insert(data),
    lambda data: _mutate_block_delete(data),
    lambda data: type_replace(data),
    lambda data: byte_shuffle(data),
    lambda data: byte_insert(data),
    lambda data: insert_ascii_num(data),
]


def _mutate_byte_flip(data: bytes) -> bytes:
    """Flip a random byte at a random position."""
    if len(data) < 1:
        return data
    buf = bytearray(data)
    pos = random.randint(0, len(buf) - 1)
    buf[pos] ^= 0xFF
    return bytes(buf)


def _mutate_random_bytes(data: bytes) -> bytes:
    """Replace a random byte with a random value."""
    if len(data) < 1:
        return data
    buf = bytearray(data)
    pos = random.randint(0, len(buf) - 1)
    buf[pos] = random.randint(0, 255)
    return bytes(buf)


def _mutate_block_insert(data: bytes) -> bytes:
    """Insert a random block at a random position."""
    if len(data) < 1:
        return data
    pos = random.randint(0, len(data))
    block_len = random.randint(1, min(16, max(1, len(data) // 4)))
    block = bytes(random.randint(0, 255) for _ in range(block_len))
    return data[:pos] + block + data[pos:]


def _mutate_block_delete(data: bytes) -> bytes:
    """Delete a random block."""
    if len(data) <= 1:
        return data
    pos = random.randint(0, len(data) - 1)
    block_len = random.randint(1, min(16, max(1, len(data) // 4)))
    end = min(pos + block_len, len(data))
    return data[:pos] + data[end:]


class GALifecycle:
    """Core GA lifecycle controller.

    Integrates into the fuzz_one() loop. Manages population,
    fitness evaluation, selection, crossover, and mutation.
    """

    def __init__(
        self,
        pop_size: int = 200,
        elite_fraction: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        generation_size: int = 500,
        speciation_threshold: float = 0.3,
        fitness: FitnessFunction | None = None,
    ):
        self.pop_size = pop_size
        self.elite_fraction = elite_fraction
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.generation_size = generation_size
        self.speciation_threshold = speciation_threshold
        self.fitness = fitness or FitnessFunction()

        self.population: list[Individual] = []
        self.generation = 0
        self.iterations_since_gen = 0
        self._speciation: Speciation | None = None

        # Stats for logging
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.species_count = 0

    def initialize(self, corpus: list[bytes], edge_tracker: EdgeTracker):
        """Seed population from existing corpus."""
        self._speciation = Speciation(edge_tracker, self.speciation_threshold)
        for data in corpus[: self.pop_size]:
            seed_key = hashlib.sha256(data).hexdigest()[:16]
            edge_set = edge_tracker.seed_edges.get(seed_key, set())
            ind = Individual(
                data=data,
                edge_count=len(edge_set),
                generation=0,
                seed_key=seed_key,
            )
            self.population.append(ind)
        self._evaluate_all(edge_tracker)

    def on_fuzz_result(
        self,
        data: bytes,
        new_coverage: bool,
        edge_count: int,
        edge_tracker: EdgeTracker,
    ) -> Individual | None:
        """Called after each fuzz_one iteration.

        Returns an Individual to add to population if it improves coverage.
        Returns None if population should not change.
        """
        self.iterations_since_gen += 1

        if new_coverage:
            seed_key = hashlib.sha256(data).hexdigest()[:16]
            ind = Individual(
                data=data,
                edge_count=edge_count,
                generation=self.generation,
                seed_key=seed_key,
            )
            return ind

        # Trigger generation boundary
        if self.iterations_since_gen >= self.generation_size:
            self._evolve(edge_tracker)
            self.generation += 1
            self.iterations_since_gen = 0

        return None

    def select_parent(self) -> Individual:
        """Fitness-proportional selection with species awareness.

        If speciation is active, selects within species with 50% probability,
        otherwise selects globally. Uses tournament selection.
        """
        if self._speciation and random.random() < 0.5:
            # Intra-species selection
            species_map = self._get_species_map()
            eligible = [s for s in species_map.values() if len(s) >= 2]
            if eligible:
                pool = random.choice(eligible)
                return self._tournament_select(pool)
        return self._tournament_select(self.population)

    def _tournament_select(self, pool: list[Individual]) -> Individual:
        """Tournament selection: pick best of k random individuals."""
        k = min(self.tournament_size, len(pool))
        candidates = random.sample(pool, k)
        return max(candidates, key=lambda i: i.fitness)

    def _evolve(self, edge_tracker: EdgeTracker):
        """Run one generation: evaluate, cull, breed."""
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
        offspring = []
        for _ in range(n_breed):
            parent_a = self.select_parent()
            parent_b = self.select_parent()

            if random.random() < self.crossover_rate:
                child_data = self._crossover(parent_a.data, parent_b.data)
            else:
                child_data = parent_a.data  # clone

            if random.random() < self.mutation_rate:
                child_data = self._mutate(child_data)

            child = Individual(
                data=child_data,
                generation=self.generation + 1,
            )
            offspring.append(child)

        # 5. Replace population
        self.population = elites + offspring
        self._update_stats()

    def _crossover(self, a: bytes, b: bytes) -> bytes:
        """Two-point crossover using existing mutations.crossover."""
        if random.random() < 0.5:
            return crossover(a, b)
        return splice(a, b)

    def _mutate(self, data: bytes) -> bytes:
        """Apply a random mutation from available operators."""
        op = random.choice(_GA_MUTATION_OPS)
        result = op(data)
        return result if result else data

    def _evaluate_all(self, edge_tracker: EdgeTracker):
        """Batch-evaluate diversity scores and compute fitness."""
        n = len(self.population)
        if n == 0:
            return

        total_edges = len(edge_tracker.cumulative_edges) if edge_tracker.cumulative_edges else 1

        # Compute pairwise diversity using Wasserstein weights
        for ind in self.population:
            # Use existing Wasserstein weight as diversity proxy
            w = edge_tracker.compute_wasserstein_weight(ind.seed_key)
            # Normalize to [0, 1] — weight is in [0.5, 2.0]
            ind.diversity_score = (w - 0.5) / 1.5

        # Score fitness
        for ind in self.population:
            self.fitness.score(ind, total_edges, self.generation)

    def _get_species_map(self) -> dict[int, list[Individual]]:
        result: dict[int, list[Individual]] = {}
        for ind in self.population:
            result.setdefault(ind.species_id, []).append(ind)
        return result

    def _update_stats(self):
        if not self.population:
            return
        fitnesses = [i.fitness for i in self.population]
        self.best_fitness = max(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)

    def pick_seed(self) -> bytes:
        """Return a seed for fuzz_one() — picks from population with fitness weighting."""
        if not self.population:
            return b"\x00" * 64
        ind = self.select_parent()
        return ind.data

    def add_to_population(self, ind: Individual):
        """Add an individual (e.g., new coverage seed) to population."""
        if len(self.population) >= self.pop_size:
            # Replace worst if new individual is better
            worst_idx = min(range(len(self.population)), key=lambda i: self.population[i].fitness)
            if ind.fitness > self.population[worst_idx].fitness:
                self.population[worst_idx] = ind
        else:
            self.population.append(ind)

    def save(self, path: Path):
        """Persist GA state to disk."""
        state = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "species_count": self.species_count,
            "population": [ind.to_dict() for ind in self.population],
        }
        path.write_text(json.dumps(state, indent=2))

    def load(self, path: Path):
        """Restore GA state from disk."""
        if not path.exists():
            return
        state = json.loads(path.read_text())
        self.generation = state.get("generation", 0)
        self.best_fitness = state.get("best_fitness", 0.0)
        self.avg_fitness = state.get("avg_fitness", 0.0)
        self.species_count = state.get("species_count", 0)
        self.population = [Individual.from_dict(d) for d in state.get("population", [])]
