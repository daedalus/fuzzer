"""Per-byte sensitivity map (fuzzing Lyapunov exponent).

For each byte position in a seed, measures how much the execution trace
diverges when that byte is perturbed. High-divergence offsets are where
the target is "chaotic" — small input changes cascade into large behavioral
changes. Low-divergence offsets (padding, unused fields) can be deprioritized.

This generalizes what cmplog/redqueen already do at comparison sites into
a target-agnostic sensitivity map computable from the edge tracker.

The sensitivity score for byte position i is:

    sensitivity[i] = 1.0 - Jaccard(original_edges, perturbed_edges)

Where Jaccard = |intersection| / |union|. A score of 1.0 means the
perturbation completely changed the execution trace (high sensitivity);
0.0 means no change (low sensitivity).
"""

import logging
import random

log = logging.getLogger(__name__)


class ByteSensitivityTracker:
    """Track per-byte sensitivity scores for seeds.

    Args:
        max_seeds: Maximum number of seeds to track sensitivity for.
        max_bytes: Maximum byte positions to analyze per seed.
        sample_rate: Fraction of byte positions to perturb (0.0-1.0).
            Full analysis requires len(seed) re-executions; sampling
            trades accuracy for speed.
    """

    def __init__(
        self,
        max_seeds: int = 100,
        max_bytes: int = 4096,
        sample_rate: float = 0.1,
    ):
        self.max_seeds = max_seeds
        self.max_bytes = max_bytes
        self.sample_rate = sample_rate
        self._sensitivity: dict[bytes, list[float]] = {}
        self._analyzed: set[bytes] = set()

    def analyze_seed(
        self,
        seed: bytes,
        original_edges: set[int],
        exec_fn,
    ) -> list[float]:
        """Analyze byte sensitivity for a seed.

        Args:
            seed: The seed input bytes.
            original_edges: Edge set from executing the original seed.
            exec_fn: Callable(bytes) -> set[int] that executes input and
                returns the edge set.

        Returns:
            List of sensitivity scores (0.0-1.0) per byte position.
        """
        seed_key = bytes(seed[:64])
        if seed_key in self._analyzed:
            return self._sensitivity.get(seed_key, [])

        n = min(len(seed), self.max_bytes)
        if n == 0:
            return []

        sample_size = max(1, int(n * self.sample_rate))
        positions = random.sample(range(n), min(sample_size, n))

        scores = [0.0] * n
        for pos in positions:
            perturbed = bytearray(seed)
            perturbed[pos] = random.randint(0, 255)
            try:
                perturbed_edges = exec_fn(bytes(perturbed))
                if original_edges and perturbed_edges:
                    intersection = len(original_edges & perturbed_edges)
                    union = len(original_edges | perturbed_edges)
                    jaccard = intersection / union if union > 0 else 1.0
                    scores[pos] = 1.0 - jaccard
            except Exception:
                log.debug("Edge computation failed for position %d", pos, exc_info=True)

        seed_key = bytes(seed[:64])
        self._sensitivity[seed_key] = scores
        self._analyzed.add(seed_key)

        if len(self._analyzed) > self.max_seeds:
            oldest = next(iter(self._analyzed))
            self._analyzed.discard(oldest)
            self._sensitivity.pop(oldest, None)

        return scores

    def get_weighted_position(self, seed: bytes, buf_len: int) -> int | None:
        """Select a byte position weighted by sensitivity.

        High-sensitivity positions are more likely to be selected.
        Falls back to uniform random if no sensitivity data exists.

        Args:
            seed: The seed input bytes.
            buf_len: Length of the buffer to select from.

        Returns:
            Byte position index, or None if no data.
        """
        seed_key = bytes(seed[:64])
        scores = self._sensitivity.get(seed_key)
        if not scores or len(scores) < buf_len:
            return None

        total = sum(scores[:buf_len])
        if total <= 0:
            return None

        r = random.random() * total
        cumulative = 0.0
        for i in range(buf_len):
            cumulative += scores[i]
            if r <= cumulative:
                return i
        return buf_len - 1

    def has_data(self, seed: bytes) -> bool:
        """Check if sensitivity data exists for a seed."""
        return bytes(seed[:64]) in self._analyzed

    def save(self) -> dict:
        """Serialize state."""
        return {
            "max_seeds": self.max_seeds,
            "max_bytes": self.max_bytes,
            "sample_rate": self.sample_rate,
            "sensitivity": {k.hex(): v for k, v in self._sensitivity.items()},
        }

    def load(self, data: dict) -> None:
        """Restore state."""
        self.max_seeds = data.get("max_seeds", self.max_seeds)
        self.max_bytes = data.get("max_bytes", self.max_bytes)
        self.sample_rate = data.get("sample_rate", self.sample_rate)
        self._sensitivity = {bytes.fromhex(k): v for k, v in data.get("sensitivity", {}).items()}
        self._analyzed = set(self._sensitivity.keys())
