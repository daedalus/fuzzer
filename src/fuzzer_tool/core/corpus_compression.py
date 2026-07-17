"""Corpus compression analysis using PPMD.

Uses PPMD compression ratio as a seed novelty signal:
- Seeds that compress poorly against the corpus model are novel/diverse
- Seeds that compress well are redundant/similar to existing corpus

Integration points:
- Seed selection: boost seeds with low PPMD ratio (novel)
- Corpus minimization: prune seeds with high PPMD ratio (redundant)
- Report: corpus compression statistics

PPMD (Prediction by Partial Matching) builds a context model as it
compresses. The model captures byte-level conditional distributions.
A seed that doesn't fit the model (high compressed size relative to
raw size) is informationally novel — it exercises different patterns.
"""

import logging
import math
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Try to import pyppmd; fall back gracefully if not installed
try:
    from pyppmd import PpmdCompressor

    PPMD_AVAILABLE = True
except ImportError:
    PPMD_AVAILABLE = False
    log.debug("pyppmd not installed — corpus compression disabled")


class CorpusCompressor:
    """Analyze corpus compressibility for seed novelty scoring.

    Maintains a running PPMD model of the corpus and computes per-seed
    compression ratios. Used for seed selection and corpus minimization.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PPMD_AVAILABLE
        self._seed_ratios: dict[str, float] = {}  # seed_key -> ratio
        self._last_computed_count = 0

    def compute_seed_ratio(self, seed: bytes) -> float:
        """Compute PPMD compression ratio for a single seed.

        Returns ratio = compressed_size / raw_size.
        Low ratio = seed is compressible (similar to known patterns).
        High ratio = seed is incompressible (novel/diverse).
        """
        if not self.enabled or not seed:
            return 1.0

        try:
            c = PpmdCompressor()
            compressed = c.compress(seed)
            compressed += c.flush()
            return len(compressed) / len(seed) if len(seed) > 0 else 1.0
        except Exception:
            return 1.0

    def compute_seed_novelty(self, seed: bytes) -> float:
        """Compute novelty score for a seed based on PPMD ratio.

        Returns a value in [0, 1]:
        - 0.0 = highly compressible (redundant with known patterns)
        - 1.0 = incompressible (maximally novel/diverse)
        """
        ratio = self.compute_seed_ratio(seed)
        # Low ratio = compressible = redundant = low novelty
        # High ratio = incompressible = novel = high novelty
        if ratio <= 0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        return ratio

    def compute_corpus_stats(self, corpus: list[bytes]) -> dict:
        """Compute compression statistics for the entire corpus.

        Returns dict with:
        - mean_ratio: average compression ratio
        - median_ratio: median compression ratio
        - min_ratio: most compressible seed
        - max_ratio: most novel seed
        - total_raw: total uncompressed bytes
        - total_compressed: total compressed bytes
        - corpus_ratio: corpus-level compression ratio
        """
        if not self.enabled or not corpus:
            return {
                "mean_ratio": 1.0,
                "median_ratio": 1.0,
                "min_ratio": 1.0,
                "max_ratio": 1.0,
                "total_raw": 0,
                "total_compressed": 0,
                "corpus_ratio": 1.0,
            }

        ratios = []
        total_raw = 0
        total_compressed = 0

        for seed in corpus:
            ratio = self.compute_seed_ratio(seed)
            ratios.append(ratio)
            total_raw += len(seed)
            total_compressed += int(len(seed) * ratio)

        ratios.sort()
        n = len(ratios)

        return {
            "mean_ratio": sum(ratios) / n if n > 0 else 1.0,
            "median_ratio": ratios[n // 2] if n > 0 else 1.0,
            "min_ratio": ratios[0] if n > 0 else 1.0,
            "max_ratio": ratios[-1] if n > 0 else 1.0,
            "total_raw": total_raw,
            "total_compressed": total_compressed,
            "corpus_ratio": total_compressed / total_raw if total_raw > 0 else 1.0,
        }

    def should_prune(self, seed: bytes, threshold: float = 0.3) -> bool:
        """Determine if a seed should be pruned based on compression ratio.

        Seeds with ratio < threshold are highly compressible (redundant)
        and candidates for pruning.
        """
        if not self.enabled:
            return False
        ratio = self.compute_seed_ratio(seed)
        return ratio < threshold

    def rank_seeds(self, corpus: list[bytes]) -> list[tuple[int, float]]:
        """Rank seeds by novelty (highest first).

        Returns list of (index, novelty_score) tuples.
        """
        if not self.enabled or not corpus:
            return [(i, 1.0) for i in range(len(corpus))]

        scored = []
        for i, seed in enumerate(corpus):
            novelty = self.compute_seed_novelty(seed)
            scored.append((i, novelty))

        # Sort by novelty descending
        scored.sort(key=lambda x: -x[1])
        return scored
