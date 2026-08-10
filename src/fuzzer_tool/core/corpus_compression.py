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

import hashlib
import logging

log = logging.getLogger(__name__)

# Try to import pyppmd; fall back gracefully if not installed
try:
    from pyppmd import PpmdCompressor

    PPMD_AVAILABLE = True
except ImportError:
    PPMD_AVAILABLE = False
    log.debug("pyppmd not installed — corpus compression disabled")


# PPMD runs at roughly 2 MB/s, so both the cache and the sample cap below are
# load-bearing rather than micro-optimisations. compute_seed_novelty is called
# once per seed per corpus minimization and once per seed-picker weighting
# pass; uncached, on a corpus that has bloated (which is exactly when
# minimization runs), that is minutes of wall clock per pass with the fuzzer
# making no progress.
#
# The ratio is a novelty estimate used to scale a scheduling weight, so a
# prefix is a sound proxy for the whole seed: the model is adaptive and its
# ratio stabilises well before 64 KiB. Capping bounds the cost per seed at
# ~30 ms instead of letting it scale with the largest seed in the corpus.
PPMD_SAMPLE_BYTES = 65536

# Cache entries are (digest -> float). Bounded because a long campaign sees
# unboundedly many distinct seeds; cleared wholesale on overflow, since
# recomputing is the same cost as a miss and an LRU would cost more
# bookkeeping than it saves.
PPMD_CACHE_MAX = 4096


def _ppmd_cache_key(seed: bytes) -> str:
    """Stable digest for the sampled prefix of *seed*."""
    return hashlib.blake2b(seed[:PPMD_SAMPLE_BYTES], digest_size=16).hexdigest()


class CorpusCompressor:
    """Analyze corpus compressibility for seed novelty scoring.

    Maintains a running PPMD model of the corpus and computes per-seed
    compression ratios. Used for seed selection and corpus minimization.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PPMD_AVAILABLE
        self._seed_ratios: dict[str, float] = {}  # seed digest -> ratio
        self._last_computed_count = 0

    def compute_seed_ratio(self, seed: bytes) -> float:
        """Compute the PPMD compression ratio for a single seed.

        Returns compressed_size / raw_size over the first PPMD_SAMPLE_BYTES.
        Low ratio = compressible (similar to known patterns); high ratio =
        incompressible (novel/diverse). Memoised by digest of the sampled
        prefix -- see the note above PPMD_SAMPLE_BYTES for why that matters.
        """
        if not self.enabled or not seed:
            return 1.0

        key = _ppmd_cache_key(seed)
        cached = self._seed_ratios.get(key)
        if cached is not None:
            return cached

        sample = seed[:PPMD_SAMPLE_BYTES]
        try:
            c = PpmdCompressor()
            compressed = c.compress(sample)
            compressed += c.flush()
            ratio = len(compressed) / len(sample)
        except Exception:
            return 1.0

        if len(self._seed_ratios) >= PPMD_CACHE_MAX:
            self._seed_ratios.clear()
        self._seed_ratios[key] = ratio
        return ratio

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
