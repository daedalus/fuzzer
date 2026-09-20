"""Skip deterministic stages for low-information seeds.

Ports AFL++'s SkipDet: decides whether a seed deserves expensive
deterministic fuzzing (bitflips, arith, interesting values) based on
how many new undetermined bits its coverage map contains.

Seeds whose coverage is largely subsumed by previously-deterministically-fuzzed
seeds skip straight to havoc, saving significant execution time.

This module is the *gate* half of SkipDet only. The *effector map* half --
deciding which individual bytes deserve the arithmetic and interesting-value
passes -- lives in ``services/operators.py``, built from the byteflip 8/8
pass the deterministic schedule already runs. Two implementations of it used
to live here, ``build_skip_eff_map`` and ``inference``; both were unreachable
from ``src/`` and ``inference`` never wrote its output map at all, returning
all-zeros ("every byte ineffective") on every call while logging a count that
was zero by construction. They are retired rather than repaired: they spend
``O(len)`` and ``O(len / MINIMAL_BLOCK_SIZE)`` extra executions respectively
to learn what the byteflip pass now yields for free.

One capability went with them and is worth naming so it is not mistaken for
an oversight: ``inference`` answered a *prior* question -- finding large inert
ranges *before* paying the 8L bitflip pass, which is the only way to skip
bitflips too. That is a real gap, and the right shape for it is a pooling
design rather than a block-halving search, so it is recorded as future work
rather than kept as dead code.
"""

import logging

log = logging.getLogger(__name__)


# Configurable thresholds (from AFL++ config.h). MINIMAL_BLOCK_SIZE and
# MAX_INF_EXECS went with the block-flip effector search; MAX_QUICK_EFF_EXECS
# stays because MAX_DET_MUTATIONS below is defined against it.
MAX_QUICK_EFF_EXECS = 64 * 1024
THRESHOLD_DEC_TIME_MS = 20 * 60 * 1000  # 20 minutes

# Deterministic-stage execution budget per seed. Mirrors MAX_QUICK_EFF_EXECS:
# a full bitflip 1/1 pass alone costs 8*len(seed) execs, so this is the
# ceiling on total mutations _deterministic_mutation_stream will yield for
# one seed regardless of how much further the stage schedule has left to run.
MAX_DET_MUTATIONS = MAX_QUICK_EFF_EXECS


def trace_mini_from_edges(edge_ids, map_size: int = 65536) -> bytearray:
    """Build a AFL-style trace_mini bitmap from a sparse edge_id set.

    ``SkipDetector.should_det_fuzz`` expects a positional bitmap (1 bit per
    coverage-map slot), the shape AFL's own bitmap naturally has. This
    fuzzer's coverage is a sparse, context-sensitive hash table instead
    (``edge_id`` values are ``ctx ^ prev_loc ^ cur_loc`` hashes, not indices
    into a fixed-size byte array), so there is no positional bitmap lying
    around to hand it.

    Folding each edge_id into ``map_size`` bits by taking it modulo the bit
    count reproduces the property should_det_fuzz actually depends on --
    "how much of this seed's coverage falls in slots no seed has
    deterministically explored before" -- without requiring the shim to
    maintain a second, byte-indexed coverage representation just for this.

    This deliberately replaces a first-cut version that indexed by
    ``edge_id // 8`` directly and dropped anything with ``idx >=
    len(trace_mini)``: since edge_ids are hash values, not small positional
    integers, that silently discarded nearly every edge for any seed whose
    coverage wasn't coincidentally under the first few KB of hash space --
    should_det_fuzz would then see an almost-always-empty bitmap and rarely
    find undetermined bits worth running a stage over, regardless of real
    coverage. Folding with modulo means every edge contributes; the only
    cost is the same bounded hash-collision noise AFL's own fixed-size
    bitmap already has, not a near-total blind spot.

    Args:
        edge_ids: Iterable of edge_id hashes covered by this seed.
        map_size: Number of addressable bit positions -- must match the
            ``SkipDetector.map_size`` this bitmap will be checked against.

    Returns:
        A packed bitmap of ``map_size // 8`` bytes (rounded up).
    """
    n_bytes = (map_size + 7) // 8
    mini = bytearray(n_bytes)
    for edge_id in edge_ids:
        bit = edge_id % map_size
        mini[bit >> 3] |= 1 << (bit & 7)
    return mini


class SkipDetector:
    """Decide whether seeds deserve deterministic fuzzing.

    Maintains a global virgin bitmap of bits that have been
    deterministically explored. Seeds whose coverage map contains
    few new undetermined bits are skipped.

    Args:
        map_size: Size of the coverage bitmap (default 65536).
    """

    def __init__(self, map_size: int = 65536):
        self.map_size = map_size
        # Global bitmap of bits explored by deterministic stages
        self.virgin_det_bits: bytearray = bytearray(map_size)
        # Threshold for deciding if a seed has enough new bits
        self.undet_bits_threshold: float = 0.0
        # Timestamp of last coverage find (for threshold decay)
        self._last_cov_undet_time: float = 0.0

    def should_det_fuzz(
        self,
        seed_trace_mini: bytearray | None,
        seed_favored: bool,
        seed_passed_det: bool,
        current_time_ms: float,
    ) -> bool:
        """Decide if a seed should undergo deterministic fuzzing.

        Args:
            seed_trace_mini: Compressed bitmap of edges hit by this seed
                (1 bit per edge, map_size/8 bytes). None if unavailable.
            seed_favored: Whether this seed is in the favored set.
            seed_passed_det: Whether this seed already passed deterministic.
            current_time_ms: Current timestamp in milliseconds.

        Returns:
            True if the seed should be deterministically fuzzed.

        This is a whole-seed verdict. Which *bytes* of an accepted seed get
        the arithmetic and interesting-value passes is decided separately, by
        the effector map ``services/operators.py`` builds from the byteflip
        8/8 pass (see ``DeterministicEffectorMap``).
        """
        # Already deterministically fuzzed or not favored
        if not seed_favored or seed_passed_det:
            return False

        if seed_trace_mini is None:
            return False

        # Decay threshold over time
        if self._last_cov_undet_time > 0:
            elapsed = current_time_ms - self._last_cov_undet_time
            if elapsed >= THRESHOLD_DEC_TIME_MS and self.undet_bits_threshold >= 2:
                self.undet_bits_threshold *= 0.75
                self._last_cov_undet_time = current_time_ms

        # Count new undetermined bits in this seed's trace
        new_det_bits = 0
        for i in range(min(len(seed_trace_mini) * 8, self.map_size)):
            byte_idx = i >> 3
            bit_idx = i & 7
            if (
                byte_idx < len(seed_trace_mini)
                and (seed_trace_mini[byte_idx] >> bit_idx) & 1
                and not self.virgin_det_bits[i]
            ):
                new_det_bits += 1

        # Initialize threshold from first seed
        if not self.undet_bits_threshold:
            self.undet_bits_threshold = max(1.0, new_det_bits * 0.05)

        if new_det_bits >= self.undet_bits_threshold:
            self._last_cov_undet_time = current_time_ms
            # Mark these bits as deterministically explored
            for i in range(min(len(seed_trace_mini) * 8, self.map_size)):
                byte_idx = i >> 3
                bit_idx = i & 7
                if byte_idx < len(seed_trace_mini) and (seed_trace_mini[byte_idx] >> bit_idx) & 1:
                    self.virgin_det_bits[i] = 1
            return True

        return False
