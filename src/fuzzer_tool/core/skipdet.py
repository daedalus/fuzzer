"""Skip deterministic stages for low-information seeds.

Ports AFL++'s SkipDet: decides whether a seed deserves expensive
deterministic fuzzing (bitflips, arith, interesting values) based on
how many new undetermined bits its coverage map contains.

Seeds whose coverage is largely subsumed by previously-deterministically-fuzzed
seeds skip straight to havoc, saving significant execution time.

Also includes an inference stage that identifies large ineffective byte
ranges by binary-searching with block flips.
"""

import logging

log = logging.getLogger(__name__)

# Configurable thresholds (from AFL++ config.h)
MINIMAL_BLOCK_SIZE = 64
MAX_INF_EXECS = 16 * 1024
MAX_QUICK_EFF_EXECS = 64 * 1024
THRESHOLD_DEC_TIME_MS = 20 * 60 * 1000  # 20 minutes


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
            if byte_idx < len(seed_trace_mini):
                if (seed_trace_mini[byte_idx] >> bit_idx) & 1:
                    if not self.virgin_det_bits[i]:
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
                if byte_idx < len(seed_trace_mini):
                    if (seed_trace_mini[byte_idx] >> bit_idx) & 1:
                        self.virgin_det_bits[i] = 1
            return True

        return False

    def build_skip_eff_map(
        self,
        data: bytes,
        exec_fn,
        max_execs: int = MAX_QUICK_EFF_EXECS,
    ) -> bytearray:
        """Build a quick effective byte map via block flipping.

        Identifies byte positions that affect execution by flipping
        blocks of bytes and checking if the execution path changes.

        Args:
            data: Input data to analyze.
            exec_fn: Callable(bytes) -> int, returns execution checksum.
            max_execs: Maximum executions before giving up.

        Returns:
            Bytearray of length len(data) where 1 = effective, 0 = skip.
        """
        length = len(data)
        if length == 0:
            return bytearray()

        eff_map = bytearray(length)  # all zeros = skip all initially
        exec_count = 0

        # Get baseline checksum
        baseline_cksum = exec_fn(data)
        exec_count += 1

        # Flip blocks of increasing size to find effective regions
        block_size = MINIMAL_BLOCK_SIZE
        while block_size <= length and exec_count < max_execs:
            for pos in range(0, length, block_size):
                if exec_count >= max_execs:
                    break

                end = min(pos + block_size, length)
                # Flip the block
                flipped = bytearray(data)
                for i in range(pos, end):
                    flipped[i] ^= 0xFF

                cksum = exec_fn(bytes(flipped))
                exec_count += 1

                if cksum != baseline_cksum:
                    # This block is effective — mark individual bytes
                    for i in range(pos, end):
                        eff_map[i] = 1

            block_size *= 2

        # Also do single-byte flips for remaining positions
        for i in range(length):
            if eff_map[i] or exec_count >= max_execs:
                continue

            flipped = bytearray(data)
            flipped[i] ^= 0xFF
            cksum = exec_fn(bytes(flipped))
            exec_count += 1

            if cksum != baseline_cksum:
                eff_map[i] = 1

        log.debug(
            "SkipDet eff map: %d/%d effective bytes (%d execs)",
            sum(eff_map),
            length,
            exec_count,
        )
        return eff_map

    def inference(
        self,
        data: bytes,
        exec_fn,
        max_execs: int = MAX_INF_EXECS,
    ) -> bytearray:
        """Inference stage: find large ineffective ranges via binary search.

        Flips progressively larger blocks starting from each position.
        If a block flip doesn't change the execution path, the entire
        block is marked as ineffective.

        Args:
            data: Input data to analyze.
            exec_fn: Callable(bytes) -> int, returns execution checksum.
            max_execs: Maximum executions before giving up.

        Returns:
            Bytearray of length len(data) where 1 = effective, 0 = skip.
        """
        length = len(data)
        if length < MINIMAL_BLOCK_SIZE * 8:
            # Too short for inference — everything is effective
            return bytearray(length)

        eff_map = bytearray(length)  # all zeros
        exec_count = 0

        baseline_cksum = exec_fn(data)
        exec_count += 1

        pos = 0
        while pos < length - 1 and exec_count < max_execs:
            cur_block = MINIMAL_BLOCK_SIZE
            max_block = length // 8

            while cur_block < max_block and exec_count < max_execs:
                flip_len = min(cur_block, length - 1 - pos)

                flipped = bytearray(data)
                for i in range(pos, pos + flip_len):
                    flipped[i] ^= 0xFF

                cksum = exec_fn(bytes(flipped))
                exec_count += 1

                if cksum == baseline_cksum:
                    # No change — this range is ineffective, try larger
                    cur_block *= 2
                else:
                    # Change detected — stop expanding
                    break

            if cur_block == MINIMAL_BLOCK_SIZE:
                # First flip already changed path — byte is effective
                pos += cur_block
            else:
                # Mark the ineffective half
                skip_len = cur_block // 2
                skip_len = min(skip_len, length - pos)
                # Don't mark in eff_map (it's 0 = skip by default)
                pos += skip_len

        log.debug(
            "SkipDet inference: %d execs, eff_map has %d effective bytes",
            exec_count,
            sum(1 for b in eff_map if b),
        )
        return eff_map
