"""The counts form of Shannon entropy must agree with the probability form.

    H = -sum_i p_i log2 p_i  with p_i = c_i / N
      = log2(N) - (1/N) sum_i c_i log2(c_i)

``byte_entropy_bits`` and ``EdgeTracker.shannon_entropy_seed`` were both
rewritten onto the right-hand side. These tests carry the left-hand side
verbatim as the oracle, so they fail if either rewrite drifts.

Both the numpy and the pure-Python branch of ``byte_entropy_bits`` are
exercised: the fallback is what runs on an install without numpy, and it
was rewritten too.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

import fuzzer_tool.core.byte_entropy as byte_entropy
from fuzzer_tool.core.byte_entropy import ENTROPY_SAMPLE_CAP, byte_entropy_bits
from fuzzer_tool.core.edge_tracker import EdgeTracker

#: AFL's count_class buckets. Real hit counts only ever take these values,
#: which is what makes the bincount step in shannon_entropy_seed cheap.
BUCKETS = (1, 2, 3, 4, 8, 16, 32, 64, 128)


def probability_form(chunk: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> float:
    """The pre-rewrite implementation, kept verbatim as the oracle."""
    chunk = bytes(chunk)[:cap]
    if not chunk:
        return 0.0
    arr = np.frombuffer(chunk, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / arr.size
    ent = float(-np.sum(probs * np.log2(probs)))
    return ent if ent > 0.0 else 0.0


def seed_probability_form(hit_counts: dict) -> float:
    """The pre-rewrite ``shannon_entropy_seed`` body, kept as the oracle."""
    total = sum(hit_counts.values())
    if total == 0:
        return 0.0
    arr = np.fromiter(hit_counts.values(), dtype=np.float64)
    arr = arr[arr > 0] / total
    return -float(np.sum(arr * np.log2(arr)))


def make_tracker(hit_counts: dict) -> EdgeTracker:
    tracker = EdgeTracker.__new__(EdgeTracker)
    tracker.seed_hit_counts = {"k": hit_counts}
    return tracker


class TestByteEntropyIdentity:
    @pytest.mark.parametrize(
        "chunk",
        [
            b"",
            b"A",
            b"A" * ENTROPY_SAMPLE_CAP,
            bytes(range(256)) * 16,
            bytes([0] * 4095 + [1]),
        ],
        ids=["empty", "single-byte", "single-symbol", "uniform", "near-degenerate"],
    )
    def test_edge_cases_match_probability_form(self, chunk):
        assert byte_entropy_bits(chunk) == pytest.approx(probability_form(chunk), abs=1e-12)

    def test_random_inputs_match_probability_form(self):
        rng = random.Random(20260902)
        for _ in range(400):
            alphabet = rng.choice([2, 4, 16, 256])
            length = rng.randrange(1, 6000)
            chunk = bytes(rng.randrange(alphabet) for _ in range(length))
            assert byte_entropy_bits(chunk) == pytest.approx(probability_form(chunk), abs=1e-12)

    def test_fallback_branch_matches_probability_form(self, monkeypatch):
        """The no-numpy branch was rewritten too and has its own oracle gap."""
        monkeypatch.setattr(byte_entropy, "_HAS_NUMPY", False)
        rng = random.Random(7)
        for _ in range(120):
            chunk = bytes(
                rng.randrange(rng.choice([2, 16, 256])) for _ in range(rng.randrange(1, 3000))
            )
            assert byte_entropy_bits(chunk) == pytest.approx(probability_form(chunk), abs=1e-12)

    def test_input_longer_than_table_does_not_fall_off_the_end(self):
        """A caller-supplied cap above the default must grow the LUT, not index past it."""
        rng = random.Random(11)
        big = bytes(rng.randrange(256) for _ in range(ENTROPY_SAMPLE_CAP * 3))
        cap = len(big)
        assert byte_entropy_bits(big, cap=cap) == pytest.approx(
            probability_form(big, cap=cap), abs=1e-12
        )

    def test_single_symbol_is_not_negative_zero(self):
        """The clamp survives the rewrite: callers format this value."""
        value = byte_entropy_bits(b"Z" * 512)
        assert value == 0.0
        assert not str(value).startswith("-")


class TestSeedEntropyIdentity:
    def test_below_numpy_threshold_matches(self):
        rng = random.Random(3)
        for _ in range(200):
            hc = {i: rng.choice(BUCKETS) for i in range(rng.randrange(1, 50))}
            tracker = make_tracker(hc)
            assert tracker.shannon_entropy_seed("k") == pytest.approx(
                seed_probability_form(hc), abs=1e-12
            )

    def test_above_numpy_threshold_matches(self):
        """The vectorized branch bins first; binning must not change the value."""
        rng = random.Random(5)
        for size in (51, 500, 8189):
            hc = {i: rng.choice(BUCKETS) for i in range(size)}
            tracker = make_tracker(hc)
            assert tracker.shannon_entropy_seed("k") == pytest.approx(
                seed_probability_form(hc), abs=1e-12
            )

    def test_unbucketed_counts_still_match(self):
        """Nothing in the rewrite may depend on counts being count_class values."""
        rng = random.Random(13)
        hc = {i: rng.randrange(1, 4096) for i in range(600)}
        tracker = make_tracker(hc)
        assert tracker.shannon_entropy_seed("k") == pytest.approx(
            seed_probability_form(hc), abs=1e-12
        )

    def test_zero_counts_are_dropped_not_logged(self):
        """c=0 contributes nothing; the bincount path must not take log2(0)."""
        hc = {0: 0, 1: 0, 2: 8, 3: 4}
        hc.update({i: 0 for i in range(4, 80)})
        tracker = make_tracker(hc)
        value = tracker.shannon_entropy_seed("k")
        assert np.isfinite(value)
        assert value == pytest.approx(seed_probability_form(hc), abs=1e-12)

    def test_all_zero_counts_returns_zero(self):
        tracker = make_tracker({i: 0 for i in range(80)})
        assert tracker.shannon_entropy_seed("k") == 0.0

    def test_missing_seed_returns_zero(self):
        tracker = make_tracker({1: 4})
        assert tracker.shannon_entropy_seed("absent") == 0.0

    @pytest.mark.parametrize("count", [1, 2, 3, 4, 8, 16, 32, 64, 128, 4095])
    def test_single_edge_is_exactly_zero(self, count):
        """H = 0 is where the counts form loses its absolute precision.

        log2(T) - (1/T) sum c log2 c is a difference of two log2(T)-sized
        quantities, so its error floor is eps*log2(T), not eps*H. At H = 0
        that error floor *is* the value: without the clamp a single-edge
        seed comes back at -4.4e-16, where the probability form returned an
        exact zero. approx() would wave this through, so it is asserted
        exactly.
        """
        tracker = make_tracker({42: count})
        assert tracker.shannon_entropy_seed("k") == 0.0

    def test_uniform_seed_entropy_is_never_negative(self):
        """Same boundary reached from the vectorized branch."""
        for size in (51, 500, 8189):
            tracker = make_tracker(dict.fromkeys(range(size), 8))
            value = tracker.shannon_entropy_seed("k")
            assert value >= 0.0
            assert value == pytest.approx(
                seed_probability_form({i: 8 for i in range(size)}), abs=1e-12
            )
