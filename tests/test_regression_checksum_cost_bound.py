"""Regression tests: unbounded pair growth made checksum recovery the
fuzzer's real bottleneck, collapsing eps to single digits.

Live-caught with py-spy on `fuzzer-tool fuzz ... --cmplog --report`: three
stack samples 15s apart all showed fuzz_one() blocked inside the exact
same call --
    add_pairs -> _maybe_recover -> _recover -> recover_polynomial_gcd
    -> poly_gcd / _poly_mod
recover_polynomial_gcd builds one Python int per (data, checksum) pair
(the data bit-shifted left by the checksum width) and reduces every pair
against a running result via poly_gcd, whose cost scales with operand
bit-length. For a PNG IDAT chunk, "data" is the whole chunk payload --
potentially several KB, i.e. tens of thousands of bits per syndrome.

Two compounding bugs, both fixed in checksum_learner.py:

1. self._pairs had no cap (unlike core/cmplog.py's CMPLOG_PAIRS_MAX for
   the exact same kind of pair pool) -- it grew for the life of the run,
   so every recovery attempt's cost grew right along with it.
2. The retry gate added in an earlier fix (2f0b657) only skipped
   *literally unchanged* pair counts. During active fuzzing with cmplog
   and format extraction on, a "new" pair shows up almost every
   iteration, so that gate was true almost every time -- for an
   unverifiable pair set (one that never satisfies _verify(), e.g. PNG
   CRCs with a mismatched init/final_xor), this meant full GCD recovery
   re-ran on essentially every iteration, at ever-growing cost.

CHECKSUM_PAIRS_MAX bounds (1); RECOVERY_RETRY_BATCH bounds (2) by
requiring real batches of new evidence, not "any change", before
retrying.
"""

from __future__ import annotations

import time

from fuzzer_tool.core.checksum_learner import (
    CHECKSUM_PAIRS_MAX,
    RECOVERY_RETRY_BATCH,
    ChecksumLearner,
)


class _FakeFuzzer:
    def __init__(self):
        self._cmplog = None


# Same 4KB-ish "data" span repeated with different checksums: never
# verifies (no single polynomial reproduces mismatched pairs for the same
# data), so recovery always fails and always retries -- worst case for
# both bugs. Mirrors a PNG IDAT chunk: several KB of chunk data feeding
# recover_polynomial_gcd's per-pair syndrome construction.
_BIG_DATA = b"\x37" * 4000


def _unverifiable_pair(i: int) -> tuple[bytes, int]:
    return (_BIG_DATA, i)


class TestPairListIsCapped:
    def test_pairs_never_exceed_the_cap(self):
        f = _FakeFuzzer()
        learner = ChecksumLearner(f, min_pairs=8, poly_width=32)
        # Feed far more than the cap, in small batches (as add_pairs is
        # actually called -- a handful of pairs per fuzz_one()).
        for batch_start in range(0, CHECKSUM_PAIRS_MAX * 5, 4):
            learner.add_pairs(
                [_unverifiable_pair(i) for i in range(batch_start, batch_start + 4)]
            )
        assert len(learner._pairs) <= CHECKSUM_PAIRS_MAX

    def test_cap_keeps_the_most_recent_pairs(self):
        """FIFO eviction: oldest pairs drop first, newest survive."""
        f = _FakeFuzzer()
        learner = ChecksumLearner(f, min_pairs=8, poly_width=32)
        total = CHECKSUM_PAIRS_MAX + 50
        learner.add_pairs([_unverifiable_pair(i) for i in range(total)])
        checksums = {c for _, c in learner._pairs}
        # The most recent CHECKSUM_PAIRS_MAX checksums (highest i values)
        # must be present; the earliest ones must have been evicted.
        assert max(checksums) == total - 1
        assert min(checksums) == total - CHECKSUM_PAIRS_MAX


class TestRetryBatching:
    def test_small_trickle_of_new_pairs_does_not_retrigger(self, monkeypatch):
        """The actual eps-collapse scenario: cmplog/format extraction
        finds a handful of new pairs almost every fuzz_one() call. That
        must not mean full recovery re-runs almost every call."""
        f = _FakeFuzzer()
        learner = ChecksumLearner(f, min_pairs=8, poly_width=32)
        recover_calls = [0]
        original = ChecksumLearner._recover

        def counting(self):
            recover_calls[0] += 1
            return original(self)

        monkeypatch.setattr(ChecksumLearner, "_recover", counting)

        learner.add_pairs([_unverifiable_pair(i) for i in range(10)])
        assert learner.ensure_poly() is None
        first_attempt_calls = recover_calls[0]
        assert first_attempt_calls >= 1

        # Simulate ~100 fuzz iterations each finding 1-2 new pairs --
        # realistic cmplog/format-extraction trickle, well under
        # RECOVERY_RETRY_BATCH per call.
        next_i = 10
        for _ in range(100):
            learner.add_pairs([_unverifiable_pair(next_i)])
            next_i += 1
            learner.ensure_poly()

        # Total new pairs added across the trickle is comparable to
        # RECOVERY_RETRY_BATCH, so at most a couple of retries should
        # have fired -- not one per iteration.
        assert recover_calls[0] - first_attempt_calls <= (100 // RECOVERY_RETRY_BATCH) + 1

    def test_real_batch_does_retrigger(self):
        """Sanity: the gate isn't just permanently closed -- crossing the
        threshold does re-attempt recovery."""
        f = _FakeFuzzer()
        learner = ChecksumLearner(f, min_pairs=8, poly_width=32)
        learner.add_pairs([_unverifiable_pair(i) for i in range(10)])
        attempted_at_1 = learner._pairs_attempted_at
        learner.add_pairs([_unverifiable_pair(i) for i in range(10, 10 + RECOVERY_RETRY_BATCH)])
        assert learner._pairs_attempted_at > attempted_at_1


class TestGcdPairSizeFilter:
    def test_large_data_pairs_excluded_from_gcd_but_recovery_still_works(self):
        """A mix of huge-data pairs (excluded from the GCD path) and
        small ones (included) must still recover correctly from the
        small ones -- the filter shouldn't silently break recovery,
        just keep it off the expensive pairs."""
        from fuzzer_tool.core.berlekamp_massey import compute_checksum
        from fuzzer_tool.core.crc32 import set_active_model

        f = _FakeFuzzer()
        learner = ChecksumLearner(f, min_pairs=4, poly_width=8)
        custom_poly = 0x1D  # x^8 + x^4 + x^3 + x^2 + 1

        small_good = [
            (bytes([i]), compute_checksum(bytes([i]), poly=custom_poly, width=8))
            for i in range(4, 12)
        ]
        huge_noise = [(b"\xAB" * 4000, 0xDEAD + i) for i in range(4)]

        try:
            learner.add_pairs(huge_noise + small_good)
            poly = learner.ensure_poly()

            assert poly is not None
            assert compute_checksum(bytes([200]), poly=poly, width=8) == compute_checksum(
                bytes([200]), poly=custom_poly, width=8
            )
        finally:
            # Successful recovery activates a global crc32 model
            # (core/crc32.py set_active_model) -- reset so other tests
            # see the standard model.
            set_active_model(None)


class TestRecoveryCostBounded:
    def test_repeated_unverifiable_batches_stay_fast(self):
        """End-to-end timing sanity: hammering an unverifiable learner
        with realistic-sized data must not blow up wall-clock time. This
        is the actual regression -- a single _recover() call on an
        uncapped pair set of big-data pairs took 30+ seconds live."""
        f = _FakeFuzzer()
        learner = ChecksumLearner(f, min_pairs=8, poly_width=32)

        start = time.monotonic()
        next_i = 0
        for _ in range(300):  # far more fuzz_one()-equivalent calls than before
            learner.add_pairs([_unverifiable_pair(next_i), _unverifiable_pair(next_i + 1)])
            next_i += 2
            learner.ensure_poly()
        elapsed = time.monotonic() - start

        # Generous ceiling: this used to take tens of seconds for far
        # fewer iterations once the pair set grew (43 iterations in 45s,
        # measured against the pre-fix code); bounded pairs + real
        # batching + the GCD data-size filter should keep 300 iterations
        # comfortably under this.
        assert elapsed < 15.0
