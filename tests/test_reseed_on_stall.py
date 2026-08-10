"""Tests for RNG reseeding on stall recovery (``--reseed-on-stall``)."""

import subprocess
import sys
import tempfile
from array import array
from unittest.mock import patch

from fuzzer_tool.core.rand_pool import _POOL_ENTRIES, RandPool


def _make_fuzzer(**kwargs):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=256,
        timeout=1,
        mutations_per_input=2,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(**defaults)


def _stalled_fuzzer(**kwargs):
    """Return a fuzzer primed so that the next stall check confirms a stall."""
    f = _make_fuzzer(**kwargs)
    f._stall_recovery_active = False
    f._stall_threshold = 100
    f.exec_count = 500
    f._last_new_edge_exec = 100
    f._entropy_execs = array("Q", (100, 200, 300, 400))
    f._entropy_vals = array("d", (1.5, 1.5, 1.5, 1.5))
    return f


# ── RandPool.reseed ─────────────────────────────────────────────────────


class TestRandPoolReseed:
    def test_reseed_invalidates_prefetched_pool(self):
        """reseed forces a refill so stale values are never dispensed."""
        p = RandPool()
        p._draw()
        p.reseed(1234)
        assert p._idx == _POOL_ENTRIES

    def test_reseed_switches_stream_immediately(self):
        """The very next draw comes from the new stream, not the old pool.

        Regression guard: reseeding without dropping the pool
        leaves up to _POOL_ENTRIES old-stream values queued, so the reseed
        would silently do nothing for thousands of draws.
        """
        expected = RandPool(seed=1234)
        expected_first = [expected._draw() for _ in range(8)]

        p = RandPool()
        for _ in range(100):  # burn part of a differently-seeded pool
            p._draw()
        p.reseed(1234)
        assert [p._draw() for _ in range(8)] == expected_first

    def test_reseed_is_reproducible(self):
        """Two pools reseeded with the same value agree."""
        a, b = RandPool(), RandPool()
        a.reseed(99)
        b.reseed(99)
        assert [a.randint(0, 255) for _ in range(64)] == [b.randint(0, 255) for _ in range(64)]

    def test_reseed_none_uses_entropy(self):
        """reseed(None) still invalidates the pool."""
        p = RandPool()
        p._refill()
        p.reseed(None)
        assert p._idx == _POOL_ENTRIES
        p._draw()  # must not raise


# ── Seed derivation ─────────────────────────────────────────────────────


class TestDeriveStallSeed:
    def test_seed_in_numpy_range(self):
        f = _make_fuzzer(seed=42)
        for _ in range(16):
            f._stall_reseed_count += 1
            assert 0 <= f._derive_stall_seed() < 2**32

    def test_deterministic_for_same_seed_and_count(self):
        """A seeded run reproduces its reseeds regardless of draw history."""
        a = _make_fuzzer(seed=7)
        b = _make_fuzzer(seed=7)
        a._stall_reseed_count = b._stall_reseed_count = 3
        for _ in range(50):  # perturb a's draw history only
            a._rand_pool._draw()
        assert a._derive_stall_seed() == b._derive_stall_seed()

    def test_consecutive_stalls_differ(self):
        """Each stall gets its own stream."""
        f = _make_fuzzer(seed=7)
        seeds = []
        for i in range(1, 9):
            f._stall_reseed_count = i
            seeds.append(f._derive_stall_seed())
        assert len(set(seeds)) == len(seeds)

    def test_adjacent_run_seeds_diverge(self):
        """Neighbouring --seed values must not map to neighbouring streams."""
        seeds = []
        for run_seed in range(1, 6):
            f = _make_fuzzer(seed=run_seed)
            f._stall_reseed_count = 1
            seeds.append(f._derive_stall_seed())
        assert len(set(seeds)) == len(seeds)
        assert all(abs(seeds[i] - seeds[i + 1]) > 1000 for i in range(len(seeds) - 1))

    def test_unseeded_run_uses_os_entropy(self):
        f = _make_fuzzer(seed=None)
        with patch("os.urandom", return_value=b"\x01\x02\x03\x04") as urandom:
            assert f._derive_stall_seed() == 0x04030201
        urandom.assert_called_once_with(4)


# ── Wiring ──────────────────────────────────────────────────────────────


class TestReseedOnStallWiring:
    def test_disabled_by_default(self):
        f = _make_fuzzer()
        assert f._reseed_on_stall is False
        assert f._stall_reseed_count == 0
        assert f._last_stall_seed is None

    def test_no_reseed_when_disabled(self):
        f = _stalled_fuzzer(reseed_on_stall=False)
        assert f._maybe_trigger_stall_recovery(400) is True
        assert f._stall_reseed_count == 0
        assert f._last_stall_seed is None

    def test_reseeds_when_enabled(self):
        f = _stalled_fuzzer(reseed_on_stall=True, seed=42)
        assert f._maybe_trigger_stall_recovery(400) is True
        assert f._stall_recovery_active is True
        assert f._stall_reseed_count == 1
        assert 0 <= f._last_stall_seed < 2**32

    def test_reseed_changes_the_mutation_stream(self):
        """The pool actually dispenses different values after the stall."""
        f = _stalled_fuzzer(reseed_on_stall=True, seed=42)
        before = [f._rand_pool.randint(0, 255) for _ in range(32)]
        f._maybe_trigger_stall_recovery(400)
        after = [f._rand_pool.randint(0, 255) for _ in range(32)]
        assert before != after

    def test_reseed_applies_to_stdlib_random_too(self):
        import random

        f = _stalled_fuzzer(reseed_on_stall=True, seed=42)
        f._maybe_trigger_stall_recovery(400)
        observed = [random.random() for _ in range(4)]
        random.seed(f._last_stall_seed)
        assert [random.random() for _ in range(4)] == observed

    def test_not_reseeded_when_stall_is_rejected(self):
        """No stall → no reseed, even with the flag on."""
        f = _make_fuzzer(reseed_on_stall=True)
        f._stall_recovery_active = False
        f._entropy_execs = array("Q", (100, 200, 300, 400))
        f._entropy_vals = array("d", (1.0, 1.5, 2.0, 2.5))  # rising: not a stall
        assert f._maybe_trigger_stall_recovery(400) is False
        assert f._stall_reseed_count == 0


class TestReseedOnStallCLI:
    def test_flag_in_help(self):
        """Verify --reseed-on-stall flag exists in fuzz subcommand."""
        result = subprocess.run(
            [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--reseed-on-stall" in result.stdout
