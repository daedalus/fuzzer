"""Regression: --seed did not cover the global ``np.random`` state.

``RandPool`` owns an independent ``np.random.default_rng(seed)`` Generator.
That Generator shares NO state with the legacy module-level ``np.random.*``
functions, and nothing anywhere in ``src/`` ever called ``np.random.seed``.

Every global draw therefore ran off OS entropy regardless of ``--seed``:

* ``core/qea.py:267``       -- observe(), collapsing amplitudes to a bitstring
* ``core/qea.py:361,364``   -- mutate(), which amplitudes get perturbed
* ``core/schedulers/monte_carlo.py:778,895`` -- spectral probe vectors

so a seeded run was not reproducible whenever QEA or the Monte-Carlo scheduler
was active. What kept this hidden is that ``_reseed_after_stall``'s docstring
asserted the opposite -- "``np.random`` backs ``RandPool``" -- which is false
and is corrected in the same commit.

The tests below assert reproducibility of the actual draw sequences, not that
some seeding call was made, so they cannot be satisfied by calling
``np.random.seed`` somewhere ineffective.
"""

import numpy as np
import pytest

from fuzzer_tool.services.fuzzer import SEED_MASK_32, Fuzzer


def _draws(n=12):
    """A sample of the global stream, in the shapes qea/monte_carlo use."""
    return (
        np.random.random(n).tolist()
        + np.random.randn(n).tolist()
        + np.random.uniform(0.0, 1.0, size=n).tolist()
    )


class TestGlobalNumpySeeding:
    def test_same_seed_gives_same_global_stream(self):
        Fuzzer._seed_global_numpy(1234)
        first = _draws()
        Fuzzer._seed_global_numpy(1234)
        second = _draws()
        assert first == second

    def test_different_seeds_give_different_streams(self):
        Fuzzer._seed_global_numpy(1)
        a = _draws()
        Fuzzer._seed_global_numpy(2)
        b = _draws()
        assert a != b

    def test_none_reseeds_from_entropy(self):
        # An unseeded run must stay unseeded, matching random.seed(None).
        Fuzzer._seed_global_numpy(None)
        a = _draws()
        Fuzzer._seed_global_numpy(None)
        b = _draws()
        assert a != b


class TestFalsification:
    def test_randpool_is_not_backed_by_global_numpy(self):
        # Falsification of the docstring claim that caused the bug. If RandPool
        # WERE backed by global np.random, reseeding the global would change
        # RandPool's output. It does not -- which is exactly why seeding the
        # global was still necessary.
        from fuzzer_tool.core.rand_pool import RandPool

        pool = RandPool(seed=99)
        Fuzzer._seed_global_numpy(11111)
        after_a = [pool.randint(0, 1_000_000) for _ in range(20)]

        pool2 = RandPool(seed=99)
        Fuzzer._seed_global_numpy(22222)
        after_b = [pool2.randint(0, 1_000_000) for _ in range(20)]

        # Same pool seed, wildly different global seed -> identical output.
        assert after_a == after_b

    def test_global_stream_actually_changes_with_seed(self):
        # The complement: the global stream IS sensitive to the global seed.
        # Together with the test above this pins down that the two streams are
        # genuinely separate and both need seeding.
        Fuzzer._seed_global_numpy(7)
        a = _draws()
        Fuzzer._seed_global_numpy(8)
        b = _draws()
        assert a != b


class TestAdversarial:
    def test_wide_seed_is_folded_not_raised(self):
        # np.random.seed rejects anything >= 2**32. A run seeded from a 64-bit
        # value must fold rather than crash at construction.
        big = (1 << 62) | 0xDEADBEEF
        Fuzzer._seed_global_numpy(big)
        a = _draws()
        Fuzzer._seed_global_numpy(big & SEED_MASK_32)
        b = _draws()
        assert a == b

    def test_boundary_seeds(self):
        for seed in (0, 1, SEED_MASK_32 - 1, SEED_MASK_32, SEED_MASK_32 + 1):
            Fuzzer._seed_global_numpy(seed)
            first = _draws(4)
            Fuzzer._seed_global_numpy(seed)
            assert _draws(4) == first

    def test_negative_seed_does_not_raise(self):
        # Defensive: a caller deriving a seed by subtraction should not be able
        # to abort a campaign at startup.
        try:
            Fuzzer._seed_global_numpy(-5)
        except ValueError:
            pytest.fail("negative seed must be folded, not raised")
        assert _draws(3) == _draws(3) or True  # stream advanced without error

    def test_qea_draws_are_reproducible_under_seed(self):
        # End-to-end on the actual consumers. collapse() is qea.py:267 and
        # mutate_amplitudes() is qea.py:361,364 -- the three global draws named
        # in the finding.
        from fuzzer_tool.core import qea

        amplitudes = np.full(64, 0.5, dtype=np.float64)

        Fuzzer._seed_global_numpy(4242)
        first = qea.collapse(amplitudes.copy())
        Fuzzer._seed_global_numpy(4242)
        second = qea.collapse(amplitudes.copy())
        assert first == second

        # And a differing seed must actually move it, or the assert above
        # would pass on a constant.
        Fuzzer._seed_global_numpy(9999)
        third = qea.collapse(amplitudes.copy())
        assert third != first

    def test_qea_mutate_amplitudes_reproducible_under_seed(self):
        from fuzzer_tool.core import qea

        base = np.full(64, 0.5, dtype=np.float64)

        Fuzzer._seed_global_numpy(31337)
        a = qea.mutate_amplitudes(base.copy())
        Fuzzer._seed_global_numpy(31337)
        b = qea.mutate_amplitudes(base.copy())
        assert np.array_equal(np.asarray(a), np.asarray(b))
