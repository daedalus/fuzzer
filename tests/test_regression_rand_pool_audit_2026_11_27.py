"""Regression tests for the 2026-11-27 rand_pool audit (R-01 .. R-08)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core import rand_pool
from fuzzer_tool.core.rand_pool import _MODULO_SAFE_MAX, RandPool


class TestR01DefaultSeed:
    def test_resolve_seed_keeps_explicit(self):
        assert commands.resolve_seed(7) == 7
        assert commands.resolve_seed(0) == 0

    def test_resolve_seed_none_is_fresh(self):
        seeds = {commands.resolve_seed(None) for _ in range(8)}
        assert len(seeds) > 1
        assert all(0 <= s < 2**63 for s in seeds)

    def test_fuzz_seed_default_is_none(self):
        ns = commands._build_parser().root.parse_args(["fuzz", "t"])
        assert ns.seed is None
        assert commands._build_parser().root.parse_args(["fuzz", "t", "-s", "9"]).seed == 9

    def test_pool_stays_deterministic(self):
        assert RandPool(5).randbytes(16) == RandPool(5).randbytes(16)
        assert RandPool().randbytes(16) == RandPool(rand_pool._NO_SEED_FALLBACK).randbytes(16)


class TestR02ModuloBias:
    def test_threshold_bounds_relative_bias(self):
        # relative bias of the over-weighted values is 1/(2**32 // n)
        assert 1 / (2**32 // _MODULO_SAFE_MAX) <= 2.5e-4

    def test_documented_bound_matches_code(self):
        assert _MODULO_SAFE_MAX == 1 << 20

    def test_wide_range_unbiased(self):
        n = 954_437_177  # q = 4 zone: 25% bias under pure modulo
        r = 2**32 % n
        p = RandPool(seed=1234)
        d = np.array([p.randint(0, n - 1) for _ in range(40000)])
        frac = float((d < r).mean())
        assert abs(frac - r / n) < 0.02  # uniform ~0.5, biased would be ~0.556

    def test_wide_list_form_unbiased(self):
        n = 954_437_177
        r = 2**32 % n
        d = np.array(RandPool(seed=99).randint_list(0, n - 1, 40000))
        assert abs(float((d < r).mean()) - r / n) < 0.02


class TestR03Categorical:
    @pytest.mark.parametrize("w", [[3.0, -1.0, 3.0], [1.0, float("nan")], [1.0, float("inf")]])
    def test_invalid_weights_raise(self, w):
        with pytest.raises(ValueError):
            RandPool(seed=2).categorical(np.array(w), 10)

    def test_valid_weights_unchanged(self):
        out = RandPool(seed=2).categorical(np.array([1.0, 0.0, 3.0]), 2000)
        counts = np.bincount(out, minlength=3)
        assert counts[1] == 0 and counts[2] > counts[0]

    def test_list_input_and_old_errors(self):
        assert RandPool(1).categorical([0.0, 1.0], 3) == [1, 1, 1]
        with pytest.raises(IndexError):
            RandPool(1).categorical([], 3)
        with pytest.raises(IndexError):
            RandPool(1).categorical([0.0, 0.0], 3)


class TestR04InjectEntropy:
    def test_draw_position_enters_the_mix(self):
        a, b = RandPool(seed=3), RandPool(seed=3)
        a.randint(0, 255)  # same batch, different position
        for _ in range(40):
            b.randint(0, 255)
        a.inject_entropy(b"xyz")
        b.inject_entropy(b"xyz")
        assert a.randbytes(32) != b.randbytes(32)

    def test_still_deterministic(self):
        a, b = RandPool(seed=3), RandPool(seed=3)
        for _ in range(10):
            a.randint(0, 255)
            b.randint(0, 255)
        a.inject_entropy(b"xyz")
        b.inject_entropy(b"xyz")
        assert a.randbytes(32) == b.randbytes(32)

    def test_empty_is_noop(self):
        a, b = RandPool(seed=3), RandPool(seed=3)
        a.inject_entropy(b"")
        assert a.randbytes(32) == b.randbytes(32)


class TestR05ThreadCheck:
    def test_second_thread_refill_raises_when_enabled(self, monkeypatch):
        monkeypatch.setattr(rand_pool, "_THREAD_CHECK", True)
        p = RandPool(seed=1)
        p.randint(0, 9)  # owner = this thread
        err: list[BaseException] = []

        def work():
            try:
                p.randbytes(4096 * 4 + 1)  # forces a refill
            except RuntimeError as e:
                err.append(e)

        t = threading.Thread(target=work)
        t.start()
        t.join()
        assert err and "not thread-safe" in str(err[0])

    def test_disabled_by_default_path_unchanged(self, monkeypatch):
        monkeypatch.setattr(rand_pool, "_THREAD_CHECK", False)
        p = RandPool(seed=1)
        p.randint(0, 9)
        out: list[bytes] = []
        t = threading.Thread(target=lambda: out.append(p.randbytes(8)))
        t.start()
        t.join()
        assert len(out[0]) == 8

    def test_handoff_before_first_draw_is_fine(self, monkeypatch):
        monkeypatch.setattr(rand_pool, "_THREAD_CHECK", True)
        p = RandPool(seed=1)
        out: list[int] = []
        t = threading.Thread(target=lambda: out.append(p.randint(0, 9)))
        t.start()
        t.join()
        assert out


class TestR06R07R08:
    def test_list_forms_empty_range_documented(self):
        p = RandPool(seed=1)
        assert p.randrange_list(0, 5) == [] and p.randint_list(5, 3, 5) == []
        assert "randrange" in (RandPool.randrange_list.__doc__ or "")
        assert "k > n" in (RandPool.sample.__doc__ or "")

    def test_random_resolution_documented(self):
        assert "2**-32" in (RandPool.random.__doc__ or "")

    def test_configure_floyd_default_enables(self):
        old = rand_pool.rand_floyd_enabled()
        try:
            rand_pool.configure_rand_floyd(False)
            rand_pool.configure_rand_floyd()
            assert rand_pool.rand_floyd_enabled() is True
        finally:
            rand_pool.configure_rand_floyd(old)
