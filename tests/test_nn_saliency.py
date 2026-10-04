"""Covers core/nn_saliency.py: closed-form gradient, fit, planted-signal recovery."""

import numpy as np
import pytest

from fuzzer_tool.core.nn_saliency import TinyMLP, encode_inputs


def _planted(n=200, length=100, seed=0):
    r = np.random.default_rng(seed)
    seeds = [bytes(r.integers(0, 256, length, dtype=np.uint8)) for _ in range(n)]
    y = np.array([[s[7] > 128, s[40] < 60, s[7] > 128 and s[40] < 60] for s in seeds], np.float32)
    return seeds, y


class TestEncode:
    def test_scales_pads_and_truncates(self):
        x = encode_inputs([b"\xff\x00", b"\x01\x02\x03\x04"], 3)
        assert x.shape == (2, 3) and x.dtype == np.float32
        assert x[0].tolist() == [1.0, 0.0, 0.0]
        assert x[1, 2] == pytest.approx(3 / 255)

    def test_empty_seed_is_zeros(self):
        assert not encode_inputs([b""], 4).any()


class TestGradient:
    def test_matches_finite_differences(self):
        seeds, y = _planted()
        x = encode_inputs(seeds, 100)
        m = TinyMLP(100, 16, 3, np.random.default_rng(1))
        m.fit(x, y, epochs=40)
        x0 = x[3].copy()
        g = m.input_grad(x0, [0, 2])
        for k, col in enumerate((0, 2)):
            for i in (7, 40, 5):
                e = 1e-3
                xp, xm = x0.copy(), x0.copy()
                xp[i] += e
                xm[i] -= e
                num = (m.logits(xp[None])[0, col] - m.logits(xm[None])[0, col]) / (2 * e)
                assert g[i, k] == pytest.approx(num, rel=2e-2, abs=1e-2)

    def test_saliency_is_nonnegative_mean_abs(self):
        m = TinyMLP(8, 4, 2, np.random.default_rng(0))
        x = np.random.default_rng(1).random(8).astype(np.float32)
        s = m.saliency(x, [0, 1])
        assert s.shape == (8,) and (s >= 0).all()
        assert np.allclose(s, np.abs(m.input_grad(x, [0, 1])).mean(axis=1))


class TestFit:
    def test_loss_drops_and_bias_starts_at_base_rate(self):
        seeds, y = _planted()
        x = encode_inputs(seeds, 100)
        m = TinyMLP(100, 32, 3, np.random.default_rng(1))
        first = m.fit(x, y, epochs=1)
        assert np.allclose(1 / (1 + np.exp(-m.b2)), y.mean(0), atol=0.05)
        m2 = TinyMLP(100, 32, 3, np.random.default_rng(1))
        assert m2.fit(x, y, epochs=150) < first

    def test_recovers_planted_bytes_above_uniform(self):
        seeds, y = _planted()
        x = encode_inputs(seeds, 100)
        m = TinyMLP(100, 64, 3, np.random.default_rng(1))
        m.fit(x, y, epochs=150, lr=3e-2)
        r = np.random.default_rng(9)
        mass = []
        for _ in range(20):
            xs = encode_inputs([bytes(r.integers(0, 256, 100, dtype=np.uint8))], 100)[0]
            g = m.saliency(xs, [0, 1, 2])
            mass.append((g[7] + g[40]) / g.sum())
        assert np.mean(mass) > 3 * (2 / 100)  # measured ~0.13 vs 0.02 uniform

    def test_deterministic_given_generator(self):
        seeds, y = _planted(n=50)
        x = encode_inputs(seeds, 100)
        a = TinyMLP(100, 8, 3, np.random.default_rng(5))
        b = TinyMLP(100, 8, 3, np.random.default_rng(5))
        assert a.fit(x, y, epochs=10) == b.fit(x, y, epochs=10)
        assert np.array_equal(a.w1, b.w1)

    def test_rejects_bad_shapes(self):
        m = TinyMLP(4, 2, 2, np.random.default_rng(0))
        with pytest.raises(ValueError):
            m.fit(np.zeros((0, 4), np.float32), np.zeros((0, 2), np.float32))
        with pytest.raises(ValueError):
            m.fit(np.zeros((3, 4), np.float32), np.zeros((3, 5), np.float32))
        with pytest.raises(ValueError):
            TinyMLP(0, 1, 1, np.random.default_rng(0))
