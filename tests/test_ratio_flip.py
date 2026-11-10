"""zzuf-style ratio flip (``core/mutations/ratio_flip.py``).

Each call draws a ratio ``r`` log-uniform in its bucket and flips
``Binomial(8n, r)`` bits (floor ``MIN_FLIPS``). Oracles: an analytic
log-uniform mean, and a bit-by-bit XOR loop. Each carries a control it must
reject.
"""

import math

import pytest

import fuzzer_tool.core.mutations.ratio_flip as R
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.scripted_rng import ScriptedRng
from .test_regression_bithacks import _MockFuzzer

LO, HI = 1e-3, 1e-2
DRAWS = 4000


class _Spy(ScriptedRng):
    """ScriptedRng that scripts ``randint_list`` positions and logs binomial args."""

    def __init__(self, positions=(), uniform=0.0, k=1):
        super().__init__(randoms=[uniform], binomial_arrays=[[k]])
        self._positions = list(positions)
        self.calls = []

    def binomial_array(self, counts, p):
        self.calls.append((list(counts), p))
        return super().binomial_array(counts, p)

    def randint_list(self, _a, _b, count):
        assert count == len(self._positions)
        return self._positions


class _PoolSpy(RandPool):
    """Real pool that records the (count, p) of every binomial draw."""

    def __init__(self, seed):
        super().__init__(seed=seed)
        self.calls = []

    def binomial_array(self, counts, p):
        out = super().binomial_array(counts, p)
        self.calls.append((int(counts[0]), p, int(out[0])))
        return out


def _xor_oracle(data, positions):
    out = bytearray(data)
    for pos in positions:
        out[pos // 8] ^= 1 << (pos % 8)
    return bytes(out)


def _popcount(a, b):
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=True))


class TestRegistration:
    @pytest.mark.parametrize("name", sorted(R.RATIO_BUCKETS))
    def test_arm_registered_in_bit_band(self, name):
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "bit"
        assert name in OPERATOR_CATEGORIES["bit"]

    def test_buckets_tile_the_range(self):
        # One decade each, contiguous, 1e-5 .. 1e-1.
        spans = sorted(R.RATIO_BUCKETS.values())
        assert spans[0][0] == pytest.approx(1e-5)
        assert spans[-1][1] == pytest.approx(1e-1)
        for (_, hi), (lo, _) in zip(spans, spans[1:], strict=False):
            assert hi == pytest.approx(lo)

    @pytest.mark.parametrize("name", sorted(R.RATIO_BUCKETS))
    def test_handler_preserves_length(self, name):
        fuzzer = _MockFuzzer()
        fuzzer._rng = RandPool(seed=7)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = bytes(range(64))
        out = dispatch[name](bytearray(data), 0, data)
        assert len(out) == len(data)
        assert _popcount(data, out) <= 8 * len(data)


class TestFlip:
    def test_positions_match_xor_oracle(self):
        data = bytes(range(3))
        positions = [0, 9, 17, 23]
        rng = _Spy(positions, k=len(positions))
        assert R.ratio_flip(data, rng, LO, HI) == _xor_oracle(data, positions)

    def test_oracle_rejects_wrong_bit_order(self):
        # Control: MSB-first numbering is a different mask.
        data = bytes(3)
        mask = _xor_oracle(data, [0])
        msb_first = bytes([0x80, 0, 0])
        assert mask != msb_first

    def test_duplicate_positions_cancel(self):
        # zzuf draws with replacement: a bit hit twice is not flipped.
        data = b"\x55\xaa"
        assert R.ratio_flip(data, _Spy([5, 5], k=2), LO, HI) == data

    def test_floor_is_one_flip(self):
        # Binomial draws 0 on short inputs at small r: the floor still flips one bit.
        data = bytes(4)
        out = R.ratio_flip(data, _Spy([13], k=0), LO, HI)
        assert _popcount(data, out) == 1

    def test_count_argument_scales_with_length(self):
        short, long_ = _Spy([0], k=1), _Spy([0], k=1)
        R.ratio_flip(bytes(16), short, LO, HI)
        R.ratio_flip(bytes(1024), long_, LO, HI)
        assert short.calls[0][0] == [16 * 8]
        assert long_.calls[0][0] == [1024 * 8]

    @pytest.mark.parametrize("u", [0.0, 0.5, 0.999])
    def test_ratio_is_log_interpolation(self, u):
        rng = _Spy([0], uniform=u, k=1)
        R.ratio_flip(bytes(8), rng, LO, HI)
        assert rng.calls[0][1] == pytest.approx(LO * (HI / LO) ** u)

    def test_input_not_modified(self):
        data = bytearray(b"\x00" * 8)
        R.ratio_flip(bytes(data), _Spy([3], k=1), LO, HI)
        assert data == bytearray(8)


class TestDistribution:
    @staticmethod
    def _ratios(seed):
        rng = _PoolSpy(seed)
        data = bytes(512)
        for _ in range(DRAWS):
            R.ratio_flip(data, rng, LO, HI)
        return [p for _n, p, _k in rng.calls]

    def test_ratio_mean_matches_log_uniform(self):
        ps = self._ratios(11)
        analytic = (HI - LO) / math.log(HI / LO)
        assert sum(ps) / DRAWS == pytest.approx(analytic, rel=0.05)

    def test_median_is_geometric_mean(self):
        # Control for the mean test: a uniform-in-r draw has mean (LO+HI)/2,
        # 40% above the log-uniform mean, so the check above can fail.
        ps = self._ratios(11)
        below = sum(p < math.sqrt(LO * HI) for p in ps) / DRAWS
        assert below == pytest.approx(0.5, abs=0.03)
        assert 1.2 * (HI - LO) / math.log(HI / LO) < (LO + HI) / 2

    def test_seed_to_seed_control(self):
        # Rule 46: the reference against a second run of itself.
        a, b = self._ratios(11), self._ratios(12)
        assert sum(a) / DRAWS == pytest.approx(sum(b) / DRAWS, rel=0.05)

    def test_flip_count_mean_is_ratio_times_bits(self):
        rng = _PoolSpy(5)
        data = bytes(512)
        for _ in range(DRAWS):
            R.ratio_flip(data, rng, LO, HI)
        nbits = 512 * 8
        want = sum(nbits * p for _n, p, _k in rng.calls) / DRAWS
        got = sum(k for _n, _p, k in rng.calls) / DRAWS
        assert got == pytest.approx(want, rel=0.05)


class TestAdversarial:
    def test_empty_input_declines(self):
        assert R.ratio_flip(b"", ScriptedRng(), LO, HI) is None

    def test_one_byte_input(self):
        out = R.ratio_flip(b"\x00", _Spy([7], k=1), LO, HI)
        assert out == b"\x80"

    @pytest.mark.parametrize("name", sorted(R.RATIO_BUCKETS))
    def test_flip_parity_matches_count(self, name):
        # Every flip toggles one mask bit; duplicates cancel in pairs, so
        # popcount(mask) and the draw count agree mod 2. Not a luck test.
        lo, hi = R.RATIO_BUCKETS[name]
        rng = _PoolSpy(3)
        data = bytes(range(256)) * 4
        for _ in range(200):
            out = R.ratio_flip(data, rng, lo, hi)
            k = max(R.MIN_FLIPS, rng.calls[-1][2])
            assert len(out) == len(data)
            assert _popcount(data, out) % 2 == k % 2
            assert _popcount(data, out) <= k

    def test_max_ratio_bucket_on_large_input_stays_bounded(self):
        # 64 KiB at r <= 0.1: at most ~52k flips, still one array pass.
        rng = _PoolSpy(9)
        data = bytes(65536)
        out = R.ratio_flip(data, rng, 1e-2, 1e-1)
        assert len(out) == len(data)
        assert rng.calls[0][0] == 65536 * 8
