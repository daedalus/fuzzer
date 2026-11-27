"""Tests for the turbo_fix operator (core/mutations/turbo_fix.py)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations.turbo_fix import (
    DEFAULT_F1,
    DEFAULT_F2,
    DEFAULT_K,
    TAIL_BITS,
    Mode,
    check_params,
    encode,
    parse_qpp,
    turbo_fix,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

K = DEFAULT_K
N = 3 * K + TAIL_BITS
# Second and last LTE table rows; check_params verifies them independently.
K48 = (48, 7, 12)
K6144 = (6144, 263, 480)


# --- independent oracle: shift registers straight from TS 36.212 5.1.3.2 -------


def _qpp(k: int, f1: int, f2: int) -> list[int]:
    return [(f1 * i + f2 * i * i) % k for i in range(k)]


def _rsc(bits: list[int]) -> tuple[list[int], list[int], list[int]]:
    """One constituent encoder: (parity z, tail x, tail z) with s1,s2,s3 = D, D^2, D^3."""
    s1 = s2 = s3 = 0
    z = []
    for c in bits:
        a = c ^ s2 ^ s3  # g0 = 1 + D^2 + D^3 feedback
        z.append(a ^ s1 ^ s3)  # g1 = 1 + D + D^3
        s1, s2, s3 = a, s1, s2
    tx, tz = [], []
    for _ in range(3):
        x = s2 ^ s3  # switch down: input equals feedback, so a = 0
        tx.append(x)
        tz.append(s1 ^ s3)
        s1, s2, s3 = 0, s1, s2
    return z, tx, tz


def _ref_encode(
    info: list[int], k: int = K, f1: int = DEFAULT_F1, f2: int = DEFAULT_F2
) -> list[int]:
    """[x K][z K][z' K][x_K..K+2][z_K..K+2][x'_K..K+2][z'_K..K+2]."""
    pi = _qpp(k, f1, f2)
    z1, tx1, tz1 = _rsc(info)
    z2, tx2, tz2 = _rsc([info[pi[i]] for i in range(k)])
    return list(info) + z1 + z2 + tx1 + tz1 + tx2 + tz2


def _end_state(bits: list[int]) -> tuple[int, int, int]:
    """Register state after feeding *bits* (data then tail inputs) to one RSC."""
    s1 = s2 = s3 = 0
    for c in bits:
        a = c ^ s2 ^ s3
        s1, s2, s3 = a, s1, s2
    return s1, s2, s3


def _to_bits(data: bytes) -> list[int]:
    return [(b >> (7 - j)) & 1 for b in data for j in range(8)]


def _to_bytes(bits: list[int]) -> bytes:
    pad = bits + [0] * (-len(bits) % 8)
    return bytes(sum(pad[i + j] << (7 - j) for j in range(8)) for i in range(0, len(pad), 8))


def _info(k: int = K, seed: int = 0) -> list[int]:
    return [((i * 37 + seed) * 11 >> 3) & 1 for i in range(k)]


def _flip(bits: list[int], positions) -> list[int]:
    out = list(bits)
    for p in positions:
        out[p] ^= 1
    return out


def _buf(cw: list[int], trailer: list[int]) -> bytes:
    return _to_bytes(cw + trailer)


TRAILER = [1, 0, 1, 1]  # 132 + 4 bits = 17 bytes exactly


class _BoundedRng(ScriptedRng):
    """ScriptedRng whose randint() rejects draws outside the caller's bounds."""

    def randint(self, a, b):
        v = super().randint(a, b)
        assert a <= v <= b, (a, v, b)
        return v


class TestReference:
    def test_control_reference_against_itself(self):
        info = _info()
        assert _ref_encode(info) == _ref_encode(list(info))

    def test_control_all_zero_info_is_all_zero_codeword(self):
        assert _ref_encode([0] * K) == [0] * N
        assert list(encode([0] * K, K, DEFAULT_F1, DEFAULT_F2)) == [0] * N

    def test_reference_tail_terminates_both_encoders(self):
        info = _info()
        cw = _ref_encode(info)
        x1, x2 = cw[3 * K : 3 * K + 3], cw[3 * K + 6 : 3 * K + 9]
        inter = [info[p] for p in _qpp(K, DEFAULT_F1, DEFAULT_F2)]
        assert _end_state(info + x1) == (0, 0, 0)
        assert _end_state(inter + x2) == (0, 0, 0)


class TestEncoder:
    @pytest.mark.parametrize("params", [(DEFAULT_K, DEFAULT_F1, DEFAULT_F2), K48, K6144])
    @pytest.mark.parametrize("seed", [0, 1, 5])
    def test_matches_reference(self, params, seed):
        info = _info(params[0], seed)
        assert list(encode(info, *params)) == _ref_encode(info, *params)

    def test_single_impulses_match_reference(self):
        for pos in (0, 1, 2, 3, K - 3, K - 1):
            info = [0] * K
            info[pos] = 1
            assert list(encode(info, K, DEFAULT_F1, DEFAULT_F2)) == _ref_encode(info)

    def test_random_info_matches_reference(self):
        rng = RandPool(seed=11)
        for _ in range(20):
            info = [rng.randint(0, 1) for _ in range(K)]
            assert list(encode(info, K, DEFAULT_F1, DEFAULT_F2)) == _ref_encode(info)

    def test_falsification_wrong_interleaver_differs(self):
        info = _info()
        assert _ref_encode(info, K, 3, 10) != _ref_encode(info, K, 13, 10)
        assert list(encode(info, K, 13, 10)) != _ref_encode(info)

    def test_falsification_flipped_parity_bit_is_not_the_encoding(self):
        info = _info()
        cw = list(encode(info, K, DEFAULT_F1, DEFAULT_F2))
        assert cw == _ref_encode(info)
        for pos in (K, 2 * K + 7, 3 * K + 11):
            assert _flip(cw, [pos]) != _ref_encode(info)

    def test_linearity(self):
        a, b = _info(seed=2), _info(seed=9)
        x = [p ^ q for p, q in zip(a, b, strict=True)]
        ea = encode(a, K, DEFAULT_F1, DEFAULT_F2)
        eb = encode(b, K, DEFAULT_F1, DEFAULT_F2)
        assert list(encode(x, K, DEFAULT_F1, DEFAULT_F2)) == list(ea ^ eb)

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            encode([0] * (K - 1), K, DEFAULT_F1, DEFAULT_F2)


class TestParams:
    @pytest.mark.parametrize(
        ("k", "f1", "f2"),
        [(39, 3, 10), (6145, 3, 10), (0, 1, 0), (-40, 3, 10), (40, 4, 10), (40, 0, 0), (40, 2, 20)],
    )
    def test_check_params_rejects(self, k, f1, f2):
        with pytest.raises(ValueError):
            check_params(k, f1, f2)

    @pytest.mark.parametrize("params", [(DEFAULT_K, DEFAULT_F1, DEFAULT_F2), K48, K6144])
    def test_check_params_accepts_lte_rows(self, params):
        check_params(*params)

    def test_check_params_agrees_with_plain_permutation_test(self):
        """Falsification: every (f1, f2) < 40 at K=40 matches a set-based oracle."""
        for f1 in range(0, 40, 3):
            for f2 in range(0, 40, 7):
                perm = sorted(_qpp(40, f1, f2)) == list(range(40))
                if perm:
                    check_params(40, f1, f2)
                    continue
                with pytest.raises(ValueError):
                    check_params(40, f1, f2)

    def test_parse_qpp(self):
        assert parse_qpp("3,10") == (3, 10)
        assert parse_qpp(" 263 , 480 ") == (263, 480)

    @pytest.mark.parametrize("bad", ["", "3", "3,", "a,b", "1,2,3", "3;10"])
    def test_parse_qpp_rejects(self, bad):
        with pytest.raises(ValueError):
            parse_qpp(bad)


class TestFix:
    def _run(self, cw: list[int], draws: list[int]) -> list[int]:
        out = turbo_fix(_buf(cw, TRAILER), _BoundedRng(randints=draws), K, DEFAULT_F1, DEFAULT_F2)
        assert out is not None and len(out) == 17
        bits = _to_bits(out)
        assert bits[N:] == TRAILER
        return bits[:N]

    def test_modes_cover_draw_values(self):
        assert [int(m) for m in Mode] == list(range(len(Mode)))

    def test_scripted_repair_flips_and_reencodes(self):
        info = _info()
        # codeword 0, REPAIR, 2 edits, sparse Fisher-Yates picks 5 then 7.
        got = self._run(_ref_encode(info), [0, Mode.REPAIR, 2, 5, 7])
        assert got == _ref_encode(_flip(info, [5, 7]))

    def test_falsification_repair_fixes_garbage_codeword(self):
        """Garbage parity in: REPAIR output is the reference encoding of its systematic bits."""
        cw = _to_bits(bytes(range(0x30, 0x30 + 17)))[:N]
        got = self._run(cw, [0, Mode.REPAIR, 1, 0])
        assert got == _ref_encode(_flip(cw[:K], [0]))
        assert _flip(got, [K + 3]) != _ref_encode(got[:K])

    def test_scripted_correctable_flips_reencoded(self):
        info = _info()
        cw = _ref_encode(info)
        garbled = _flip(cw, [K + 1, 2 * K + 2])  # input parity damaged; re-encode heals it
        # codeword 0, CORRECTABLE, 2 flips: i=0 picks 3, i=1 picks 100.
        got = self._run(garbled, [0, Mode.CORRECTABLE, 2, 3, 100])
        assert got == _flip(cw, [3, 100])

    def test_scripted_uncorrectable_burst_hits_all_streams(self):
        info = _info()
        cw = _ref_encode(info)
        pi = _qpp(K, DEFAULT_F1, DEFAULT_F2)
        inv = {p: m for m, p in enumerate(pi)}
        burst = [38, 39, 0, 1]  # start 38, length 4, wraps mod K
        want = burst + [K + j for j in burst] + [2 * K + inv[j] for j in burst]
        got = self._run(cw, [0, Mode.UNCORRECTABLE, 38, 4])
        assert got == _flip(cw, want)
        assert sum(a != b for a, b in zip(got, cw, strict=True)) == 3 * len(burst)

    def test_scripted_tail_break_leaves_trellis_open(self):
        info = _info()
        cw = _ref_encode(info)
        # codeword 0, TAIL_BREAK, 2 flips: tail index 0 (x_K) then 5 (z_K+2).
        got = self._run(cw, [0, Mode.TAIL_BREAK, 2, 0, 5])
        assert got == _flip(cw, [3 * K, 3 * K + 5])
        assert got[: 3 * K] == cw[: 3 * K]
        assert _end_state(info + got[3 * K : 3 * K + 3]) != (0, 0, 0)

    def test_noop_guard_flips_first_systematic_bit(self):
        """Input already equals re-encode + scripted flip: output still differs."""
        info = _info()
        cw = _ref_encode(info)
        got = self._run(_flip(cw, [K + 3]), [0, Mode.CORRECTABLE, 1, K + 3])
        assert got == _flip(cw, [K + 3, 0])

    def test_only_chosen_codeword_changes(self):
        a, b = _ref_encode(_info(seed=1)), _ref_encode(_info(seed=4))
        data = _to_bytes(a + b + TRAILER)
        out = turbo_fix(
            data, _BoundedRng(randints=[1, Mode.CORRECTABLE, 1, 0]), K, DEFAULT_F1, DEFAULT_F2
        )
        bits = _to_bits(out)
        assert bits[:N] == a
        assert bits[N : 2 * N] == _flip(b, [0])
        assert bits[2 * N : 2 * N + 4] == TRAILER

    def test_seeded_runs_keep_length_and_neighbours(self):
        a, b = _ref_encode(_info(seed=1)), _ref_encode(_info(seed=4))
        data = _to_bytes(a + b + TRAILER)
        rng = RandPool(seed=4)
        for _ in range(200):
            out = turbo_fix(data, rng, K, DEFAULT_F1, DEFAULT_F2)
            assert out is not None and len(out) == len(data) and out != data
            bits = _to_bits(out)
            assert bits[:N] == a or bits[N : 2 * N] == b
            assert bits[2 * N : 2 * N + 4] == TRAILER

    def test_large_block(self):
        info = _info(K6144[0], 3)
        data = _to_bytes(_ref_encode(info, *K6144))
        out = turbo_fix(data, _BoundedRng(randints=[0, Mode.REPAIR, 1, 6143]), *K6144)
        assert _to_bits(out)[: 3 * K6144[0] + TAIL_BITS] == _ref_encode(_flip(info, [6143]), *K6144)

    def test_declines_when_no_codeword_fits(self):
        assert turbo_fix(b"", RandPool(seed=1), K, DEFAULT_F1, DEFAULT_F2) is None
        assert turbo_fix(bytes(N // 8), RandPool(seed=1), K, DEFAULT_F1, DEFAULT_F2) is None

    def test_adversarial_random_buffers_never_raise(self):
        rng = RandPool(seed=7)
        for size in range(0, 400):
            data = bytes(rng.randint(0, 255) for _ in range(size))
            out = turbo_fix(data, rng, K, DEFAULT_F1, DEFAULT_F2)
            if size * 8 < N:
                assert out is None
                continue
            assert out is not None and len(out) == size and out != data

    def test_adversarial_hostile_fill_bytes(self):
        rng = RandPool(seed=3)
        for fill in (0x00, 0xFF, 0xAA, 0x55):
            for size in (17, 33, 34, 200):
                data = bytes([fill]) * size
                for _ in range(20):
                    out = turbo_fix(data, rng, K, DEFAULT_F1, DEFAULT_F2)
                    assert out is not None and len(out) == size and out != data

    @pytest.mark.parametrize(("k", "f1", "f2"), [(39, 3, 10), (40, 4, 10), (7000, 1, 0)])
    def test_adversarial_bad_params_raise_value_error(self, k, f1, f2):
        with pytest.raises(ValueError):
            turbo_fix(bytes(1000), RandPool(seed=1), k, f1, f2)
