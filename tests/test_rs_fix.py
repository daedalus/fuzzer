"""Tests for the rs_fix operator (core/mutations/rs_fix.py)."""

from __future__ import annotations

import argparse

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.mutations.rs_fix import (
    DEFAULT_POLY,
    block_spans,
    check_params,
    parse_poly,
    rs_fix,
    rs_parity,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

# Published QR example "HELLO WORLD" 1-M (16 data bytes, 10 EC bytes).
QR_DATA = bytes([32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17])
QR_EC = bytes([196, 35, 39, 119, 235, 215, 231, 226, 93, 23])
# Parity of the second published example (data in test_second_published_vector).
WIKI_EC = bytes.fromhex("bc2a90136bafeffd4be0")


# --- independent oracle: carryless multiply + Horner, no GF2n -------------------


def _gmul(a: int, b: int, poly: int = DEFAULT_POLY) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= poly
    return out


def _alpha(e: int, poly: int = DEFAULT_POLY) -> int:
    x = 1
    for _ in range(e):
        x = _gmul(x, 2, poly)
    return x


def _syndromes(block: bytes, nsym: int, fcr: int = 0, poly: int = DEFAULT_POLY) -> list[int]:
    """S_i = c(alpha^(fcr+i)), c(x) with block[0] the highest-degree coefficient."""
    out = []
    for i in range(nsym):
        root, acc = _alpha(fcr + i, poly), 0
        for byte in block:
            acc = _gmul(acc, root, poly) ^ byte
        out.append(acc)
    return out


def _valid(block: bytes, nsym: int, fcr: int = 0, poly: int = DEFAULT_POLY) -> bool:
    return not any(_syndromes(block, nsym, fcr, poly))


def _hamming(a: bytes, b: bytes) -> int:
    return sum(x != y for x, y in zip(a, b, strict=True))


def _sample(n: int) -> bytes:
    return bytes((i * 7 + 3) % 256 for i in range(n))


class TestEncoder:
    def test_qr_vector(self):
        assert rs_parity(QR_DATA, 10) == QR_EC

    def test_second_published_vector(self):
        data = bytes([0x40, 0xD2, 0x75, 0x47, 0x76, 0x17, 0x32, 0x06])
        data += bytes([0x27, 0x26, 0x96, 0xC6, 0xC6, 0x96, 0x70, 0xEC])
        assert rs_parity(data, 10) == WIKI_EC

    @pytest.mark.parametrize(("nsym", "fcr"), [(2, 0), (8, 0), (10, 1), (32, 112), (254, 0)])
    def test_codeword_has_zero_syndromes(self, nsym, fcr):
        data = _sample(min(40, 255 - nsym))
        block = data + rs_parity(data, nsym, fcr)
        assert _valid(block, nsym, fcr)

    def test_falsification_wrong_fcr_or_nsym_is_not_a_codeword(self):
        block = QR_DATA + QR_EC
        assert not _valid(block, 10, fcr=1)
        assert _valid(block, 10)
        assert not _valid(QR_DATA + rs_parity(QR_DATA, 8), 10)

    def test_adversarial_empty_and_zero_data(self):
        assert rs_parity(b"", 4) == bytes(4)
        assert rs_parity(bytes(20), 6) == bytes(6)

    def test_adversarial_full_length_block(self):
        data = _sample(255 - 32)
        assert _valid(data + rs_parity(data, 32), 32)

    def test_adversarial_single_symbol_change_changes_parity(self):
        base = rs_parity(QR_DATA, 10)
        for i in range(len(QR_DATA)):
            mut = bytearray(QR_DATA)
            mut[i] ^= 0x01
            assert rs_parity(bytes(mut), 10) != base

    def test_linearity(self):
        """parity(a ^ b) == parity(a) ^ parity(b): the code is GF(2)-linear."""
        a, b = _sample(30), bytes(reversed(_sample(30)))
        x = bytes(p ^ q for p, q in zip(a, b, strict=True))
        pa, pb = rs_parity(a, 12), rs_parity(b, 12)
        assert rs_parity(x, 12) == bytes(p ^ q for p, q in zip(pa, pb, strict=True))


class TestParams:
    @pytest.mark.parametrize(
        ("nsym", "block"), [(0, 255), (1, 255), (10, 10), (10, 256), (-2, 255)]
    )
    def test_check_params_rejects(self, nsym, block):
        with pytest.raises(ValueError):
            check_params(nsym, block)

    def test_check_params_accepts_bounds(self):
        check_params(2, 3)
        check_params(254, 255)

    def test_parse_poly_default(self):
        assert parse_poly("0x11d") == DEFAULT_POLY
        assert parse_poly("285") == DEFAULT_POLY

    @pytest.mark.parametrize("bad", ["", "zz", "0x11b", "0x1", "0x200", "0x11c"])
    def test_parse_poly_rejects(self, bad):
        """0x11b (AES) is irreducible but x has order 51: not usable as alpha."""
        with pytest.raises(ValueError):
            parse_poly(bad)

    def test_argparse_type_turns_error_into_usage_error(self):
        with pytest.raises(argparse.ArgumentTypeError):
            commands._rs_poly("0x11b")

    def test_block_spans_skips_blocks_without_data(self):
        assert block_spans(25, 4, 10) == [(0, 10), (10, 20), (20, 25)]
        assert block_spans(24, 4, 10) == [(0, 10), (10, 20)]
        assert block_spans(3, 4, 10) == []
        assert block_spans(0, 4, 10) == []


class TestFix:
    def test_scripted_repair_flips_data_and_reencodes(self):
        blk = QR_DATA + QR_EC
        # span 0, mode repair, 1 edit, pos 2 (swap with itself), bit-flip branch, bit 3.
        rng = ScriptedRng(randints=[0, 0, 1, 2, 1, 3])
        out = rs_fix(blk, rng, 10)
        want = bytearray(QR_DATA)
        want[2] ^= 1 << 3
        assert out == bytes(want) + rs_parity(bytes(want), 10)
        assert _valid(out, 10)

    def test_scripted_repair_redraw_byte(self):
        blk = QR_DATA + QR_EC
        rng = ScriptedRng(randints=[0, 0, 1, 5, 0, 0x55])
        out = rs_fix(blk, rng, 10)
        want = bytearray(QR_DATA)
        want[5] ^= 0x55
        assert out == bytes(want) + rs_parity(bytes(want), 10)

    def test_outputs_are_valid_or_within_2t_plus_1_of_original(self):
        """Repair gives a changed valid codeword; injection stays within 2t+1 symbols."""
        rng = RandPool(seed=11)
        data = QR_DATA + QR_EC  # a valid codeword, nsym 10 -> t = 5
        seen = set()
        for _ in range(200):
            out = rs_fix(data, rng, 10)
            assert out is not None and len(out) == len(data)
            if _valid(out, 10):
                assert out != data
                seen.add("repaired")
                continue
            dist = _hamming(out, data)
            assert 1 <= dist <= 2 * 5 + 1
            seen.add("correctable" if dist <= 5 else "uncorrectable")
        assert seen == {"repaired", "correctable", "uncorrectable"}

    def test_correctable_mode_distance_within_t(self):
        # span 0, mode 1 (correctable), 3 errors, positions 0,0,0 (identity picks), values 1,2,3.
        blk = QR_DATA + QR_EC
        rng = ScriptedRng(randints=[0, 1, 3, 0, 1, 2, 1, 2, 3])
        out = rs_fix(blk, rng, 10)
        assert _hamming(out, blk) == 3
        assert not _valid(out, 10)

    def test_correctable_word_has_unique_nearest_codeword(self):
        """MDS: d = nsym+1, so t errors are always decoded to the original.

        Brute force at nsym=2 (t=1) on a 6-byte block: the only codeword within
        Hamming distance 1 of a 1-error word is the original.
        """
        nsym = 2
        data = bytes([9, 8, 7, 6])
        orig = data + rs_parity(data, nsym)
        # span 0, mode 1, 1 error, pos 4 (swap with 2... identity pick idx 4), value 0x2A.
        rng = ScriptedRng(randints=[0, 1, 1, 4, 0x2A])
        out = rs_fix(orig, rng, nsym)
        assert _hamming(out, orig) == 1 and not _valid(out, nsym)
        near = []
        for pos in range(len(out)):
            for val in range(256):
                cand = bytearray(out)
                cand[pos] = val
                if _valid(bytes(cand), nsym):
                    near.append(bytes(cand))
        assert near == [orig]

    def test_uncorrectable_mode_exceeds_t(self):
        # span 0, mode 2, count t+1 = 6 (nsym 10), positions identity, values 1..6.
        blk = QR_DATA + QR_EC
        rng = ScriptedRng(randints=[0, 2, 6, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6])
        out = rs_fix(blk, rng, 10)
        assert _hamming(out, blk) == 6

    def test_only_chosen_block_changes_and_length_is_kept(self):
        a, b = QR_DATA + QR_EC, bytes(reversed(QR_DATA)) + rs_parity(bytes(reversed(QR_DATA)), 10)
        data = a + b
        rng = RandPool(seed=4)
        for _ in range(100):
            out = rs_fix(data, rng, 10, block=26)
            assert len(out) == len(data)
            assert out[:26] == a or out[26:] == b

    def test_short_final_block(self):
        data = _sample(26 + 14)
        rng = RandPool(seed=8)
        assert any(rs_fix(data, rng, 6, block=26) != data for _ in range(20))

    def test_adversarial_declines_when_no_block_holds_data(self):
        assert rs_fix(b"", RandPool(seed=1), 4) is None
        assert rs_fix(bytes(4), RandPool(seed=1), 4) is None

    def test_adversarial_data_length_one(self):
        """nsym+1 bytes: one data symbol, every mode must still be well formed."""
        rng = RandPool(seed=2)
        data = b"\x42" + rs_parity(b"\x42", 4)
        for _ in range(60):
            out = rs_fix(data, rng, 4)
            assert out is not None and len(out) == 5

    def test_adversarial_error_count_capped_at_block_length(self):
        rng = RandPool(seed=9)
        data = _sample(3) + rs_parity(_sample(3), 4)  # 7 bytes, t = 2, up to 5 errors
        for _ in range(100):
            out = rs_fix(data, rng, 4)
            assert out is not None and _hamming(out, data) <= 7

    def test_custom_fcr_roundtrip(self):
        rng = ScriptedRng(randints=[0, 0, 1, 0, 1, 0])
        data = _sample(20) + rs_parity(_sample(20), 8, 112)
        out = rs_fix(data, rng, 8, fcr=112)
        assert _valid(out, 8, 112) and not _valid(out, 8, 0)


class _MockFuzzer:
    def __init__(self, nsym=None, block=255, fcr=0, poly=DEFAULT_POLY):
        self.dictionary = []
        self.markov_trained = False
        self.mc = None
        self.mc_cem = False
        self.grammar = None
        self._cmplog = None
        self.enable_regex_bomb = False
        self.enable_x86_mutator = False
        self.enable_arm_mutator = False
        self.seed_meta = {}
        self.corpus = []
        self.max_len = 4096
        self.rs_nsym = nsym
        self.rs_block = block
        self.rs_fcr = fcr
        self.rs_poly = poly
        self._op_declines = {}


class TestWiring:
    def test_registered_format(self):
        assert "rs_fix" in REGISTRY.names()
        assert REGISTRY.category_of("rs_fix") == "format"

    def test_available_only_with_nsym(self):
        data = QR_DATA + QR_EC
        assert "rs_fix" not in REGISTRY.available(_MockFuzzer(), data)
        assert "rs_fix" in REGISTRY.available(_MockFuzzer(10), data)

    def test_dispatch_mutates_and_keeps_length(self):
        fuzzer = _MockFuzzer(10)
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        data = QR_DATA + QR_EC
        outs = {bytes(dispatch["rs_fix"](bytearray(data), 0, data)) for _ in range(30)}
        assert len(outs) > 1 and all(len(o) == len(data) for o in outs)

    def test_dispatch_declines_on_short_input(self):
        fuzzer = _MockFuzzer(10)
        fuzzer._rng = RandPool(seed=5)
        dispatch = REGISTRY.dispatch(OperatorEngine(fuzzer))
        buf = bytearray(b"tiny")
        dispatch["rs_fix"](buf, 0, bytes(buf))
        assert bytes(buf) == b"tiny"
        assert fuzzer._op_declines == {"rs_fix": 1}
