"""Tests for the ldpc_fix operator (core/mutations/ldpc_fix.py)."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from fuzzer_tool.core.mutations.ldpc_fix import (
    MAX_DATA_EDITS,
    MAX_N,
    LdpcCode,
    Mode,
    check_params,
    ldpc_fix,
    load_alist,
    parse_alist,
)
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.ldpc_codes import (
    GALLAGER_N,
    HAMMING_N,
    HAMMING_ROWS,
    RANK_DEF_ROWS,
    TWIN_N,
    TWIN_ROWS,
    Pad,
    alist_text,
    gallager_rows,
)
from tests.support.scripted_rng import ScriptedRng

# A valid Hamming codeword (oracle-checked in TestOracle).
HAMMING_WORD = (1, 1, 1, 0, 0, 0, 0)


# --- independent oracle: plain nested loops over the 0-based row lists ---------


def _syndrome(rows, word) -> list[int]:
    out = []
    for row in rows:
        acc = 0
        for j in row:
            acc ^= word[j]
        out.append(acc)
    return out


def _valid(rows, word) -> bool:
    return not any(_syndrome(rows, word))


def _rank(rows, n: int) -> int:
    mat = [[1 if j in row else 0 for j in range(n)] for row in rows]
    rank = 0
    for col in range(n):
        piv = next((r for r in range(rank, len(mat)) if mat[r][col]), None)
        if piv is None:
            continue
        mat[rank], mat[piv] = mat[piv], mat[rank]
        for r in range(len(mat)):
            if r != rank and mat[r][col]:
                mat[r] = [a ^ b for a, b in zip(mat[r], mat[rank], strict=True)]
        rank += 1
    return rank


def _bits(data: bytes) -> list[int]:
    return [(b >> (7 - i)) & 1 for b in data for i in range(8)]


def _pack(bits) -> bytes:
    bits = list(bits)
    return bytes(
        sum(bit << (7 - i) for i, bit in enumerate(bits[p : p + 8])) for p in range(0, len(bits), 8)
    )


def _word(data: bytes, n: int, idx: int = 0) -> list[int]:
    return _bits(data)[idx * n : (idx + 1) * n]


def _codewords(rows, n: int) -> list[tuple[int, ...]]:
    return [w for w in itertools.product((0, 1), repeat=n) if _valid(rows, w)]


def _completion(rows, n: int, free, info: dict[int, int]) -> tuple[int, ...]:
    """The unique codeword carrying *info* on the *free* positions (brute force)."""
    hits = [w for w in _codewords(rows, n) if all(w[j] == info[j] for j in free)]
    assert len(hits) == 1
    return hits[0]


def _hamming() -> LdpcCode:
    return parse_alist(alist_text(HAMMING_N, HAMMING_ROWS))


def _buf(word, tail=(1, 0)) -> bytes:
    """Two copies of *word* plus trailing bits: n=7 -> 16 bits, 2 bytes."""
    return _pack([*word, *word, *tail])


class _BoundedRng(ScriptedRng):
    """ScriptedRng whose randint() rejects draws outside the caller's bounds."""

    def randint(self, a, b):
        v = super().randint(a, b)
        assert a <= v <= b, (a, v, b)
        return v


class TestOracle:
    def test_control_known_codeword_and_planted_error(self):
        assert _syndrome(HAMMING_ROWS, HAMMING_WORD) == [0, 0, 0]
        bad = list(HAMMING_WORD)
        bad[6] ^= 1
        assert _syndrome(HAMMING_ROWS, bad) == [1, 1, 1]

    def test_control_hamming_has_16_codewords(self):
        assert len(_codewords(HAMMING_ROWS, HAMMING_N)) == 16

    def test_control_pack_roundtrip(self):
        data = bytes(range(0, 256, 17))
        assert _pack(_bits(data)) == data


class TestAlist:
    def test_hamming(self):
        code = _hamming()
        assert (code.n, code.m, code.k, code.rank) == (7, 3, 4, 3)

    def test_padding_optional(self):
        a = parse_alist(alist_text(HAMMING_N, HAMMING_ROWS, Pad.ZEROS))
        b = parse_alist(alist_text(HAMMING_N, HAMMING_ROWS, Pad.NONE))
        assert a.rows == b.rows

    def test_row_masks_msb_first(self):
        """Bit n-1-j of a row mask is variable j (reads like the stream)."""
        code = _hamming()
        for mask, row in zip(code.rows, HAMMING_ROWS, strict=True):
            assert [j for j in range(7) if mask >> (6 - j) & 1] == list(row)

    @pytest.mark.parametrize(
        ("n", "rows"),
        [(HAMMING_N, HAMMING_ROWS), (HAMMING_N, RANK_DEF_ROWS), (GALLAGER_N, gallager_rows())],
    )
    def test_k_is_n_minus_rank(self, n, rows):
        code = parse_alist(alist_text(n, rows))
        assert code.k == n - _rank(rows, n)
        assert sorted((*code.free, *code.pivots)) == list(range(n))

    def test_gallager_is_rank_deficient(self):
        rows = gallager_rows()
        assert _rank(rows, GALLAGER_N) < len(rows)

    @pytest.mark.parametrize(("n", "rows"), [(HAMMING_N, HAMMING_ROWS), (HAMMING_N, RANK_DEF_ROWS)])
    def test_free_positions_are_an_information_set(self, n, rows):
        code = parse_alist(alist_text(n, rows))
        words = _codewords(rows, n)
        assert len(words) == 2**code.k
        assert len({tuple(w[j] for j in code.free) for w in words}) == len(words)

    def test_load_alist_file(self, tmp_path):
        path = tmp_path / "h.alist"
        path.write_text(alist_text(HAMMING_N, HAMMING_ROWS))
        assert load_alist(str(path)).rows == _hamming().rows

    def test_load_alist_non_utf8(self, tmp_path):
        path = tmp_path / "bad.alist"
        path.write_bytes(b"\xff\xfe7 3\n")
        with pytest.raises(ValueError):
            load_alist(str(path))

    def test_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            _hamming().n = 8  # type: ignore[misc]


def _ok() -> list[str]:
    return alist_text(HAMMING_N, HAMMING_ROWS).splitlines()


def _with(idx: int, text: str) -> str:
    lines = _ok()
    lines[idx] = text
    return "\n".join(lines)


MALFORMED = {
    "empty": "",
    "header_text": _with(0, "seven 3"),
    "header_short": _with(0, "7"),
    "n_zero": _with(0, "0 3"),
    "m_negative": _with(0, "7 -3"),
    "n_too_big": _with(0, f"{MAX_N + 1} 3"),
    "max_weight_mismatch": _with(1, "3 5"),
    "col_weight_count": _with(2, "1 1 2 1 2 2"),
    "row_weight_sum": _with(3, "4 4 3"),
    "col_list_zero_inside": _with(4, "0 1 0"),
    "col_index_too_big": _with(4, "4 0 0"),
    "col_duplicate": _with(6, "1 1 0"),
    "col_list_long": _with(4, "1 0 0 0"),
    "row_index_too_big": _with(11, "1 3 5 8"),
    "lists_disagree": _with(11, "1 3 5 6"),
    "negative_entry": _with(4, "-1 0 0"),
    "huge_int": _with(4, "9" * 5000),
    "missing_lines": "\n".join(_ok()[:-1]),
    "trailing_data": "\n".join([*_ok(), "1 2 3"]),
}


class TestMalformedAlist:
    @pytest.mark.parametrize("name", sorted(MALFORMED))
    def test_raises_value_error(self, name):
        with pytest.raises(ValueError):
            parse_alist(MALFORMED[name])

    def test_falsification_unmodified_text_parses(self):
        """Control for MALFORMED: the base text it edits is accepted."""
        assert parse_alist("\n".join(_ok())).k == 4

    def test_adversarial_random_token_edits_only_value_error(self):
        rng = RandPool(seed=11)
        tokens = ["0", "1", "2", "3", "7", "8", "-1", "x", "", "99999999", "1.5"]
        for _ in range(500):
            lines = [line.split() for line in _ok()]
            for _ in range(rng.randint(1, 3)):
                li = rng.randint(0, len(lines) - 1)
                row = lines[li] or ["0"]
                row[rng.randint(0, len(row) - 1)] = rng.choice(tokens)
                lines[li] = row
            text = "\n".join(" ".join(r) for r in lines)
            try:
                check_params(parse_alist(text))
            except ValueError:
                continue


class TestParams:
    def test_rejects_k_zero(self):
        code = parse_alist(alist_text(3, ((0,), (1,), (2,))))
        assert code.k == 0
        with pytest.raises(ValueError):
            check_params(code)

    def test_rejects_rank_zero(self):
        code = parse_alist(alist_text(3, ((),)))
        with pytest.raises(ValueError):
            check_params(code)

    def test_accepts(self):
        check_params(_hamming())
        check_params(parse_alist(alist_text(TWIN_N, TWIN_ROWS)))


class TestFix:
    def test_scripted_repair_flips_info_and_reencodes(self):
        code = _hamming()
        data = _buf(HAMMING_WORD)
        # codeword 1, repair, 1 edit, info index 2.
        out = ldpc_fix(data, _BoundedRng(randints=[1, Mode.REPAIR, 1, 2]), code)
        info = {j: HAMMING_WORD[j] for j in code.free}
        info[code.free[2]] ^= 1
        want = _completion(HAMMING_ROWS, 7, code.free, info)
        assert out == _pack([*HAMMING_WORD, *want, 1, 0])

    def test_falsification_repair_output_valid_and_parity_flip_detected(self):
        code = _hamming()
        data = _buf((0, 0, 0, 0, 0, 0, 1))  # invalid input words
        out = ldpc_fix(data, _BoundedRng(randints=[0, Mode.REPAIR, 2, 0, 3]), code)
        word = _word(out, 7)
        assert _valid(HAMMING_ROWS, word)
        for p in code.pivots:
            bad = list(word)
            bad[p] ^= 1
            assert not _valid(HAMMING_ROWS, bad)

    def test_repair_gallager(self):
        rows = gallager_rows()
        code = parse_alist(alist_text(GALLAGER_N, rows))
        data = bytes(range(7, 7 + 10))  # 80 bits = 4 codewords
        out = ldpc_fix(data, _BoundedRng(randints=[2, Mode.REPAIR, 1, code.k - 1]), code)
        bits_in, bits_out = _bits(data), _bits(out)
        assert bits_out[:40] == bits_in[:40] and bits_out[60:] == bits_in[60:]
        word = bits_out[40:60]
        assert _valid(rows, word)
        flipped = [j for j in code.free if word[j] != bits_in[40 + j]]
        assert flipped == [code.free[-1]]

    def test_correctable_flips_reencoded_word(self):
        code = _hamming()
        data = _buf(HAMMING_WORD)
        out = ldpc_fix(data, _BoundedRng(randints=[0, Mode.CORRECTABLE, 1, 4]), code)
        want = list(HAMMING_WORD)
        want[4] ^= 1
        assert out == _pack([*want, *HAMMING_WORD, 1, 0])
        assert sum(_syndrome(HAMMING_ROWS, want)) > 0

    def test_correctable_count_bounded_by_t_guess(self):
        code = parse_alist(alist_text(GALLAGER_N, gallager_rows()))
        data = bytes(5)
        top = code.t_guess
        draws = [0, Mode.CORRECTABLE, top, *range(top)]
        out = ldpc_fix(data, _BoundedRng(randints=draws), code)
        assert _bits(out)[:GALLAGER_N] == [1] * top + [0] * (GALLAGER_N - top)

    def test_uncorrectable_flips_whole_check(self):
        code = _hamming()
        data = _buf(HAMMING_WORD)
        out = ldpc_fix(data, _BoundedRng(randints=[0, Mode.UNCORRECTABLE, 2]), code)
        want = [b ^ (j in HAMMING_ROWS[2]) for j, b in enumerate(HAMMING_WORD)]
        assert out == _pack([*want, *HAMMING_WORD, 1, 0])

    @pytest.mark.parametrize("check", [0, 1, 2])
    def test_syndrome_single_unsatisfied_check(self, check):
        code = _hamming()
        data = _buf(HAMMING_WORD)
        out = ldpc_fix(data, _BoundedRng(randints=[0, Mode.SYNDROME, check]), code)
        syn = _syndrome(HAMMING_ROWS, _word(out, 7))
        assert syn == [int(r == check) for r in range(3)]
        assert _word(out, 7, 1) == list(HAMMING_WORD)

    def test_syndrome_rank_deficient_only_feasible_checks(self):
        """Brute force: only check 2 can fail alone; the code must agree."""
        reach = set()
        for e in itertools.product((0, 1), repeat=HAMMING_N):
            syn = _syndrome(RANK_DEF_ROWS, e)
            if sum(syn) == 1:
                reach.add(syn.index(1))
        code = parse_alist(alist_text(HAMMING_N, RANK_DEF_ROWS))
        assert code.feasible == tuple(sorted(reach)) == (2,)
        out = ldpc_fix(_buf(HAMMING_WORD), _BoundedRng(randints=[0, Mode.SYNDROME, 0]), code)
        assert _syndrome(RANK_DEF_ROWS, _word(out, 7)) == [0, 0, 1, 0]

    def test_syndrome_falls_back_to_one_flip_when_none_feasible(self):
        code = parse_alist(alist_text(TWIN_N, TWIN_ROWS))
        assert code.feasible == ()
        data = bytes([0b11000000])  # 2 codewords of 4 bits
        out = ldpc_fix(data, _BoundedRng(randints=[1, Mode.SYNDROME, 3]), code)
        assert out == bytes([0b11000001])

    def test_noop_guard_returns_reencoded_word(self):
        """Re-flipping the only bad bit would echo the input: emit the codeword."""
        code = _hamming()
        bad = list(HAMMING_WORD)
        bad[code.pivots[0]] ^= 1
        data = _buf(bad)
        out = ldpc_fix(data, _BoundedRng(randints=[0, Mode.CORRECTABLE, 1, code.pivots[0]]), code)
        assert out == _pack([*HAMMING_WORD, *bad, 1, 0])
        assert out != data

    def test_repair_edit_count_capped(self):
        code = _hamming()
        draws = [0, Mode.REPAIR, min(MAX_DATA_EDITS, code.k), *range(code.k)]
        out = ldpc_fix(_buf(HAMMING_WORD), _BoundedRng(randints=draws), code)
        assert _valid(HAMMING_ROWS, _word(out, 7))

    def test_declines_when_no_codeword_fits(self):
        code = parse_alist(alist_text(GALLAGER_N, gallager_rows()))
        assert ldpc_fix(b"", RandPool(seed=1), code) is None
        assert ldpc_fix(bytes(2), RandPool(seed=1), code) is None

    def test_only_one_codeword_and_tail_untouched(self):
        code = _hamming()
        data = _buf(HAMMING_WORD, tail=(1, 1))
        rng = RandPool(seed=4)
        for _ in range(200):
            out = ldpc_fix(data, rng, code)
            b, o = _bits(data), _bits(out)
            assert len(out) == len(data) and out != data
            assert o[14:] == b[14:]
            assert o[:7] == b[:7] or o[7:14] == b[7:14]

    def test_adversarial_random_buffers_never_raise(self):
        codes = [
            _hamming(),
            parse_alist(alist_text(HAMMING_N, RANK_DEF_ROWS)),
            parse_alist(alist_text(TWIN_N, TWIN_ROWS)),
            parse_alist(alist_text(GALLAGER_N, gallager_rows())),
        ]
        rng = RandPool(seed=7)
        for size in range(0, 64):
            for code in codes:
                data = bytes(rng.randint(0, 255) for _ in range(size))
                out = ldpc_fix(data, rng, code)
                if size * 8 < code.n:
                    assert out is None
                    continue
                assert len(out) == size and out != data

    def test_larger_code_repair_valid(self):
        n = 1180  # coprime to the band multipliers 3 and 7
        rows = gallager_rows(n)
        code = parse_alist(alist_text(n, rows))
        check_params(code)
        rng = RandPool(seed=3)
        data = bytes(rng.randint(0, 255) for _ in range(n // 8 * 2 + 3))
        out = ldpc_fix(data, ScriptedRng(randints=[1, Mode.REPAIR, 1, 0]), code)
        assert _valid(rows, _word(out, n, 1))
        assert _word(out, n, 0) == _word(data, n, 0)
