"""Tests for core/token_superstring.py (dictionary tokens -> one short seed)."""

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.core.mutations import load_dictionary
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.token_superstring import (
    EXACT_MAX_LEN,
    ILS_MAX_PAIRS,
    _Net,
    _sweep,
    build,
    exact_order,
    greedy_order,
    ils_order,
    overlap_map,
    reduce_input,
    token_seed,
)

ROOT = Path(__file__).resolve().parents[2]
DICTS = ROOT / "dictionaries"
TOOL = ROOT / "tools" / "gen_dict_seed.py"
KICKS = 200

# Greedy gives 10 bytes, exact optimum is 9 (found by exhaustive search).
GREEDY_LOSES = [b"bbc", b"aabac", b"cba", b"bac"]
GREEDY_LEN = 10
OPT_LEN = 9


def _instances(count, seed):
    """Deterministic small token sets over a 3-letter alphabet."""
    rng = RandPool(seed=seed)
    out = []
    for _ in range(count):
        n = 1 + rng.randrange(6)
        out.append(
            [bytes(97 + rng.randrange(3) for _ in range(1 + rng.randrange(5))) for _ in range(n)]
        )
    return out


def _opt(tokens):
    """Exact shortest common superstring length (bitmask DP, small n)."""
    ss = reduce_input(tokens)
    n = len(ss)
    if n == 0:
        return 0
    ov = overlap_map(ss)
    inf = 10**9
    dp = [[inf] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = len(ss[i])
    for mask in range(1 << n):
        for i in range(n):
            if dp[mask][i] >= inf:
                continue
            for j in range(n):
                if mask >> j & 1:
                    continue
                v = dp[mask][i] + len(ss[j]) - ov.get((i, j), 0)
                dp[mask | 1 << j][j] = min(dp[mask | 1 << j][j], v)
    return min(dp[(1 << n) - 1])


def _naive_overlap(a, b):
    for k in range(min(len(a), len(b)) - 1, 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


# ---- reduce_input ----


def test_reduce_drops_empty_dupes_and_substrings():
    assert reduce_input([b"", b"abc", b"abc", b"bc", b"xyz"]) == [b"abc", b"xyz"]


def test_reduce_keeps_all_when_independent():
    assert sorted(reduce_input([b"ab", b"cd", b"ef"])) == [b"ab", b"cd", b"ef"]


# ---- overlap map vs naive oracle ----


def test_overlap_map_matches_naive_oracle():
    for toks in _instances(60, seed=11):
        ss = reduce_input(toks)
        ov = overlap_map(ss)
        for i, a in enumerate(ss):
            for j, b in enumerate(ss):
                if i == j:
                    continue
                assert ov.get((i, j), 0) == _naive_overlap(a, b), (ss, i, j)


# ---- validity: every token is in the seed ----


def test_seed_contains_every_token():
    for toks in _instances(150, seed=3):
        out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
        assert all(t in out for t in toks), toks


def test_small_sets_hit_opt():
    for toks in _instances(120, seed=5):
        ss = reduce_input(toks)
        ov = overlap_map(ss)
        glen = len(build(greedy_order(ss, ov), ss, ov)) if ss else 0
        out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
        assert len(out) == _opt(toks) <= glen, toks


def test_control_opt_dp_matches_permutations():
    """The DP oracle against an independent brute force."""
    for toks in _instances(60, seed=8):
        ss = reduce_input(toks)
        ov = overlap_map(ss)
        perms = itertools.permutations(range(len(ss)))
        brute = min((len(build(list(p), ss, ov)) for p in perms), default=0)
        assert _opt(toks) == brute, toks


def test_exact_order_is_a_permutation():
    ss = reduce_input([b"abab", b"baba", b"abba", b"bbaa", b"aabb"])
    assert sorted(exact_order(ss, overlap_map(ss))) == list(range(len(ss)))


# ---- falsification: ILS must beat greedy where greedy is known to lose ----


def test_greedy_loses_on_known_instance():
    ss = reduce_input(GREEDY_LOSES)
    ov = overlap_map(ss)
    assert len(build(greedy_order(ss, ov), ss, ov)) == GREEDY_LEN
    assert _opt(GREEDY_LOSES) == OPT_LEN


def test_ils_reaches_opt_where_greedy_loses():
    out = token_seed(GREEDY_LOSES, rng=RandPool(seed=1), kicks=KICKS)
    assert len(out) == OPT_LEN
    assert all(t in out for t in GREEDY_LOSES)


def test_known_chain_collapses():
    assert token_seed([b"abc", b"bcd", b"cde"], rng=RandPool(seed=1), kicks=KICKS) == b"abcde"


# ---- control (AGENTS.md rule 46): the oracle against itself ----


def test_control_same_seed_same_output():
    toks = [b"GIF", b"IFA", b"89a", b"9a;", b"a;G"]
    a = token_seed(toks, rng=RandPool(seed=9), kicks=KICKS)
    b = token_seed(toks, rng=RandPool(seed=9), kicks=KICKS)
    assert a == b


def test_control_opt_oracle_on_known_values():
    assert _opt([b"abc", b"bcd", b"cde"]) == 5
    assert _opt([b"ab", b"ba"]) == 3
    assert _opt([]) == 0


# ---- ILS pieces ----


def _wide(count, seed, n):
    rng = RandPool(seed=seed)
    return [
        [bytes(97 + rng.randrange(3) for _ in range(2 + rng.randrange(5))) for _ in range(n)]
        for _ in range(count)
    ]


def test_ils_never_worse_than_input_order():
    for toks in _instances(80, seed=21) + _wide(20, 22, 25):
        ss = reduce_input(toks)
        if not ss:
            continue
        ov = overlap_map(ss)
        go = greedy_order(ss, ov)
        better = ils_order(go, ov, RandPool(seed=2), KICKS)
        assert sorted(better) == sorted(go)
        assert len(build(better, ss, ov)) <= len(build(go, ss, ov))


def test_order_is_a_permutation():
    ss = reduce_input([b"abab", b"baba", b"abba", b"bbaa", b"aabb"])
    ov = overlap_map(ss)
    assert sorted(greedy_order(ss, ov)) == list(range(len(ss)))


# ---- adversarial ----


def test_empty_input():
    assert token_seed([], rng=RandPool(seed=1), kicks=KICKS) == b""


def test_only_empty_tokens():
    assert token_seed([b"", b""], rng=RandPool(seed=1), kicks=KICKS) == b""


def test_single_token():
    assert token_seed([b"\x89PNG"], rng=RandPool(seed=1), kicks=KICKS) == b"\x89PNG"


def test_all_duplicates():
    assert token_seed([b"zz"] * 50, rng=RandPool(seed=1), kicks=KICKS) == b"zz"


def test_token_contained_in_another():
    assert token_seed([b"abcdef", b"cd"], rng=RandPool(seed=1), kicks=KICKS) == b"abcdef"


def test_nul_and_high_bytes():
    toks = [b"\x00\x00\x01", b"\x00\x01\xff", b"\xff\xff\x00"]
    out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
    assert all(t in out for t in toks)
    assert len(out) <= sum(map(len, toks))


def test_cyclic_pair_does_not_loop():
    out = token_seed([b"ab", b"ba"], rng=RandPool(seed=1), kicks=KICKS)
    assert len(out) == 3
    assert b"ab" in out and b"ba" in out


def test_periodic_tokens():
    toks = [b"aaaa", b"aaab", b"baaa", b"abab"]
    out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
    assert all(t in out for t in toks)
    assert len(out) >= _opt(toks)


def test_all_single_bytes_are_all_kept():
    toks = [bytes([b]) for b in range(256)]
    out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
    assert sorted(out) == list(range(256))


def test_no_overlap_means_concatenation_length():
    toks = [b"aaa", b"bbb", b"ccc"]
    assert len(token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)) == 9


def test_ils_beats_or_ties_greedy_on_wide_sets():
    for toks in _wide(15, 31, 30):
        ss = reduce_input(toks)
        ov = overlap_map(ss)
        glen = len(build(greedy_order(ss, ov), ss, ov))
        out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
        assert all(t in out for t in toks)
        assert len(out) <= glen


# 14 tokens (> EXACT_MAX_LEN): greedy gives 31 bytes, ILS 30.
ILS_WINS = [
    b"aabb", b"abaa", b"aaaba", b"cbbcb", b"ccabc", b"cbc", b"aac",
    b"ab", b"bcbcbb", b"caa", b"acbcac", b"bcaca", b"abbb", b"abbaaa",
]  # fmt: skip


def test_ils_strictly_beats_greedy_on_known_wide_set():
    ss = reduce_input(ILS_WINS)
    ov = overlap_map(ss)
    assert len(ss) > EXACT_MAX_LEN
    assert len(build(greedy_order(ss, ov), ss, ov)) == 31
    out = token_seed(ILS_WINS, rng=RandPool(seed=1), kicks=KICKS)
    assert len(out) == 30
    assert all(t in out for t in ILS_WINS)


def test_many_tokens_stay_valid_and_bounded():
    rng = RandPool(seed=4)
    toks = [bytes(rng.randrange(4) for _ in range(3 + rng.randrange(6))) for _ in range(300)]
    out = token_seed(toks, rng=RandPool(seed=1), kicks=20)
    assert all(t in out for t in toks)
    assert len(out) <= sum(map(len, reduce_input(toks)))


# ---- real dictionaries ----


@pytest.mark.parametrize("name", ["png.dict", "gif.dict", "css.dict"])
def test_real_dictionary_seed(name):
    toks = load_dictionary(str(DICTS / name))
    out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
    assert all(t in out for t in toks)
    assert len(out) <= sum(map(len, set(toks)))


# ---- CLI ----


def test_cli_writes_seed(tmp_path):
    out = tmp_path / "corpus"
    run = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "-D",
            str(DICTS / "gif.dict"),
            "--out",
            str(out),
            "--rng-seed",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    files = list((out / "seeds").glob("id_*_dictseed.bin"))
    assert len(files) == 1
    data = files[0].read_bytes()
    toks = load_dictionary(str(DICTS / "gif.dict"))
    assert all(t in data for t in toks)


def test_cli_missing_dict_fails(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "-D",
            str(tmp_path / "nope.dict"),
            "--out",
            str(tmp_path / "c"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert not (tmp_path / "c" / "seeds").exists()


def test_cli_empty_dict_fails(tmp_path):
    empty = tmp_path / "empty.dict"
    empty.write_text("# nothing\n")
    run = subprocess.run(
        [sys.executable, str(TOOL), "-D", str(empty), "--out", str(tmp_path / "c")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert not (tmp_path / "c" / "seeds").exists()


def test_output_independent_of_hash_seed():
    code = (
        "from fuzzer_tool.core.rand_pool import RandPool\n"
        "from fuzzer_tool.core.token_superstring import token_seed\n"
        "t=[b'GIF',b'IFA',b'89a',b'9a;',b'a;G',b'xyz',b'zyx',b'abc',b'cab']\n"
        "print(token_seed(t, rng=RandPool(seed=1), kicks=50).hex())\n"
    )
    outs = set()
    for hs in ("0", "1", "2", "3"):
        env = {"PYTHONHASHSEED": hs, "PYTHONPATH": str(ROOT / "src")}
        run = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outs.add(run.stdout)
    assert len(outs) == 1


def test_long_token_is_bounded():
    big = bytes(range(256)) * 400
    toks = [big, big[-100:] + b"tail", b"head" + big[:100]]
    out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
    assert all(t in out for t in toks)
    assert len(out) < sum(map(len, toks))


# ---- differential: neighbour-slot sweep vs the full-scan sweep ----


def _ref_link(a, b, ov):
    return 0 if a is None or b is None else ov.get((a, b), 0)


def _ref_sweep(cur, ov):
    """Reference or-opt sweep: tries every slot (improve.py's `local`)."""
    total = 0
    for seg_len in (1, 2, 3):
        i = 0
        while i + seg_len <= len(cur):
            seg, rest = cur[i : i + seg_len], cur[:i] + cur[i + seg_len :]
            prev = cur[i - 1] if i > 0 else None
            after = cur[i + seg_len] if i + seg_len < len(cur) else None
            lost = (
                _ref_link(prev, seg[0], ov)
                + _ref_link(seg[-1], after, ov)
                - _ref_link(prev, after, ov)
            )
            best, slot = 0, -1
            for p in range(len(rest) + 1):
                if p == i:
                    continue
                a = rest[p - 1] if p > 0 else None
                b = rest[p] if p < len(rest) else None
                g = _ref_link(a, seg[0], ov) + _ref_link(seg[-1], b, ov) - _ref_link(a, b, ov)
                if g > best:
                    best, slot = g, p
            i += 1
            if slot < 0 or best <= lost:
                continue
            cur[:] = rest[:slot] + seg + rest[slot:]
            total += best - lost
    return total


def test_sweep_matches_full_scan_reference():
    for toks in _wide(25, 41, 20) + _instances(40, seed=42):
        ss = reduce_input(toks)
        ov = overlap_map(ss)
        order = list(range(len(ss)))
        fast, slow = list(order), list(order)
        assert _sweep(fast, _Net(len(ss), ov)) == _ref_sweep(slow, ov)
        assert fast == slow


def test_control_reference_sweep_against_itself():
    ss = reduce_input(_wide(1, 43, 20)[0])
    ov = overlap_map(ss)
    a, b = list(range(len(ss))), list(range(len(ss)))
    assert _ref_sweep(a, ov) == _ref_sweep(b, ov)
    assert a == b


def test_dense_graph_skips_ils_and_equals_greedy():
    toks = [bytes(97 + (n >> b & 1) for b in range(9)) for n in range(512)]
    ss = reduce_input(toks)
    ov = overlap_map(ss)
    assert len(ov) > ILS_MAX_PAIRS
    out = token_seed(toks, rng=RandPool(seed=1), kicks=KICKS)
    assert out == build(greedy_order(ss, ov), ss, ov)
    assert all(t in out for t in toks)
    assert len(out) < sum(map(len, toks))
