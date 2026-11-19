"""token_tile: max-coverage dictionary tiling, mutated a whole token at a time."""

import itertools
import time

import pytest

from fuzzer_tool.core.operator_registry import _CATEGORIES, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.token_tile import (
    MAX_TILE_MATCHES,
    MAX_TILE_SCAN,
    MAX_TILE_TOKENS,
    TILE_NP_MIN_MATCHES,
    Edit,
    _prepare,
    _tile_np,
    _tile_py,
    tile,
    token_tile,
)
from tests.support.scripted_rng import ScriptedRng

_RNG = RandPool(seed=0x711E)
_BIG = 1 << 20
_EDITS = tuple(Edit)
_TOKS = [b"GET", b"HTTP", b"/a"]
_REQ = b"GET /a HTTP"
#: Seconds a capped worst case may take (generous: CI noise).
_WORST_CASE_S = 2.0


def _edit(e: Edit, *extra) -> ScriptedRng:
    return ScriptedRng(randints=[_EDITS.index(e), *extra])


def _covered(tiles) -> int:
    return sum(len(t) for _, t in tiles)


def _assert_valid(data: bytes, tiles) -> None:
    """Ascending, non-overlapping, each token really at its offset."""
    end = 0
    for at, tok in tiles:
        assert at >= end
        assert data[at : at + len(tok)] == tok
        end = at + len(tok)


def _dense(data: bytes, toks) -> int:
    """Independent oracle: dense position DP, best[i] = max covered in data[:i]."""
    best = [0] * (len(data) + 1)
    for i in range(1, len(data) + 1):
        best[i] = best[i - 1]
        for t in toks:
            if i >= len(t) and data[i - len(t) : i] == t:
                best[i] = max(best[i], best[i - len(t)] + len(t))
    return best[-1]


def _brute(data: bytes, toks) -> int:
    """Every subset of matches; keep the best non-overlapping one."""
    spans = [(i, i + len(t)) for t in toks for i in range(len(data)) if data[i : i + len(t)] == t]
    best = 0
    for r in range(len(spans) + 1):
        for pick in itertools.combinations(spans, r):
            ordered = sorted(pick)
            if all(a[1] <= b[0] for a, b in itertools.pairwise(ordered)):
                best = max(best, sum(e - s for s, e in ordered))
    return best


def _rand_case(n: int, alphabet: int, n_toks: int):
    data = bytes(b % alphabet + 0x61 for b in _RNG.randbytes(n))
    toks = list(
        dict.fromkeys(
            bytes(b % alphabet + 0x61 for b in _RNG.randbytes(_RNG.randint(1, 4)))
            for _ in range(n_toks)
        )
    )
    return data, toks


# ── tiling ─────────────────────────────────────────────────────────────


def test_tile_exact():
    assert tile(_REQ, _TOKS) == [(0, b"GET"), (4, b"/a"), (7, b"HTTP")]


def test_tile_prefers_fewer_tiles_on_ties():
    assert tile(b"abcd", [b"ab", b"cd", b"abcd"]) == [(0, b"abcd")]


def test_tile_overlap_picks_max_coverage():
    # "abc" + "cdef" overlap at c; "ab" + "cdef" covers all six.
    assert tile(b"abcdef", [b"abc", b"cdef", b"ab"]) == [(0, b"ab"), (2, b"cdef")]


def test_control_dense_oracle_matches_brute_force():
    # Hard Rule 46: the oracle itself checked on cases small enough to enumerate.
    for _ in range(40):
        data, toks = _rand_case(8, 3, 3)
        assert _dense(data, toks) == _brute(data, toks)


def test_tile_matches_dense_oracle():
    for _ in range(60):
        data, toks = _rand_case(200, 4, 8)
        tiles = tile(data, toks)
        _assert_valid(data, tiles)
        assert _covered(tiles) == _dense(data, toks)


@pytest.mark.parametrize("fn", [_tile_py, _tile_np])
def test_both_dp_paths_match_dense_oracle(fn):
    # Small and large cases on each path, whichever one tile() would dispatch to.
    for n in (40, 400):
        data, toks = _rand_case(n, 4, 8)
        found = _prepare(tuple(toks))[1].scan(data)
        tiles = fn(found)
        _assert_valid(data, tiles)
        assert _covered(tiles) == _dense(data, toks)


def test_dispatch_threshold_reached_by_dense_input():
    data, toks = _rand_case(1000, 2, 6)
    found = _prepare(tuple(toks))[1].scan(data)
    assert sum(map(len, found.values())) >= TILE_NP_MIN_MATCHES
    assert _covered(tile(data, toks)) == _dense(data, toks)


def test_tile_str_and_duplicate_tokens():
    assert tile(b"key=val", ["key", b"key", "", b"val"]) == [(0, b"key"), (4, b"val")]


# ── edits ──────────────────────────────────────────────────────────────


def test_replace():
    out = token_tile(_REQ, _TOKS, 0, _edit(Edit.REPLACE, 1), _BIG)
    assert out == b"HTTP /a HTTP"


def test_duplicate():
    assert token_tile(_REQ, _TOKS, 0, _edit(Edit.DUPLICATE), _BIG) == b"GETGET /a HTTP"


def test_delete():
    assert token_tile(_REQ, _TOKS, 0, _edit(Edit.DELETE), _BIG) == b" /a HTTP"


def test_swap():
    assert token_tile(_REQ, _TOKS, 0, _edit(Edit.SWAP, 2), _BIG) == b"HTTP /a GET"


def test_site_anchors_tile_at_or_after_start():
    # start inside "/a": that tile is the one deleted.
    assert token_tile(_REQ, _TOKS, 5, _edit(Edit.DELETE), _BIG) == b"GET  HTTP"
    # Past the last tile: wraps to the first.
    assert token_tile(_REQ + b"  ", _TOKS, 12, _edit(Edit.DELETE), _BIG) == b" /a HTTP  "


# ── falsification ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("data", "toks"),
    [
        (b"", _TOKS),
        (_REQ, []),
        (_REQ, [b"", ""]),
        (b"nothing matches", [b"XYZ", b"QQ"]),
    ],
)
def test_falsification_nothing_to_tile_declines(data, toks):
    # Empty script: the op must decline before drawing.
    assert token_tile(data, toks, 0, ScriptedRng(), _BIG) is None


def test_falsification_noop_edits_decline():
    # Replacing a token by itself, or swapping a tile with itself, changes nothing.
    assert token_tile(_REQ, _TOKS, 0, _edit(Edit.REPLACE, 0), _BIG) is None
    assert token_tile(_REQ, _TOKS, 0, _edit(Edit.SWAP, 0), _BIG) is None
    assert token_tile(b"ab-ab", [b"ab"], 0, _edit(Edit.SWAP, 1), _BIG) is None


# ── adversarial ────────────────────────────────────────────────────────


def test_adversarial_growth_past_max_len_declines():
    assert token_tile(_REQ, _TOKS, 0, _edit(Edit.DUPLICATE), len(_REQ)) is None


def test_adversarial_tokens_past_scan_cap_ignored():
    data = bytes(MAX_TILE_SCAN) + b"GET"
    assert token_tile(data, [b"GET"], 0, ScriptedRng(), _BIG) is None


def test_adversarial_tail_past_scan_cap_kept():
    data = b"GET" + bytes(MAX_TILE_SCAN) + b"GET"
    out = token_tile(data, [b"GET"], 0, _edit(Edit.DELETE), _BIG)
    assert out == data[len(b"GET") :]


def test_adversarial_token_window_wraps():
    # More tokens than the cap: the window starts at the drawn index and wraps.
    filler = [b"\xfe%d\xfe" % i for i in range(MAX_TILE_TOKENS)]
    toks = [b"GET", *filler]
    assert token_tile(_REQ, toks, 0, ScriptedRng(randints=[1]), _BIG) is None
    rng = ScriptedRng(randints=[2, _EDITS.index(Edit.DELETE)])
    assert token_tile(_REQ, toks, 0, rng, _BIG) == b" /a HTTP"


def test_adversarial_zero_fill_run_tokens_bounded():
    # Every offset matches every run token: matches capped, still a valid tiling.
    data = bytes(1 << 16)
    toks = [bytes(k) for k in range(1, 9)]
    t0 = time.perf_counter()
    tiles = tile(data, toks)
    assert time.perf_counter() - t0 < _WORST_CASE_S
    _assert_valid(data, tiles)
    assert sum(len(t) for _, t in tiles) <= MAX_TILE_SCAN
    assert len(tiles) <= MAX_TILE_MATCHES

    out = token_tile(data, toks, 0, _edit(Edit.DELETE), _BIG)
    assert out is not None
    assert len(out) == len(data) - len(tiles[0][1])


# ── registry / handler wiring ──────────────────────────────────────────


def test_registered_in_dict_category():
    assert "token_tile" in _CATEGORIES["dict"]


def test_gated_on_literals():
    from types import SimpleNamespace

    bare = SimpleNamespace(dictionary=[], _cmplog=None)
    with_dict = SimpleNamespace(dictionary=_TOKS, _cmplog=None)
    assert "token_tile" not in REGISTRY.available(bare, _REQ)
    assert "token_tile" in REGISTRY.available(with_dict, _REQ)


def test_handler_tiles_and_declines():
    from fuzzer_tool.services.operators import OperatorEngine
    from tests.support.operator_env import make_minimal_fuzzer

    engine = OperatorEngine(make_minimal_fuzzer())
    engine.f.dictionary = _TOKS
    engine.f._rng = _edit(Edit.SWAP, 2)
    assert bytes(engine._op_token_tile(bytearray(_REQ), 0, _REQ)) == b"HTTP /a GET"
    engine.f.dictionary = []
    assert bytes(engine._op_token_tile(bytearray(_REQ), 0, _REQ)) == _REQ
