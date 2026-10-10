"""condstmt_solve builds CondStmt objects on demand, not the whole pool.

The CondStmt cache was keyed on the cmplog pool's identity, which eviction
rebinds every drain at cap, so each drain rebuilt ~10k CondStmt objects
(6% of FFmpeg fuzz-loop wall time) for an operator that touches one per
call. ``LazyConds`` keeps the same order, cmpids, selection and RNG draws.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.core import cond_stmt
from fuzzer_tool.core.cond_stmt import CondState, LazyConds, conds_from_cmplog_pairs
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer

_DATA = b"GIF89a....RIFFxxxxWAVEfmt ....fLaC..OggS..\x89PNG"


def _pool(n):
    """n pairs; every 7th op_a occurs in _DATA so solves hit both branches."""
    hits = [b"GIF8", b"RIFF", b"WAVE", b"fLaC", b"OggS", b"\x89PNG"]
    out = []
    for i in range(n):
        a = hits[i % len(hits)] if i % 7 == 0 else i.to_bytes(4, "little")
        out.append((a, (i * 2654435761 % 2**32).to_bytes(4, "big")))
    return out


def _engine(pairs, seed):
    f = make_minimal_fuzzer(pool=RandPool(seed))
    f._cmplog = SimpleNamespace(pairs=pairs, tokens=[], _pair_cmp={}, _pair_pc={})
    f._use_wall_order = False
    return OperatorEngine(f)


def _ref_solve(eng, conds, buf):
    """Oracle: the pre-change selection over an eager CondStmt list."""
    rng = eng.ctx._rng
    unsolved = [c for c in conds if c.state is CondState.UNSOLVED]
    target = rng.choice(unsolved) if unsolved else rng.choice(conds)
    data = bytes(buf)
    target_value = target.base.op_b if rng.randint(0, 1) == 0 else target.base.op_a
    source_value = target.base.op_a if target_value is target.base.op_b else target.base.op_b
    width = target.base.width
    idx = eng._weizz_restricted_find(data, source_value[:width])
    if idx != -1 and idx + width <= len(buf):
        buf[idx : idx + width] = target_value[:width]
        target.mark_solved()
        return bytes(buf)
    if len(buf) + width <= eng.ctx.max_len:
        pos = rng.randint(0, len(buf))
        buf[pos:pos] = target_value[:width]
        target.mark_solved()
        return bytes(buf)
    target.mark_unsolvable()
    return None


def _run_ref(pairs, seed, calls):
    eng = _engine(pairs, seed)
    conds = conds_from_cmplog_pairs(pairs)
    return [_ref_solve(eng, conds, bytearray(_DATA)) for _ in range(calls)]


def _run_new(pairs, seed, calls):
    eng = _engine(pairs, seed)
    out = []
    for _ in range(calls):
        r = eng._op_condstmt_solve(bytearray(_DATA), 0, _DATA)
        out.append(bytes(r) if r is not None else None)
    return out, eng


def test_control_oracle_matches_itself():
    """Rule 46: the eager oracle is deterministic per seed."""
    pairs = _pool(300)
    assert _run_ref(pairs, 3, 40) == _run_ref(pairs, 3, 40)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n_pairs", [5, 300, 600])
def test_same_mutations_as_eager(seed, n_pairs):
    """Falsification: identical outputs, draw for draw, across 40 calls --
    including picks after earlier solves shrink the unsolved set.
    (600 pairs exercises RandPool.choice's >256 branch.)"""
    pairs = _pool(n_pairs)
    got, _ = _run_new(pairs, seed, 40)
    assert got == _run_ref(pairs, seed, 40)


def test_builds_only_touched_conds(monkeypatch):
    """Falsification: k calls build at most k CondStmt objects, not the pool."""
    calls = {"n": 0}
    real = cond_stmt.CondStmt.from_cmplog_pair.__func__

    def counting(cls, *a, **k):
        calls["n"] += 1
        return real(cls, *a, **k)

    monkeypatch.setattr(cond_stmt.CondStmt, "from_cmplog_pair", classmethod(counting))
    _run_new(_pool(600), 1, 10)
    assert calls["n"] <= 10


def _key(c):
    return (c.base.cmpid, c.base.op_a, c.base.op_b, c.base.width, c.base.result, c.base.pc)


def test_iteration_and_indexing_match_eager():
    """Adversarial: duplicate pairs dedupe to first occurrence; cmpids,
    order and metadata equal conds_from_cmplog_pairs."""
    pairs = _pool(20)
    pairs = pairs + pairs[:5]  # duplicates
    meta = {pairs[1]: (1, 2)}
    pcs = {pairs[2]: 0x401000}
    lazy = LazyConds(pairs, meta, pcs)
    eager = conds_from_cmplog_pairs(pairs, pair_meta=meta, pair_pc=pcs)
    assert len(lazy) == len(eager)
    assert [_key(c) for c in lazy] == [_key(c) for c in eager]
    assert _key(lazy[3]) == _key(eager[3])
    assert lazy[3] is lazy[3]  # cached object: state sticks


def test_all_solved_falls_back_to_any():
    """Adversarial: once every cond is solved the pick spans the whole pool."""
    pairs = _pool(3)
    lazy = LazyConds(pairs, {}, {})
    for c in lazy:
        c.mark_solved()
    assert lazy.unsolved_count() == 0
    assert len(lazy.unsolved_view()) == 0


def test_empty_pool():
    """Adversarial: no pairs -> empty, falsy."""
    lazy = LazyConds([], {}, {})
    assert len(lazy) == 0 and not lazy
