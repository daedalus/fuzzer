"""PositionHotspotScheduler: offsets ranked by how many unsolved comparisons they block.

Covers core/schedulers/pos_hotspot.py. Uses real ``LazyConds`` so the
CondStmt state machine (``mark_solved``) the arm prunes on is pinned here.
"""

from fuzzer_tool.core.cond_stmt import CondState, LazyConds
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_hotspot import (
    EPSILON,
    MAX_CONDS,
    MAX_HITS,
    MAX_SEEDS,
    MIN_OPERAND,
    SCAN_BYTES,
    PositionHotspotScheduler,
)
from tests.support.scripted_rng import ScriptedRng

NO_ESCAPE = 0.99  # random() draw above EPSILON

# "ABCDEF" at 4: cmp 0 (ABCD) blocks 4..7, cmp 1 (CDEF) blocks 6..9.
SEED = b"\x00" * 4 + b"ABCDEF" + b"\x00" * 6
PAIRS = [(b"ABCD", b"WXYZ"), (b"CDEF", b"QRST")]


def _brute(data: bytes, conds) -> dict[int, int]:
    """Oracle: offset -> number of UNSOLVED conds whose operand covers it."""
    out: dict[int, set[int]] = {}
    for c in conds:
        if c.state is not CondState.UNSOLVED:
            continue
        for op in (c.base.op_a, c.base.op_b):
            if len(op) < MIN_OPERAND:
                continue
            for i in range(len(data) - len(op) + 1):
                if data[i : i + len(op)] != op:
                    continue
                for j in range(i, i + len(op)):
                    out.setdefault(j, set()).add(c.base.cmpid)
    return {o: len(ids) for o, ids in out.items()}


def _hottest(counts: dict[int, int], buf_len: int) -> list[int]:
    inside = {o: n for o, n in counts.items() if o < buf_len}
    top = max(inside.values(), default=0)
    return sorted(o for o, n in inside.items() if n == top)


class _Spy:
    """``conds_of`` that counts how often a table is (re)built."""

    def __init__(self, conds):
        self.conds = conds
        self.views = 0
        real = conds.unsolved_view

        def view():
            self.views += 1
            return real()

        conds.unsolved_view = view  # instance attribute shadows the method

    def __call__(self):
        return self.conds


class _SpyConds(LazyConds):
    """LazyConds with a ``__dict__``, so the spy can shadow ``unsolved_view``."""


def _conds(pairs=PAIRS):
    return _SpyConds(pairs, None, None)


def _sched(conds, randoms=(NO_ESCAPE,), choices=(0,)):
    rng = ScriptedRng(randoms=randoms, choice_idxs=choices)
    spy = _Spy(conds)
    return PositionHotspotScheduler(rng, conds_of=spy), spy


class TestProtocol:
    def test_satisfies_the_protocol(self):
        s, _ = _sched(_conds())
        assert isinstance(s, PositionScheduler)
        assert s.name == "hotspot"

    def test_record_is_a_noop(self):
        s, _ = _sched(_conds())
        s.record(SEED, [5], Outcome.GAIN)
        assert s.blocked(SEED) == _brute(SEED, _conds())


class TestTable:
    def test_counts_match_brute_force(self):
        conds = _conds()
        s, _ = _sched(conds)
        assert s.blocked(SEED) == _brute(SEED, conds)

    def test_proposes_from_the_hottest_tier(self):
        conds = _conds()
        hot = _hottest(_brute(SEED, conds), len(SEED))
        s, _ = _sched(conds, choices=(len(hot) - 1,))
        assert s.propose(SEED, len(SEED)) == hot[-1]

    def test_falsification_solved_cond_leaves_the_table(self):
        conds = _conds()
        s, spy = _sched(conds, randoms=(NO_ESCAPE, NO_ESCAPE), choices=(0, 0))
        before = s.propose(SEED, len(SEED))
        conds[1].mark_solved()

        after = s.blocked(SEED)
        assert after == _brute(SEED, conds)
        assert before in after and after[before] < 2  # the overlap cooled
        assert spy.views == 1  # pruned in place, not rebuilt

    def test_new_cond_list_rebuilds(self):
        conds = _conds()
        holder = {"c": conds}
        s = PositionHotspotScheduler(ScriptedRng(), conds_of=lambda: holder["c"])
        assert s.blocked(SEED)
        holder["c"] = _conds([(b"\x00\x00\x00", b"\xff\xff\xff")])
        assert s.blocked(SEED) == _brute(SEED, holder["c"])

    def test_escape_declines(self):
        s, _ = _sched(_conds(), randoms=(EPSILON / 2,), choices=())
        assert s.propose(SEED, len(SEED)) is None

    def test_shrunk_buffer_uses_the_hottest_offset_inside_it(self):
        conds = _conds()
        buf_len = 6  # only 4, 5 (count 1) remain
        hot = _hottest(_brute(SEED, conds), buf_len)
        s, _ = _sched(conds, choices=(len(hot) - 1,))
        assert s.propose(SEED, buf_len) == hot[-1] < buf_len

    def test_emptied_table_refills_from_the_next_unsolved(self):
        pairs = [(bytes([65 + i % 26, 97 + i // 26]), b"\xfe\xfe") for i in range(MAX_CONDS + 1)]
        data = b"".join(a for a, _ in pairs)
        conds = _conds(pairs)
        s, spy = _sched(conds)
        s.blocked(data)
        for k in range(MAX_CONDS):
            conds[k].mark_solved()

        assert s.blocked(data) == _brute(data, conds) != {}
        assert spy.views == 2

    def test_empty_build_is_not_rebuilt_every_call(self):
        s, spy = _sched(_conds([(b"none", b"zzzz")]))
        s.blocked(SEED)
        s.blocked(SEED)
        assert spy.views == 1


class TestAdversarial:
    def test_no_conds_declines(self):
        s, _ = _sched(_conds([]), randoms=(), choices=())
        assert s.propose(SEED, len(SEED)) is None

    def test_empty_inputs_decline(self):
        s, _ = _sched(_conds(), randoms=(), choices=())
        assert s.propose(b"", 8) is None
        assert s.propose(SEED, 0) is None

    def test_all_solved_declines(self):
        conds = _conds()
        for c in conds:
            c.mark_solved()
        s, _ = _sched(conds, randoms=(), choices=())
        assert s.propose(SEED, len(SEED)) is None

    def test_buffer_shorter_than_every_hotspot_declines(self):
        s, _ = _sched(_conds(), randoms=(NO_ESCAPE,), choices=())
        assert s.propose(SEED, 4) is None

    def test_short_operands_are_ignored(self):
        # One-byte operands match everywhere: noise, not a blocker.
        s, _ = _sched(_conds([(b"\x00", b"\x01")]), randoms=(), choices=())
        assert s.blocked(SEED) == {}
        assert s.propose(SEED, len(SEED)) is None

    def test_operand_past_the_scan_prefix_is_ignored(self):
        data = b"\x00" * SCAN_BYTES + b"ABCD"
        s, _ = _sched(_conds([(b"ABCD", b"WXYZ")]))
        assert s.blocked(data) == {}

    def test_repeated_operand_is_capped(self):
        data = b"AB" * (MAX_HITS * 4)
        s, _ = _sched(_conds([(b"AB", b"ZZ")]))
        assert len(s.blocked(data)) == MAX_HITS * 2

    def test_cond_window_is_bounded(self):
        pairs = [(bytes([65 + i % 26, 97 + i // 26]), b"\xfe\xfe") for i in range(MAX_CONDS * 2)]
        data = b"".join(a for a, _ in pairs)
        s, _ = _sched(_conds(pairs))
        assert len(s.blocked(data)) <= MAX_CONDS * 2

    def test_lru_evicts_oldest_seed(self):
        conds = _conds()
        s, spy = _sched(conds)
        seeds = [SEED + i.to_bytes(4, "little") for i in range(MAX_SEEDS + 1)]
        for d in seeds:
            s.blocked(d)
        assert s.seed_count() == MAX_SEEDS
        s.blocked(seeds[0])  # evicted: built again
        assert spy.views == MAX_SEEDS + 2


def test_real_randpool_stays_in_bounds():
    s = PositionHotspotScheduler(RandPool(seed=3), conds_of=lambda: _conds())
    for n in (1, 6, len(SEED)):
        pos = s.propose(SEED, n)
        assert pos is None or 0 <= pos < n
