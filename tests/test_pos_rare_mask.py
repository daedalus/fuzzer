"""PositionRareMaskScheduler: FairFuzz-style rare-branch mask over offsets.

Covers core/schedulers/pos_rare_mask.py.
"""

from fuzzer_tool.core.schedulers.pos_base import Outcome, PositionScheduler
from fuzzer_tool.core.schedulers.pos_rare_mask import (
    PRIOR_A,
    PRIOR_B,
    RARE_MAX,
    RETARGET_EVERY,
    PositionRareMaskScheduler,
)

SEED = bytes(64)


class ScriptedRng:
    def random(self):
        return 0.5

    def randint(self, a, b):
        return a


class _Env:
    """Seed edges, owner counts and this exec's hit set, all scripted."""

    def __init__(self, edges, owners, hit):
        self.edges = edges
        self.owners = owners
        self.hit = hit
        self.hit_queries = []

    def sched(self):
        return PositionRareMaskScheduler(
            ScriptedRng(),
            edges_of=lambda d: self.edges,
            owner_count=lambda e: self.owners.get(e, 0),
            hit=self._hit,
        )

    def _hit(self, e):
        self.hit_queries.append(e)
        return None if self.hit is None else e in self.hit


def _rate(s, n):
    return (s + PRIOR_A) / (n + PRIOR_A + PRIOR_B)


class TestProtocol:
    def test_satisfies_the_protocol(self):
        assert isinstance(_Env({1}, {1: 1}, hit={1}).sched(), PositionScheduler)


class TestTarget:
    def test_rarest_edge_is_the_target(self):
        env = _Env({1, 2, 3}, {1: 9, 2: 1, 3: 3}, hit={2})
        assert env.sched().target(SEED) == 2

    def test_ties_break_to_the_lowest_id(self):
        env = _Env({5, 4}, {4: 1, 5: 1}, hit=set())
        assert env.sched().target(SEED) == 4

    def test_no_rare_edge_no_target(self):
        env = _Env({1}, {1: RARE_MAX + 1}, hit={1})
        s = env.sched()
        assert s.target(SEED) is None
        s.record(SEED, [3], Outcome.MISS)
        assert s.propose(SEED, len(SEED)) is None
        assert env.hit_queries == []  # nothing to check, nothing scanned

    def test_adversarial_no_edges(self):
        for edges in (None, set()):
            env = _Env(edges, {}, hit=set())
            assert env.sched().target(SEED) is None


class TestMask:
    def test_preserving_offsets_gain_rate(self):
        env = _Env({2}, {2: 1}, hit={2})
        s = env.sched()
        s.record(SEED, [4, 9], Outcome.MISS)
        w = s.weights(SEED, len(SEED))
        assert w[4] == w[9] == _rate(1, 1)

    def test_breaking_offsets_lose_rate(self):
        env = _Env({2}, {2: 1}, hit=set())
        s = env.sched()
        s.record(SEED, [4], Outcome.GAIN)  # gain is irrelevant: the edge was lost
        assert s.weights(SEED, len(SEED))[4] == _rate(0, 1)

    def test_falsification_mask_separates_bins(self):
        env = _Env({2}, {2: 1}, hit={2})
        s = env.sched()
        for _ in range(10):
            s.record(SEED, [10], Outcome.MISS)
        env.hit = set()
        for _ in range(10):
            s.record(SEED, [20], Outcome.MISS)
        w = s.weights(SEED, len(SEED))
        assert w[10] > w[0] > w[20]

    def test_unknown_hit_is_not_credited(self):
        env = _Env({2}, {2: 1}, hit=None)
        s = env.sched()
        s.record(SEED, [4], Outcome.MISS)
        assert s.weights(SEED, len(SEED)) is None

    def test_retarget_resets_the_mask(self):
        env = _Env({2, 3}, {2: 1, 3: 2}, hit={2})
        s = env.sched()
        s.record(SEED, [4], Outcome.MISS)
        env.owners = {2: 3, 3: 1}  # edge 3 is now the rarest
        for _ in range(RETARGET_EVERY):
            s.record(SEED, [5], Outcome.MISS)
        assert s.target(SEED) == 3
        # Evidence collected against edge 2 is gone.
        assert s.weights(SEED, len(SEED))[4] == _rate(0, 0)

    def test_regression_target_is_cached_between_retargets(self):
        env = _Env({2, 3}, {2: 1, 3: 2}, hit={2})
        s = env.sched()
        s.record(SEED, [4], Outcome.MISS)
        env.owners = {2: 3, 3: 1}
        s.record(SEED, [4], Outcome.MISS)
        assert s.target(SEED) == 2


class TestShmHasEdge:
    """The adapter probe rare_mask's ``hit`` reads (Fuzzer._edge_hit)."""

    def test_reports_this_table_only(self):
        from fuzzer_tool.adapters.shm import ShmCoverage

        cov = ShmCoverage(size=1024)
        try:
            cov.record_edge(5)
            assert cov.has_edge(5)
            assert not cov.has_edge(6)
            cov.reset_edge_map()  # adversarial: a stale entry must not count
            assert not cov.has_edge(5)
        finally:
            cov.cleanup()
