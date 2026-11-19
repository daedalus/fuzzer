"""Observed branch counts -> likelihood distance (services/branch_counts.py).

Fakes stand in for the SHM segments and TargetDistance; the oracle is the
site -> block sum done by hand.
"""

import ast
import inspect

from fuzzer_tool.core.cadence import bucket
from fuzzer_tool.services.branch_counts import REFRESH_EXECS, BranchCounts

_SITE = "fuzzer.branch_counts"


class _FakeDistance:
    def __init__(self, blocks):
        self._blocks = blocks
        self.reweights = []

    def site_blocks(self):
        return dict(self._blocks)

    def reweight(self, block_hits):
        self.reweights.append(dict(block_hits))

    def pc_distance_table(self):
        return {"table": len(self.reweights)}


class _FakeTable:
    def __init__(self):
        self.rewrites = []

    def rewrite(self, entries):
        self.rewrites.append(entries)


class _FakeHits:
    def __init__(self, script):
        self._script = iter(script)

    def site_hits(self):
        return next(self._script)


def _counts(blocks, script):
    dist, table = _FakeDistance(blocks), _FakeTable()
    return BranchCounts(dist, table, _FakeHits(script)), dist, table


def test_sites_sum_into_blocks():
    bc, dist, table = _counts(
        {0x10: 0x100, 0x20: 0x100, 0x30: 0x200}, [{0x10: 3, 0x20: 4, 0x30: 9}]
    )
    assert bc.refresh() is True
    assert dist.reweights == [{0x100: 7, 0x200: 9}]
    assert table.rewrites == [{"table": 1}]


def test_falsification_unchanged_hits_skip_reweight():
    hits = {0x10: 5}
    bc, dist, table = _counts({0x10: 0x100}, [hits, dict(hits)])
    assert bc.refresh() is True
    assert bc.refresh() is False
    assert len(dist.reweights) == 1 and len(table.rewrites) == 1


def test_adversarial_unmapped_sites_ignored():
    bc, dist, _ = _counts({0x10: 0x100}, [{0x10: 2, 0xDEAD: 10**9}])
    bc.refresh()
    assert dist.reweights == [{0x100: 2}]


def test_adversarial_no_hits_no_reweight():
    bc, dist, table = _counts({0x10: 0x100}, [{}])
    assert bc.refresh() is False
    assert dist.reweights == [] and table.rewrites == []


def test_tick_once_per_bucket():
    script = [{0x10: n} for n in range(1, 10)]
    bc, dist, _ = _counts({0x10: 0x100}, script)
    start = next(c for c in range(REFRESH_EXECS * 2) if bucket(c, REFRESH_EXECS, _SITE) == 1)
    assert bc.tick(start) is True
    assert bc.tick(start + 1) is False  # same bucket
    assert bc.tick(start + REFRESH_EXECS) is True
    assert len(dist.reweights) == 2


def test_periodic_ticks_branch_counts():
    from fuzzer_tool.services import fuzz_round

    src = inspect.getsource(fuzz_round.FuzzRound._periodic)
    calls = [
        n.func.attr
        for n in ast.walk(ast.parse(src.strip().replace("\n    ", "\n")))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "tick" in calls


def test_activation_binds_only_for_likelihood(monkeypatch):
    import os

    from fuzzer_tool.core import analyzer_registry
    from fuzzer_tool.core.analyzers.analyzer_distance import DistanceMetric

    class _TD:
        def __init__(self, *a, **kw):
            self.metric = kw["metric"]
            self._bb_value = {0x100: 1.0}
            self._base_addr = 0
            self.dead_targets = set()
            self.target_addrs = {0x100}
            self.functions = {"f": (0x100, 0x200)}

        def load(self):
            return True

        def pc_distance_table(self):
            return {0x10: 1.0}

        def site_blocks(self):
            return {0x10: 0x100}

    monkeypatch.setattr("fuzzer_tool.core.analyzers.analyzer_distance.TargetDistance", _TD)
    seen = {}
    for metric in (DistanceMetric.HARMONIC, DistanceMetric.LIKELIHOOD):
        os.environ.pop("__AFL_DIST_HITS_ID", None)
        f = type(
            "F",
            (),
            {
                "target": "/bin/true",
                "_distance_targets": ["main"],
                "_use_cfg_cache": False,
                "debug": False,
                "_distance_metric": metric,
            },
        )()
        try:
            analyzer_registry._activate_distance(f)
            seen[metric] = (f._branch_counts, os.environ.get("__AFL_DIST_HITS_ID"))
        finally:
            if f._branch_counts is not None:
                f._branch_counts.cleanup()
            f._dist_table_shm.cleanup()
            os.environ.pop("__AFL_DIST_HITS_ID", None)
            os.environ.pop("__AFL_DIST_SHM_ID", None)

    assert seen[DistanceMetric.HARMONIC] == (None, None)
    bc, env = seen[DistanceMetric.LIKELIHOOD]
    assert isinstance(bc, BranchCounts) and env
