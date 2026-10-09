"""collect_structure_map: hand the collector's own pool to the caches.

It copied ``cmplog.pairs`` and ``cmplog._pair_pc`` per call. The copy is a
new object, so ``scanner_for_pairs`` (keyed on pool identity) rebuilt the
Aho-Corasick automaton and ``_tag_plan`` (skipped whenever ``pair_pcs`` was
passed) re-sorted the pool every admitted seed: 54 ms per call, 7% of a
2000-exec ``--hail-mary`` profile. The plan cache now also keys on the
identity and length of the pc map; the collector only adds a pc when it
appends that pair, and rebinds ``pairs`` on eviction.
"""

import random
from types import SimpleNamespace

import pytest

from fuzzer_tool.core import aho_corasick as ac
from fuzzer_tool.core import weizz_tags as wt


@pytest.fixture(autouse=True)
def _fresh_caches():
    ac._reset_scanner_cache()
    wt._plans.clear()
    yield
    ac._reset_scanner_cache()
    wt._plans.clear()


@pytest.fixture
def plans(monkeypatch):
    """Count full plan builds (one _pair_key sort per build)."""
    count = []
    real = wt._sized_operands

    def _counting(op_a, op_b, cfg):
        count.append(1)
        return real(op_a, op_b, cfg)

    monkeypatch.setattr(wt, "_sized_operands", _counting)
    return count


def _collector(rnd, n):
    pairs = [(rnd.randbytes(rnd.choice((2, 4))), rnd.randbytes(4)) for _ in range(n)]
    pcs = {p: rnd.randrange(1, 1 << 20) for p in pairs[::3]}
    return SimpleNamespace(pairs=pairs, _pair_pc=pcs)


def _grow(cmplog, rnd, k):
    """Collector growth: append new pairs, record a pc for some (cmplog._ingest_conds)."""
    for _ in range(k):
        pair = (rnd.randbytes(4), rnd.randbytes(4))
        cmplog.pairs.append(pair)
        if rnd.random() < 0.5:
            cmplog._pair_pc[pair] = rnd.randrange(1, 1 << 20)


def _seed(rnd, cmplog):
    return rnd.randbytes(300) + b"".join(a for a, _ in cmplog.pairs[::40])


def _old(data, cmplog):
    """Pre-change behaviour: fresh copies, uncached plan."""
    return wt.build_tag_map_from_cmplog(
        data, list(cmplog.pairs), pair_pcs=dict(cmplog._pair_pc), config=wt.TagCollectorConfig()
    )


# ---------------------------------------------------------------------------


def test_regression_weizz_collect_caches(plans):
    """Same pool, second seed: no new plan, no new automaton."""
    rnd = random.Random(0)
    cmplog = _collector(rnd, 700)
    wt.collect_structure_map(_seed(rnd, cmplog), cmplog)
    first = len(plans)
    scanner = ac.scanner_for_pairs(cmplog.pairs)

    wt.collect_structure_map(_seed(rnd, cmplog), cmplog)
    assert len(plans) == first
    assert ac.scanner_for_pairs(cmplog.pairs) is scanner


def test_tags_match_uncached_copies_through_growth():
    """Falsification: identical tag maps to the copy path, across growth with new pcs."""
    rnd = random.Random(1)
    cmplog = _collector(rnd, 600)
    for step in range(12):
        data = _seed(rnd, cmplog)
        got = wt.collect_structure_map(data, cmplog)
        want = _old(data, cmplog)
        # Control (Hard Rule 46): the oracle against a second run of itself.
        assert _old(data, cmplog).tags == want.tags
        assert got.tags == want.tags
        assert (got.ntypes, got.max_counter) == (want.ntypes, want.max_counter)
        _grow(cmplog, rnd, 1 + step * 3)


def test_eviction_rebind_rebuilds(plans):
    """Adversarial: eviction rebinds pairs and drops pcs; nothing stale survives."""
    rnd = random.Random(2)
    cmplog = _collector(rnd, 600)
    wt.collect_structure_map(_seed(rnd, cmplog), cmplog)
    before = len(plans)

    victims = set(cmplog.pairs[:100])
    for p in victims:
        cmplog._pair_pc.pop(p, None)
    cmplog.pairs = [p for p in cmplog.pairs if p not in victims]

    data = _seed(rnd, cmplog)
    assert wt.collect_structure_map(data, cmplog).tags == _old(data, cmplog).tags
    assert len(plans) > before


def test_mutator_and_seed_paths_keep_separate_plans(plans):
    """Adversarial: alternating pc-less and pc'd builds on one pool do not evict."""
    rnd = random.Random(4)
    cmplog = _collector(rnd, 600)
    data = _seed(rnd, cmplog)
    wt.collect_structure_map(data, cmplog)
    wt.build_tag_map_from_cmplog(data, cmplog.pairs)
    first = len(plans)
    for _ in range(3):
        wt.collect_structure_map(data, cmplog)
        wt.build_tag_map_from_cmplog(data, cmplog.pairs)
    assert len(plans) == first


def test_fresh_pc_dict_still_uncached(plans):
    """Adversarial: same pool, a new pc dict per call: a fresh plan each time."""
    rnd = random.Random(3)
    cmplog = _collector(rnd, 50)
    data = _seed(rnd, cmplog)

    def _call():
        wt.build_tag_map_from_cmplog(data, cmplog.pairs, pair_pcs=dict(cmplog._pair_pc))

    _call()
    first = len(plans)
    _call()
    assert len(plans) == 2 * first
