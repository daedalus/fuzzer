"""``_EdgeVectors.prepare`` re-reads only changed owner counts.

Every weight pass gathered the owner count of every indexed edge id: 88k
dict lookups, ~19 ms of a ~100 ms pass at 9.4k ffmpeg seeds. The tracker now
logs the ids whose count moved; a prune, reload or overflow forces the full
gather. Reference: a fresh full gather.
"""

from collections import defaultdict

import numpy as np

from fuzzer_tool.core import edge_tracker as et_mod
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.seed_picker import _EdgeVectors


def _full_owners(vectors: _EdgeVectors, et: EdgeTracker) -> list[int]:
    get = et._edge_owner_count.get
    return [get(e, 0) for e in vectors._ids]


def _pass(vectors: _EdgeVectors, et: EdgeTracker) -> None:
    sets = [s for s in et.seed_edges.values() if s]
    vectors.prepare(sets, et._edge_owner_count, [], changes=et.drain_owner_changes())


def _record(et: EdgeTracker, rng: RandPool, key: str) -> None:
    edges = set(rng.randint_list(1, 600, rng.randint(1, 60)))
    et.record_edges(key, edges)


def test_regression_owner_gather_control():
    # Control (Hard Rule 46): two full gathers agree.
    et, rng = EdgeTracker(), RandPool(seed=1)
    for i in range(10):
        _record(et, rng, f"s{i}")
    v = _EdgeVectors()
    _pass(v, et)
    assert _full_owners(v, et) == _full_owners(v, et)


def test_regression_owner_gather_matches_full():
    for seed in range(5):
        et, rng, v = EdgeTracker(), RandPool(seed=seed), _EdgeVectors()
        for i in range(40):
            _record(et, rng, f"s{rng.randint(0, 12)}")
            if i % 3 == 0:
                _pass(v, et)
                assert v.owners.tolist() == _full_owners(v, et), (seed, i)


def test_regression_owner_gather_prune_rebuilds():
    # Adversarial: a prune replaces the owner dict and decrements counts.
    et, rng, v = EdgeTracker(), RandPool(seed=9), _EdgeVectors()
    et.max_tracked_seeds = 8
    for i in range(30):
        _record(et, rng, f"s{i}")
        _pass(v, et)
        assert v.owners.tolist() == _full_owners(v, et), i


def test_regression_owner_gather_reload_rebuilds():
    et, rng, v = EdgeTracker(), RandPool(seed=2), _EdgeVectors()
    for i in range(6):
        _record(et, rng, f"s{i}")
    _pass(v, et)
    et.from_dict(et.to_dict())
    _record(et, rng, "s0")
    _pass(v, et)
    assert v.owners.tolist() == _full_owners(v, et)


def test_regression_owner_gather_reads_only_changes():
    # Falsification: an incremental pass looks up the changed ids only.
    looked = []

    class _Spy(defaultdict):
        def get(self, k, d=None):
            looked.append(k)
            return super().get(k, d)

    et, rng, v = EdgeTracker(), RandPool(seed=3), _EdgeVectors()
    et._edge_owner_count = _Spy(int)
    for i in range(10):
        _record(et, rng, f"s{i}")
    _pass(v, et)
    looked.clear()
    et.record_edges("s0", {1, 2})
    _pass(v, et)
    assert set(looked) <= {1, 2}
    assert v.owners.tolist() == _full_owners(v, et)


def test_regression_owner_log_bounded(monkeypatch):
    # Memory bound: past the cap the log drops and asks for a rebuild.
    monkeypatch.setattr(et_mod, "OWNER_LOG_MAX", 5)
    et = EdgeTracker()
    assert et.drain_owner_changes() is None  # first drain: rebuild
    et.record_edges("a", {1, 2, 3})
    assert et.drain_owner_changes() == [{1, 2, 3}]
    et.record_edges("b", set(range(10, 20)))
    assert et.drain_owner_changes() is None
    assert np.array_equal([], et._owner_log)
