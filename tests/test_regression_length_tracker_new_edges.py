"""The input-length tracker is credited with the edges a round discovered.

``FuzzRound._track_edges`` passed the input's whole edge trace to
``LengthEdgeTracker.record``, whose contract is "this length produced these
new edges". That scored a length by how much code its inputs execute -- long
inputs win by construction -- and on FFmpeg, where a trace is thousands of
context-sensitive edges, every call overflowed the 200-edge per-length cap
and sorted the whole bucket (~2ms per admitted input).
"""

import types

from fuzzer_tool.core.analyzers.analyzer_length_mi import LengthEdgeTracker
from fuzzer_tool.services.fuzz_round import FuzzRound


def _round(*, novel, new_ids, trace):
    f = types.SimpleNamespace(
        _inprocess_runner=None,
        ptrace_cov=None,
        shm_cov=None,
        _length_tracker=LengthEdgeTracker(),
        _last_new_edge_ids=list(new_ids),
        _current_edges_cache=set(trace),
        _distance=None,
        exec_count=1,
    )
    r = FuzzRound.__new__(FuzzRound)
    r._f = f
    r._meta = None
    r._has_new_coverage = novel
    r._mutated = b"x" * 37
    return r


def test_only_discovered_edges_are_credited():
    r = _round(novel=True, new_ids=[7, 9], trace=range(1, 5000))
    r._track_edges()
    counts = r._f._length_tracker.length_edge_counts[37]
    assert dict(counts) == {7: 1, 9: 1}


def test_bucket_only_novelty_credits_nothing():
    r = _round(novel=True, new_ids=[], trace=range(1, 50))
    r._track_edges()
    assert r._f._length_tracker.total_execs == 0


def test_no_novelty_credits_nothing():
    r = _round(novel=False, new_ids=[3], trace=range(1, 50))
    r._track_edges()
    assert r._f._length_tracker.total_execs == 0
