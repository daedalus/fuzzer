"""--renyi-weight dump: ``drift_half`` = D_1/2(current hits || previous dump's).

Bhattacharyya form, independent of renyi_divergence:
D_1/2(P||Q) = -2 log2 sum_i sqrt(p_i q_i) after normalization.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

from fuzzer_tool.services.stats import _INF_JSON, _renyi_stats


def _fuzzer(hits: dict[int, int]) -> SimpleNamespace:
    return SimpleNamespace(
        _use_renyi_weight=True,
        _edge_tracker=SimpleNamespace(_global_edge_hits=hits),
        _renyi_prev_hits=None,
    )


def _dump(f: SimpleNamespace, hits: dict[int, int]) -> dict:
    f._edge_tracker._global_edge_hits = hits
    stats: dict = {}
    _renyi_stats(f, stats)
    return stats["renyi"]


def _bhattacharyya(p: dict[int, int], q: dict[int, int]) -> float:
    ps, qs = sum(p.values()), sum(q.values())
    bc = sum(math.sqrt(p[k] / ps * q.get(k, 0) / qs) for k in p)
    return -2 * math.log2(bc)


def test_first_dump_has_no_drift():
    f = _fuzzer({1: 5})

    assert "drift_half" not in _dump(f, {1: 5, 2: 3})


def test_identical_dumps_zero():
    """Control: a distribution against itself drifts 0."""
    f = _fuzzer({})
    _dump(f, {1: 5, 2: 3})

    assert _dump(f, {1: 5, 2: 3})["drift_half"] == 0.0


def test_scaled_dumps_zero():
    """Falsification: counts that only grow proportionally are not drift
    (a raw-count diff would report it)."""
    f = _fuzzer({})
    _dump(f, {1: 5, 2: 3})

    assert _dump(f, {1: 50, 2: 30})["drift_half"] == 0.0


def test_shift_matches_bhattacharyya():
    prev, cur = {1: 8, 2: 2}, {1: 3, 2: 5, 3: 2}
    f = _fuzzer({})
    _dump(f, prev)

    drift = _dump(f, cur)["drift_half"]

    assert drift > 0
    assert math.isclose(drift, round(_bhattacharyya(cur, prev), 4))


def test_snapshot_replaced_each_dump():
    """Bounded: only the latest dump is kept, so drift is vs the last one."""
    a, b = {1: 8, 2: 2}, {1: 3, 2: 5}
    f = _fuzzer({})
    _dump(f, a)
    _dump(f, b)

    assert _dump(f, b)["drift_half"] == 0.0
    assert f._renyi_prev_hits == b


def test_adversarial_disjoint_supports_json_safe():
    """inf (no shared edge) is written as _INF_JSON: strict JSON has no Infinity."""
    f = _fuzzer({})
    _dump(f, {1: 4})

    renyi = _dump(f, {2: 4})

    assert renyi["drift_half"] == _INF_JSON
    json.dumps(renyi, allow_nan=False)


def test_adversarial_empty_hits_keep_snapshot():
    """No hits (e.g. after a map resize): no entry, previous snapshot kept."""
    f = _fuzzer({})
    _dump(f, {1: 4})
    stats: dict = {}
    f._edge_tracker._global_edge_hits = {}
    _renyi_stats(f, stats)

    assert "renyi" not in stats
    assert f._renyi_prev_hits == {1: 4}
