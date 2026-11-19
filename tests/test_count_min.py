"""Count-min sketch (core/count_min.py) and its buffered path counter."""

from __future__ import annotations

import io
import pickle
from collections import Counter

import numpy as np
import pytest

from fuzzer_tool.core.count_min import CountMin, PathFreq, Update
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import safe_load

_WIDE = 1 << 16  # wide enough that a few hundred keys never collide in every row


def _keys(n: int, distinct: int, seed: int = 7) -> np.ndarray:
    """*n* uint64 keys drawn (with repeats) from *distinct* random 64-bit values."""
    rng = RandPool(seed=seed)
    pool = np.frombuffer(rng.randbytes(8 * distinct), dtype=np.uint64)
    return pool[np.asarray(rng.randrange_list(distinct, n), dtype=np.int64)]


def _truth(keys: np.ndarray) -> Counter:
    return Counter(int(k) for k in keys)


# --- Exact regime -------------------------------------------------------------


@pytest.mark.parametrize("update", list(Update))
def test_exact_when_wide(update):
    keys = _keys(2000, 50)
    cms = CountMin(depth=4, width=_WIDE, update=update)
    cms.add(keys)

    truth = _truth(keys)
    uniq = np.fromiter(truth, dtype=np.uint64)
    assert cms.estimate(uniq).tolist() == [truth[int(k)] for k in uniq]
    assert cms.total == 2000


def test_scalar_count_matches_batch():
    cms = CountMin(width=_WIDE)
    cms.add(np.array([5, 5, 9], dtype=np.uint64))
    assert (cms.count(5), cms.count(9), cms.count(11)) == (2, 1, 0)


def test_batch_add_equals_one_by_one():
    keys = _keys(500, 400)
    batch = CountMin(depth=3, width=64)
    batch.add(keys)
    single = CountMin(depth=3, width=64)
    for k in keys:
        single.add(np.array([k], dtype=np.uint64))
    assert np.array_equal(batch.table, single.table)


# --- Bounds -------------------------------------------------------------------


@pytest.mark.parametrize("update", list(Update))
def test_never_underestimates(update):
    # Narrow table: heavy collisions, the estimate must still dominate.
    keys = _keys(5000, 3000)
    cms = CountMin(depth=4, width=256, update=update)
    for chunk in np.array_split(keys, 7):
        cms.add(chunk)

    truth = _truth(keys)
    uniq = np.fromiter(truth, dtype=np.uint64)
    true = np.array([truth[int(k)] for k in uniq])
    assert (cms.estimate(uniq) >= true).all()


def test_plain_error_within_bound():
    # CMS guarantee: est - true <= e*N/width with prob >= 1 - e^-depth.
    keys = _keys(20000, 5000)
    width, depth = 1024, 4
    cms = CountMin(depth=depth, width=width)
    cms.add(keys)

    truth = _truth(keys)
    uniq = np.fromiter(truth, dtype=np.uint64)
    over = cms.estimate(uniq) - np.array([truth[int(k)] for k in uniq])
    bound = np.e * len(keys) / width
    assert np.mean(over > bound) <= np.exp(-depth)


def test_conservative_tighter_than_plain():
    keys = _keys(8000, 4000)
    plain = CountMin(depth=4, width=512)
    cu = CountMin(depth=4, width=512, update=Update.CONSERVATIVE)
    for chunk in np.array_split(keys, 16):
        plain.add(chunk)
        cu.add(chunk)

    uniq = np.unique(keys)
    est_p, est_c = plain.estimate(uniq), cu.estimate(uniq)
    assert (est_c <= est_p).all()
    assert est_c.sum() < est_p.sum()


def test_control_identical_builds_agree():
    # Hard Rule 46: the reference against itself first.
    keys = _keys(3000, 900)
    a, b = CountMin(width=512), CountMin(width=512)
    a.add(keys)
    b.add(keys)
    assert np.array_equal(a.table, b.table)


# --- Falsification / adversarial ----------------------------------------------


def test_falsify_rows_are_independent():
    # Identical row hashes would make depth useless: rows must differ.
    cms = CountMin(depth=4, width=1 << 12)
    cms.add(_keys(64, 64))
    rows = [tuple(np.flatnonzero(r)) for r in cms.table]
    assert len(set(rows)) == len(rows)


def test_adversarial_single_cell():
    # width 1: every key collides; estimate is the stream length.
    cms = CountMin(depth=2, width=1)
    cms.add(np.array([0, 1, 2**64 - 1, 1], dtype=np.uint64))
    assert cms.count(12345) == 4


def test_adversarial_heavy_hitter_and_extremes():
    cms = CountMin(width=_WIDE, update=Update.CONSERVATIVE)
    hot = np.full(10_000, 2**64 - 1, dtype=np.uint64)
    cms.add(hot)
    cms.add(np.array([0], dtype=np.uint64))
    cms.add(np.array([], dtype=np.uint64))
    assert (cms.count(2**64 - 1), cms.count(0)) == (10_000, 1)


@pytest.mark.parametrize("kw", [dict(width=100), dict(width=0), dict(depth=0)])
def test_rejects_bad_shape(kw):
    with pytest.raises(ValueError):
        CountMin(**kw)


def test_merge_adds_streams():
    a, b = CountMin(width=_WIDE), CountMin(width=_WIDE)
    a.add(np.array([1, 2], dtype=np.uint64))
    b.add(np.array([2, 2], dtype=np.uint64))
    a.merge(b)
    assert (a.count(1), a.count(2), a.total) == (1, 3, 4)


@pytest.mark.parametrize("kw", [dict(width=512), dict(depth=3), dict(seed=1)])
def test_merge_rejects_mismatch(kw):
    with pytest.raises(ValueError):
        CountMin(width=_WIDE).merge(CountMin(**{"width": _WIDE, **kw}))


# --- Persistence --------------------------------------------------------------


def test_pickle_roundtrip():
    cms = CountMin(width=256, update=Update.CONSERVATIVE)
    cms.add(_keys(300, 100))
    back = pickle.loads(pickle.dumps(cms))
    assert np.array_equal(back.table, cms.table) and back.total == cms.total


def test_state_dict_survives_safe_loader():
    cms = CountMin(width=256, seed=3)
    cms.add(_keys(300, 100))
    raw = pickle.dumps(cms.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    back = CountMin.from_dict(safe_load(io.BytesIO(raw)))
    assert np.array_equal(back.table, cms.table)
    assert back.count(int(_keys(1, 100)[0])) == cms.count(int(_keys(1, 100)[0]))


def test_from_dict_rejects_corrupt_table():
    state = CountMin(width=256).to_dict()
    state["table"] = state["table"][:-8]
    with pytest.raises(ValueError):
        CountMin.from_dict(state)


# --- PathFreq -----------------------------------------------------------------


def test_path_freq_buffers_then_counts():
    paths = PathFreq(CountMin(width=_WIDE), flush_at=4)
    for h in (7, 7, 9):
        paths.note(h)
    assert paths.pending == 3
    assert paths.count(7) == 2  # count flushes first
    assert paths.pending == 0


def test_path_freq_buffer_is_bounded():
    paths = PathFreq(CountMin(width=_WIDE), flush_at=8)
    for h in range(100):
        paths.note(h)
        assert paths.pending < 8
    assert paths.count(99) == 1


def test_path_freq_counts_batch_and_state():
    paths = PathFreq(CountMin(width=_WIDE))
    for h in (3, 3, 3, 5):
        paths.note(h)
    assert paths.counts([3, 5, 6]).tolist() == [3, 1, 0]

    back = PathFreq.from_dict(pickle.loads(pickle.dumps(paths.to_dict())))
    assert back.count(3) == 3


def test_regression_state_section_survives_store(tmp_path):
    # Protocol-5 ndarrays were refused by the safe loader and the section
    # silently dropped from state.pkl.gz.
    from fuzzer_tool.core.state_store import StateStore

    paths = PathFreq(CountMin(width=256))
    for h in (4, 4, 8):
        paths.note(h)
    store = StateStore(tmp_path)
    store.set("path_freq", paths.to_dict())
    assert store.save()

    back = StateStore(tmp_path)
    back.load()
    assert PathFreq.from_dict(back.get("path_freq")).count(4) == 2
