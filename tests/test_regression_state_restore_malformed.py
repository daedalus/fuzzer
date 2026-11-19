"""Regression: corrupt --resume state crashed the run instead of being dropped.

Found by tools/fuzz.py (`state`, `markov`, `edge_tracker`, `learner_state`):
- `StateStore.load` let TypeError / AttributeError from a malformed pickle
  escape (APPEND on a str, a dict used as a set key).
- Section restores (`from_dict` / `load_state`) crashed on wrong-typed or
  junk-keyed sections, and left the object half-restored.

Contract now: a restore either applies fully or raises ValueError and leaves
the object as it was; `StateStore.restore` logs and drops such a section.
"""

from __future__ import annotations

import gzip
import pickle

import pytest

from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatLearner
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import STATE_FILENAME, StateStore

# Hand-built opcodes: valid pickles that fail while being built.
APPEND_ON_STR = b"\x80\x04X\x01\x00\x00\x00aK\x01a."  # str.append(1)
DICT_IN_SET = b"\x80\x04\x8f(}\x90."  # EMPTY_SET, MARK, {}, ADDITEMS


def _markov() -> MarkovEnsemble:
    m = MarkovEnsemble(rng=RandPool(1))
    m.train(b"\x89PNG\r\n\x1a\nIHDR" * 4)
    return m


def _tracker() -> EdgeTracker:
    et = EdgeTracker(map_size=1024)
    et.record_edges("s0", {1, 2, 3})
    return et


def _learner() -> FormatLearner:
    fl = FormatLearner(rng=RandPool(1))
    fl.record_transition(b"ABCDEFGH", "bit_flip", 2, 1, 10, 12, {5}, set())
    return fl


# (factory, restore, snapshot, malformed sections that crashed before)
_CASES = {
    "markov": (
        _markov,
        MarkovEnsemble.from_dict,
        MarkovEnsemble.to_dict,
        [{"chains": {"zz": {}}}, {"chains": None}, {"transitions": {"zz": {}}}],
    ),
    "edge_tracker": (
        _tracker,
        EdgeTracker.from_dict,
        EdgeTracker.to_dict,
        [{"seed_edges": None}, {"seed_hit_counts": {"s": {"zz": 1}}}, {"cumulative_edges": 5}],
    ),
    "learner_state": (
        _learner,
        FormatLearner.load_state,
        FormatLearner.get_state,
        [{"clusters": {"": 5}}, {"timeline": [{"op": "x"}]}, {"hypotheses": None}],
    ),
}


_EMPTY = {
    "markov": lambda: MarkovEnsemble(rng=RandPool(1)),
    "edge_tracker": lambda: EdgeTracker(map_size=1024),
    "learner_state": lambda: FormatLearner(rng=RandPool(1)),
}


def _gz_state(tmp_path, raw: bytes) -> StateStore:
    (tmp_path / STATE_FILENAME).write_bytes(gzip.compress(raw))
    return StateStore(tmp_path)


@pytest.mark.parametrize("raw", [APPEND_ON_STR, DICT_IN_SET])
def test_regression_corrupt_pickle_loads_empty(tmp_path, raw):
    with pytest.raises(Exception):  # noqa: B017 - precondition: the pickle is malformed
        pickle.loads(raw)

    assert _gz_state(tmp_path, raw).load() == {}


@pytest.mark.parametrize("section", sorted(_CASES))
def test_regression_malformed_section_raises_and_rolls_back(section):
    factory, restore, snapshot, bad = _CASES[section]
    for data in bad:
        obj = factory()
        before = snapshot(obj)

        with pytest.raises(ValueError, match=section):
            restore(obj, data)

        assert snapshot(obj) == before


@pytest.mark.parametrize("section", sorted(_CASES))
def test_falsify_valid_section_still_restores(section):
    factory, restore, snapshot, _ = _CASES[section]
    want = snapshot(factory())
    obj = _EMPTY[section]()

    restore(obj, want)

    assert snapshot(obj) == want


@pytest.mark.parametrize("section", sorted(_CASES))
def test_adversarial_non_dict_section(section):
    factory, restore, snapshot, _ = _CASES[section]
    obj = factory()
    before = snapshot(obj)

    for junk in (None, 5, "x", [1, 2]):
        with pytest.raises(ValueError):
            restore(obj, junk)

    assert snapshot(obj) == before


def test_regression_store_restore_drops_bad_section(tmp_path, caplog):
    store = _gz_state(tmp_path, pickle.dumps({"edge_tracker": {"seed_edges": None}}))
    et = _tracker()
    before = et.to_dict()

    assert store.restore("edge_tracker", et) is False
    assert et.to_dict() == before
    assert "edge_tracker" in caplog.text


def test_store_restore_applies_good_section(tmp_path):
    good = _tracker().to_dict()
    store = _gz_state(tmp_path, pickle.dumps({"edge_tracker": good}))
    et = EdgeTracker(map_size=1024)

    assert store.restore("edge_tracker", et) is True
    assert et.to_dict() == good


def test_store_restore_missing_section(tmp_path):
    store = _gz_state(tmp_path, pickle.dumps({}))

    assert store.restore("edge_tracker", EdgeTracker()) is False


def test_chain_from_dict_is_atomic_too():
    # MarkovChain is restored standalone by legacy callers.
    chain = MarkovChain(order=1)
    chain.train(b"abcabc")
    before = chain.to_dict()

    with pytest.raises(ValueError, match="markov"):
        chain.from_dict({"transitions": {"61": {"zz": 1}}})

    assert chain.to_dict() == before


def test_regression_empty_learner_state_round_trips():
    # `clusters: {}` is falsy: load_state took the legacy path and invented a
    # default cluster, so an empty model did not survive save -> load.
    fl = FormatLearner(rng=RandPool(1))
    state = fl.get_state()

    fl.load_state(state)

    assert fl.get_state() == state
