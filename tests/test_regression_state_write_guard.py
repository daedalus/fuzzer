"""Tests for the write-time state guard (§7.2 of the thermo handover).

``StateStore`` validated on *read* and not on *write*, so a component could
serialise a payload its own loader would reject -- and the reader's failure is
all-or-nothing, so one bad field cost every section. That is what
``CoverageRegimeDetector`` did with ``regime_history``: a measured campaign
resumed with **zero of nineteen** sections recovered, logging a message that
blamed tampering for a payload this codebase wrote itself.

The sweep that motivated this found no second instance -- all 22 persisted
sections are clean, checked by auditing a real campaign's state file (19
sections) plus static construction of the three it did not reach (``mi``,
``cmaes``, ``katz``). So this guard is not fixing a live bug. It exists
because enumerating ``save()`` methods cannot prevent the next one, and
because the cost of the next one is total rather than local.

The guard runs the real unpickler over the pickled bytes before compressing
them, so it cannot drift from the reader or be wrong about the reader's rules.
It costs a second pass -- 64 ms against the 2.26 s a real 5.9 MiB save already
spends, ~2.8%, nearly all of which is gzip.
"""

from __future__ import annotations

import enum
import gzip
import pickle

import pytest

from fuzzer_tool.core.percolation import CoverageRegime
from fuzzer_tool.core.state_store import (
    _ALLOWED_GLOBALS,
    StateStore,
    UnsafeStateError,
    _safe_loads,
    _verify_readable,
)


class _Custom:
    pass


class _Colour(enum.Enum):
    RED = "red"


def _dumps_guarded(obj) -> None:
    """Assert the reader would accept *obj* if it were saved."""
    _verify_readable(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


# --- symmetry with the reader ---------------------------------------------


def test_the_motivating_payload_is_refused_on_the_way_out() -> None:
    with pytest.raises(UnsafeStateError) as exc:
        _dumps_guarded({"regime": {"regime_history": [(1, CoverageRegime.CRITICAL)]}})
    assert "CoverageRegime" in str(exc.value)


@pytest.mark.parametrize("obj", [_Custom(), _Colour.RED, object()])
def test_arbitrary_objects_are_refused(obj) -> None:
    with pytest.raises(UnsafeStateError):
        _dumps_guarded({"section": obj})


def test_an_unpicklable_object_is_also_isolated(tmp_path, caplog) -> None:
    """Two failure modes, one blast radius.

    A lambda, a lock or an open file raises PicklingError/TypeError at dump
    time rather than UnsafeStateError at verify time. Catching only the
    latter would leave that case aborting the whole save from inside one
    section -- the same all-or-nothing failure this guard exists to break up.
    """
    store = StateStore(tmp_path)
    store.set("good", {"x": 1})
    store.set("unpicklable", {"f": lambda: None})
    assert store.save() is True
    assert sorted(StateStore(tmp_path).load()) == ["good"]


def test_a_refused_type_is_refused_however_deeply_nested() -> None:
    """Depth is what made the original defect survive: the sibling field one
    line above was converted correctly."""
    payload = {"a": {"b": [{"c": (1, 2, {"d": [CoverageRegime.CRITICAL]})}]}}
    with pytest.raises(UnsafeStateError):
        _dumps_guarded(payload)


def test_writer_and_reader_agree_on_numpy_scalars() -> None:
    """Numpy scalars are on the allowlist and must pass both sides."""
    import numpy as np

    payload = {"f": np.float64(1.5), "i": np.int64(7)}
    _dumps_guarded(payload)
    assert _safe_loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)) == payload


def test_the_writer_refuses_the_ndarray_the_reader_already_rejects() -> None:
    """A finding, not a design choice: the allowlist is stale for numpy 2.x.

    ``_ALLOWED_GLOBALS`` lists ``numpy.ndarray`` and its comment says
    "numpy scalars/arrays appear in a few saved payloads". Under numpy 2.4.4
    an ndarray does not reduce through the allowlisted
    ``numpy._core.multiarray._reconstruct`` but through
    ``numpy._core.numeric._frombuffer``, which is not listed -- so an
    ndarray written to state today cannot be read back, and under the old
    unguarded writer it would have taken every other section with it. Scalars
    are unaffected.

    No current section carries one: all 22 persisted sections were audited
    clean (19 from a real campaign's state file, plus ``mi``, ``cmaes`` and
    ``katz`` by static construction -- ``katz.state_dict()`` already converts
    with ``.tolist()``). So this is latent rather than live, and the guard
    downgrades it from silent total loss at resume to a named section dropped
    at write.

    Asserted in both directions so whichever way the allowlist question is
    settled -- widen it to include the numpy 2.x reduction, or require
    components to use ``.tolist()`` -- this test fails and has to be updated
    deliberately.
    """
    import numpy as np

    blob = pickle.dumps({"arr": np.arange(4)}, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(UnsafeStateError):
        _safe_loads(blob)
    with pytest.raises(UnsafeStateError):
        _dumps_guarded({"arr": np.arange(4)})


def test_writer_and_reader_share_one_allowlist_table() -> None:
    """Not a copy of it: a divergent copy is how the two sides drift apart."""
    import inspect

    from fuzzer_tool.core import state_store

    src = inspect.getsource(state_store._SafeUnpickler)
    assert "_ALLOWED_GLOBALS" in src
    assert ("builtins", "dict") in _ALLOWED_GLOBALS


def test_everything_a_clean_payload_contains_is_accepted() -> None:
    import collections

    _dumps_guarded(
        {
            "scalars": [1, 1.5, True, None, complex(1, 2), "s", b"b", bytearray(b"c")],
            "containers": ({1, 2}, frozenset({3}), [4], (5,), {"k": "v"}),
            "collections": {
                "od": collections.OrderedDict(a=1),
                "dd": collections.defaultdict(int, {"a": 1}),
                "dq": collections.deque([1, 2], maxlen=4),
                "ct": collections.Counter("aab"),
            },
        }
    )


# --- save() behaviour ------------------------------------------------------


def test_a_bad_section_is_dropped_and_the_others_survive(tmp_path, caplog) -> None:
    """The point of the fallback. Before the guard this lost all three."""
    store = StateStore(tmp_path)
    store.set("good_a", {"x": [1, 2, 3]})
    store.set("bad", {"history": [(1, CoverageRegime.CRITICAL)]})
    store.set("good_b", {"y": b"abc"})
    assert store.save() is True

    recovered = StateStore(tmp_path).load()
    assert sorted(recovered) == ["good_a", "good_b"]
    assert recovered["good_a"] == {"x": [1, 2, 3]}


def test_the_dropped_section_is_named_in_the_log(tmp_path, caplog) -> None:
    """The read-side message named nothing and blamed tampering."""
    store = StateStore(tmp_path)
    store.set("regime", {"history": [(1, CoverageRegime.CRITICAL)]})
    store.set("corpus", {"seeds": 1})
    with caplog.at_level("ERROR"):
        store.save()
    text = caplog.text
    assert "regime" in text
    assert "CoverageRegime" in text
    assert "tamper" not in text.lower()


def test_a_clean_payload_round_trips_untouched(tmp_path) -> None:
    store = StateStore(tmp_path)
    sections = {
        "corpus": {"seeds": 12, "hashes": ["a", "b"]},
        "edge_tracker": {"edges": [1, 2, 3], "owners": {"1": 2}},
        "regime": {"regime": "critical", "regime_history": [(1, "critical")]},
    }
    for k, v in sections.items():
        store.set(k, v)
    assert store.save() is True
    assert StateStore(tmp_path).load() == sections


def test_no_temp_file_is_left_behind_after_the_fallback(tmp_path) -> None:
    """The fallback allocates a second temp file; the first must be cleaned."""
    store = StateStore(tmp_path)
    store.set("ok", {"a": 1})
    store.set("bad", {"b": CoverageRegime.CRITICAL})
    store.save()
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / store.path.name).exists()


def test_the_written_file_is_still_a_gzipped_pickle(tmp_path) -> None:
    """Format unchanged: the guard swaps the Pickler, not the container."""
    store = StateStore(tmp_path)
    store.set("a", {"x": 1})
    store.save()
    with gzip.open(store.path, "rb") as fh:
        assert pickle.loads(fh.read()) == {"a": {"x": 1}}


def test_an_all_bad_payload_writes_an_empty_dict_rather_than_failing(tmp_path) -> None:
    store = StateStore(tmp_path)
    store.set("bad", {"b": CoverageRegime.CRITICAL})
    assert store.save() is True
    assert StateStore(tmp_path).load() == {}


# --- the guard must not be satisfiable by widening the allowlist ----------


def test_the_guard_is_the_reader_and_not_a_second_implementation() -> None:
    """Pin the mechanism, because the alternative was wrong three times.

    A heuristic ``reducer_override`` written against ``_ALLOWED_GLOBALS``
    looked cheaper (0.9% against 2.8%) and got three cases wrong in a row:
    class objects passed as reduction arguments, ``_frombuffer``, and
    ``numpy.float64``, whose reduction names ``scalar`` rather than the
    scalar's own type. Running the real unpickler cannot be wrong by
    construction, and cannot drift from the reader because it *is* the
    reader. If this is ever swapped for a type walk, that is a deliberate
    trade and this test should make it visible.
    """
    import inspect

    from fuzzer_tool.core import state_store

    assert "_safe_loads" in inspect.getsource(state_store._verify_readable)


def test_the_saved_bytes_are_an_ordinary_pickle() -> None:
    """The guard changes what is refused, never what is written."""
    payload = {"a": [1, 2, {"b": (3, 4)}], "c": b"x", "d": {5, 6}}
    _verify_readable(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert _safe_loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)) == payload
