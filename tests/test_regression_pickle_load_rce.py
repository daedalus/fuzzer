"""Regression: pickle loaders must refuse globals outside the state allowlist.

A crafted pickle (edge export, minimize batch state, tool cache) used to run
arbitrary callables on load. Each payload here reduces to ``_record``; a
call to it means the loader executed attacker-chosen code.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.edge_pickle import (
    EdgePickleWriter,
    iter_edges_pickle,
    read_edges_pickle,
)
from fuzzer_tool.core import state_store
from fuzzer_tool.core.state_store import StateStore, UnsafeStateError
from fuzzer_tool.services import minimize

_CALLS: list[str] = []

# Edge exports write protocol 3; the splice layout depends on it.
_EDGE_PROTO = 3
_BATCH_STATE = ".minimize_batch_state.pkl.gz"


def _record(tag: str) -> str:
    _CALLS.append(tag)
    return tag


class _Evil:
    """Pickles as ``_record(tag)``: the stand-in for os.system."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __reduce__(self):
        return _record, (self.tag,)


class _EvilBuiltin:
    """Pickles as ``builtins.eval(...)``: same module as allowed types, other name."""

    def __reduce__(self):
        return eval, ("0",)


@pytest.fixture(autouse=True)
def _clear_calls():
    _CALLS.clear()
    yield
    _CALLS.clear()


def _evil_export(path: Path, tag: str) -> Path:
    path.write_bytes(pickle.dumps({"seed": _Evil(tag)}, _EDGE_PROTO))
    return path


# ── edge exports ─────────────────────────────────────────────────────


def test_regression_edge_open_refuses_globals(tmp_path):
    writer = EdgePickleWriter(_evil_export(tmp_path / "e.pkl", "open"))
    with pytest.raises(UnsafeStateError):
        writer.open()
    assert _CALLS == []


def test_regression_read_edges_refuses_globals(tmp_path):
    with pytest.raises(UnsafeStateError):
        read_edges_pickle(_evil_export(tmp_path / "e.pkl", "read"))
    assert _CALLS == []


def test_regression_iter_edges_refuses_globals(tmp_path):
    with pytest.raises(UnsafeStateError):
        list(iter_edges_pickle(_evil_export(tmp_path / "e.pkl", "iter")))
    assert _CALLS == []


def test_regression_iter_edges_old_layout_refuses(tmp_path):
    """Old layout (protocol 5, no splice head) takes the other branch."""
    path = tmp_path / "e.pkl"
    path.write_bytes(pickle.dumps({"seed": _Evil("old")}, 5))
    with pytest.raises(UnsafeStateError):
        list(iter_edges_pickle(path))
    assert _CALLS == []


def test_edge_export_round_trip_still_loads(tmp_path):
    """Falsification: genuine exports (1-D and 2-D arrays) keep loading."""
    path = tmp_path / "e.pkl"
    flat = np.arange(5, dtype=np.uint32)
    square = np.arange(4, dtype=np.uint32).reshape(2, 2)
    writer = EdgePickleWriter(path)
    writer.write([("a", flat), ("b", square)])
    writer.close()

    assert EdgePickleWriter(path).open() == {"a", "b"}
    got = read_edges_pickle(path)
    np.testing.assert_array_equal(got["a"], flat)
    np.testing.assert_array_equal(got["b"], square)

    # 1-D only: _batch_spans misses batches holding 2-D arrays (docs/TODO.md).
    flat_only = tmp_path / "flat.pkl"
    writer = EdgePickleWriter(flat_only)
    writer.write([("a", flat)])
    writer.close()
    streamed = {k: v for batch in iter_edges_pickle(flat_only) for k, v in batch.items()}
    np.testing.assert_array_equal(streamed["a"], flat)


def test_edge_old_protocol5_export_still_loads(tmp_path):
    """Falsification: protocol-5 exports reduce via numpy's _frombuffer."""
    path = tmp_path / "e.pkl"
    ids = np.arange(3, dtype=np.uint32)
    path.write_bytes(pickle.dumps({"a": ids}, 5))
    np.testing.assert_array_equal(read_edges_pickle(path)["a"], ids)


def test_edge_open_refuses_builtin_eval(tmp_path):
    """Adversarial: an allowed module (builtins) does not admit every name."""
    path = tmp_path / "e.pkl"
    path.write_bytes(pickle.dumps({"seed": _EvilBuiltin()}, _EDGE_PROTO))
    with pytest.raises(UnsafeStateError):
        EdgePickleWriter(path).open()
    with pytest.raises(UnsafeStateError):
        read_edges_pickle(path)


# ── minimize batch state ─────────────────────────────────────────────


def _corpus(tmp_path: Path, state: object) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"a")
    with gzip.open(corpus / _BATCH_STATE, "wb") as f:
        pickle.dump(state, f)
    return corpus


def test_regression_minimize_state_refuses_globals(tmp_path):
    corpus = _corpus(tmp_path, {"processed": _Evil("minimize")})
    minimize.minimize_corpus("unused", str(corpus), batch_mode=True)
    assert _CALLS == []


def test_regression_commit_state_refuses_globals(tmp_path):
    corpus = _corpus(tmp_path, {"processed": _Evil("commit"), "survivors": set()})
    seed = corpus / "seed"
    minimize._commit_results([seed], [str(seed)], None, corpus, batch_mode=True)
    assert _CALLS == []


def test_minimize_state_skips_processed(tmp_path):
    """Falsification: a genuine batch state still filters processed seeds."""
    from fuzzer_tool.adapters.filesystem import hash_data

    corpus = _corpus(tmp_path, {"processed": {hash_data(b"a")}, "survivors": set()})
    assert minimize.minimize_corpus("unused", str(corpus), batch_mode=True) == (0, 0)


def test_minimize_state_not_a_dict(tmp_path):
    """Adversarial: a non-dict state restarts the batch instead of crashing."""
    corpus = _corpus(tmp_path, ["not", "a", "dict"])
    kept, _ = minimize.minimize_corpus("unused", str(corpus), batch_mode=True)
    assert kept >= 1


# ── state store size cap ─────────────────────────────────────────────


def test_regression_state_gzip_bomb_capped(tmp_path, monkeypatch):
    """Adversarial: a tiny gzip that inflates past the cap is not read whole."""
    cap = 1 << 12
    monkeypatch.setattr(state_store, "_MAX_STATE_BYTES", cap)
    bomb = pickle.dumps({"pad": bytes(cap * 4)})
    with gzip.open(tmp_path / state_store.STATE_FILENAME, "wb") as f:
        f.write(bomb)
    assert (tmp_path / state_store.STATE_FILENAME).stat().st_size < cap
    assert StateStore(tmp_path).load() == {}


def test_state_under_cap_loads(tmp_path, monkeypatch):
    """Falsification: a payload under the cap round-trips."""
    monkeypatch.setattr(state_store, "_MAX_STATE_BYTES", 1 << 12)
    store = StateStore(tmp_path)
    store.set("markov", {1: 2})
    store.save()
    assert StateStore(tmp_path).load()["markov"] == {1: 2}


# ── tool caches ──────────────────────────────────────────────────────


_TOOLS = ("tools.corpus_ffmpeg_issues", "tools.extract_ffmpeg_seeds")


@pytest.mark.parametrize("modname", _TOOLS)
def test_regression_tool_cache_not_in_tmp(modname):
    """/tmp is world-writable: another user could plant the cache."""
    mod = pytest.importorskip(modname)
    assert not mod.CACHE_PATH.startswith("/tmp")


@pytest.mark.parametrize("modname", _TOOLS)
def test_regression_tool_cache_refuses_globals(modname, tmp_path, monkeypatch):
    mod = pytest.importorskip(modname)
    path = tmp_path / "cache.pkl"
    path.write_bytes(pickle.dumps({"k": _Evil(modname)}))
    monkeypatch.setattr(mod, "CACHE_PATH", str(path))
    with pytest.raises(UnsafeStateError):
        mod.load_cache()
    assert _CALLS == []


@pytest.mark.parametrize("modname", _TOOLS)
def test_tool_cache_round_trip(modname, tmp_path, monkeypatch):
    """Falsification: save_cache creates the dir; load_cache reads it back."""
    mod = pytest.importorskip(modname)
    path = tmp_path / "sub" / "cache.pkl"
    monkeypatch.setattr(mod, "CACHE_PATH", str(path))
    cache = {"cve": {"CVE-1/poc.bin": "/x/poc.bin"}}
    mod.save_cache(cache)
    assert mod.load_cache() == cache
