"""--path-freq cms wiring: registry, per-exec note, n_fuzz lookup, CLI."""

from __future__ import annotations

import ast
import inspect
import types

import numpy as np

from fuzzer_tool.core.analyzer_registry import REGISTRY
from fuzzer_tool.core.count_min import CountMin, PathCount, PathFreq
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer


class _Store(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _wired(mode=PathCount.CMS, store=None):
    f = types.SimpleNamespace(_path_freq_mode=mode, _state_store=store or _Store())
    spec = REGISTRY._specs["path_freq"]
    if spec.available(f):
        spec.activate(f)
    else:
        spec.deactivate(f)
    return f


def _fuzzer(paths, seed_paths):
    f = Fuzzer.__new__(Fuzzer)
    f._path_freq = paths
    f._edge_tracker = types.SimpleNamespace(
        seed_path_hash=seed_paths, get_seed_path_hash=lambda k: seed_paths.get(k, 0)
    )
    f.seed_meta = {b"a": {"fuzz_count": 5}, b"b": {"fuzz_count": 0}}
    return f


# --- Registry -----------------------------------------------------------------


def test_default_is_fuzz_count():
    assert inspect.signature(Fuzzer.__init__).parameters["path_freq"].default is PathCount.FUZZ
    assert _wired(PathCount.FUZZ)._path_freq is None


def test_cms_activates_and_restores():
    saved = PathFreq(CountMin(width=64))
    for h in (9, 9, 9):
        saved.note(h)
    f = _wired(store=_Store(path_freq=saved.to_dict()))
    assert isinstance(f._path_freq, PathFreq)
    assert f._path_freq.count(9) == 3


# --- Per-exec note ------------------------------------------------------------


def _round(paths, path_hash=77, shm=True):
    r = FuzzRound.__new__(FuzzRound)
    r._f = types.SimpleNamespace(_path_freq=paths)
    r._scanned_shm = types.SimpleNamespace(read_path_hash=lambda: path_hash) if shm else None
    return r


def test_round_notes_path_hash():
    paths = PathFreq(CountMin(width=64))
    r = _round(paths)
    r._note_path()
    r._note_path()
    assert paths.count(77) == 2


def test_round_note_noop_off_and_without_shm():
    _round(None)._note_path()  # off: must not touch the shm
    paths = PathFreq(CountMin(width=64))
    _round(paths, shm=False)._note_path()
    assert paths.pending == 0


# --- n_fuzz lookup ------------------------------------------------------------


def test_n_fuzz_off_is_fuzz_level():
    assert _fuzzer(None, {})._path_n_fuzz("k", 5) == 5


def test_n_fuzz_cms_reads_path_count():
    paths = PathFreq(CountMin(width=64))
    for _ in range(40):
        paths.note(123)
    f = _fuzzer(paths, {"k": 123})
    assert f._path_n_fuzz("k", 5) == 40
    # No path hash recorded for this seed: fall back to its fuzz_count.
    assert f._path_n_fuzz("unknown", 5) == 5


def test_n_fuzz_values_follow_mode():
    assert sorted(_fuzzer(None, {})._n_fuzz_values()) == [0, 5]

    paths = PathFreq(CountMin(width=64))
    for h in (1, 1, 2):
        paths.note(h)
    vals = _fuzzer(paths, {"x": 1, "y": 2, "z": 3})._n_fuzz_values()
    assert sorted(np.asarray(vals).tolist()) == [0, 1, 2]


# --- CLI ----------------------------------------------------------------------


def test_cli_dest_registered():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    assert "path_freq" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))


def test_lite_round_notes_path():
    # --lightweight skips most steps; n_fuzz would read 0 for every path.
    src = inspect.getsource(FuzzRound._lite_steps)
    assert "self._note_path()" in src
