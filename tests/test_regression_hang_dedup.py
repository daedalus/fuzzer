"""Regression: every input that timed out was saved, one file per input hash.

One hang loop fed different bytes filled ``seeds/timeouts/`` without bound.
A hang is now kept only when its trace hits an edge no earlier hang hit
(AFL's virgin-tmout rule); without a trace, input-hash dedup as before.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.services.corpus_manager import CorpusManager
from tests.test_regression_crash_triage_cost import _Fuzzer


class _Shm:
    """Scripted trace: the edge ids of the last execution."""

    def __init__(self) -> None:
        self.edges: list[int] = []

    def edge_id_array(self) -> np.ndarray:
        return np.array(sorted(self.edges), dtype=np.uint32)


def _setup(tmp_path, shm=True):
    f = _Fuzzer(tmp_path)
    f.multi_targets = None
    f.shm_cov = _Shm() if shm else None
    return f, CorpusManager(f)


def _hang(f, cm, data: bytes, edges: list[int]) -> None:
    if f.shm_cov is not None:
        f.shm_cov.edges = edges
    cm.save_timeout(data)


def _saved(f) -> int:
    return sum(1 for p in (f.corpus_dir / "seeds" / "timeouts").rglob("id_*") if p.is_file())


def test_regression_same_loop_saved_once(tmp_path) -> None:
    f, cm = _setup(tmp_path)
    _hang(f, cm, b"loop-a", [1, 2, 3])
    _hang(f, cm, b"loop-b", [1, 2, 3])
    _hang(f, cm, b"loop-c", [2, 3])  # subset: same loop, cut earlier

    assert _saved(f) == 1


def test_new_edge_new_hang(tmp_path) -> None:
    # Falsification: a hang reaching new code is kept.
    f, cm = _setup(tmp_path)
    _hang(f, cm, b"a", [1, 2, 3])
    _hang(f, cm, b"b", [1, 2, 3, 9])

    assert _saved(f) == 2


def test_no_trace_falls_back_to_input_hash(tmp_path) -> None:
    # Edge: no SHM: each new input saved, a repeat is not.
    f, cm = _setup(tmp_path, shm=False)
    for data in (b"a", b"b", b"a"):
        _hang(f, cm, data, [])

    assert _saved(f) == 2


def test_empty_trace_falls_back(tmp_path) -> None:
    # Edge: a hang before any edge fired says nothing about its loop.
    f, cm = _setup(tmp_path)
    _hang(f, cm, b"a", [])
    _hang(f, cm, b"b", [])

    assert _saved(f) == 2


def test_counts_hangs_deduped(tmp_path) -> None:
    # Adversarial: dropped hangs still count as timeouts elsewhere; here they
    # are tallied so the report can tell "no hangs" from "deduped".
    f, cm = _setup(tmp_path)
    for i in range(5):
        _hang(f, cm, b"x%d" % i, [4, 5])

    assert _saved(f) == 1
    assert f.hangs_deduped == 4
