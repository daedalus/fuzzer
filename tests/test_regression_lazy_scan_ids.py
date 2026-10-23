"""The scanned edge-id set is built on first read, not on every scan.

``set(active_ids.tolist())`` cost ~0.33 ms per ffmpeg exec (4% of wall)
and --lightweight rounds never read it. ``is_new_coverage()`` now answers
without it; ``is_new_coverage_with_edges()`` and ``_last_ids`` still see
the same ids.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.run_mode import RunMode
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.test_regression_scan_new_ids_prefilter import _exec, _shim_write


def _scan_only(shm: ShmCoverage, hits: dict[int, int]) -> bool:
    shm.reset_edge_map()
    for eid, count in hits.items():
        _shim_write(shm, eid, count)
    return shm.is_new_coverage()


def test_regression_verdict_without_id_set():
    # Falsification: the boolean scan leaves the id set unbuilt.
    shm = ShmCoverage(size=4096)
    assert _scan_only(shm, {1: 1, 2: 1}) is True
    assert shm._ids_set is None
    assert _scan_only(shm, {1: 1, 2: 1}) is False
    assert shm._ids_set is None


def test_regression_ids_read_on_demand():
    # Control: the lazily built set equals the live ids.
    shm = ShmCoverage(size=4096)
    _scan_only(shm, {5: 1, 6: 3})
    assert shm._last_ids == {5, 6}
    assert shm.get_edge_ids() == {5, 6}


def test_regression_with_edges_unchanged():
    shm = ShmCoverage(size=4096)
    _exec(shm, {1: 1, 2: 1})
    has_new, ids = shm.is_new_coverage_with_edges()
    # Same table: fast path, same ids.
    assert (has_new, ids) == (False, {1, 2})


def test_regression_reject_phantoms_on_lazy_ids():
    # Adversarial: withdrawing ids from a set that was never materialized.
    shm = ShmCoverage(size=4096)
    _scan_only(shm, {1: 1, 2: 1, 3: 1})
    assert shm.reject_phantoms({3}) == 1
    assert shm._last_ids == {1, 2}


def _lite_fuzzer(tmp: str) -> Fuzzer:
    corpus = Path(tmp) / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"SEED-0123456789")
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(Path(tmp) / "crashes"),
            max_len=256,
            timeout=1,
            run_mode=RunMode.LITE,
        )


def test_regression_lite_scan_skips_edge_set():
    # Falsification: a LITE round's scan takes the verdict, not the set.
    with tempfile.TemporaryDirectory(prefix="lazy_ids_") as tmp:
        f = _lite_fuzzer(tmp)
        f.shm_cov = ShmCoverage(size=4096)
        _shim_write(f.shm_cov, 9, 1)
        rnd = FuzzRound(f, f.corpus[0])
        rnd._is_timeout = False
        with patch.object(f.shm_cov, "is_new_coverage_with_edges") as with_edges:
            rnd._lite_scan()
        with_edges.assert_not_called()
        assert rnd._has_new_coverage is True
        assert f._current_edges_cache is None
        assert rnd._edges_now() == {9}
