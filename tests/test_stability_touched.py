"""Seed stability calibration over the touched-slot bitmap.

With a bitmap-capable shim the verdict is OR & ~AND over per-run bitmaps
instead of a full table scan per run plus Python set algebra. It must reach
the same verdict as the edge-set path wherever both are sound, and step aside
where slot identity is not: when the generation tag wrapped (the table was
wiped, so a slot may now hold a different edge) and when edges were dropped.
"""

from __future__ import annotations

import random
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

SIZE = 256


def _edge_of(slot: int) -> int:
    return slot * 10 + 1


class _TouchedShm:
    """Scripted runs of live *slots*; edge id at a slot is _edge_of(slot)."""

    def __init__(self, runs, *, hashes=None, gens=None, drops=None, supported=True):
        self._runs = [set(r) for r in runs]
        self._hashes = hashes
        self._gens = gens
        self._drops = drops
        self.touched_supported = supported
        self._i = -1
        self._last_drop_raw = 0
        self.log: list[str] = []
        self.masked: set[int] = set()
        self._seen_edge_ids: set[int] = set()
        self._masked_edge_ids: set[int] = set()

    def advance(self):
        self.log.append("run")
        self._i += 1

    def _cur(self):
        return self._runs[min(self._i, len(self._runs) - 1)]

    def touched_clear(self):
        self.log.append("clear")

    def touched_snapshot(self):
        self.log.append("snap")
        words = np.zeros((SIZE + 63) // 64, dtype=np.uint64)
        for s in self._cur():
            words[s >> 6] |= np.uint64(1) << np.uint64(s & 63)
        return words

    def slot_edge_ids(self, words):
        bits = np.unpackbits(words.view(np.uint8), bitorder="little")
        return np.array([_edge_of(int(s)) for s in np.flatnonzero(bits)], dtype=np.uint32)

    def read_generation(self):
        if self._gens is None:
            return 7 + max(self._i, 0)
        return self._gens[min(self._i, len(self._gens) - 1)]

    def get_edge_ids(self):
        self.log.append("scan")
        return {_edge_of(s) for s in self._cur()}

    def read_path_hash(self):
        if self._hashes is not None:
            return self._hashes[min(self._i, len(self._hashes) - 1)]
        return hash(frozenset(self._cur())) & 0xFFFFFFFF

    def dropped_edges_delta(self):
        raw = 0
        if self._drops is not None and self._i >= 0:
            raw = sum(self._drops[: min(self._i + 1, len(self._drops))])
        delta = raw - self._last_drop_raw
        self._last_drop_raw = raw
        return delta

    def mask_edges(self, ids):
        ids = {int(i) for i in ids}
        newly = ids - self._masked_edge_ids
        self.masked |= ids
        self._masked_edge_ids |= ids
        return len(newly)


def _fuzzer(**kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.mkdtemp(prefix="stab_touched_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmp}/c",
            crashes_dir=f"{tmp}/x",
            max_len=64,
            timeout=1,
            mutations_per_input=2,
            **kw,
        )


def _wire(shm):
    f = _fuzzer()
    f.shm_cov = shm
    f._run_target = lambda data: (shm.advance(), (0, ""))[1]
    return f


class TestBitmapVerdict:
    def test_slot_in_one_run_only_masks_its_edge_without_a_table_scan(self):
        shm = _TouchedShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}])
        f = _wire(shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {_edge_of(9)}
        assert shm.masked == {_edge_of(9)}
        assert "scan" not in shm.log

    def test_stable_seed_masks_nothing(self):
        shm = _TouchedShm([{1, 2, 3}] * 3)
        f = _wire(shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()
        assert shm.masked == set()

    def test_the_bitmap_is_cleared_before_every_run_and_read_after_it(self):
        shm = _TouchedShm([{1}, {1}, {1}])
        f = _wire(shm)
        f._calibrate_seed_stability(b"x", n_runs=3)
        assert shm.log == ["clear", "run", "snap"] * 3

    def test_diverging_hash_with_identical_slots_masks_nothing(self):
        shm = _TouchedShm([{1, 2}] * 3, hashes=[1, 2, 3])
        f = _wire(shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()
        assert shm.masked == set()

    def test_the_calibration_is_counted(self):
        shm = _TouchedShm([{1}, {1, 4}, {1}])
        f = _wire(shm)
        before = f._stability_calibrations
        f._calibrate_seed_stability(b"x", n_runs=3)
        assert f._stability_calibrations == before + 1

    def test_unstable_edges_are_remembered(self):
        shm = _TouchedShm([{1}, {1, 4}, {1}])
        f = _wire(shm)
        f._calibrate_seed_stability(b"x", n_runs=3)
        assert f._unstable_edges == {_edge_of(4)}


class TestWhereSlotsCannotBeTrusted:
    def test_drops_veto_the_verdict(self):
        shm = _TouchedShm([{1, 2}, {1, 2, 9}, {1, 2}], drops=[0, 1, 0])
        f = _wire(shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()
        assert shm.masked == set()

    def test_a_table_wipe_between_runs_falls_back_to_edge_ids(self):
        # Generation 0 on a later run means the table was wiped after run 1.
        # The bitmap attempt consumes the first three scripted runs; the
        # edge-id re-measure then sees the next three.
        runs = [{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}] * 2
        shm = _TouchedShm(runs, gens=[255, 0, 1])
        f = _wire(shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {_edge_of(9)}
        assert "scan" in shm.log  # decided from edge ids, not slots

    def test_generation_zero_on_the_first_run_is_fine(self):
        shm = _TouchedShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}], gens=[0, 1, 2])
        f = _wire(shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {_edge_of(9)}
        assert "scan" not in shm.log

    def test_an_old_shim_uses_the_edge_set_path(self):
        shm = _TouchedShm([{1, 2, 3}, {1, 2, 3, 9}, {1, 2, 3}], supported=False)
        f = _wire(shm)
        assert f._calibrate_seed_stability(b"x", n_runs=3) == {_edge_of(9)}
        assert "scan" in shm.log
        assert "snap" not in shm.log

    def test_a_seed_that_fails_to_rerun_is_left_alone(self):
        shm = _TouchedShm([{1}, {1, 4}, {1}])
        f = _wire(shm)

        def boom(data):
            raise RuntimeError("no")

        f._run_target = boom
        assert f._calibrate_seed_stability(b"x", n_runs=3) == set()
        assert shm.masked == set()


class TestAgreesWithTheEdgeSetPath:
    @pytest.mark.parametrize("seed", range(40))
    def test_same_verdict_on_random_runs(self, seed):
        rng = random.Random(seed)
        core = set(rng.sample(range(SIZE), rng.randint(0, 30)))
        runs = [
            core | set(rng.sample(range(SIZE), rng.randint(0, 6))) for _ in range(rng.randint(2, 5))
        ]
        verdicts = []
        for supported in (True, False):
            shm = _TouchedShm(runs, supported=supported)
            f = _wire(shm)
            verdicts.append((f._calibrate_seed_stability(b"x", n_runs=len(runs)), shm.masked))
        assert verdicts[0] == verdicts[1]


class TestConstruction:
    @pytest.mark.parametrize(("n", "expected"), [(0, False), (3, True)])
    def test_bitmap_is_allocated_only_when_calibration_is_on(self, n, expected):
        with patch("fuzzer_tool.services.fuzzer.ShmCoverage") as cls:
            cls.return_value = MagicMock()
            _fuzzer(use_coverage=True, calibrate_stability=n)
        assert cls.call_args.kwargs.get("touched_bitmap", False) is expected


# argv[1:]: block ids, fired in order through the real per-edge path (path hash
# included, which is what the calibration's cheap screen reads).
_LOC_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        __afl_map_loc((uint32_t)strtoul(argv[i], NULL, 10));
    return 0;
}
"""


@pytest.fixture(scope="module")
def loc_target(tmp_path_factory):
    from tests.test_regression_shim_audit import _build

    return _build(tmp_path_factory.mktemp("stab_e2e"), _LOC_DRIVER, "-D__AFL_CTX_SENSITIVE=0")


def _calibrate_for_real(exe, scripts, *, bitmap: bool):
    """Calibrate against a real shim and a real ShmCoverage; return (verdict, masked)."""
    import subprocess

    from fuzzer_tool.adapters.shm import ShmCoverage
    from tests.test_regression_shim_audit import _clean_env

    cov = ShmCoverage(size=1000, touched_bitmap=bitmap)
    try:

        def run(ids):
            cov.reset_edge_map()
            env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE="1000")
            r = subprocess.run([str(exe), *map(str, ids)], env=env, capture_output=True, timeout=30)
            assert r.returncode == 0, r.stderr

        run(scripts[0])  # the run that accepted the seed: the shim attaches and announces itself
        assert cov.touched_supported is bitmap
        queue = list(scripts)
        f = _fuzzer()
        f.shm_cov = cov
        f._run_target = lambda data: (run(queue.pop(0)), (0, ""))[1]
        verdict = f._calibrate_seed_stability(b"x", n_runs=len(scripts))
        return verdict, cov.masked_edges
    finally:
        cov.cleanup()


class TestAgainstTheRealShim:
    from tests.conftest import requires_clang

    @requires_clang
    def test_bitmap_and_edge_set_paths_agree_on_a_flaky_tail_edge(self, loc_target):
        base = [0x1111, 0x2222, 0x3333]
        scripts = [base, [*base, 0x9999], base]  # run 2 fires one extra block at the end
        via_bitmap = _calibrate_for_real(loc_target, scripts, bitmap=True)
        via_scan = _calibrate_for_real(loc_target, scripts, bitmap=False)
        assert via_bitmap == via_scan
        assert len(via_bitmap[0]) == 1
        assert via_bitmap[1] == via_bitmap[0]

    @requires_clang
    def test_a_deterministic_seed_masks_nothing(self, loc_target):
        scripts = [[0x1111, 0x2222, 0x3333]] * 3
        assert _calibrate_for_real(loc_target, scripts, bitmap=True) == (set(), set())


class TestTouchedScanConstruction:
    @pytest.mark.parametrize("flag", [False, True])
    def test_scan_flag_reaches_the_coverage_object(self, flag):
        with patch("fuzzer_tool.services.fuzzer.ShmCoverage") as cls:
            cls.return_value = MagicMock()
            _fuzzer(use_coverage=True, touched_scan=flag)
        assert cls.call_args.kwargs["touched_scan"] is flag

    def test_cli_exposes_the_flag_and_passes_it_through(self):
        import inspect

        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--touched-scan"' in src
        assert 'touched_scan=getattr(args, "touched_scan", None)' in src
