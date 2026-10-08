"""Directed-distance follow-ups (docs/TODO.md "Directed distance follow-ups").

(a) Coverage edge ids are hashes, not block addresses: they must never reach
    ``seed_distance``. Only ptrace hits real block addresses.
(b) Corpus seeds are measured at calibration, so aflgo/go rank from exec 0.
(c) No per-seed edge trace is filed (it had no reader, and used the parent key).
(d) A trimmed seed keeps a distance: its own measurement, else the original's.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.core.analyzers.analyzer_distance import _NO_VALUE_DISTANCE
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.ptrace_coverage import PtraceCoverage

PARENT = b"PARENT-SEED"
CHILD = b"CHILD-MUTANT"
EDGE_IDS = {5, 7}
BLOCKS = {0x1000, 0x1040}
DIST = 4.5
OLD_DIST = 2.25
TAIL_SCALE = 100.0  # shim reports distance x100 (Fuzzer._read_runtime_avg_distance)
TAIL = (1200, 2)  # 1200 / 2 / 100 = 6.0
NO_TAIL = (0, 0)
CRASH_RC = -11
_CLEAN = (0, "")


def _tail_value(tail: tuple[int, int]) -> float:
    return tail[0] / tail[1] / TAIL_SCALE


class _Dist:
    """TargetDistance stand-in: fixed seed distance, records its input."""

    max_distance = 2 * _NO_VALUE_DISTANCE

    def __init__(self, value: float):
        self.value = value
        self.traces: list = []

    def seed_distance(self, trace):
        self.traces.append(trace)
        return self.value


class _Ptrace:
    """PtraceCoverage stand-in: a fixed set of hit block addresses."""

    def __init__(self, blocks: set[int]):
        self._blocks = set(blocks)
        self.edge_map = bytearray(4)

    def blocks_hit(self) -> set[int]:
        return self._blocks


class _Shm:
    """ShmCoverage stand-in: fixed edges, settable distance tail."""

    def __init__(self):
        self.tail = NO_TAIL
        self.edges = {1, 2}

    def is_new_coverage_with_edges(self):
        return True, set(self.edges)

    def get_edge_ids(self):
        return set(self.edges)

    def get_edge_counts(self):
        return {e: 1 for e in self.edges}

    def read_stack_depth(self):
        return 0

    def read_path_hash(self):
        return 1

    def read_distance_tail(self):
        return self.tail


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="dist_followups_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
                max_len=256,
                timeout=1,
                mutations_per_input=2,
                cmplog=False,  # no ~/.cache cmplog fifo left behind
            )
        f.save_to_corpus(PARENT)
        yield f


def _py_round(f, dist: _Dist) -> FuzzRound:
    # Off the SHM tail: new coverage, edge ids in the scan cache.
    f._distance = dist
    f._current_edges_cache = set(EDGE_IDS)
    rnd = FuzzRound(f, PARENT)
    rnd._meta = f.seed_meta.get(PARENT)
    rnd._mutated = CHILD
    rnd._has_new_coverage = True
    with patch.object(f, "_read_runtime_avg_distance", return_value=None):
        rnd._update_distance()
    return rnd


# ── (a) edge ids are not block addresses ─────────────────────────────


def test_regression_edge_ids_not_blocks(fuzzer):
    """Falsification: edge-id hashes never reach seed_distance or stats."""
    dist = _Dist(DIST)
    rnd = _py_round(fuzzer, dist)

    assert dist.traces == []
    assert rnd._distance is None
    assert fuzzer._dist_last_value is None


def test_regression_ptrace_blocks_feed_distance(fuzzer):
    """ptrace hits real block addresses: those, not edge ids, are measured."""
    fuzzer.ptrace_cov = _Ptrace(BLOCKS)
    dist = _Dist(DIST)
    rnd = _py_round(fuzzer, dist)
    fuzzer.save_to_corpus(CHILD, parent=PARENT)
    rnd._tag_distance()

    assert dist.traces == [{(b, b) for b in BLOCKS}]
    assert fuzzer.seed_meta[CHILD]["avg_distance"] == DIST
    assert fuzzer._dist_last_value == DIST


def test_adversarial_ptrace_no_blocks(fuzzer):
    """Adversarial: ptrace with no block hit yields no value, not the sentinel."""
    fuzzer.ptrace_cov = _Ptrace(set())
    dist = _Dist(DIST)
    rnd = _py_round(fuzzer, dist)

    assert dist.traces == []
    assert rnd._distance is None


def _ptrace_cov(tmp_path, map_size: int) -> PtraceCoverage:
    # Non-ELF file: no breakpoints collected, record_edge still works.
    target = tmp_path / "not_elf"
    target.write_bytes(b"not an elf")
    return PtraceCoverage(str(target), map_size=map_size)


def test_adversarial_bucket_collision_keeps_blocks(tmp_path):
    """Adversarial: map_size 1 folds every edge into one bucket; blocks stay distinct."""
    cov = _ptrace_cov(tmp_path, map_size=1)
    base = 0x5555_0000_0000
    cov._base_address = base
    for rel in sorted(BLOCKS):
        cov.record_edge(base + rel)

    assert cov.total_edges == 1
    assert cov.blocks_hit() == BLOCKS

    cov.reset_edge_map()
    assert cov.blocks_hit() == set()


def test_adversarial_non_pie_keeps_vaddr(tmp_path):
    """Adversarial: non-PIE breakpoints sit at link vaddrs; no base is subtracted."""
    cov = _ptrace_cov(tmp_path, map_size=64)
    cov._is_pie = False
    cov._base_address = 0x400000
    vaddr = 0x401000
    cov.record_edge(vaddr)

    assert cov.blocks_hit() == {vaddr}


def test_regression_warn_no_distance_source(fuzzer, capsys):
    """--target-functions without a block-address channel says so once."""
    fuzzer._distance = _Dist(DIST)
    fuzzer._warn_no_dist_source()
    fuzzer._warn_no_dist_source()

    assert capsys.readouterr().out.count("WARNING") == 1


def test_falsify_warn_silent_with_source(fuzzer, capsys):
    """Falsification: ptrace or no directed mode prints nothing."""
    fuzzer._distance = None
    fuzzer._warn_no_dist_source()
    fuzzer._distance = _Dist(DIST)
    fuzzer.ptrace_cov = _Ptrace(BLOCKS)
    fuzzer._warn_no_dist_source()

    assert "WARNING" not in capsys.readouterr().out


# ── (b) seeds measured at calibration ────────────────────────────────

SEED_A = b"SEED-A-calibrate"
SEED_B = b"SEED-B-resumed"
CRASHER = b"SEED-C-crasher"


def _calibrate(f, tails: dict[bytes, tuple[int, int]], crashers=frozenset()):
    shm = _Shm()
    f.shm_cov = shm
    f.use_coverage = True
    f._distance = _Dist(DIST)

    def run(seed):
        shm.tail = tails.get(seed, NO_TAIL)
        return (CRASH_RC, "") if seed in crashers else _CLEAN

    with (
        patch.object(f, "_run_target", side_effect=run),
        patch.object(f, "_is_crash", side_effect=lambda rc, _e: rc == CRASH_RC),
        patch.object(f, "_seed_crash_path"),
        patch.object(f, "_confirm_new_coverage", side_effect=lambda _s, _m, h, e: (h, e)),
        patch.object(f, "_report_comparison_reach"),
        patch.object(f, "_report_edge_id_stability"),
        patch.object(f, "_report_coverage_noise"),
    ):
        f._calibrate_seed_baselines()


def test_regression_calibration_measures_seeds(fuzzer):
    """A fresh seed leaves calibration with its SHM-tail distance."""
    fuzzer.save_to_corpus(SEED_A)
    _calibrate(fuzzer, {SEED_A: TAIL})

    assert fuzzer.seed_meta[SEED_A]["avg_distance"] == _tail_value(TAIL)


def test_falsify_calibration_skips_crasher(fuzzer):
    """Falsification: a crashing seed is not tagged; a clean one is."""
    fuzzer.save_to_corpus(SEED_A)
    fuzzer.save_to_corpus(CRASHER)
    _calibrate(fuzzer, {SEED_A: TAIL, CRASHER: TAIL}, crashers={CRASHER})

    assert "avg_distance" not in fuzzer.seed_meta[CRASHER]
    assert fuzzer.seed_meta[SEED_A]["avg_distance"] == _tail_value(TAIL)


def test_adversarial_resumed_seed_keeps_distance(fuzzer):
    """Adversarial: a seed already tagged keeps its value; an untagged one is measured."""
    fuzzer.save_to_corpus(SEED_A)
    fuzzer.save_to_corpus(SEED_B)
    fuzzer.seed_meta[SEED_B]["avg_distance"] = OLD_DIST
    _calibrate(fuzzer, {SEED_A: TAIL, SEED_B: TAIL})

    assert fuzzer.seed_meta[SEED_B]["avg_distance"] == OLD_DIST
    assert fuzzer.seed_meta[SEED_A]["avg_distance"] == _tail_value(TAIL)


# ── (c) no per-seed edge trace ───────────────────────────────────────


def test_regression_no_parent_edge_trace(fuzzer):
    """Falsification: a round files no edge trace under any key (parent included)."""
    fuzzer.ptrace_cov = _Ptrace(BLOCKS)
    _py_round(fuzzer, _Dist(DIST))

    assert not getattr(fuzzer._edge_tracker, "seed_edge_traces", {})


def test_adversarial_legacy_edge_traces_state():
    """Adversarial: a state file with legacy edge_traces loads and is not re-saved."""
    tracker = EdgeTracker()
    state = tracker.to_dict()
    state["edge_traces"] = {"k": [[1, 1], [2, 2]]}
    tracker.from_dict(state)

    assert "edge_traces" not in tracker.to_dict()


# ── (d) trimmed seeds keep a distance ────────────────────────────────

LONG = b"L" * 64
TRIMMED = LONG[:32]


def _trim(f, trimmed_tail: tuple[int, int]) -> None:
    shm = _Shm()
    f.shm_cov = shm
    f._distance = _Dist(DIST)
    f.save_to_corpus(LONG, parent=PARENT)
    f.seed_meta[LONG]["avg_distance"] = OLD_DIST

    def run(data):
        shm.tail = trimmed_tail if data == TRIMMED else NO_TAIL
        return _CLEAN

    with patch.object(f._runner, "run_target", side_effect=run):
        CorpusManager(f).trim_new_coverage(LONG, PARENT)
    assert TRIMMED in f.corpus


def test_regression_trim_measures_distance(fuzzer):
    """The trimmed seed carries the distance measured on its own run."""
    _trim(fuzzer, TAIL)

    assert fuzzer.seed_meta[TRIMMED]["avg_distance"] == _tail_value(TAIL)


def test_adversarial_trim_without_source_inherits(fuzzer):
    """Adversarial: no tail on the trimmed run -> the original's value carries over."""
    _trim(fuzzer, NO_TAIL)

    assert fuzzer.seed_meta[TRIMMED]["avg_distance"] == OLD_DIST


def test_falsify_rejected_trim_tags_nothing(fuzzer):
    """Falsification: a trim whose trace differs adds no seed; the original keeps its value."""
    shm = _Shm()
    fuzzer.shm_cov = shm
    fuzzer._distance = _Dist(DIST)
    fuzzer.save_to_corpus(LONG, parent=PARENT)
    fuzzer.seed_meta[LONG]["avg_distance"] = OLD_DIST

    def run(data):
        shm.tail = TAIL
        shm.edges = {9} if data == TRIMMED else {1, 2}
        return _CLEAN

    with patch.object(fuzzer._runner, "run_target", side_effect=run):
        CorpusManager(fuzzer).trim_new_coverage(LONG, PARENT)

    assert TRIMMED not in fuzzer.seed_meta
    assert fuzzer.seed_meta[LONG]["avg_distance"] == OLD_DIST
