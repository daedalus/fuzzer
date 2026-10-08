"""A mutant's directed distance belongs to the mutant, never to its parent.

FuzzRound._update_distance wrote each mutant's measured distance into the
PARENT's ``seed_meta["avg_distance"]``, so aflgo/go ranked the parent by
whichever child ran last.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.core.analyzers.analyzer_distance import _NO_VALUE_DISTANCE
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer

PARENT = b"PARENT-SEED"
CHILD = b"CHILD-MUTANT"
PARENT_DIST = 3.0
CHILD_DIST = 9.0
_CLEAN = (0, "")


class _Dist:
    """TargetDistance stand-in: fixed seed distance, records its input."""

    max_distance = 2 * _NO_VALUE_DISTANCE

    def __init__(self, value: float):
        self.value = value
        self.traces: list = []

    def seed_distance(self, trace):
        self.traces.append(trace)
        return self.value


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="dist_parent_") as tmp:
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
        f.seed_meta[PARENT]["avg_distance"] = PARENT_DIST
        yield f


def _shm_round(f, runtime: float | None, admit: bool) -> bool:
    # SHM-tail path: the runtime average arrives on every execution.
    with (
        patch.object(f, "_dedup_mutate", return_value=CHILD),
        patch.object(f, "_run_target", return_value=_CLEAN),
        patch.object(f, "_is_crash", return_value=False),
        patch.object(f, "_is_interesting", return_value=admit),
        patch.object(f, "_read_runtime_avg_distance", return_value=runtime),
    ):
        return f.fuzz_one(PARENT)


def _py_round(f, dist: _Dist) -> FuzzRound:
    # Python path: new coverage, no SHM tail; distance from the edge trace.
    f._distance = dist
    f._current_edges_cache = {5, 7}
    rnd = FuzzRound(f, PARENT)
    rnd._meta = f.seed_meta.get(PARENT)
    rnd._mutated = CHILD
    rnd._has_new_coverage = True
    with patch.object(f, "_read_runtime_avg_distance", return_value=None):
        rnd._update_distance()
    return rnd


def test_regression_distance_parent_overwrite(fuzzer):
    """SHM path: parent keeps its distance; the admitted child carries its own."""
    fuzzer._distance = _Dist(CHILD_DIST)

    assert _shm_round(fuzzer, CHILD_DIST, admit=True) is True

    assert fuzzer.seed_meta[PARENT]["avg_distance"] == PARENT_DIST
    assert fuzzer.seed_meta[CHILD]["avg_distance"] == CHILD_DIST


def test_regression_distance_parent_overwrite_python(fuzzer):
    """Python path: same contract when distance comes from the edge trace."""
    rnd = _py_round(fuzzer, _Dist(CHILD_DIST))
    assert fuzzer.seed_meta[PARENT]["avg_distance"] == PARENT_DIST

    rnd._admit()

    assert fuzzer.seed_meta[PARENT]["avg_distance"] == PARENT_DIST
    assert fuzzer.seed_meta[CHILD]["avg_distance"] == CHILD_DIST
    assert fuzzer._dist_last_value == CHILD_DIST


def test_falsify_no_distance_mode_tags_nothing(fuzzer):
    """Falsification: without --distance no seed gains an avg_distance."""
    fuzzer._distance = None

    assert _shm_round(fuzzer, CHILD_DIST, admit=True) is True

    assert "avg_distance" not in fuzzer.seed_meta[CHILD]
    assert fuzzer.seed_meta[PARENT]["avg_distance"] == PARENT_DIST


def test_adversarial_rejected_mutant(fuzzer):
    """Adversarial: a non-admitted mutant leaves no meta; stats still observe it."""
    fuzzer._distance = _Dist(CHILD_DIST)

    assert _shm_round(fuzzer, CHILD_DIST, admit=False) is False

    assert CHILD not in fuzzer.seed_meta
    assert fuzzer.seed_meta[PARENT]["avg_distance"] == PARENT_DIST
    assert fuzzer._dist_last_value == CHILD_DIST


def test_adversarial_parent_without_meta(fuzzer):
    """Adversarial: a parent with no seed_meta still lets the child be tagged."""
    fuzzer._distance = _Dist(CHILD_DIST)
    del fuzzer.seed_meta[PARENT]

    assert _shm_round(fuzzer, CHILD_DIST, admit=True) is True

    assert PARENT not in fuzzer.seed_meta
    assert fuzzer.seed_meta[CHILD]["avg_distance"] == CHILD_DIST


def test_adversarial_sentinel_distance(fuzzer):
    """Adversarial: the no-valued-blocks sentinel tags the child but skips stats."""
    rnd = _py_round(fuzzer, _Dist(_NO_VALUE_DISTANCE))

    rnd._admit()

    assert fuzzer.seed_meta[PARENT]["avg_distance"] == PARENT_DIST
    assert fuzzer.seed_meta[CHILD]["avg_distance"] == _NO_VALUE_DISTANCE
    assert fuzzer._dist_last_value is None
