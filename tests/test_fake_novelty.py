"""Per-seed fake novelty (Fuzzification BranchTrap, USENIX Sec '19 §4.1).

BranchTrap routes function returns through input-selected gadgets, so most
mutants of a trapped seed hit a "new" edge id and are admitted. Real parsers
admit a small fraction of a seed's mutants; a seed whose children flood the
corpus is down-weighted in seed selection.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.core.coverage_noise import (
    FAKE_NOVELTY_MIN_FUZZ,
    FAKE_NOVELTY_PENALTY,
    AdmissionMonitor,
    fake_novelty_factor,
)
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import SeedPicker

_FLOOD = int(FAKE_NOVELTY_MIN_FUZZ * AdmissionMonitor.FLOOD_RATE)


def test_flooding_seed_is_penalized() -> None:
    assert fake_novelty_factor(FAKE_NOVELTY_MIN_FUZZ, _FLOOD) == FAKE_NOVELTY_PENALTY


def test_healthy_seed_keeps_weight() -> None:
    # Falsification: 1% of mutants admitted is a productive real seed.
    fuzzed = FAKE_NOVELTY_MIN_FUZZ * 10
    assert fake_novelty_factor(fuzzed, fuzzed // 100) == 1.0


def test_young_seed_is_not_judged() -> None:
    # Adversarial: a fresh seed's first mutants are often all new.
    early = FAKE_NOVELTY_MIN_FUZZ - 1
    assert fake_novelty_factor(early, early) == 1.0
    assert fake_novelty_factor(0, 0) == 1.0


def test_admissions_above_execs_still_bounded() -> None:
    # Adversarial: counters out of step (initial replay, resume) never amplify.
    assert (
        fake_novelty_factor(FAKE_NOVELTY_MIN_FUZZ, 10 * FAKE_NOVELTY_MIN_FUZZ)
        == FAKE_NOVELTY_PENALTY
    )


def _weight(meta: dict) -> float:
    return SeedPicker.__dict__["_weight_fake_novelty"](None, meta, 1.0)


def test_picker_applies_penalty() -> None:
    meta = {"fuzz_count": FAKE_NOVELTY_MIN_FUZZ, "child_count": _FLOOD}
    assert _weight(meta) == pytest.approx(FAKE_NOVELTY_PENALTY)


def test_picker_noop_without_counter() -> None:
    # Falsification: a seed with no admitted child keeps its weight.
    assert _weight({"fuzz_count": FAKE_NOVELTY_MIN_FUZZ}) == 1.0


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="fake_novelty_") as tmp:
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
            )
        yield f


def test_regression_admit_counts_parent_children(fuzzer) -> None:
    parent = fuzzer.corpus[0]
    mutants = [b"TRAP%04d" % i for i in range(3)]

    for mutant in mutants:
        with (
            patch.object(fuzzer, "_dedup_mutate", return_value=mutant),
            patch.object(fuzzer, "_run_target", return_value=(0, "")),
            patch.object(fuzzer, "_is_crash", return_value=False),
            patch.object(FuzzRound, "_admits", return_value=True),
        ):
            fuzzer.fuzz_one(parent)

    assert fuzzer.seed_meta[parent]["child_count"] == len(mutants)
    assert fuzzer.seed_meta[mutants[0]].get("child_count", 0) == 0


def test_compute_weights_demotes_flooding_seed(fuzzer) -> None:
    parent = fuzzer.corpus[0]
    meta = fuzzer.seed_meta[parent]
    meta["fuzz_count"] = FAKE_NOVELTY_MIN_FUZZ
    now = meta["added_at"]
    meta["coverage_edges"] = 10
    fuzzer._cached_weights = {}  # set by the run loop's weight pass

    meta["child_count"] = 0
    base = fuzzer._compute_weights(now)[0]
    meta["child_count"] = _FLOOD
    flooded = fuzzer._compute_weights(now)[0]

    assert flooded == pytest.approx(base * FAKE_NOVELTY_PENALTY)


# ── End to end: simulated BranchTrap through the real fuzz loop ──────────
# The target is modelled in Python behind a minimal SHM stand-in, so
# admission, the phantom rerun and edge tracking run unpatched. Only
# execution and mutation are scripted (no RNG).

_REAL_EDGES = (1, 2, 3)
_GADGET_BASE = 1000  # fake edge ids start here


def _branchtrap(data: bytes) -> set[int]:
    """Real edges plus one gadget edge picked by the last two input bytes."""
    return {*_REAL_EDGES, _GADGET_BASE + int.from_bytes(data[-2:], "little")}


_SMALL_POOL = 8  # gadgets per trapped return in a cheap trap


def _honest(data: bytes) -> set[int]:
    """Real edges plus one branch on the low bit of the mutated byte."""
    return {*_REAL_EDGES, 10 + (data[-2] & 1)}


def _small_trap(data: bytes) -> set[int]:
    """BranchTrap with few gadgets: XOR of the input bytes picks one of 8."""
    xor = 0
    for b in data:
        xor ^= b
    return {*_REAL_EDGES, _GADGET_BASE + xor % _SMALL_POOL}


class _ModelShm:
    """SHM stand-in: the last executed input's edges under *model*."""

    last_old_bucket_novel = False
    new_max_edges = 0  # no hit-count maxima here

    def __init__(self, model) -> None:
        self._model = model
        self._edges: set[int] = set()
        self._seen: set[int] = set()
        self.last_new_ids: frozenset[int] = frozenset()

    def run(self, data: bytes):
        self._edges = self._model(data)
        return 0, ""

    def is_new_coverage_with_edges(self):
        self.last_new_ids = frozenset(self._edges - self._seen)
        self._seen |= self._edges
        return bool(self.last_new_ids), set(self._edges)

    def get_edge_ids(self) -> set[int]:
        return set(self._edges)

    def get_edge_counts(self) -> dict[int, int]:
        return dict.fromkeys(self._edges, 1)

    def reject_phantoms(self, phantoms) -> None:
        self._seen -= set(phantoms)

    def read_stack_depth(self) -> int:
        return 0

    def read_path_hash(self) -> int:
        return hash(frozenset(self._edges)) or 1


def _campaign(fuzzer, model) -> dict:
    """Run FAKE_NOVELTY_MIN_FUZZ rounds on one parent; return its meta."""
    shm = _ModelShm(model)
    fuzzer.shm_cov = shm
    parent = fuzzer.corpus[0]

    # Mutant i carries i in its last two bytes (LE).
    # Short mutants keep coverage trimming (len > 10) out of the loop.
    mutants = iter(b"M" + i.to_bytes(2, "little") for i in range(FAKE_NOVELTY_MIN_FUZZ))
    with (
        patch.object(fuzzer, "_dedup_mutate", side_effect=lambda _d: next(mutants)),
        patch.object(fuzzer, "_run_target", side_effect=shm.run),
        patch.object(fuzzer, "_is_crash", return_value=False),
        patch.object(fuzzer, "_is_interesting", return_value=False),
    ):
        for _ in range(FAKE_NOVELTY_MIN_FUZZ):
            fuzzer.fuzz_one(parent)

    return fuzzer.seed_meta[parent]


def test_branchtrap_parent_is_demoted(fuzzer) -> None:
    meta = _campaign(fuzzer, _branchtrap)

    assert meta["child_count"] / meta["fuzz_count"] >= AdmissionMonitor.FLOOD_RATE
    assert _weight(meta) == pytest.approx(FAKE_NOVELTY_PENALTY)


def test_honest_parent_keeps_weight(fuzzer) -> None:
    # Falsification: two real outcomes admit at most two children.
    meta = _campaign(fuzzer, _honest)

    assert meta.get("child_count", 0) <= len({10, 11})
    assert _weight(meta) == 1.0


def test_small_gadget_pool_exhausts_without_penalty(fuzzer) -> None:
    # Adversarial: a trap with N gadgets yields at most N fake ids, then
    # stops flooding. Its parent is not demoted: the cost it imposes is
    # bounded by N admissions, not by the campaign length.
    meta = _campaign(fuzzer, _small_trap)

    assert meta.get("child_count", 0) <= _SMALL_POOL
    assert _weight(meta) == 1.0
