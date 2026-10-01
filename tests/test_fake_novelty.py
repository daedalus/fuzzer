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
