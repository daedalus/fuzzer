"""Entropic seed arm: libFuzzer's -entropic (Böhme, Manès, Cha, FSE'20).

Energy is the smoothed Shannon entropy of the rare edges a seed's mutants
hit: high while its mutants keep reaching rare, varied edges (information
left to gain), low once they repeat themselves. Expected values come from
``_spec_energy`` below, a transcription of libFuzzer's
``InputInfo::UpdateEnergy``, not from the module under test.
"""

from __future__ import annotations

import ast
import inspect
import math
from unittest.mock import patch

import pytest

from fuzzer_tool.core.schedulers.seed_entropic import (
    FREQ_SATURATE,
    EntropicSeedStrategy,
)


def _spec_energy(local: dict[int, int], n_execs: int, n_rare: int) -> float:
    """libFuzzer InputInfo::UpdateEnergy (FuzzerCorpus.h), no time scaling."""
    energy = 0.0
    incidence = 0.0
    for freq in local.values():
        li = freq + 1
        energy -= li * math.log(li)
        incidence += li
    incidence += n_rare - len(local)
    abundant = n_execs + 1
    energy -= abundant * math.log(abundant)
    incidence += abundant
    return energy / incidence + math.log(incidence)


class _RecordingRng:
    """Scripted weighted_choice: returns seq[idx], keeps the weights it saw."""

    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.weights: list[float] | None = None

    def weighted_choice(self, seq, weights):
        self.weights = list(weights)
        return seq[self.idx]


A, B, C = b"seed-a", b"seed-b", b"seed-c"


def _strategy(**kw) -> EntropicSeedStrategy:
    return EntropicSeedStrategy(_RecordingRng(0), **kw)


# ── Energy ─────────────────────────────────────────────────────────────


def test_energy_matches_spec():
    """Falsification: local counts, exec count and rare total per the spec."""
    s = _strategy()
    for edges in ({1, 2}, {1}, {1, 3}):
        s.observe(A, edges)

    expected = _spec_energy({1: 3, 2: 1, 3: 1}, n_execs=3, n_rare=3)
    assert s.energy(A) == pytest.approx(expected)


def test_unfuzzed_seed_gets_spec_energy():
    s = _strategy()
    s.observe(A, {1, 2, 3, 4})

    assert s.energy(B) == pytest.approx(_spec_energy({}, 0, 4))


def test_repetitive_mutants_lose_energy_to_varied_ones():
    """Adversarial: a seed whose mutants repeat one edge is exhausted."""
    s = _strategy()
    for i in range(20):
        s.observe(A, {1})
        s.observe(B, {100 + i})

    assert s.energy(B) > s.energy(A)


def test_empty_execution_counts_as_abundant():
    """A mutant hitting nothing rare still spends the seed's budget."""
    s = _strategy()
    s.observe(A, {1})
    s.observe(B, {2})
    before = s.energy(B)
    s.observe(B, set())

    assert s.energy(B) == pytest.approx(_spec_energy({2: 1}, 2, 2))
    assert s.energy(B) < before


# ── Bounds ─────────────────────────────────────────────────────────────


def test_rare_set_bounded_and_locals_follow():
    """Abundant edges are evicted past the cap, and leave every seed table."""
    cap, threshold = 4, 2
    s = _strategy(rare_cap=cap, freq_threshold=threshold)
    for edge in range(10):
        for _ in range(threshold + 3):
            s.observe(A, {edge})

    assert s.rare_count <= cap + 1
    assert set(s.local_counts(A)) <= s.rare_edges()


def test_frequency_saturates():
    s = _strategy()
    for _ in range(FREQ_SATURATE + 5):
        s.observe(A, {7})

    assert s.freq(7) == FREQ_SATURATE


def test_seed_tables_bounded():
    s = _strategy(seed_cap=2)
    s.observe(A, {1})
    s.observe(B, {2})
    s.observe(C, {3})

    assert s.local_counts(A) == {}
    assert s.energy(A) == pytest.approx(_spec_energy({}, 0, 3))


# ── Selection ──────────────────────────────────────────────────────────


def test_select_draws_by_energy():
    rng = _RecordingRng(1)
    s = EntropicSeedStrategy(rng)
    s.observe(A, {1})
    s.observe(A, {1})
    s.observe(B, {2, 3})

    assert s.select([A, B, C]) == B
    assert rng.weights == pytest.approx([s.energy(A), s.energy(B), s.energy(C)])


def test_select_declines_without_rare_edges():
    """No rare edge seen: every energy is log(1) = 0, nothing to rank."""
    s = _strategy()
    s.observe(A, set())

    assert s.select([A, B]) is None
    assert s.select([]) is None


# ── Wiring ─────────────────────────────────────────────────────────────


def test_registered_as_seed_strategy():
    from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES, Fuzzer

    assert "entropic" in _SEED_STRATEGY_NAMES
    assert inspect.signature(Fuzzer.__init__).parameters["entropic_seed"].default is False


def test_cli_passes_flag():
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    assert "entropic_seed" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
    calls = [
        {k.arg for k in c.keywords}
        for c in ast.walk(ast.parse(inspect.getsource(commands.cmd_fuzz)))
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "Fuzzer"
    ]
    assert calls and all("entropic_seed" in k for k in calls)


@pytest.fixture
def fuzzer(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            entropic_seed=True,
        )
    yield f


def test_every_round_feeds_parent(fuzzer):
    """Each executed mutant is credited to the seed it was mutated from."""
    s = fuzzer._entropic_seed
    assert isinstance(s, EntropicSeedStrategy)

    parent = fuzzer.corpus[0]
    with (
        patch.object(fuzzer, "_dedup_mutate", return_value=b"MUTANT01"),
        patch.object(fuzzer, "_run_target", return_value=(0, "")),
    ):
        fuzzer.fuzz_one(parent)
        fuzzer.fuzz_one(parent)

    assert s.executions(parent) == 2


def test_elo_arm_available(fuzzer):
    from fuzzer_tool.services.seed_picker import SeedPicker

    cold: list[str] = []
    SeedPicker._elo_entropy_arms(fuzzer, cold)
    assert "entropic" not in cold, "listed before any edge is rare"

    fuzzer._entropic_seed.observe(fuzzer.corpus[0], {1})
    warm: list[str] = []
    SeedPicker._elo_entropy_arms(fuzzer, warm)
    assert "entropic" in warm


def test_numpy_and_scalar_entropy_agree():
    """Large tables take the numpy path; both must equal the spec."""
    s = _strategy()
    big = set(range(200))
    s.observe(A, big)
    s.observe(A, set(range(0, 200, 2)))

    expected = {e: (2 if e % 2 == 0 else 1) for e in big}
    assert s.energy(A) == pytest.approx(_spec_energy(expected, 2, 200))
