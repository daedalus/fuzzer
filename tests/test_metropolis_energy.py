"""Metropolis energy term (order_theory P0-3).

``ΔE = |P| / |M ∪ P|`` (parent's share of the combined path), ``p = min(1,
exp(-ΔE/T))``. Design and rejected alternatives: the P0-3 section of
``docs/handover/handover_order_theory_ecc_ising_2026-09-20.md``.
"""

import math
from unittest.mock import patch

import pytest

from fuzzer_tool.core.metropolis import MIN_TEMPERATURE, accept_prob, path_energy

PARENT = set(range(100))


def _floor(t):
    return math.exp(-1.0 / max(t, MIN_TEMPERATURE))


# --------------------------------------------------------------------------
# path_energy
# --------------------------------------------------------------------------


def test_identical_path_is_worst_energy():
    assert path_energy(set(PARENT), PARENT) == 1.0


def test_early_exit_subset_is_worst_energy():
    assert path_energy(set(range(5)), PARENT) == 1.0


def test_half_reroute_matches_definition():
    mutant = set(range(50)) | set(range(1000, 1050))
    union = len(mutant | PARENT)
    assert path_energy(mutant, PARENT) == len(PARENT) / union


def test_falsification_small_error_path_not_rewarded():
    """1 - Jaccard would call this near-total divergence; ours stays near 1."""
    mutant = {0, 1, 2, 5000, 5001}
    jaccard = len(mutant & PARENT) / len(mutant | PARENT)
    assert 1.0 - jaccard > 0.95
    assert path_energy(mutant, PARENT) > 0.98


def test_adversarial_empty_sets():
    assert path_energy(set(), PARENT) == 1.0
    assert path_energy({1, 2, 3}, set()) == 1.0
    assert path_energy(set(), set()) == 1.0


def test_energy_monotone_in_reroute_size():
    energies = [
        path_energy(set(range(100 - k)) | set(range(1000, 1000 + k)), PARENT)
        for k in (0, 10, 50, 100)
    ]
    assert energies == sorted(energies, reverse=True)
    assert len(set(energies)) == len(energies)


# --------------------------------------------------------------------------
# accept_prob
# --------------------------------------------------------------------------


def test_worst_energy_keeps_legacy_rate():
    for t in (1.0, 0.5, 0.1):
        assert accept_prob(1.0, t) == pytest.approx(_floor(t))


def test_clamped_to_one():
    assert accept_prob(0.0, 1.0) == 1.0
    assert accept_prob(-3.0, 0.5) == 1.0


def test_temperature_floor():
    assert accept_prob(1.0, 0.0) == pytest.approx(math.exp(-1.0 / MIN_TEMPERATURE))


def test_monotone_in_temperature():
    ps = [accept_prob(0.7, t) for t in (0.1, 0.3, 0.6, 1.0)]
    assert ps == sorted(ps)


# --------------------------------------------------------------------------
# Fuzzer wiring
# --------------------------------------------------------------------------


def _fuzzer(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            anneal_budget=1000,
            metropolis=True,
        )


def test_regression_metropolis_energy_varies_with_mutant(tmp_path):
    """Pre-fix p was exp(-1/T) for every mutant; divergence must now matter."""
    f = _fuzzer(tmp_path)
    f._temperature = 0.5
    parent = b"parent"
    f._edge_tracker.seed_edges[f._seed_key(parent)] = set(PARENT)

    same = f._metropolis_accept_p(parent, set(PARENT))
    rerouted = f._metropolis_accept_p(parent, set(range(50)) | set(range(1000, 1050)))
    assert same == pytest.approx(_floor(0.5))
    assert rerouted > same


def test_untracked_parent_gets_floor(tmp_path):
    f = _fuzzer(tmp_path)
    f._temperature = 1.0
    assert f._metropolis_accept_p(b"unknown", {1, 2, 3}) == pytest.approx(_floor(1.0))
    assert f._seed_key(b"unknown") not in f._edge_tracker.seed_edges


def test_fuzz_one_uses_energy_not_constant():
    import inspect

    from fuzzer_tool.services.fuzzer import Fuzzer

    src = inspect.getsource(Fuzzer.fuzz_one)
    assert "_metropolis_accept_p(" in src
    assert "math.exp(-1.0 / max(self._temperature" not in src
