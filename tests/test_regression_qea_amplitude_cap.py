"""Regression tests for QEA's amplitude-array memory blowup.

_bias_amplitudes_from() converts every input byte into 8 float64
amplitudes (one per bit): a 64x memory amplification with no length
guard. Reached from initialize(), on_fuzz_result(), and the breeding
path in _evolve(), any oversized input flowing into QEA (e.g. via the
_op_fuse_this bug in operators.py, which could grow a seed past the
fuzzer's max_len undetected) turned into a multi-GB allocation and OOM'd
the fuzzer. A 186.9 MB seed measured ~12.5 GB peak RSS at the time this
was diagnosed.

QEA_MAX_INPUT_BYTES / _qea_cap() bound what actually gets converted to
amplitudes, independent of whatever produced the oversized input.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from fuzzer_tool.core.qea import (
    QEA_MAX_INPUT_BYTES,
    QEAIndividual,
    QEALifecycle,
    _bias_amplitudes_from,
    _qea_cap,
)


def _edge_tracker(total_edges=8, weight=1.0):
    et = MagicMock()
    et.cumulative_edges = set(range(total_edges))
    et.seed_edges = {}
    et.compute_wasserstein_weight.return_value = weight
    return et


class TestQeaCapHelper:
    def test_leaves_small_input_untouched(self):
        data = b"A" * 1024
        assert _qea_cap(data) == data

    def test_truncates_oversized_input(self):
        data = b"B" * (QEA_MAX_INPUT_BYTES * 10)
        capped = _qea_cap(data)
        assert len(capped) == QEA_MAX_INPUT_BYTES
        assert capped == data[:QEA_MAX_INPUT_BYTES]

    def test_boundary_exact_cap_size_untouched(self):
        data = b"C" * QEA_MAX_INPUT_BYTES
        assert _qea_cap(data) is data


class TestBiasAmplitudesMemoryBound:
    def test_amplitude_array_bounded_for_huge_input(self):
        """The actual regression: this used to allocate 8 bytes per BIT of
        the raw input (64x the input size). A 50 MB input would have
        produced a ~3.2 GB float64 array; capped, it must stay under the
        QEA_MAX_INPUT_BYTES * 8 floats * 8 bytes ceiling (4 MiB)."""
        huge = b"\x41" * (50 * 1024 * 1024)  # 50 MB
        capped = _qea_cap(huge)
        amps = _bias_amplitudes_from(capped)
        max_expected_bytes = QEA_MAX_INPUT_BYTES * 8 * 8  # bits * float64
        assert amps.nbytes <= max_expected_bytes
        assert amps.nbytes < 5 * 1024 * 1024  # well under 5 MiB, not GB

    def test_amplitude_length_matches_capped_input(self):
        data = b"\x00" * (QEA_MAX_INPUT_BYTES * 3)
        capped = _qea_cap(data)
        amps = _bias_amplitudes_from(capped)
        assert len(amps) == len(capped) * 8


class TestQeaLifecycleCapsOversizedSeeds:
    """Exercise the three real call sites, not just the helper directly."""

    def test_initialize_caps_oversized_corpus_entries(self):
        qea = QEALifecycle(pop_size=4, generation_size=10**9)
        oversized = b"\x01" * (QEA_MAX_INPUT_BYTES * 4)
        corpus = [oversized, b"small seed"]
        qea.initialize(corpus, _edge_tracker())

        assert len(qea.population) == 2
        big_ind = next(i for i in qea.population if len(i.best_collapsed) > 100)
        assert len(big_ind.best_collapsed) == QEA_MAX_INPUT_BYTES
        assert len(big_ind.amplitudes) == QEA_MAX_INPUT_BYTES * 8
        # seed_key must still key off the REAL data (matches edge_tracker's
        # keying elsewhere in the fuzzer), not the truncated bytes.
        import hashlib

        assert big_ind.seed_key == hashlib.sha256(oversized).hexdigest()[:16]

    def test_on_fuzz_result_caps_oversized_new_coverage_input(self):
        qea = QEALifecycle(pop_size=4, generation_size=10**9)
        qea.initialize([b"seed one", b"seed two"], _edge_tracker())

        oversized = b"\x02" * (QEA_MAX_INPUT_BYTES * 4)
        new_ind = qea.on_fuzz_result(
            oversized, new_coverage=True, edge_count=3, edge_tracker=_edge_tracker()
        )

        assert new_ind is not None
        assert len(new_ind.best_collapsed) == QEA_MAX_INPUT_BYTES
        assert new_ind.amplitudes.nbytes <= QEA_MAX_INPUT_BYTES * 8 * 8

    def test_breed_never_produces_oversized_amplitudes(self):
        """Defense-in-depth: even though parents should already be capped,
        the breeding path caps child_bytes explicitly too."""
        qea = QEALifecycle(pop_size=6, generation_size=1, elite_fraction=0.2)
        # Seed with already-large-but-capped individuals to make the
        # crossover path exercise near-cap-size inputs.
        for i in range(6):
            data = bytes([i % 256]) * QEA_MAX_INPUT_BYTES
            qea.population.append(
                QEAIndividual(
                    amplitudes=_bias_amplitudes_from(data),
                    fitness=0.5,
                    edge_count=i,
                    best_collapsed=data,
                )
            )
        qea._evolve(_edge_tracker())
        for ind in qea.population:
            assert len(ind.best_collapsed) <= QEA_MAX_INPUT_BYTES
            assert ind.amplitudes.nbytes <= QEA_MAX_INPUT_BYTES * 8 * 8


class TestQeaCapDoesNotAffectNormalSeeds:
    """Make sure the cap is a no-op for realistic, small fuzzing inputs."""

    def test_typical_seed_size_unaffected(self):
        qea = QEALifecycle(pop_size=4, generation_size=10**9)
        seeds = [b"PNG seed data " * 20 for _ in range(4)]  # ~280 bytes each
        qea.initialize(seeds, _edge_tracker())
        for ind, seed in zip(qea.population, seeds):
            assert ind.best_collapsed == seed
            assert len(ind.amplitudes) == len(seed) * 8
