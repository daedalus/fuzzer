"""Edges per bit of seed: corpus summary line and per-seed report column."""

from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.edge_density import BITS_PER_BYTE, edges_per_bit
from fuzzer_tool.services.report import _seed_contribution
from fuzzer_tool.services.stats import StatsReporter


def test_edges_per_bit_counts_bits_not_bytes():
    """Falsification: 8 edges on a 1-byte seed is 1 edge/bit, not 8 edges/byte."""
    assert edges_per_bit(BITS_PER_BYTE, 1) == 1.0
    assert edges_per_bit(3, 2) == 3 / (2 * BITS_PER_BYTE)


@pytest.mark.parametrize(
    ("edges", "nbytes"),
    [(5, 0), (0, 10), (0, 0), (-1, 10), (5, -3)],
)
def test_edges_per_bit_degenerate_inputs(edges, nbytes):
    """Adversarial: empty seed, no edges, negative counts -> 0.0, never raise."""
    assert edges_per_bit(edges, nbytes) == 0.0


def _summary_fuzzer(corpus: list[bytes], edges: int) -> MagicMock:
    f = MagicMock()
    f.corpus = corpus
    f.ptrace_cov = None
    f.shm_cov._peak_cumulative_edges = edges
    f._edge_tracker.get_cumulative_edge_count.return_value = edges
    f._edge_tracker.bitmap_density.return_value = 0.0
    f._edge_tracker.birthday_collision_risk.return_value = 0.0
    f._edge_tracker.recommended_map_size.return_value = None
    f._edge_tracker.good_turing_estimate.return_value = {"n": 0}
    f._edge_tracker.coverage_growth_model.return_value = {"confidence": 0.0}
    f._edge_tracker.bayesian_coverage_growth_model.return_value = {}
    return f


def _print_coverage(f: MagicMock) -> None:
    """Run the coverage summary with its trailing sub-reports stubbed."""
    with (
        patch.object(StatsReporter, "_print_summary_zipf"),
        patch.object(StatsReporter, "_print_summary_eta"),
        patch.object(StatsReporter, "_print_summary_edge_stats"),
        patch.object(StatsReporter, "_print_summary_confirm"),
    ):
        StatsReporter(f)._print_summary_coverage(f)


def test_summary_prints_corpus_edges_per_bit(capsys):
    corpus = [b"abcd", b"ef"]
    edges = 24
    _print_coverage(_summary_fuzzer(corpus, edges))

    line = next(ln for ln in capsys.readouterr().out.splitlines() if "Edges per bit" in ln)
    nbits = sum(map(len, corpus)) * BITS_PER_BYTE
    assert f"{edges / nbits:.4f}" in line
    assert f"{nbits:,} corpus bits" in line


def test_summary_empty_corpus_no_crash(capsys):
    """Adversarial: empty corpus prints 0, no ZeroDivisionError."""
    _print_coverage(_summary_fuzzer([], 10))
    line = next(ln for ln in capsys.readouterr().out.splitlines() if "Edges per bit" in ln)
    assert "0.0000" in line


def test_seed_contribution_shows_edges_per_bit():
    f = MagicMock()
    f.corpus = [b"ab", b"abcdefgh"]
    f.shm_cov.cumulative_edges = 100
    f.seed_meta = {
        b"ab": {"coverage_edges": 16, "fuzz_count": 1},
        b"abcdefgh": {"coverage_edges": 32, "fuzz_count": 1},
    }
    lines = _seed_contribution(f).splitlines()

    short = next(ln for ln in lines if ln.rstrip().endswith("ab") and "16 edges" in ln)
    long_ = next(ln for ln in lines if "32 edges" in ln)
    assert f"{16 / (2 * BITS_PER_BYTE):.3f} e/bit" in short
    assert f"{32 / (8 * BITS_PER_BYTE):.3f} e/bit" in long_
