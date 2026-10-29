"""Tests for core/rank_stats.py: tied ranks, Spearman, rank AUC.

Oracles count pairs directly (Hard Rule 46 control: oracle vs itself first).
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.core import edge_matrix, rank_stats
from fuzzer_tool.core.rank_stats import average_ranks, rank_auc, rank_corr


def _pair_auc(scores, labels):
    """All (pos, neg) pairs: 1 for pos > neg, 0.5 for a tie."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


class TestRankAuc:
    @pytest.mark.parametrize("seed", range(8))
    def test_matches_pair_count(self, seed):
        rng = np.random.default_rng(seed)
        # Integer scores force ties; labels mix both classes.
        scores = rng.integers(0, 6, 60).astype(float)
        labels = rng.random(60) < 0.3
        labels[:2] = (True, False)

        # Control (Hard Rule 46): the oracle against a second run of itself.
        assert _pair_auc(scores, labels) == _pair_auc(scores, labels)
        assert rank_auc(scores, labels) == pytest.approx(_pair_auc(scores, labels), abs=1e-12)

    def test_perfect_and_reversed(self):
        """Falsification: separation reads 1.0, its mirror 0.0."""
        scores = [0.1, 0.2, 0.8, 0.9]
        assert rank_auc(scores, [0, 0, 1, 1]) == 1.0
        assert rank_auc(scores, [1, 1, 0, 0]) == 0.0

    def test_all_ties_is_chance(self):
        assert rank_auc([3.0] * 5, [1, 0, 1, 0, 0]) == 0.5

    @pytest.mark.parametrize("labels", [[1, 1, 1], [0, 0, 0], []])
    def test_one_class_is_nan(self, labels):
        assert math.isnan(rank_auc([1.0, 2.0, 3.0][: len(labels)], labels))

    def test_adversarial_nan_score_raises(self):
        with pytest.raises(ValueError):
            rank_auc([0.1, float("nan"), 0.3], [1, 0, 1])

    def test_adversarial_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rank_auc([0.1, 0.2], [1, 0, 1])


class TestSingleSource:
    def test_edge_matrix_reexports(self):
        """edge_matrix keeps its names; the code lives in rank_stats once."""
        assert edge_matrix.average_ranks is rank_stats.average_ranks
        assert edge_matrix.rank_corr is rank_stats.rank_corr

    def test_rank_corr_on_ties(self):
        # Hand-derived: ranks [0,1,2,3] vs [0.5,0.5,2,3]: cov 4.5, sx^2 5, sy^2 4.5.
        got = rank_corr(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 5.0, 6.0]))
        assert got == pytest.approx(4.5 / math.sqrt(5 * 4.5))

    def test_average_ranks_ties(self):
        assert average_ranks(np.array([5.0, 1.0, 5.0, 3.0])).tolist() == [2.5, 0.0, 2.5, 1.0]


TOOLS = Path(__file__).resolve().parent.parent / "tools"
#: Scripts that each carried a private rank/Spearman copy (ordinal in two of them).
RANK_TOOLS = ("frontier.py", "edge_diagnostic.py", "entropy_shapley_vs_loo.py")


@pytest.mark.parametrize("script", RANK_TOOLS)
def test_regression_tools_use_shared_ranks(script):
    """Falsification: a private copy is back (argsort-of-argsort ranks ties apart)."""
    tree = ast.parse((TOOLS / script).read_text())
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not defined & {"_rank", "_ranks", "_spearman"}
    imported = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "fuzzer_tool.core.rank_stats"
        for a in n.names
    }
    assert "rank_corr" in imported


def test_ordinal_ranks_bias_ties():
    """Adversarial: the replaced ordinal ranking reads tied data as correlated."""
    x = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ordinal = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    biased = float(np.corrcoef(ordinal, y)[0, 1])
    assert biased == pytest.approx(1.0)
    assert rank_corr(x, y) < biased
