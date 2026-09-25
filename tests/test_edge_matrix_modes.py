"""``tools/edge_matrix_modes.py``: the extra ``edge_diagnostic.py matrix`` analyses.

Each analysis carries a falsification test (a corpus built so the answer is known
and the statistic must say so) and an adversarial one (degenerate or corrupted
input). Rule 46 controls are inline: the reference is compared against a second run
of itself before it is trusted.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.rand_pool import RandPool

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EMM = _load("edge_matrix_modes")
ED = _load("edge_diagnostic")


def _mat(rows) -> np.ndarray:
    return np.array(rows, dtype=float)


def _runs(mat: np.ndarray, ids=None):
    ids = np.arange(mat.shape[1], dtype=np.int64) if ids is None else np.asarray(ids)
    out = []
    for row in mat:
        nz = np.flatnonzero(row)
        out.append((ids[nz].astype(np.int64), row[nz].astype(np.int64)))
    return out


def _rec(*pairs) -> np.ndarray:
    """A tracer record array: (prev, cur, site=0) per pair."""
    return np.array([(p, c, 0) for p, c in pairs], dtype=np.uint64)


# ── Flaky edges ───────────────────────────────────────────────────────


def _rep(ids, counts):
    return (np.array(ids, dtype=np.int64), np.array(counts, dtype=np.int64))


def test_flaky_presence_and_count_variance():
    reps = [
        [_rep([1, 2, 3], [1, 1, 5]), _rep([1, 2], [1, 1]), _rep([1, 2, 3], [1, 1, 5])],
        [_rep([1, 2, 4], [1, 2, 1])] * 3,
    ]
    out = EMM.flaky_edges(reps)
    assert out["presence_flaky"] == 1  # edge 3, present in 2 of 3
    assert out["count_flaky"] == 0
    assert out["union_edges"] == 4
    assert out["flaky_ids"] == [3]


def test_flaky_count_only_variance():
    reps = [[_rep([1], [1]), _rep([1], [4])]]
    out = EMM.flaky_edges(reps)
    assert (out["presence_flaky"], out["count_flaky"]) == (0, 1)


def test_flaky_control_identical_repeats_report_nothing():
    reps = [[_rep([1, 2], [1, 3])] * 4, [_rep([2, 5], [2, 2])] * 4]
    out = EMM.flaky_edges(reps)
    assert out["flaky_fraction"] == 0.0
    assert out["flaky_ids"] == []


def test_flaky_private_edges_that_are_all_noise():
    reps = [
        [_rep([1], [1]), _rep([1], [1])],
        [_rep([1, 9], [1, 1]), _rep([1], [1])],  # 9 private to seed 1, and flaky
    ]
    out = EMM.flaky_edges(reps)
    assert out["seeds_with_private"] == 1
    assert out["private_all_flaky"] == 1


def test_flaky_adversarial_single_repeat_is_unmeasurable():
    assert EMM.flaky_edges([[_rep([1], [1])]])["available"] is False
    assert EMM.flaky_edges([])["available"] is False


# ── Subsumption ───────────────────────────────────────────────────────

POSET = _mat([[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]])
POSET_IDS = np.array([10, 20, 30, 40])


def test_subsumption_seed_side():
    s = EMM.subsumption(POSET, POSET_IDS)["seeds"]
    assert (s["rows"], s["distinct_rows"], s["duplicate_rows"]) == (4, 3, 1)
    assert s["subsumed_distinct"] == 1
    assert s["maximal_rows"] == 2
    assert (s["width"], s["height"]) == (2, 2)


def test_subsumption_edge_side_implications():
    e = EMM.subsumption(POSET, POSET_IDS)["edges"]
    assert (e["edges"], e["classes"], e["duplicate_edges"]) == (4, 3, 1)
    assert e["implication_pairs"] == 1
    assert e["deepest_classes"] == 2
    top = e["gateways"][0]
    assert top["ids"] == [30]
    assert top["implied_edges"] == 2


def test_subsumption_falsification_chain_and_antichain():
    chain = np.tril(np.ones((6, 6)))
    s = EMM.subsumption(chain, np.arange(6))["seeds"]
    assert (s["maximal_rows"], s["width"], s["height"]) == (1, 1, 6)
    anti = np.eye(6)
    s = EMM.subsumption(anti, np.arange(6))["seeds"]
    assert (s["maximal_rows"], s["width"], s["height"]) == (6, 6, 1)


def _brute(rows: list[int]):
    n = len(rows)
    below = [[i != j and rows[i] & rows[j] == rows[i] for j in range(n)] for i in range(n)]
    comp = [[below[i][j] or below[j][i] for j in range(n)] for i in range(n)]
    width = max(
        len(sub)
        for k in range(1, n + 1)
        for sub in itertools.combinations(range(n), k)
        if all(not comp[a][b] for a, b in itertools.combinations(sub, 2))
    )
    memo: dict[int, int] = {}

    def depth(i):
        if i not in memo:
            memo[i] = 1 + max((depth(j) for j in range(n) if below[j][i]), default=0)
        return memo[i]

    maximal = sum(1 for i in range(n) if not any(below[i]))
    return maximal, width, max(depth(i) for i in range(n))


def test_subsumption_matches_brute_force_on_random_matrices():
    pool = RandPool(11)
    for _ in range(6):
        bits = np.array(pool.random_list(9 * 7)).reshape(9, 7) < 0.45
        bits[:, 0] |= ~bits.any(axis=1)  # no empty rows
        ints = [int("".join(str(int(b)) for b in r), 2) for r in bits]
        distinct = sorted(set(ints))
        got = EMM.subsumption(bits.astype(float), np.arange(7))["seeds"]
        assert (got["maximal_rows"], got["width"], got["height"]) == _brute(distinct)


def test_subsumption_adversarial_degenerate_shapes():
    one = EMM.subsumption(_mat([[1, 1]]), np.arange(2))["seeds"]
    assert (one["distinct_rows"], one["width"], one["height"]) == (1, 1, 1)
    same = EMM.subsumption(_mat([[1, 0]] * 5), np.arange(2))["seeds"]
    assert (same["distinct_rows"], same["duplicate_rows"], same["maximal_rows"]) == (1, 4, 1)
    assert EMM.subsumption(np.zeros((0, 0)), np.array([]))["available"] is False


# ── Admission replay and resolution ladder ────────────────────────────


def test_admission_rules_on_a_hand_built_corpus():
    mat = _mat([[1, 0], [2, 0], [3, 0], [1, 1]])
    rules = EMM.admission_replay(mat, shuffles=0, seed=1)["rules"]
    assert rules["edge"]["corpus_order"] == 2
    assert rules["bucket"]["corpus_order"] == 4
    assert rules["maxcount"]["corpus_order"] == 4


def test_admission_bucket_and_maxcount_disagree_inside_a_bucket():
    up = EMM.admission_replay(_mat([[4], [5]]), shuffles=0, seed=1)["rules"]
    assert (up["bucket"]["corpus_order"], up["maxcount"]["corpus_order"]) == (1, 2)
    down = EMM.admission_replay(_mat([[5], [4]]), shuffles=0, seed=1)["rules"]
    assert (down["bucket"]["corpus_order"], down["maxcount"]["corpus_order"]) == (1, 1)


def test_admission_control_edge_rule_never_loses_an_edge_in_any_order():
    pool = RandPool(21)
    mat = np.array(pool.randint_list(0, 6, 30 * 12), dtype=float).reshape(30, 12)
    mat[mat < 3] = 0
    out = EMM.admission_replay(mat, shuffles=8, seed=2)
    assert out["rules"]["edge"]["retains_union"] is True
    for rule in ("bucket", "maxcount"):
        assert out["rules"][rule]["shuffled_min"] >= out["rules"]["edge"]["shuffled_min"]
        assert out["rules"][rule]["extra_over_edge_mean"] >= 0.0


def test_admission_adversarial_empty_seed_is_never_admitted():
    out = EMM.admission_replay(_mat([[0, 0], [1, 0]]), shuffles=3, seed=1)
    assert out["rules"]["edge"]["shuffled_max"] == 1


def test_resolution_ladder_is_monotone_in_distinct_rows():
    mat = _mat([[1, 0], [2, 0], [4, 0], [4, 1], [5, 1]])
    ladder = EMM.resolution_ladder(mat)
    rows = [ladder[k]["distinct_rows"] for k in ("binary", "bucket", "raw")]
    assert rows == sorted(rows)
    assert ladder["binary"]["distinct_rows"] == 2
    assert ladder["raw"]["distinct_rows"] == 5


# ── Rarefaction / Chao2 calibration ───────────────────────────────────


def _live_estimate(owner_counts, m):
    ns = SimpleNamespace(
        cumulative_edges=set(range(len(owner_counts))),
        _edge_owner_count=dict(enumerate(owner_counts)),
        seed_edges={i: None for i in range(m)},
    )
    return EdgeTracker.good_turing_estimate(ns)


@pytest.mark.parametrize(
    ("owners", "m"),
    [
        ([1] * 5 + [2] * 3 + [3] * 4, 6),  # Q2 < 10: bias-corrected form
        ([1] * 30 + [2] * 12 + [5] * 10, 40),  # Q2 >= 10: classic form
        ([3] * 8, 5),  # no singletons
    ],
)
def test_chao2_matches_the_live_estimator(owners, m):
    mine, live = EMM.chao2(np.array(owners), m), _live_estimate(owners, m)
    for key in ("chao2", "ci_low", "ci_high", "sample_coverage"):
        assert mine[key] == pytest.approx(live[key])


def test_rarefaction_curve_is_monotone_and_ends_at_the_union():
    pool = RandPool(31)
    mat = (np.array(pool.random_list(40 * 30)).reshape(40, 30) < 0.2).astype(float)
    out = EMM.rarefaction(mat, resamples=20, seed=3)
    curve = out["curve_mean"]
    assert curve == sorted(curve)
    assert curve[-1] == out["union_edges"] == int((mat.sum(axis=0) > 0).sum())


def test_rarefaction_falsification_front_loaded_corpus_beats_its_null():
    mat = np.zeros((10, 10))
    mat[0, :] = 1
    for i in range(1, 10):
        mat[i, i] = 1
    out = EMM.rarefaction(mat, resamples=50, seed=4)
    assert out["order"]["obs_area"] == pytest.approx(1.0)
    assert out["order"]["z"] > 0


def test_rarefaction_control_shuffled_corpus_reads_null():
    pool = RandPool(41)
    mat = (np.array(pool.random_list(60 * 80)).reshape(60, 80) < 0.1).astype(float)
    order = list(range(60))
    RandPool(5).shuffle(order)
    out = EMM.rarefaction(mat[order], resamples=100, seed=6)
    assert abs(out["order"]["z"]) < 3.0


def test_rarefaction_calibration_reports_bias_and_coverage():
    pool = RandPool(51)
    mat = (np.array(pool.random_list(60 * 50)).reshape(60, 50) < 0.15).astype(float)
    out = EMM.rarefaction(mat, resamples=30, seed=7)
    for row in out["calibration"]:
        assert 0.0 <= row["ci_covers_full"] <= 1.0
        assert 0.0 < row["obs_ratio"] <= 1.0
        assert np.isfinite(row["mean_rel_error"])


def test_rarefaction_adversarial_too_few_seeds_or_no_edges():
    assert EMM.rarefaction(_mat([[1], [1]]), resamples=5, seed=1)["available"] is False
    assert EMM.rarefaction(np.zeros((8, 4)), resamples=5, seed=1)["available"] is False


# ── Bootstrap ─────────────────────────────────────────────────────────


def _distinct_rows(mat):
    return {"distinct_rows": float(len({r.tobytes() for r in mat}))}


def test_bootstrap_is_deterministic_per_seed():
    mat = np.eye(8)
    a = EMM.bootstrap_ci(mat, _distinct_rows, resamples=25, seed=9)
    b = EMM.bootstrap_ci(mat, _distinct_rows, resamples=25, seed=9)
    assert a == b


def test_bootstrap_falsification_identical_rows_have_no_spread():
    out = EMM.bootstrap_ci(np.ones((6, 3)), _distinct_rows, resamples=30, seed=1)
    s = out["stats"]["distinct_rows"]
    assert (s["sd"], s["low"], s["high"], s["point"]) == (0.0, 1.0, 1.0, 1.0)


def test_bootstrap_resamples_shrink_a_distinct_row_count():
    out = EMM.bootstrap_ci(np.eye(12), _distinct_rows, resamples=60, seed=2)
    s = out["stats"]["distinct_rows"]
    assert s["point"] == 12.0
    assert s["mean"] < 12.0  # ~63% distinct under with-replacement resampling
    assert s["low"] <= s["mean"] <= s["high"]


def test_bootstrap_adversarial_nothing_to_resample():
    assert EMM.bootstrap_ci(np.eye(1), _distinct_rows, resamples=10, seed=1)["available"] is False
    assert EMM.bootstrap_ci(np.eye(4), _distinct_rows, resamples=0, seed=1)["available"] is False


def test_headline_stat_runs_on_the_real_matrix_code():
    mat = _mat([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    head = ED._headline(mat)
    assert head["union_edges"] == 3
    assert head["distinct_rows"] == 4
    assert head["gf2_rank"] == ED.gf2_structure(mat)["gf2_rank"]


# ── Length confound ───────────────────────────────────────────────────


def test_length_confound_volume_is_size_when_rows_scale_with_size():
    base = np.array([1, 2, 0, 3, 1, 0, 2, 1, 1, 4, 0, 1], dtype=float)
    mat = np.array([base * (i + 1) for i in range(20)])
    out = EMM.length_confound(mat, np.arange(12), np.arange(20) + 1)
    assert out["rho_size_total"] == pytest.approx(1.0)
    assert "rho_size_residual" in out


def test_length_confound_falsification_shuffled_sizes_decorrelate():
    base = np.array([1, 2, 0, 3, 1, 0, 2, 1, 1, 4, 0, 1], dtype=float)
    mat = np.array([base * (i + 1) for i in range(20)])
    sizes = list(range(1, 21))
    RandPool(3).shuffle(sizes)
    out = EMM.length_confound(mat, np.arange(12), np.array(sizes))
    assert abs(out["rho_size_total"]) < 0.6


def test_length_confound_adversarial_constant_sizes_and_mismatch():
    mat = np.eye(6)
    flat = EMM.length_confound(mat, np.arange(6), np.full(6, 100))
    assert flat["rho_size_total"] == 0.0
    with pytest.raises(ValueError, match="sizes"):
        EMM.length_confound(mat, np.arange(6), np.arange(5))
    assert EMM.length_confound(mat, np.arange(6), None)["available"] is False


# ── Prefix fold ───────────────────────────────────────────────────────


def test_prefix_fold_separates_context_only_rows():
    ids = np.array([0x100, 0x101, 0x102, 0x200])
    mat = _mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    out = EMM.prefix_fold(mat, ids, ctx_bits=2)
    assert out["per_bit"][0]["rows_binary"] == 4
    assert out["per_bit"][2]["rows_binary"] == 2
    assert out["ctx_only_rows_binary"] == 2


def test_prefix_fold_falsification_no_tag_variation_means_no_ctx_rows():
    ids = np.array([0x100, 0x200, 0x300])
    out = EMM.prefix_fold(np.eye(3), ids, ctx_bits=4)
    assert out["ctx_only_rows_binary"] == 0
    assert out["ctx_only_rows_raw"] == 0


def test_prefix_fold_adversarial_zero_bits_and_unsorted_ids():
    assert EMM.prefix_fold(np.eye(2), np.array([1, 2]), ctx_bits=0)["available"] is False
    ids = np.array([0x201, 0x100])
    mat = _mat([[1, 0], [0, 1]])
    out = EMM.prefix_fold(mat, ids, ctx_bits=1)
    assert out["per_bit"][0]["edges"] == 2


# ── Score audit ───────────────────────────────────────────────────────


def _audit_matrix():
    mat = np.zeros((12, 7))
    mat[:, 0] = 1  # everyone
    mat[0, 1] = 1  # private to seed 0
    mat[1, 2] = 1  # private to seed 1
    for i in range(2, 12):
        mat[i, 3 + i % 3] = 1
    mat[::4, 6] = 1  # varies degree without adding private edges
    return mat


def test_loo_edges_counts_private_columns():
    loo = EMM.loo_edges(_audit_matrix() > 0)
    assert loo[0] == 1 and loo[1] == 1
    assert loo[2:].sum() == 0


def test_score_audit_reports_every_candidate():
    out = EMM.score_audit(_audit_matrix(), np.arange(7), rank=2)
    for name in ("mass", "residual", "pc1", "leverage", "subspace_distance"):
        row = out["candidates"][name]["loo_edges"]
        assert set(row) >= {"rho", "partial_total", "partial_degree", "dead"}


def test_score_audit_falsification_partial_controls_for_the_outcome_proxy():
    mat = _audit_matrix()
    degree = (mat > 0).sum(axis=1)
    out = EMM.score_audit(mat, np.arange(7), rank=2, outcomes={"deg": degree.astype(float)})
    assert out["baselines"]["degree"]["deg"]["rho"] == pytest.approx(1.0)
    assert out["baselines"]["degree"]["deg"]["partial_degree"] == 0.0
    assert out["candidates"]["mass"]["deg"]["partial_degree"] == 0.0
    assert out["candidates"]["mass"]["deg"]["dead"] is True


def test_score_audit_adversarial_tiny_corpus_is_unavailable():
    assert EMM.score_audit(np.eye(3), np.arange(3), rank=2)["available"] is False


# ── Wiring through edge_diagnostic ────────────────────────────────────


def _saved_corpus(tmp_path: Path) -> Path:
    pool = RandPool(61)
    dense = (np.array(pool.random_list(30 * 40)).reshape(30, 40) < 0.25).astype(float)
    dense *= np.array(pool.randint_list(1, 9, 30 * 40)).reshape(30, 40)
    dense[:, 0] = 3
    ids = (np.arange(40, dtype=np.int64) + 1) * 0x101
    runs = []
    for row in _runs(dense, ids):
        runs.append((np.arange(len(row[0]), dtype=np.int64), row[0], row[1]))
    path = tmp_path / "runs.npz"
    ED.save_runs(path, runs, 65536, sizes=np.arange(30) * 7 + 5)
    return path


def test_sizes_round_trip_through_the_npz(tmp_path):
    path = _saved_corpus(tmp_path)
    assert ED.load_sizes(path).tolist() == (np.arange(30) * 7 + 5).tolist()


def test_load_sizes_is_none_for_legacy_collections(tmp_path):
    path = tmp_path / "legacy.npz"
    ED.save_runs(path, [(np.array([0]), np.array([1]), np.array([1]))], 512)
    assert ED.load_sizes(path) is None


def test_all_offline_runs_every_analysis_and_writes_json(tmp_path, capsys):
    path, out = _saved_corpus(tmp_path), tmp_path / "out.json"
    rc = ED._matrix_main(
        ["--load", str(path), "--all-offline", "--perms", "20", "--resamples", "6",
         "--bootstrap", "5", "--ctx-bits", "4", "--json", str(out)]
    )  # fmt: skip
    assert rc == 0
    got = json.loads(out.read_text())
    for key in ("subsumption", "admission", "rarefaction", "bootstrap", "length_confound",
                "prefix_fold", "score_audit"):  # fmt: skip
        assert key in got
    text = capsys.readouterr().out
    for tag in ("[12]", "[13]", "[14]", "[15]", "[16]", "[17]", "[18]"):
        assert tag in text


def test_flaky_needs_a_live_target(tmp_path):
    path = _saved_corpus(tmp_path)
    with pytest.raises(SystemExit):
        ED._matrix_main(["--load", str(path), "--flaky", "3"])
