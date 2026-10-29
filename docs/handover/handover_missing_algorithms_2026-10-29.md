# Handover: algorithms present in tests, missing from src

Status: **PLAN ONLY — not implemented.** Source: 2026-10-29 audit of all
1,149 test files (six parallel passes, claims re-checked against src).

## Verdicts

| # | Algorithm | Test location | Verdict |
|---|---|---|---|
| 1 | ROC AUC (Mann-Whitney U) + shared tied ranks / Spearman | `test_entropy_kl_miller_madow.py:23`, `test_target_arena.py:232` | **Implement** (P1) |
| 2 | Poisson / geometric samplers | `test_gravity.py:29`, `test_discovery_uniformity.py:27,161` | **Implement** (P2) |
| 3 | Hadamard by Sylvester / Paley | `tools/lib/factorial_design.py` | **Implement** (P3) |
| 4 | Permutation sign / cycle type | `test_swap_tuple.py:31`, `test_structured_mutations.py:326` | Implement only with P4's consumer |
| 5 | XXH64 second preimage | `tests/support/xxh64_collide.py` | Keep test-only: no target-side consumer (zstd checksums are recomputed, not collided) |
| 6 | McNemar / Fisher exact / Wilcoxon | `tools/lib/bench_paired.py` | Keep in tools: no src consumer |
| 7 | Dilworth width, rank mod p | `tools/edge_matrix_modes.py`, `tools/edge_diagnostic.py` | Keep in tools |
| 8 | Steinhaus-Johnson-Trotter | `tests/support/sjt.py` | Keep test-only (n! oracle) |
| 9 | Panzeri-Treves TE correction | `test_transfer_entropy_bias_correction.py:25` | Rejected by design (`analyzer_transfer_entropy.py:118-130`) |
| 10 | Euclid step count, signal generators (Cox, AR(1), 1/f, GARCH) | `test_structured_mutations.py:427`, `test_regression_periodicity_null.py` | Test data / metrics; skip |
| 11 | Rényi divergence | `test_regressions.py:243` | Never written, no consumer; skip |

---

## P1 — `core/rank_stats.py`: tied ranks, Spearman, AUC

Three Spearman/rank copies exist (`edge_matrix.rank_corr`,
`analyzer_coverage_regime._spearman_rank_correlation`, plus
`tools/{frontier,edge_diagnostic,entropy_shapley_vs_loo}.py`), and the AUC
that `seed_entropy_kl.py:45` quotes (0.73 -> 0.98) has no code. Hard Rule 1:
one place.

```python
def tied_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged: [3, 1, 3] -> [2.5, 1, 2.5]."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(x, y) -> float:
    """Pearson on tied ranks; nan when either side is constant."""


def rank_auc(scores, labels) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie) = U / (n1 n0); nan if a class is empty."""
    r = tied_ranks(np.asarray(scores, dtype=np.float64))
    pos = np.asarray(labels, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return float("nan")
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
```

**Wiring**
- `edge_matrix.rank_corr` and `analyzer_coverage_regime._spearman_rank_correlation`
  call `spearman` (keep their signatures; the latter keeps its p-value).
- `tools/frontier.py`, `tools/edge_diagnostic.py`, `tools/entropy_shapley_vs_loo.py`
  import `tied_ranks`/`spearman` instead of local copies.
- Seed-score AUC in the report (`services/report.py`): per seed scheduler with
  `scores()`, AUC of score vs label "seed has >= 1 lineage child"
  (`LineageTree`). Report-only; no per-exec cost.
- Tests: `test_entropy_kl_miller_madow.py` and `test_target_arena.py` import
  `rank_auc`; new `tests/test_rank_stats.py`.

**Tests:** oracle = all-pairs count over (pos, neg) with ties as 0.5 (control:
oracle vs itself); falsification: perfect separation -> 1.0, reversed -> 0.0;
adversarial: all ties -> 0.5, empty class -> nan, NaN scores rejected.
Equivalence: the replaced Spearman copies give identical floats on seeded data.

---

## P2 — `RandPool.poisson_array` / `geometric`

Hard Rule 16: tests hand-roll Knuth's sampler because `RandPool` has none.

```python
def poisson_array(self, lams) -> np.ndarray:
    """One Poisson(lams[i]) draw per element; one C-level numpy call."""
    return self._rng.poisson(lams)


def geometric(self, p: float) -> int:
    """Failures before the first success, P(k) = (1-p)^k p; one uniform draw.

    Inversion: floor(log(U) / log(1 - p)). p = 1 -> 0; p <= 0 raises.
    """
```

**Wiring**
- `operator_registry._trickle`: TODO "Arithmetic hotspots" (c) wants a geometric
  skip (~2 draws instead of ~48). It changes the seeded stream that registry
  tests pin, so it needs a go/no-go first. Without it, P2 has only test consumers.
- `ScriptedRng` gains `poisson_arrays=` and `geometrics=` iterators.
- `test_gravity.py` / `test_discovery_uniformity.py` drop `_poisson` /
  `_poisson_sample` / `_geometric`.

**Tests:** mean/variance against λ and (1-p)/p via the in-house chi-squared
(`core/chi_squared.py`, Hard Rule 51), with the Rule 46 control (sample vs a
second seeded sample, `chi2_contingency`-style, not `chisquare`); p = 1 and
λ = 0 edges; p <= 0 raises.

---

## P3 — Hadamard constructions for `hadamard_oa`

`fxt.hadamard_oa` is fixed at order 16 (16-bit flag words). Sylvester gives
8/16/32; Paley I gives 12/20/24. Move `_paley`, `_sylvester_double`,
`_hadamard` from `tools/lib/factorial_design.py` to `core/hadamard.py`;
the tool imports them.

```python
def hadamard_rows(order: int) -> tuple[int, ...]:
    """Rows as words, bit j set where H[i][j] = -1, row 0 all +1 (normalized)."""


OA_WIDTHS = (1, 2, 4)  # flag word bytes -> orders 8, 16, 32
```

**Wiring:** `hadamard_oa` draws a width from `OA_WIDTHS` (bounded by the
region), uses `hadamard_rows(8 * width)`. Order 16 must stay bit-identical
to `HADAMARD_ROWS` (or keep the m-sequence for 16 and document why).
Registry unchanged (same op name).

**Tests:** H Hᵀ = nI for each order; strength-2 OA property per column pair;
falsification: order 12 is not a power of 2 and still passes the identity;
adversarial: order 28 (Paley needs q = 27, not prime) returns None.
Speed: tables cached per order; per-call cost unchanged (Rule 58 check).

---

## P4 — permutation sign / cycle type (conditional)

Implement only if a parity-preserving mutator needs it: `generic.py:~92-110`
describes a parity trap and `cycle_lock` builds single cycles, but neither
checks a permutation's sign. Sketch: `core/permutation.py`
`cycle_type(perm) -> list[int]` (visited-array walk), `sign(perm)` =
(-1)^(n - cycles). Wiring would be an assertion-free fast path in
`_swap_tuple` choosing an even/odd composition; without it, skip.

---

## Order

P1 (dedup + real consumer), then P3, then P2 once `_trickle` is decided.
Each lands with TDD (test first, observe fail), lizard on new files,
DEEP_DIVE entry, and a TODO close in the same commit.
