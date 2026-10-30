# Handover: algorithms present in tests, missing from src

Status: **All shipped (P1 2026-10-29, rest 2026-10-30).** See DEEP_DIVE §Test/tool-only algorithms moved into core. Source: 2026-10-29 audit of all
1,149 test files (six parallel passes, claims re-checked against src).

## Verdicts

| # | Algorithm | Test location | Verdict |
|---|---|---|---|
| 1 | ROC AUC (Mann-Whitney U) + shared tied ranks / Spearman | `test_entropy_kl_miller_madow.py:23` | **Shipped** (P1) |
| 2 | Poisson / geometric samplers | `test_gravity.py:29`, `test_discovery_uniformity.py:27,161` | Shipped: `RandPool.poisson_array` / `geometric` |
| 3 | Hadamard by Sylvester / Paley | `tools/lib/factorial_design.py` | Shipped: `core/hadamard.py`, `hadamard_oa` widths 1/2/4 |
| 4 | Permutation sign / cycle type | `test_swap_tuple.py:31`, `test_structured_mutations.py:326` | Shipped: `core/permutation.py` (test callers only) |
| 5 | XXH64 second preimage | `tests/support/xxh64_collide.py` | Shipped: `hash_inverse.xxh64_collide` (test callers only) |
| 6 | McNemar / Fisher exact / Wilcoxon | `tools/lib/bench_paired.py` | Shipped: `core/exact_tests.py` (tools import it) |
| 7 | Dilworth width, rank mod p | `tools/edge_matrix_modes.py`, `tools/edge_diagnostic.py` | Shipped: `core/poset.py`, `null_lattice.rank_mod_p` |
| 8 | Steinhaus-Johnson-Trotter | `tests/support/sjt.py` | Shipped: `permutation.sjt_swaps` |
| 9 | Panzeri-Treves TE correction | `test_transfer_entropy_bias_correction.py:25` | Shipped for comparison: `panzeri_treves_bias`; scoring keeps the shuffle correction |
| 10 | Euclid step count, signal generators (Cox, AR(1), 1/f, GARCH) | `test_structured_mutations.py:427`, `test_regression_periodicity_null.py` | Shipped: `numtheory.euclid_steps`, `core/synthetic_series.py` |
| 11 | Rényi divergence | `test_regressions.py:243` | Shipped: `renyi.renyi_divergence` (no production caller) |

---

## P1 — `core/rank_stats.py` (shipped 2026-10-29)

- `average_ranks`, `rank_corr` moved from `edge_matrix` (re-exported there); new `rank_auc`.
- `tools/frontier.py`, `edge_diagnostic.py`, `entropy_shapley_vs_loo.py` use `rank_corr`.
- Report: `auc=` under Entropy KL (`EntropyKLSeedStrategy.cached_scores()`,
  `LineageTree.child_count`), end-of-run only.
- Kept: `analyzer_coverage_regime._spearman_rank_correlation` — numpy is slower at
  its n <= 100 (12.5x at n=3, 0.92x at 100). See DEEP_DIVE §Byte-entropy seed arms.
- Dropped from the plan: `test_target_arena.py:_auc` is an `AuctionTarget` fixture,
  not ROC AUC.

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

P3, then P2 once `_trickle` is decided.
Each lands with TDD (test first, observe fail), lizard on new files,
DEEP_DIVE entry, and a TODO close in the same commit.
