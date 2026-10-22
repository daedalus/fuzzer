# Plan: DPP seed selector (2026-10-22)

## Ask
Seed arm that favours seeds whose coverage differs from the seeds picked just
before them. The aim is fewer near-duplicate seeds back to back without
starving high-value seeds.

## Model
L-ensemble kernel over corpus seeds:

```
L_ij = q_i · S_ij · q_j        S = Φ̂ Φ̂ᵀ  (cosine of hashed edge sets, PSD)
```

- `Φ_i ∈ R^D` hashes seed i's edge set into D buckets with a fixed hash, so
  it needs no RNG. Rows are L2-normalised, which gives `S_ii = 1`.
- `q_i` is quality: `(Σ_{e∈i} 1/freq(e))^θ`, where `freq(e)` = number of
  seeds hitting e. Rare-edge seeds score high.

Sequential pick conditioned on window W (the last ≤ K picks):

```
P(i | W) ∝ det(L_{W∪i}) / det(L_W) = d_i²   (Schur complement)
```

`d²` is updated incrementally with fast greedy-MAP Cholesky (Chen et al.,
NeurIPS 2018). Each pick costs O(N·(D+K)) and needs no N×N matrix. This is
sampling, not greedy MAP: a single `rng.random()` inverse-CDF draw keeps
exploration and makes the pick scriptable (Hard Rule 39).

```
 refit (fuzz_round, every R execs)        select_seed(keys)  (per pick)
 ┌──────────────────────────────┐         ┌───────────────────────────────┐
 │ seed_edges ─hash→ Φ̂ (N×D f32)│         │ d² ← q²  (window empty)       │
 │ freq(e) ─→ q (N)             │ ──────▶ │ j ~ d² / Σd²   (rng.random)   │
 │ row index: key → row         │         │ e = (L_j − Cᵀc_j)/√d²_j       │
 └──────────────────────────────┘         │ C.append(e); d² −= e²         │
                                          │ |W| = K or Σd² ≤ ε → reset    │
                                          └───────────────────────────────┘
```

## Code sketch — `core/schedulers/seed_dpp.py`

```python
HASH_DIM = 256          # Φ width; N×256 float32 = 10 MB at 10k seeds
WINDOW = 8              # K: picks conditioned on before reset
REFIT_EVERY = 2000      # execs between kernel rebuilds (Tang's interval)
QUALITY_EXP = 0.5       # θ
RESIDUAL_EPS = 1e-9     # Σd² at or below this: window exhausted, reset
EDGE_MULT = 0x9E3779B97F4A7C15  # Fibonacci hash, edge id → bucket


class SeedDPPScheduler(ArmCounts):
    """Sequential DPP seed sampler over hashed edge-set features."""

    supports_priors = False  # seed arm; Beta priors unused

    def __init__(self, rng: RandPool, window: int = WINDOW, refit_every: int = REFIT_EVERY):
        if rng is None:
            raise ValueError("SeedDPPScheduler needs a RandPool (Hard Rule 16)")
        ...
        self._phi = np.zeros((0, HASH_DIM), np.float32)  # q-scaled rows
        self._row: dict[str, int] = {}
        self._chol: list[np.ndarray] = []                 # C rows, ≤ K
        self._d2 = np.zeros(0)
        self._last_refit = -refit_every                   # refit on first call

    def maybe_refit(self, tracker, exec_count: int) -> bool: ...
    def refit(self, tracker) -> None: ...            # builds _phi, _row; resets window

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]

        rows, novel = self._rows(seed_ids)            # novel: keys newer than refit
        d2 = self._live_d2(rows, novel)               # novel keys take max(d²)
        if d2.sum() <= RESIDUAL_EPS:
            self._reset()
            d2 = self._live_d2(rows, novel)

        pick = _cdf_index(d2, self._rng.random())
        self._condition(rows[pick])                   # Cholesky step, K cap
        return seed_ids[pick]
```

`_condition` is only a Cholesky update over all rows:
`L_j = self._phi @ self._phi[j]`, so it reads one row and never builds N×N.

## Wiring
OS-style arm (`select_seed(keys)`). It also needs a model-style refit hook
because OS arms never get the tracker.

| File | Change |
|---|---|
| `core/schedulers/seed_dpp.py` | new |
| `core/schedulers/__init__.py` | export `SeedDPPScheduler` |
| `services/seed_picker.py` | `_OS_SEED_ARMS["dpp"] = "_seed_dpp"`; no `_OS_SEED_SIGNALS` entry |
| `services/fuzzer.py` | `seed_dpp_scheduler=False` appended after `negative_bloom`; `self._seed_dpp = SeedDPPScheduler(rng=self._rng) if seed_dpp_scheduler else None`; add to `_seed_os_arms`; `"dpp"` in `_SEED_STRATEGY_NAMES`; "Seed selection" banner line |
| `services/fuzz_round.py` | next to the Tang refit: `if f._seed_dpp is not None: f._seed_dpp.maybe_refit(f._edge_tracker, f.exec_count)` |
| `cli/commands.py` | `--seed-dpp-scheduler` (`store_true`); kwarg pass-through in `cmd_fuzz`. **Not** in `_HAIL_MARY_FLAGS` until it wins an A/B |
| `tools/lib/bench_paired.py` | `ARMS["seed-dpp"] = ["--elo", "--seed-dpp-scheduler"]`; `ARM_BASELINES` → `p2c`, `drr` |
| tests to update | `test_os_net_scheduler_wiring.py` (`SEED_ARMS`, `UNMEASURED_ARMS`); `test_regression_scheduler_independence.py::TestSeedSchedulerExports`; `test_regression_scheduler_operator_reach.py` exclusion set; the appended-last constructor tests (`test_kruskal_count.py`, `test_target_schedule.py`, `test_position_arena.py`); `test_regression_hail_mary_gates.py` `_EXCLUDED_OPT_IN` |
| docs | `DEEP_DIVE.md` §Scheduling Intelligence bullet; `TODO.md` A/B item |

There is no persistence, the same as the other OS arms: the kernel is rebuilt
on the first refit after `--resume`. `architecture.png` is unchanged because
this adds an arm, not a subsystem.

## Tests (TDD: write each, watch it fail, then implement) — `tests/test_seed_dpp.py`

Each test is checked against a pure-Python oracle `_brute_ratio(L, W, i)` =
`det(L_{W∪i})/det(L_W)` computed by Gaussian elimination. The oracle is
written first; then the numpy version, kept if faster (Hard Rule 14).

- **Policy**
  - `d²` equals the oracle ratio for every i after 0..K picks, with random PSD L.
  - First pick ∝ `q²`.
  - Window resets at K.
- **Falsification** (Hard Rule 23): seeds A, B (identical edge sets) and C
  (disjoint). `ScriptedRng(randoms=[0.0, u])` picks A, then a `u` placed at
  B's old CDF slot must return C, because `d²_B = 0`. Under uniform it would
  return B, so the test fails against a uniform stub.
- **Adversarial**
  - 1000 identical seeds: the residual collapses, the window resets, and a
    live key comes back with no NaN.
  - Zero-edge seeds: `Φ` row is 0.
  - Keys absent from the index (added since refit).
  - Keys dropped since refit.
  - Single key, empty list.
  - Refit mid-window.
- **Distribution, with control** (Hard Rule 46): compare 20k picks of the
  sampler against oracle-driven sampling with `core/chi_squared.chi_squared_test`
  (contingency, no scipy). First assert oracle-vs-oracle passes.
- **Wiring**: Elo pool dispatch via `_pick_os_seed`; the `fuzz_round` refit
  hook fires on the interval; the CLI flag reaches `Fuzzer` (AST walk in the
  wiring test).
- **Bounds** (Hard Rule 54): `_phi.nbytes == N·D·4`; `len(_chol) ≤ K`.

## Performance (Hard Rule 41)
- Micro-bench `select_seed` at N ∈ {1k, 10k} and `refit` at N = 10k on ffmpeg-scale edge sets.
- Budget: pick < 1 ms; refit < 100 ms, which stays off the pick path as Tang's does.
- `--profile-hotpath` on fuzzgoat with the arm on vs off: no regression in exec/s.

## Measurement
`tools/benchmark.py paired seed-dpp` vs `p2c` and `drr` on clang-built
ASAN fuzzgoat (Hard Rule 52). Add it to `_HAIL_MARY_FLAGS` only on a
significant McNemar win.

## Open
- (a) Hashed cosine vs exact Jaccard. The collision rate at D=256 needs
  measuring on ffmpeg; raise D or switch to MinHash signatures
  (`EdgeTracker.signatures`) if it is too high.
- (b) `θ`, `K`, `REFIT_EVERY` are untuned.
- (c) Whether novel keys (since refit) should get max `d²` or the median.
- (d) Pick probabilities are not ratios of exact k-DPP marginals; they are
  sequential volume sampling. That is fine for diversity, but state it in the
  docs.
