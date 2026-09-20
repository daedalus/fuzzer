# Strata schedulers: a seed arm and an op arm built on the edge-matrix findings

**Status:** design plus measurements. Section 3.1 (`--confirm-novelty`) is built,
off by default, unproven (see 8). Sections 3.2-3.4 are design only.
Base `0b6d323`; rebased onto `fbd4cd3e`.
Source: `handover_edge_id_axis_2026-09-18.md` (F1-F13, P0-P4).
Repro: `tools/phantom_edge_probe.py`, `tools/edge_matrix_analysis.py`.
Measured on one target (fuzzgoat, clang 18, `-fsanitize-coverage=trace-pc-guard`,
ASLR off, 250 synthetic JSON inputs). P1-1 (three targets) gates every constant.

## 1. What the measurements say

### 1.1 F2 is live on the default path, not only on the first execution

One persistent table, one discarded warm-up, inputs in order (the fuzzer's own
arrangement). Phantom = id absent from the input's steady state.

| | corpus order | shuffle 0 | shuffle 1 |
|---|---|---|---|
| execs carrying phantom ids | 16/250 | 16/250 | 17/250 |
| new-edge successes observed / steady | 40 / 35 | 40 / 35 | 40 / 33 |
| successes caused only by phantoms | 6 | 5 | 7 |
| singleton edges, phantom share | 12/17 | 13/18 | 10/15 |
| owner<=3 edges, phantom share | 16/36 | 16/36 | 16/35 |
| seeds boosted by phantoms (real `_weight_edge_penalties`) | 12 | 14 | 12 |
| max boost | x2.16 | x2.00 | x2.00 |

Fresh table: every input's first run had 4-6 extra and 5-8 missing ids. After a
warm-up on a *different* input, 4 of 10 still diverged (3-4 extra ids). A
discarded warm-up does not cure F2; the mechanism is still P0-1.

Why nothing downstream catches it (code reads):

- `_calibrate_seed_stability` compares the reruns with each other and never with
  the original run. Phantoms live only in the original run, so reruns agree and
  the verdict is "stable". Confirmed: all 250 inputs' reruns were identical.
- Where it does mask, it masks in the SHM adapter only. `_unstable_edges` is
  written (`fuzzer.py:5817`) and never read. `EdgeTracker` has no purge, so
  `seed_edges`, `_edge_owner_count` and `cumulative_edges` keep the ids.
- `success` for the op reward fan-out is `has_new` from
  `is_new_coverage_with_edges`; phantoms set it.

### 1.2 Family level (`id >> ctx_bits`) is immune to F1

Same 250 inputs, three builds.

| | clean ctx | ASLR raw | ctx-free |
|---|---|---|---|
| distinct edge ids | 349 | 1460 | 256 |
| family owner counts equal to clean | - | 22/22 | 18/22 |
| seeds with identical family set | - | 250/250 | - |
| edge-id Jaccard vs clean | - | 0.198 | 0.232 |
| execs with a new edge (edge / family) | 40 / 5 | 164 / 5 | 38 / 5 |
| families with tag occupancy >= 0.95 | 0 | 6 | 0 |
| max tag occupancy (of 128) | 0.81 | 1.00 | 0.55 |

Family owner counts (clean): seven "prologue" families at 215-250 of 250 seeds,
fifteen at 10-35. Gap between 0.14 and 0.86; any threshold inside it separates.

### 1.3 Ideas measured and dropped

- **Class-collapsed rarity** (partition refinement, exact, incremental,
  verified against brute force): 349 edges -> 261 classes, but
  rare-bonus credits 66 -> 59, seed-rank rho 0.999, top-20 overlap 19/20.
  The duplicate edges are high-owner. Kept as a P3-1 diagnostic only.
- **Spectral residual / leverage (P3-2)**: rank-1-removed row norm on the binary
  matrix has top-20 overlap 20/20 with plain live-edge count; log1p variant
  rho 0.83 with volume. IDF *sums* are also size proxies (rho 0.91 with live
  edges). No independent signal on this target.
- **Tag-level reward discount** (F4): success events 40 (ctx) vs 38 (ctx-free).
  Context variants add almost no success events. No basis for a discount.
- **GF(2) as scheduler**: F8 already shows 133 vs 31 for the cover. Not revisited.

## 2. Coverage of the handover

| finding | disposition |
|---|---|
| F1 ASLR ids | family-only resolution when the stability probe fails (1.2) |
| F2 phantoms | confirm-on-novelty gate, section 3.1 (1.1) |
| F3 blocked axis | family = stratum; only prefix/equality ops exposed |
| F4 tag saturation | per-family occupancy >= cap -> family resolution; no reward discount (1.3) |
| F5 bimodal y, `2^H` | `eff_edges()` logged; consumers gated on E-1 |
| F6 substituted axes | owner count only (already used); `node_idx` stays with aflgo arm |
| F7 spectrum, P3-2 leverage | dropped (1.3); revisit on ffmpeg (P1-1) |
| F8 GF(2) | dropped |
| F9/F10 LLL, classes, Kirchhoff | classes diagnostic only; P1-2 unchanged |
| F11 build defects | arm abstains unless `sancov_guard_status` says instrumented |
| F12 fold, F13 eigen | rejected; PC2 validity waits on P1-3 |
| "not defined on id axis" | enforced by API + relabeling test (section 5) |
| P2-1 `2^H` in stall reason | independent, one commit |

## 3. Design

```
afl shim ids -> [confirm gate] -> EdgeLedger -+-> StrataSeedScheduler  (seed arm "strata")
                                              +-> StrataOpScheduler    (op strategy "op_strata")
                                              +-> confirmed `success` for the op reward fan-out
```

### 3.1 `--confirm-novelty` (shared, the change with the most leverage)

When `is_new_coverage_with_edges` reports new coverage, rerun the input once and
keep `ids & rerun_ids`. Phantoms never reach `record_edges`, `has_new` or the
reward fan-out, so no purge API is needed. Also applied per seed in
`_calibrate_seed_baselines`. Cost: one exec per new-coverage event. In a 12,001
exec fuzzgoat run 137 seeds were admitted, so about 1-2%; measure it (Hard Rule 41),
default off until the A/B.

Open: `_check_new_coverage` already marks phantoms as seen in the SHM adapter, which
is benign (`_seen_edge_ids.update(new)`, `shm.py:876`). `has_new` can also come from
count-bucket novelty on a real edge; keep it when no id is new to the tracker.
`new_max_edges` admits on a separate path and is not gated here. `_repeat_edge_sets` refuses `n_runs < 2`; a
one-rerun helper is needed.

### 3.2 `EdgeLedger` (`core/edge_ledger.py`)

```python
class Res(Enum): TAG = 1; FAMILY = 2
class Trust(Enum): UNKNOWN = 0; STABLE = 1; UNSTABLE = 2

PROLOGUE_FRAC = 0.5      # midpoint of the measured 0.14..0.86 gap
OCC_CAP = 0.95           # measured: 0 clean families vs 6 under ASLR
FAMILY_SHIFT_DEFAULT = 8 # ctx-free builds: families still form at 8

class EdgeLedger:
    def __init__(self, ctx_bits: int): self.shift = ctx_bits or FAMILY_SHIFT_DEFAULT
    def observe(self, seed_key, confirmed: frozenset[int]) -> Novelty: ...
    def set_trust(self, t: Trust) -> None: ...      # from _report_edge_id_stability
    def res(self, fam: int) -> Res:                 # FAMILY if UNSTABLE or occupancy >= OCC_CAP
    def frontier(self) -> list[int]:                # families with owner/n_seeds < PROLOGUE_FRAC
    def seeds_in(self, fam: int) -> list[str]: ...
    def eff_edges(self) -> float: ...               # 2^H of the hit marginal; logged only
    def to_dict(self) / from_dict(...)              # existing pickle machinery
```

`Novelty(edges, families, level)`; `level` is `Res.FAMILY` when the family's
resolution is `FAMILY`. Ids are only masked, shifted and compared for equality;
no arithmetic on their order.

### 3.3 Seed arm `strata` (`core/schedulers/seed_strata.py`)

1. `phi ~ Thompson(Beta(a,b))` over `ledger.frontier()`.
2. Seed within `phi` proportional to `mean_e log1p(N / owner(e))` over its
   confirmed edges in `phi` (mean, not sum: 1.3). At `Res.FAMILY` the score is
   uniform.
3. `record(success_families)`: `a[phi] += 1` if the run confirmed a new edge in
   `phi`, else `b[phi] += 1`.

Elo arm like `kruskal_count`; off by default; abstains from the ballot when the
guard tri-state is not "present" or the frontier is empty.

### 3.4 Op arm `op_strata` (`core/schedulers/op_strata.py`)

Thompson over cells `(op, stratum)` with partial pooling: cell prior
`Beta(K*p_op + s, K*(1-p_op) + f)`, `p_op` the operator's pooled mean, `K` the
pseudo-count. Stratum = the seed arm's `phi`, or the current seed's rarest family
under other seed arms. Reward is the confirmed `success` and the existing cost
adjusted weight in `[0, 1]`. `supports_priors = False`: the pooled posterior is
the prior. Elo-only, absent from `_FALLBACK_PRECEDENCE`, like the other
unproven arms.

## 4. Wiring (Hard Rule 48)

- Ledger: build after `detect_ctx_bits` (`fuzzer.py:824`, `:7059`); `observe` at the
  two `record_edges` sites (`:5061`, `:6978`); `set_trust` from
  `_report_edge_id_stability`.
- Confirm gate: after `is_new_coverage_with_edges` (`:4764`, `:4779`, `:6968`).
- Seed arm: ctor flag `strata` next to `kruskal_count` (`:1212`, `:2150`); name in
  the seed-strategy tuple (`:155`); `_pick_seed_elo` `available` and `strategy_map`;
  `pick_seed` fallthrough; persist/load next to `_load_kruskal_count`;
  `parallel.py` (two signatures, two forwards); `cli/commands.py` flag and help;
  `report.py`/`stats.py` sections.
- Op arm: `_OPERATOR_STRATEGY_NAMES` (`:111`); `operator_strategy_pool`
  (`operators.py:490`); `select_op` branch (`:4532` pattern); `record` fan-out
  (`fuzzer.py:5361`); `_register_arms` (`:2862`).
- Docs: `DEEP_DIVE.md`, `architecture.png` (Rules 11, 44); `lizard --CCN=15` on
  each new file (Rule 45).

## 5. Tests (Rules 23, 38, 39, 46)

- **Phantom injection.** Inject k phantom singletons into observed sets:
  with the gate, picks and ledger stats equal the phantom-free run; without it,
  they differ. The second half proves the test can fail.
- **Relabeling invariance.** Permute tags within each family by a random
  bijection; all family-level outputs are identical. This is "id axis is
  blocked" as an executable statement.
- **Trust flip.** `UNSTABLE` mid-run switches every family to `Res.FAMILY`
  without changing family owner counts.
- **Adversarial.** Empty corpus; one family; all families prologue; `ctx_bits = 0`;
  occupancy 1.0 everywhere; a confirm rerun that crashes or times out.
- **Control.** Same seed, same scripted RNG -> identical picks (scheduler vs
  itself) before any comparison.

## 6. Evaluation (pre-registered; `bench_paired.py`)

E-1 (passive, before building 3.3/3.4): log `(stratum picked, families of
confirmed new edges)` and `(op, stratum, success)` over >= 50k execs on three
targets.
- Kill the seed arm if `P(new edge in phi | picked phi) / P(same | uniform pick) <= 1.5`.
- Kill the op arm if the op x stratum interaction (own likelihood-ratio test,
  Rule 51) is not significant after op main effects.

Arms: A0 baseline; A1 `--confirm-novelty`; A2 A1 + `--strata`; A3 A1 + `op_strata`
with `--elo`; A4 A1 + both. Metric: edges at 10k execs, paired cells, McNemar
exact on discordant pairs, median delta with IQR, Holm across the three
comparisons to A1. A1 must be non-inferior on edges and lose < 2% execs/s.
Control: A0 against a second A0 run must not reject.

## 7. Not proposed

Arithmetic or regression on ids, spectral leverage, GF(2), class-collapsed
rarity, tag-level reward discount, PC2 validity (needs P1-3), an unmeasured
`2^H` temperature. Each has a measurement or a handover line above.

## 8. Status after the build

**Upstream overlap.** `fbd4cd3e` added `seed_residual`, `op_credit` and a
`coverage_trust` gate (`handover_matrix_schedulers_2026-09-19.md`). Their gate is
the F1 Jaccard from a steady-state probe, which by design skips F2, so it passes
while phantoms sit in the tracker. Sections 3.3 and 3.4 overlap those arms:
class-deduplicated mass and credit there versus family strata here. Section 1.3
measured class collapse only against the rare-edge bonus (rho 0.999); it says
nothing about their `1/owners` mass with volume regressed out. Reconcile through
`bench_paired`, not by argument. `--confirm-novelty` is upstream-independent and
feeds their tracker rows and rewards.

**Built:** `core/novelty_confirm.py` (`confirm`), `ShmCoverage.last_new_ids`,
`last_old_bucket_novel`, `reject_phantoms`, `Fuzzer._confirm_new_coverage`,
`--confirm-novelty`, a summary line. 40 tests in `tests/test_novelty_confirm.py`.
Open points from 3.1 resolved: the bucket event survives a phantom neighbour
(the adapter splits the virgin fold so a new id's own bucket does not read as an
old edge moving); no new id means the verdict is left as reported;
`new_max_edges` is not gated.

**Two defects found only by running it, both pinned by tests:**
- The fuzz_one call site first passed `data` (the parent seed); the executed
  input is `mutated`. Every rerun then measured a different input: a fuzzgoat
  campaign found 60 edges instead of about 178 and withdrew 21 of 30 successes.
  The helper's own tests could not see it. `TestCallSitesRerunTheInputThatRan`
  pins both call sites to the variable that ran.
- An unescaped `%` in the flag's help made `fuzz --help` raise. `TestHelp`.

**Measured, fuzzgoat ctx build, one campaign, `--mc-bandit`, `-n 12000`:**
8,473 executions in 99 s, 155 reruns (1.8% of executions), 2 successes withdrawn,
29 phantom ids rejected, 166 edges, 162 seeds added. Only 2 of 155 successes
were withdrawn against 12-18% on the static corpus of section 1.1, so mutation
campaigns may carry less phantom contamination than that corpus suggests. There
is no paired baseline on the fixed build and the run stopped at 8,473 of 12,000
executions for a reason not investigated: this shows the gate runs end to end and
costs about what 3.1 estimated. It shows nothing about coverage.

**Complexity:** `fuzz_one` CCN 384 -> 385 (`is_crash or is_timeout`); new helpers
CCN <= 9.

**Not done:** the E-1 passive logging, arms A0-A4 of section 6, any target other
than fuzzgoat, and the full test suite (only the affected tests were run).
