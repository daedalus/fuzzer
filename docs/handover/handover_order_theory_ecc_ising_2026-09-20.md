# Handover — order theory, coding theory, Ising, partitions, combinatorics

**Date:** 2026-09-20
**Base:** `504e1a4e` (`perf(bayes-ucb): shortlist candidate arms with a probit estimate before bisecting`)
**Status:** analysis only. No production code changed by this patch; this
document is the whole patch. Every number below was measured in-container at
`504e1a4e` and every reproduction is inlined in §9 so it can be re-run from a
clean checkout.

**Tiering** follows `handover_pending_2026-09-06.md` §0 — P0 defect a running
campaign can hit, P1 measured win with the correctness argument settled, P2
shipped-and-unwired, P3 design work gated on a stated question, P4 genuine but
blocking nothing, plus a Rejected section that must be argued against rather
than ignored.

## Trigger

"What from Combinatorics / Error correction code / Ising model / Integer
partition / Order theory applies directly to the fuzzer?" Four of the five
articles turned out to name a concept the tree *already implements under that
name but does not actually compute*. That is the same failure mode as the
thermo/stochastic round (`handover_thermo_stochastic_concepts_2026-09-12`,
where `allan_variance` was the variogram): a file or a function existing with
the right name is not evidence the concept is covered. Each item below was
established by deriving what the code computes and comparing it against what
its own docstring claims, not by inventory.

---

## Summary table

| Tier | Item | File | One line |
|---|---|---|---|
| **P0-1** | Subsumption weight is a size signal | `core/edge_tracker.py:1436` | Measures `1 − \|S\|/\|U\|`; Spearman **−0.106** against true uniqueness |
| **P0-2** | 3-D Pareto sweep is not a maxima computation | `services/seed_picker.py:1743` | Drops **38.9%** of maximal points; disagrees with the exact 4-D branch beside it |
| **P0-3** | Metropolis acceptance has no energy term | `services/fuzzer.py:5972` | `ΔE ≡ 1.0`; admits **16.5%** of every boring exec, campaign-averaged |
| **P1-1** | The effector map is observed and discarded | `services/operators.py:285` | 33 mutants/byte unconditionally; AFL gates 24 of them for free |
| **P2-1** | `skipdet.inference()` returns all zeros, always | `core/skipdet.py:230` | Never writes `eff_map`; also unreferenced from `src/` |
| **P3-1** | Pooled (group-testing) byte probing | — | Gated on measuring the effective-byte count `d` first |
| **P4-1** | Block-count collapse in `block_shuffle_variable` | `core/mutations/generic.py` | 68.3% collapse at len 8, 0.2% at 4 KiB |
| **Rejected** | CRC repair by GF(2) linearity | — | Measured 6× **slower** than `zlib.crc32` at 1 MiB |

---

## P0-1. `compute_subsumption_weight` measures size, not subsumption

**Where.** `src/fuzzer_tool/core/edge_tracker.py:1436`, via
`MinHashLSH.corpus_minhash` (`:579`) and `approximate_union_jaccard` (`:606`).

**What it claims.** Its docstring says "based on how much this seed's coverage
overlaps with other seeds". `EdgeTracker`'s class docstring (`:618-623`) says
"seeds fully subsumed by others get deprioritized". `stats.py:456` prints
"Parasitic seeds: N (fully subsumed)".

**What it computes.** `corpus_minhash()` is called with `seed_keys=None`
(`:585`), which defaults to *all* signatures — including the queried seed's.
So the union `U` contains `S`, hence `S ∩ U = S` and `S ∪ U = U`, and

    Jaccard(S, U) = |S| / |U|      (identically, not approximately)

so the returned weight is `max(0.1, 1 − |S|/|U|)` plus MinHash sampling noise.
Subsumption is set *inclusion* — a partial order — and inclusion never enters
the computation. Two seeds of equal size score the same whether one is fully
redundant or owns every edge it touches.

**Measured.** 480 seeds over 12 random corpora (40 seeds each, mixed redundant
/ random / private edge regions; §9.1 reproduces it):

| correlation with | Spearman |
|---|---|
| true unique-edge fraction | **−0.106** |
| `\|S\| / \|corpus union\|` | −0.658 |

Mean absolute residual against the closed form `1 − |S|/|U|`: **0.0203** — the
whole function is that expression to within MinHash noise. The `−0.658` rather
than `−1.000` is the `max(0.1, …)` floor flattening the tail.

Worked cases:

| corpus | seed | `\|S\|` | uniquely owned | weight |
|---|---|---|---|---|
| A | `redundant` | 200 | **0** | 0.797 |
| A | `unique` | 200 | **200** | 0.906 |
| A | `big` | 1000 | 800 | **0.100** (floor) |
| B | `small_redundant` | 50 | **0** | **0.969** |
| B | `large_unique` | 800 | **800** | 0.406 |

The two 200-edge seeds in corpus A sit inside the noise band despite being at
opposite ends of the property the function names. In corpus B the seed with
zero unique coverage gets the *highest* priority in the corpus and the seed
that owns 800 edges alone gets less than half of it.

**Why it matters beyond the multiplier.** `sub` is not only a multiplicand
(`seed_picker.py:1051`, `w *= sub * div * spa`) — it is **coordinate 0 of the
Pareto score** (`:1720/1722`) and therefore the primary sort key of
`_pareto_front` (`:1744`). The front is currently ordered smallest-coverage
first.

**Fix.** The correct primitive already exists and is already maintained
incrementally: `EdgeTracker.edge_owner_count(e)` (`:2540`), the number of
distinct corpus seeds covering `e`. The order-theoretic quantity is

    unique_fraction(S) = |{ e ∈ S : owner_count(e) == 1 }| / |S|

which is 0.0 for a fully-subsumed seed and 1.0 for a seed no other seed
overlaps. Replace the body of `compute_subsumption_weight` with
`max(0.1, unique_fraction(seed_key))`, keeping the signature, the `[0.1, 1.0]`
range, the `seed_key not in seed_edges → 1.0` guard and the empty-edges → 0.5
guard exactly as they are.

**Cost.** O(|S|) dict reads. This is *not* a per-pass cost: the value is cached
in `f._cached_weights[seed_key]` (`seed_picker.py:1048`) and only recomputed on
cache fill. At the measured FFmpeg shape (~460 edges/seed) that is single-digit
microseconds per seed per cache epoch; at 1800 edges the owner-count loop was
measured at ~110 µs in the edge-owner-count round. Affordable at cache-fill
cadence, and it replaces a 64-permutation MinHash comparison that was
answering a different question.

Read the map with `.get(e, 0)`, **not** a bare subscript: `_edge_owner_count`
is a `defaultdict` and subscripting inserts — the exact trap documented at
`edge_tracker.py:2549-2553`. The existing loop in `_weight_edge_penalties`
(`seed_picker.py:1133-1144`) uses bare subscripts because its write sites
guarantee key presence; a new read path has no such guarantee.

**Calibration warning, and it is not optional.** The value *distribution*
changes, not just the values. Today the weight clusters in a narrow band
around `1 − |S|/|U|` (most seeds 0.4–1.0); under the fix it spreads across the
full `[0.1, 1.0]` with mass at both ends, because "owns nothing uniquely" and
"owns everything uniquely" are both common. Two downstream effects:

1. `w *= sub * div * spa` gains dynamic range. Keep the `0.1` floor.
2. Coordinate 0 of the Pareto score stops being a proxy for size, so the front
   composition changes even before P0-2 is fixed. **Do not land P0-1 and P0-2
   in the same commit** — the front will change for two independent reasons
   and neither A/B will be readable.

Also re-point, in the same commit, `cli/commands.py:1372` (display) and the
`stats.py:456` "fully subsumed" wording, which is currently describing a
classification that this function does not produce.

**Test plan.** `tests/test_subsumption_is_inclusion.py`:

* Two seeds of identical size, one fully contained in a third and one disjoint
  from everything → the disjoint one scores strictly higher. **This test fails
  on current `master`** (0.797 vs 0.906 is within noise and the ordering is not
  stable across seeds) — verify that before trusting the fix.
* A fully-subsumed seed scores the floor, a fully-unique seed scores 1.0,
  exactly, for several sizes — pin the endpoints as `== 0.1` and `== 1.0`, not
  `approx`, per the H=0 lesson from the entropy identity round.
* Monotonicity: adding a seed that duplicates `S`'s edges strictly lowers
  `S`'s weight. The current implementation *raises* it (|U| grows while |S| is
  fixed), so this is the falsification test.
* Invariance: scaling every seed's edge count by a constant factor leaves the
  weights unchanged. Current implementation fails this; it is the cleanest
  single statement of the defect.

**Not in scope.** MinHash stays where it is used for what it is good at —
`find_near_duplicate_seeds` and the LSH pre-filter are untouched. This changes
one consumer of it, not the structure.

---

## P0-2. The 3-D Pareto fast path is not a maxima computation

**Where.** `src/fuzzer_tool/services/seed_picker.py:1743-1752`.

```python
if dims <= 3:
    indices.sort(key=lambda i: (-scores[i][0], -scores[i][1], -scores[i][2]))
    result = []
    max_b = max_c = float("-inf")
    for i in indices:
        _a, b, c = scores[i][0], scores[i][1], scores[i][2]
        if b > max_b or c > max_c:
            result.append(i)
            max_b = max(max_b, b)
            max_c = max(max_c, c)
    return set(result)
```

**The defect.** After sorting by the first coordinate descending, a point is
non-dominated iff no *earlier* point dominates it in the remaining two
coordinates. That is a 2-D staircase query. `(max_b, max_c)` is the
componentwise maximum of the accepted set, which is not a member of that set
and dominates points nothing actually dominates.

**Minimal witness:**

    scores = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.9, 0.5, 0.5)]
    fast path returns {0, 1};  point 2 is dominated by neither
      (1.0,1.0,0.0) loses on c;  (1.0,0.0,1.0) loses on b

**Measured** (§9.2), 200 random windows of 100 points:

| point distribution | true front | fast path | maximal points dropped |
|---|---|---|---|
| uniform continuous | 13.8 | 8.4 | **38.9%** |
| quantised to 0.1 (the realistic regime) | 7.1 | 4.0 | **43.3%** |

**The argument that settles it.** The `dims >= 4` branch immediately below
(`:1756-1768`) is an exact dominance test. The two branches therefore return
different answers on the same data, and `--overlap-mode pareto4d` silently
switches between them. Whatever the intent of "backward compatible path", the
module cannot be holding both semantics on purpose.

**Fix.** Delete the `dims <= 3` branch and let the exact branch handle all
dimensionalities. Measured at `window=100`: sweep **24.1 µs**, exact
**226.4 µs**. `_pareto_front` is called on the cached-weights cadence
(`due(f.exec_count, 100, "seed_picker.pareto")`, `:1792`), so the exact path
costs ~2 µs/exec amortised. That is under the noise floor of `_pick_seed`,
which the FFmpeg profile measured at 51% of runtime before the seed-picker
work.

If the window is ever raised above ~1000, the correct O(N log N) 3-D sweep is
available: sort by `a` descending, maintain the 2-D staircase of accepted
`(b, c)` in a list sorted by `b`, and test a candidate with one `bisect` —
dominated iff the staircase entry at the insertion point has `c >= c_i`. Do
not write that now; `window=100` does not pay for it.

**Tie semantics — decide explicitly, do not inherit it.** The exact branch
uses non-strict `>=` in both directions, so equal points dominate each other
and only the first survives. Strict Pareto dominance (`>=` in all, `>` in at
least one) would put every tied point on the front. That matters here because
ties are not rare:

* seeds without metadata keep the placeholder `(1.0, 1.0, 1.0)` (`:1495`);
* under `_saturation_gated`, `_cached_weights` is forced to `(1.0, 1.0, 1.0,
  0.5)` (`:1042`), so `sub` **and** `spa` are constant 1.0 for every seed and
  the score collapses to the burst-factor axis alone.

In that collapsed state the current sweep returns **exactly one seed** — the
one with maximal `bf` — which then takes `×2.0` while every other seed in the
corpus takes `×0.5`, a 4× spread decided by a single coordinate. Strict
dominance would instead put every placeholder seed on the front. Neither is
obviously right; pick one, write the reason in the docstring, and pin it with
a test. The recommendation is: keep non-strict (so ties do not flood the
front) **and** exclude placeholder-scored seeds from the front computation
entirely, since "no metadata" is an absence of evidence, not a maximal point.

**Test plan.** `tests/test_pareto_front_maximality.py`:

* The three-point witness above, asserted as an exact set.
* Property test over random windows: every index returned is non-dominated
  (soundness) **and** every non-dominated index is returned (completeness).
  Only the second fails today; a test that asserts only soundness passes
  against the broken sweep and is worthless — Hard Rule 39.
* A control that the 3-D and 4-D paths agree when the 4th coordinate is
  constant. This is the Hard Rule 46 shape: run the two implementations
  against each other on data where they must agree.
* Saturation case: all-`(1.0, bf, 1.0)` input, assert the documented tie
  semantics rather than whatever falls out.

---

## P0-3. The Metropolis acceptance rule has no energy term

**Where.** `src/fuzzer_tool/services/fuzzer.py:5970-5982`.

```python
if self._metropolis and self._anneal_budget > 0 and not is_timeout:
    p_accept = math.exp(-1.0 / max(self._temperature, 0.01))
```

`ΔE` is the literal `1.0`. The Metropolis criterion is `min(1, exp(−ΔE/T))`;
with a constant numerator the acceptance probability is a function of the
clock alone and carries no information about the candidate. Every boring
execution is equally acceptable, so this is a uniform random sample of
executed mutants at a decaying rate — a random walk, not annealing.

**Measured** under the live schedule `T = max(0.1, 1 − exec/anneal_budget)`
(`seed_picker.py:647`):

| T | `p_accept` |
|---|---|
| 1.0 | 0.3679 |
| 0.75 | 0.2636 |
| 0.5 | 0.1353 |
| 0.3 | 0.0357 |
| 0.2 | 0.0067 |
| 0.1 | 4.54e-05 |

Campaign-averaged over the linear anneal: **16.5% of every boring execution is
saved to the corpus**, and **35.1%** over the first tenth of the budget.

**Severity note, stated rather than buried.** `--metropolis` is off by default
(`fuzzer.py:990`, `commands.py:3410`), so no default campaign hits this. It is
in P0 because it is a shipped flag whose behaviour does not match its name,
and because the corpus-growth figure is large enough that anyone who switches
it on and sees the corpus explode will diagnose the wrong thing.

**The tree already contains the correct form.** `op_monte_carlo.py:511`:

```python
delta_e = worst_score - score
acceptance = math.exp(-delta_e / temperature)
```

and the same call site already passes a score into it — `2` on the
new-coverage path (`fuzzer.py:5959`), `1` on the Metropolis path (`:5980`).

**Fix, and the paper question it is gated on.** An energy must be *defined*.
By construction `has_new_coverage` is False at this branch, so "new edges" is
not available; the candidates that are:

1. **Path divergence from the parent.** `self._get_current_edge_set()` is
   already called two lines later (`:5981`). `ΔE = 1 − Jaccard(mutant_edges,
   parent_edges)` rewards a mutant that took a *different* route through known
   code. Cheap, available, and it is the quantity annealing over a corpus
   actually wants.
2. **Rarity of the edges hit.** `rare_edge_count` against
   `_edge_owner_count` — but the mutant is not yet a tracked seed, so this
   needs an unowned-edge path that does not insert into the defaultdict.
3. **Hit-count novelty** against the count-class ladder, which is the axis the
   Wasserstein/CRPS family was moved onto.

Recommendation: (1), because it needs no new state and no new accessor. But
**write the choice and the rejected alternatives into the docstring at the
call site**, the same way `core/gaussian.py` records the two discarded
Bayes-UCB paths, so the next pass does not re-derive this.

Whatever is chosen, normalise so that `ΔE = 0` means "as good as the incumbent"
and keep the `min(1, …)` clamp — the current expression has no clamp because
with `ΔE ≡ 1` it can never exceed 1, which is exactly the kind of guard that
disappears when the constant it depended on goes away.

**Test plan.** `tests/test_metropolis_energy.py`: a mutant whose edge set
equals the parent's is accepted with probability ~`exp(0/T)` clamped to 1; a
maximally divergent mutant is accepted with `exp(−1/T)`; the acceptance rate
is monotone in divergence at fixed T and monotone in T at fixed divergence.
The regression that pins the defect: two mutants with very different
divergence must **not** get the same acceptance probability — that assertion
fails on `master` for every pair.

---

## P1-1. The effector map is observed on every deterministic exec and discarded

**Where.** `src/fuzzer_tool/services/operators.py:285`
(`_deterministic_mutation_stream`), consumed at `:4805`
(`_next_deterministic_mutation`) and `:4823` (`maybe_deterministic_mutation`).

**What happens today.** The stream yields **33 mutants per byte** — 8 bitflip,
1 byteflip 8/8, 16 arithmetic (8 deltas × ±), 8 interesting — for every byte
of the seed, unconditionally. AFL gates the arithmetic and interesting-value
passes on an *effector map* built during the 8/8 byteflip pass, which it has
already paid for: if flipping a whole byte does not change the execution
trace, the parser is not reading that byte and 24 of the 33 mutants at that
position cannot produce anything.

Here the information is observed and thrown away. `maybe_deterministic_mutation`
routes every mutant "through the exact same execution/coverage/corpus-save
path `fuzz_one` already gives every mutation" (`:4831-4834`) — so the coverage
result of each byteflip is computed, used for corpus admission, and then
forgotten.

**Size of the prize.** Up to **24/33 = 72.7%** of the deterministic schedule on
inert bytes. What fraction of bytes are inert is target- and format-dependent
and is the one number this item needs before it ships (see "Measure first").

**The pass order is already correct.** The stream runs bitflip → byteflip →
arithmetic → interesting. The eff map is complete at the end of the byteflip
pass, before the two passes it gates. No reordering is required — which is
worth saying because AFL's own order (1/1, 2/1, 4/1, 8/8, 16/8, 32/8, arith,
interest) is different and a port could easily "fix" the order into being
wrong.

**Implementation, in the order the commits should land.**

1. **Feedback channel.** `_deterministic_mutation_stream` yields
   `(mutant, byte_idx, pass_id)` instead of bare `bytes`.
   `_next_deterministic_mutation` keeps the public return type (`bytes | None`)
   and stashes `(byte_idx, pass_id)` on the provider context, so no caller
   signature changes.
2. **Recording.** New `ProviderContext._det_eff: dict[str, bytearray]`, one
   entry per in-flight seed_key, allocated lazily at queue creation. Add
   `note_deterministic_result(changed: bool)` on the provider, called from
   `fuzz_one` after the execution it already performs.
3. **What counts as "changed".** AFL's criterion is *trace differs from the
   seed's baseline*, not *new coverage*. Use the path hash the SHM path
   already produces, compared against the seed's calibrated baseline — the
   same baseline `_calibrate_seed_baselines` establishes. Using
   `has_new_coverage` instead would be far too strict: almost every byteflip
   changes the trace and almost none finds new coverage, so the eff map would
   be nearly all zeros and the arithmetic and interesting passes would be
   deleted rather than gated. **This is the single most likely way to get this
   item wrong.**
4. **Gating.** The arithmetic and interesting loops skip `byte_idx` where
   `eff[byte_idx] == 0`. Keep AFL's belt-and-braces rule: if the eff map marks
   *everything* inert (a target whose path hash is unstable or a seed that
   times out under byteflips), treat the map as absent and run the full
   schedule. A silent total skip is the failure mode that looks like a
   throughput win.
5. **Quota interaction — the real subtlety.** Per-pass quotas
   (`operators.py:328-357`) are computed up front from the *natural* costs
   `[8L, L, 16L, 8L]`. With gating, the true arithmetic and interesting costs
   become `16·|eff|` and `8·|eff|`, which is not known until the byteflip pass
   ends. The quota split must therefore be **recomputed lazily on entry to the
   arithmetic pass**, from the realised eff map. Leaving the up-front split in
   place would hand the gated passes a quota sized for bytes they now skip,
   and the budget would go unspent instead of reaching further into the seed.
6. **Bonus interaction, worth naming.** `handover_pending_2026-09-06.md`
   records that P0-1 shipped as per-pass quotas but left the passes truncating
   as a **prefix of bytes**, so the tail of a long seed gets no deterministic
   treatment. Gating frees budget from inert prefix bytes, which pushes the
   prefix further. It does not close that item — a rotating start offset by
   `fuzz_count` still would — but it shifts the number, so re-measure it
   afterwards rather than carrying the old figure forward.

**Test plan.** A synthetic `exec_fn` over a 512-byte buffer where only `k`
known positions change the returned trace hash:

* Every arithmetic and interesting mutant occurs at one of the `k` positions.
* Bitflip and byteflip mutants still occur at all 512.
* Total mutant count drops from `33·512` to `9·512 + 24·k`, exactly.
* All-inert map → full schedule runs (the fail-safe in step 4).
* Quota case: with `max_mutations` below the natural cost, the realised budget
  is fully consumed and all four passes appear in the output — the property
  the per-pass quota work exists to hold.

**Measure first.** Before landing, record the eff-map density on png, zlib and
ffmpeg corpora. That number is what makes P3-1 decidable and is also the
honest form of the "up to 72.7%" claim above.

---

## P2-1. `skipdet.inference()` cannot return anything but zeros

**Where.** `src/fuzzer_tool/core/skipdet.py:230-298`.

The function allocates `eff_map = bytearray(length)` (`:255`) and **never
writes to it**. Both assignment sites in the module (`:205`, `:220`) belong to
`build_skip_eff_map`, a different function. The comment at `:290` — "Don't
mark in eff_map (it's 0 = skip by default)" — contradicts the docstring at
`:252` — "1 = effective, 0 = skip". The log line at `:294-296` reports
"eff_map has %d effective bytes" and that count is 0 by construction.

Neither `inference` nor `build_skip_eff_map` is called from anywhere in
`src/`; `grep` finds only `tests/test_skipdet.py`. Only `should_det_fuzz` is
wired (`fuzzer.py:2123`). The existing tests pass because they assert lengths
and exec-budget behaviour, never the content of the returned map — the same
shape as the shm-hang lesson: a suite that cannot detect the feature is
inert.

**Decide, do not leave it in the middle.** Two options, and the choice must be
recorded either way so that "absent from the tree" and "considered and
rejected" stay distinguishable:

* **Retire.** P1-1 supersedes both functions — it obtains the same map for
  zero extra executions, where `inference` spends `O(L / 64)` probes and
  `build_skip_eff_map` spends `O(L)`. Delete both, and note in
  `should_det_fuzz`'s docstring that the eff-map half of AFL's SkipDet lives
  in the deterministic stream now.
* **Fix and keep as a pre-pass.** `inference` answers a *prior* question than
  P1-1: it finds large inert ranges **before** paying the 8L bitflip pass,
  which is the only way to skip bitflips too. If kept, write the accept branch
  (`:355-360` equivalent — mark `eff_map[pos:pos+cur_block] = b"\x01" *
  cur_block` when the first flip changes the trace) and wire it ahead of the
  stream.

Recommendation: **retire**, and re-propose the pre-pass as P3-1 below, where
it belongs — because a block-halving search is a weak special case of the
pooling design, and implementing the weak version first makes the strong one
look like a rewrite.

---

## P3-1. Pooled byte probing (group testing), gated on one measurement

**The gate, stated before the design.** How many bytes of a real seed are
effective? Call it `d`. Nothing below is decidable without it, and P1-1
produces it as a by-product. **Do not write code for this item until `d` has
been measured on png, zlib and ffmpeg corpora.**

**The idea.** Finding which of `L` bytes affect the trace is non-adaptive
group testing. A `d`-disjunct pooling matrix identifies any `d` defectives in
`O(d² log L)` tests instead of `L`; Kautz–Singleton builds one from a
Reed–Solomon code, which is the direct link from the coding-theory article to
this tree — and the combinatorial-designs half of the combinatorics article is
the same object seen from the other side.

**What makes this stronger here than textbook group testing.** Classic group
testing gets one bit back per test: defective present, or not. A pooled byte
flip here returns the *set of edges that changed* — far more than one bit, and
enough to attribute changes to sub-pools directly rather than by decoding.
That raises the effective information rate per test well above the `log₂`
bound the classic construction is sized against, so the classic test counts
are an upper bound on what is needed, not an estimate.

**The crossover, stated honestly.** Pooling only wins when `d ≪ L`. At
`L = 4096`: `d ≈ 8` needs on the order of 256 tests against 4096 — a ~16×
win; `d ≈ 64` needs more tests than the linear scan and is a loss. The
byteflip pass in P1-1 costs exactly `L` execs and is already paid for by the
schedule, so pooling has to beat *free* on that pass and only earns its keep
if it also lets the 8L bitflip pass be skipped on pooled-inert ranges. That is
the real argument for it and it should be made explicitly rather than
inherited from the group-testing literature.

**Second gate.** Pooled flips change many bytes at once, so a pool that
crosses a length field or a checksum region can fail the parser outright and
report "all bytes effective" for a structural reason. The tree already knows
where those regions are — `core/field_constraints.py`,
`analyzer_checksum_learner.py`, `core/weizz_tags.py`. Whether pools must
respect those boundaries is the second paper question, and the answer changes
the construction (a design with constrained supports is not Kautz–Singleton).

---

## P4-1. Block-count collapse in `block_shuffle_variable`

**Where.** `src/fuzzer_tool/core/mutations/generic.py`, the
`cuts = sorted(set(cuts))` line.

The operator draws `k ∈ [2,5]` and generates `k−1` cut points from normalised
exponential spacings, then deduplicates integer collisions — which silently
reduces `k`. The clamp `max(1, min(pos, len(data) − 1))` additionally piles
mass onto exactly the two endpoints, which is where collisions concentrate.

**Measured** (§9.4), 20 000 draws per length:

| len | mean blocks lost | P(collapse) at k=5 |
|---|---|---|
| 8 | 0.359 | **68.3%** |
| 16 | 0.167 | 36.4% |
| 32 | 0.079 | 18.4% |
| 64 | 0.038 | 9.3% |
| 256 | 0.009 | 2.3% |
| 4096 | 0.001 | 0.2% |

**The textbook form.** A composition of `L` into exactly `k` positive parts is
in bijection with a `(k−1)`-subset of `{1, …, L−1}`, of which there are
`C(L−1, k−1)`. So `sorted(rng.sample(range(1, L), k − 1))` samples uniformly
from compositions, cannot collide, and needs no clamp.

**Why this is P4 and not higher.** The bias only bites below ~64 bytes, and
the spacings construction is explicitly documented as part of what `--seed`
reproduces (its own comment records that it *loses* to `sorted(uniforms)` by
2.3–2.6× and is kept as the operator's defining construction). Changing the
draw sequence is a reproducibility break across the change boundary for a
small correctness gain on short inputs. Record it; do not fix it as a
drive-by. If it is fixed, note in the commit message that seeded runs are not
comparable across it — the same statement `perf(execution-time)` had to make
about CRPS.

---

## Rejected — argue against these, do not re-propose them

**CRC repair by GF(2) linearity.** CRC is affine, so
`crc(m ⊕ δ) = crc(m) ⊕ crc(δ)`, and a single-byte mutation's CRC could in
principle be repaired in `O(log n)` by multiplying by `x^(8n)` mod the
polynomial — the `crc32_combine` construction. **Measured and falsified for
this tree** (§9.3), pure-Python 32×32 GF(2) matrix exponentiation against
`zlib.crc32`:

| chunk | `zlib.crc32` | GF(2) `x^(8n)` shift |
|---|---|---|
| 4 KiB | 1.1 µs | 941.0 µs |
| 64 KiB | 15.6 µs | 1258.7 µs |
| 1 MiB | 241.0 µs | 1553.2 µs |

`zlib.crc32` is hardware-accelerated C; the identity is exact and the
implementation of it loses by 6× even at 1 MiB, where the asymptotics are
supposed to be overwhelming. The real cost at `core/mutations/png.py:40` is
that `to_bytes()` recomputes every chunk's CRC on every serialisation even
when one chunk was mutated — that is a caching fix with no field theory in
it, and it belongs to the format-mutator perf thread, not here.

The remaining genuine use of CRC linearity is *forcing* a target CRC value by
solving for four free bytes. That capability is already covered from the other
direction by `ChecksumLearner` + `int_checksum_solver` recovering the model and
recomputing; no consumer wants a bit-identical stale CRC field.

**Combinatorics generally.** The surface is mined out.
`handover_done_2026-09-06.md` §10 is the survey; `handover_pending_2026-09-06.md`
item 13 names the open gaps with file:line (10b `allow_bulk` over-conservative,
10c the ~20 `rng.random() < p` sites that defeat enumeration, 10d first-order
Markov only, 10e unreachable derivation space, 10g `byte_shuffle` registered
without its implementation, 10h unverified `markov.py` state transfer). §10i is
a **verified no-op**. The `k=1` parity trap (3-cycles are even, reaching only
`A_n`), the Cayley diameter measurements, the de Bruijn construction cache and
the Gray-code rejection are all settled. `_swap_tuple` remains the one unwired
permutation primitive (P2-2 there) and any new permutation generator would be
the second before the first is connected.

The **one** thing the article offers that the survey did not take is design
theory — covering and packing designs — and that is P3-1 above, not a new
mutation operator.

**Number partitioning.** `core/parallel_cost_partition.py` already ports
Multifit with the non-monotonicity and the hysteresis reasoned out;
`core/job_scheduling.py` has Lawler/EDF/LST/MDD. The frequency-spectrum sense
of integer partitions is what Chao2 already consumes. Nothing open.

**Ising-model diagnostics.** `analyzer_critical_slowing.py` already ports the
rising-variance / lag-1-autocorrelation precursor signature, and Boltzmann
energy-by-cost was settled in the `bench_paired` A/B thread. The only live
Ising content is P0-3.

---

## Ordering

`P0-1 → P0-2 → P1-1 → P0-3 → P2-1`, then `P3-1` once P1-1 has produced `d`,
then `P4-1` if ever.

P0-1 before P0-2 because P0-1 changes what the Pareto coordinates *mean* and
P0-2 changes which points are selected from them — landing them together makes
both unreadable. **Separate commits, separate A/Bs**, and re-measure the front
composition between them.

P1-1 before P0-3 because P1-1 produces a measurement (eff-map density) that
several later items need, and P0-3 is gated on a design choice that should be
made on paper first.

---

## 9. Reproductions

All four run against a clean checkout at `504e1a4e` with `PYTHONPATH=src`.
They are inlined rather than committed as scripts because two previous rounds
lost their measurements to uncommitted instrumentation
(`item4_png_real_corpus_samples.tsv` and friends).

### 9.1 — subsumption weight vs true uniqueness

```python
import random
from fuzzer_tool.core.edge_tracker import EdgeTracker

def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for pos, i in enumerate(s): r[i] = pos
        return r
    rx, ry = rk(x), rk(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0

random.seed(11)
W, UF, SZ = [], [], []
for _ in range(12):
    et = EdgeTracker(); sets = {}
    universe = list(range(4000))
    for i in range(40):
        n = random.randint(30, 600)
        if i % 3 == 0:                      # highly redundant
            s = set(random.sample(universe[:600], min(n, 600)))
        elif i % 3 == 1:                    # random
            s = set(random.sample(universe, n))
        else:                               # private region
            s = set(random.sample(range(10000 + i * 1000, 11000 + i * 1000), min(n, 1000)))
        sets[f"s{i}"] = s; et.record_edges(f"s{i}", s)
    et._corpus_sig = None
    U = set().union(*sets.values())
    for k, s in sets.items():
        others = set().union(*[v for k2, v in sets.items() if k2 != k])
        W.append(et.compute_subsumption_weight(k))
        UF.append(len(s - others) / len(s))
        SZ.append(len(s) / len(U))
# -0.106 against uniqueness, -0.658 against relative size, residual 0.0203
```

### 9.2 — Pareto front completeness

```python
def fast3d(scores):
    idx = sorted(range(len(scores)), key=lambda i: (-scores[i][0], -scores[i][1], -scores[i][2]))
    res, mb, mc = [], float("-inf"), float("-inf")
    for i in idx:
        _a, b, c = scores[i]
        if b > mb or c > mc:
            res.append(i); mb = max(mb, b); mc = max(mc, c)
    return set(res)

def true_front(scores):
    out = set()
    for i in range(len(scores)):
        if not any(i != j
                   and all(scores[j][d] >= scores[i][d] for d in range(3))
                   and any(scores[j][d] >  scores[i][d] for d in range(3))
                   for j in range(len(scores))):
            out.add(i)
    return out

# witness: fast3d -> {0, 1}; true_front -> {0, 1, 2}
print(fast3d([(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.9, 0.5, 0.5)]))
```

### 9.3 — CRC linearity vs `zlib.crc32`

`zlib.crc32(buf)` against square-and-multiply on 32×32 GF(2) matrices for
`x^(8n)` mod `0xEDB88320`, minimum of 5 repeats. Table in the Rejected section.

### 9.4 — block-count collapse

Replay the cut-point construction of `block_shuffle_variable` (normalised
`expovariate(1.0)` spacings, `int(L * cum / s)`, clamp, `sorted(set(...))`) for
`k ∈ [2,5]`, 20 000 draws per length, and count realised blocks.

---

## 10. Method notes that generalise

* **Four of five items are name/behaviour mismatches, not missing features.**
  `compute_subsumption_weight`, `_pareto_front`, the Metropolis rule and
  `skipdet.inference` all exist, are named correctly, and compute something
  else. Searching for absent modules would have found none of them. The
  productive question is "derive what this returns and compare it to its own
  docstring", which is the same lesson `handover_thermo_stochastic_concepts`
  recorded and which has now paid out twice.
* **A defect in a cached value hides behind its cache.** `sub` is cached in
  `_cached_weights`, so it is computed rarely and its distribution never shows
  up in a profile. Cheap-and-wrong is harder to find than expensive-and-wrong.
* **Two implementations of the same predicate in one function body is the
  finding.** `_pareto_front` holding an exact branch and an inexact branch,
  selected by dimensionality, is stronger evidence than any measurement — the
  measurement only quantifies it.
* **An identity being exact says nothing about whether implementing it wins.**
  The CRC linearity result is correct mathematics that loses by 6× to a C
  builtin. That is the third time this thread has produced that shape
  (spacings vs sort; inverse-transform normals vs ziggurat; now CRC). Measure
  against the *implementation* that exists, never against the asymptotics.
