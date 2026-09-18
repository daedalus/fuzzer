# Handover: Cloudflare "saving 100TB of RAM with math" — applicability to this repo

Source: https://blog.cloudflare.com/saving-100-tb-of-ram-with-math/ (PBR /
pingora-ketama memory reduction: struct-packing fix + closed-form
coefficient-of-variation math to cut hash-ring size 90%). Status: **analysis
only, no production code changed**. Two threads investigated; findings and
evidence below.

## 1. `struct __afl_entry` — the Point-struct lesson already applied, both ways

`adapters/afl_shim.c`/`adapters/shm.py`:

```c
struct __afl_entry { uint32_t edge_id; uint32_t count; };  // 8 bytes
```

`count` is bit-packed on purpose: top 8 bits are a generation tag, bottom 24
are a saturating hit count (`(gen << 24) | 1`, capped at `0x00FFFFFF`). This
is safe packing in the blog's sense — gen and count share one writer and one
lifecycle, so there is no read-modify-write hazard. Contrast with the
metadata header word at offset 4, which `shm.py`'s own comment says used to
pack ctx-width + 16-bit drop count + generation into one `uint32` and caused
two shipped bugs (plus a third latent one) from exactly the RMW hazard the
blog's `#[repr(packed)]` caution is about. This codebase independently drew
the same line the blog draws: pack fields with one writer, never fields
with two.

Separately: the earlier PC-field-in-`__afl_entry` analysis (see
`handover_edge_pc_field_2026-09-13.md`) already ran the blog's exact
cost-benefit arithmetic in the opposite direction — doubling the entry from
8→16 bytes was rejected specifically because of SHM-size doubling, doubled
`memset` cost, and cache-line packing dropping from 8 entries/line to 4.
Same struct-alignment reasoning as Zaidoon's `Point` fix, no code change
needed here since the earlier analysis already reached the right call.

**No parallel action item.** Nothing to port; this section documents
convergent design, not a gap.

## 2. `__afl_map_size` sizing — closed form vs. simulated reality

`core/edge_tracker.py:2446` (`recommended_map_size`) already exists and
resizes reactively: trigger on `dropped_edges > 0` or load factor ≥ 0.7,
target 2x headroom (load factor < 0.5) after resize, doubling until reached.

Built the blog's comparison for this table: simulated the shim's actual
mechanism (n edges hashed uniformly into m=8192 slots, bounded linear probe,
window=64, drop if window exhausted) rather than trusting a formula on
paper — this repo's own house rule (validate against real execution before
trusting a closed form; see `fuzzer.md`'s sqlite/ffmpeg validation turns and
the Shapley/KL-UCB paper-fidelity checks).

Result: the naive independent-slot model (`P(drop) ≈ ρ^w`, ρ = load factor)
badly underpredicts real drops at high load — at ρ=0.9 it predicts ~9 drops
per 8192-slot table, simulation shows ~69. Cause: linear probing suffers
primary clustering (occupied runs grow and swallow their own neighbors),
which an independent-slot model can't see. This is the same *shape* of
surprise as the blog's own "the math only holds on a continuous ring; real
32-bit hashes collide faster than the naive theory predicts" — reality
diverges from the closed form in the same direction (worse than predicted),
for a structurally different reason (clustering vs. birthday paradox).

```
n=5734  rho=0.700  observed_drops≈1     naive_pred≈0.0
n=6554  rho=0.800  observed_drops≈8     naive_pred≈0.004
n=7000  rho=0.854  observed_drops≈26    naive_pred≈0.30
n=7373  rho=0.900  observed_drops≈69    naive_pred≈8.7
```

Drop rate is flat and near-zero through ρ≈0.7–0.75, then knees sharply
upward through 0.85–0.9. **Conclusion: the existing 0.7 trigger sits right
at the start of the cliff, not dangerously late, and the 0.5 post-resize
target is comfortably clear of it.** Unlike ketama's 90%-of-hashes-bought-
nothing finding, there is no comparable free win sitting in this heuristic
— it is already well-calibrated. No implementation followed from this
thread; documenting the negative result so it isn't re-derived.

Simulation script used: `sim_probe.py` (not committed — throwaway,
reproducible in a few lines: place n random slots into an m-array with an
`w`-wide bounded linear probe, count placements that exhaust the window).

## 3. Ketama-weighted virtual nodes vs. Multifit — open, not implemented

`core/parallel_fractal_partition.py`'s Voronoi-root-cell assignment is
single-point consistent hashing (one hash point per worker, no virtual
nodes) — the same "server A gets a disproportionate range" problem the
blog opens with, for workers instead of backends.

`core/parallel_cost_partition.py` is the fix for that imbalance via
Multifit instead of more hash points, and its own docstring names the
price: reassignment churn on any cost drift, dampened only by
`maybe_repartition`'s hysteresis threshold — i.e. the same
stability-vs-balance tradeoff ketama's weighted hash counts
(`H₁ = w × H₂`) exist to avoid without a global repack.

**Candidate not yet built:** give each worker `k` hash points weighted by
estimated cost (disk-space analogue = `cost_ledger` per-seed cost, or the
existing input-size proxy at campaign start) instead of one point via
Voronoi root cell. A seed's cost drifting would only move the boundary of
that worker's own cells, not reassign unrelated seeds on other cells — no
global repack, so `maybe_repartition`'s hysteresis becomes unnecessary
rather than merely damped.

**What's unverified before building this:** balance quality under
k-point weighted hashing is bounded by the post's CV(k) curve — it
approaches but does not reach Multifit's near-optimal makespan, only for
large enough k. Whether "close enough" holds here depends on the actual
variance in per-seed cost that `cost_ledger` has measured, which was not
pulled in this turn. Next step, if picked up: measure that variance, pick
a k from the CV(k) curve for the target error margin, and only then decide
whether replacing `parallel_cost_partition.py`'s worker assignment is worth
doing.

## Status summary

| Thread | Outcome |
|---|---|
| 1. Struct packing | No gap found — already applied both directions |
| 2. Map-size closed form | Simulated and confirmed existing 0.7/0.5 heuristic is well-calibrated; no change |
| 3. Ketama-weighted partitioning | Open — candidate described, needs `cost_ledger` variance pulled before implementing |
