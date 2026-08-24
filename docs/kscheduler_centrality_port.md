# Porting K-Scheduler into `daedalus/fuzzer`

**Paper:** She, Shah, Jana — *Effective Seed Scheduling for Fuzzing with Graph
Centrality Analysis*, IEEE S&P 2022 (arXiv:2203.12064).
**Repo state audited:** `d5058c0` (live source, not docs).

---

## 1. What the paper actually requires

Five pieces, in dependency order:

1. **Inter-procedural CFG**, nodes = basic blocks.
2. **Per-seed visited *node* set** (not edge set).
3. **Horizon nodes** `H = {u unvisited : ∃ parent v ∈ V}`.
4. **Edge horizon graph**: delete visited nodes (preserving connectivity),
   convert to a DAG by removing loops, insert one node per seed with edges to
   the horizon nodes whose visited parent lies on that seed's path.
5. **Out-degree Katz centrality** by power iteration:
   `c(t) = αA·c(t−1) + β`, with `α = 0.5` and `βᵢ = 1 − Rᵢ/T`
   (`Rᵢ` = mutations reaching node *i*'s parents, `T` = total mutations).
   Recompute on new coverage **or** a timer.

Energy integration (their AFL variant) is trivial: seed energy = its centrality.

---

## 2. Audit: what the repo already has

| Paper requirement | Existing asset | Status |
|---|---|---|
| BB-level CFG | `core/cfg.py` — `build_function_cfg`, successors, `callees`, `indirect_call`/`indirect_jump` flags | **Done, better than theirs** |
| Caller→callee merge | `core/distance.py` `_build_call_graph`, `_resolve_callee_name` (PLT-aware), `_addr_to_function` | **Done** |
| CFG scope | `distance.py:430 _build_cfgs` — *target functions only* | **Needs widening** |
| Node identity at runtime | `distance.py:544 pc_distance_table()` recovers the exact PCs the shim observes by scanning REL32 calls to `__sanitizer_cov_trace_pc` | **This is the bridge** |
| Fuzzer→shim keyed table | `adapters/shm.py:970 DistanceTableShm` + `__AFL_DIST_SHM_ID`, open-addressed probe mirrored byte-exactly in C | **Reusable verbatim** |
| Per-seed coverage | `edge_tracker.seed_edges: dict[str, set[int]]`, persisted | Exists, but **wrong granularity** (see §3) |
| Scheduler arm registry | `seed_picker.py:34 _pick_seed_elo` — `available` list + `strategy_map` | **Drop-in slot** |
| Energy assignment | `core/schedules.py SeedScorer` | **Drop-in slot** |
| Linear algebra | `numpy>=2.0` hard dep; **no scipy, no networkx** | Hand-roll Katz (~15 lines) |

Their implementation used wllvm + LLVM `opt` + networkit. You have the
equivalent in pure Python already, plus per-block indirect-call flags that their
version lacks — which matters, because indirect calls are the paper's stated
limitation (§D).

---

## 3. The blocker: node identity

`seed_edges` cannot be used. `__afl_map_edge` (`afl_shim.c:573`) computes

```
edge_id = caller_ctx ^ __afl_prev_loc ^ cur_loc
```

and in trace-pc mode `cur_loc = (pc − base) >> 1` (`afl_shim.c:737`). XOR is not
invertible — you cannot recover *which basic blocks* a seed visited from the
stored edge IDs. K-Scheduler is node-based, so this is load-bearing.

**Option A — new SHM node-visit bitmap (recommended).**
Trace-pc mode already probes a fuzzer-uploaded table keyed on `pc − base`.
Widen the entry from `{u64 key; u32 dist}` to `{u64 key; u32 dist; u32 node_idx}`
(keep `__attribute__((packed))` — the existing comment at `afl_shim.c` about the
12-byte stride is there because padding silently misreads every entry) and set
`bitmap[node_idx >> 3] |= 1 << (node_idx & 7)` in the same probe that already
runs. Read and clear it in `__afl_map_reset` next to the distance tail.

Cost: ~40 lines of C, one `NodeBitmapShm` class, one read per iteration. **Zero
new runtime hashing** — the lookup is already on the hot path. It composes with
distance mode instead of competing with it.

*Trap:* the distance tail has a separate `atexit` writer
(`afl_shim.c:863 __afl_write_distance_tail_exit`) because one-shot subprocess
runs never call `__afl_map_reset`. The node bitmap needs the identical treatment
or you silently lose the last iteration on every non-forkserver run.

**Option B — guard IDs.** `__sanitizer_cov_trace_pc_guard_init`
(`afl_shim.c:671`) assigns sequential IDs from a link-order counter. Recovering
the static mapping means replaying section order, and it breaks on every relink.
Not worth it.

**Option C — offline re-execution** of each seed under a PC-tracing mode. Correct
but O(corpus) re-execs per recompute; destroys the paper's <1% overhead claim.

**Build-scope caveat:** `tools/build_targets.sh:1181-1183` — the trace-pc
distance path instruments only the wrapper `.so`; vendored libraries keep
trace-pc-guard. A first K-Scheduler campaign therefore sees a *partial* CFG.
Rebuilding vendor libs with trace-pc is a prerequisite for a fair evaluation, not
a nice-to-have.

---

## 4. Phased plan

### W1 — whole-program ICFG (`core/icfg.py`)
Lift `_build_cfgs` out of the target-function restriction into
`build_interprocedural_cfg(td: TargetDistance) -> (node_index, src[], dst[])` as
numpy arrays. Inter-procedural edges: for each block with a resolved callee,
add `blk → entry(callee)`. Caller→callee only — the return back-edge is exactly
what the paper's loop-removal step deletes anyway.

`_MAX_CFG_FUNC_SIZE` (256 KiB, `distance.py:58`) already bounds per-function
cost, but whole-program decode is a new regime.

> **Time-box this first.** The pure-Python x86-64 decoder over an entire binary
> is the real cost of this port, not Katz. Measure decode wall-time on your
> largest target before writing W3/W4. If it's minutes, the answer is a
> build-id-keyed on-disk CFG cache, not a faster decoder.

Surface `indirect_call`/`indirect_jump` blocks as candidates for a β penalty —
the paper explicitly leaves this to future work and you already have the flags.

### W2 — node channel (`afl_shim.c` + `adapters/shm.py`)
Per §3 Option A. Ship with a round-trip test: known target, known PC set,
assert the bitmap matches the CFG node indices.

### W3 — edge horizon graph (`core/horizon.py`)
- `V` = union of per-seed node bitmaps; `U` = complement.
- `H`: vectorized as `dst[visited[src] & ~visited[dst]]`.
- Unvisited-only subgraph: mask `~visited[src] & ~visited[dst]` — free.
- Seed→horizon edges: mask per seed — free.
- **Visited-node deletion with connectivity preservation** is the one non-trivial
  part. It only matters for U→V→…→V→U paths. Skipping it is a cheap
  approximation; doing it properly needs a contracted-graph pass. Decide
  explicitly and record the choice.
- **Loop removal → DAG.** Tarjan SCC on the U-subgraph, drop intra-SCC edges.
  Pure Python and numpy-hostile, but the U-subgraph shrinks monotonically over a
  campaign.

### W4 — Katz (`core/schedulers/katz.py`)
No scipy needed. The SpMV is one line:

```python
c = alpha * np.bincount(src, weights=c[dst], minlength=n) + beta
```

Iterate to tolerance or ~30 steps. `α = 0.5` default (their Table XI), tunable.

β from W2's accumulated per-node hit counts (`1 − Rᵢ/T`) — note `Rᵢ` counts
**all** mutations reaching node *i*'s parents, not just corpus-adds, so the
bitmap must be sampled every exec, not only on new coverage.

Convergence needs `α < 1/λ_max`. On a DAG `λ_max = 0`, so any α converges in
≤ depth iterations — assert the DAG property, since W3's loop removal is
precisely what makes W4 safe.

### W5 — wiring
- `seed_picker._pick_katz_seed()`; append `"katz"` to `available` in
  `_pick_seed_elo` when a node channel exists, and to `strategy_map`.
  **This is strictly better than the paper's evaluation setup**: Elo will rate
  it head-to-head against `aflgo`, `weighted`, `pareto` etc. for free, instead
  of you asserting a win from a single run.
- Energy: add `"katz"` to `SeedScorer.SCHEDULES`. Their AFL integration sets
  energy = centrality directly; clamp against `max_mult` or one high-centrality
  seed starves the queue.
- Recompute trigger (their Algorithm 2): on new coverage **or** timer `k`.
  Reuse the existing corpus-add hook.
- Persist the centrality vector in `state.pkl.gz`; cache the ICFG keyed on
  target build-id.

---

## 5. Risks and calibration

1. **Decode cost dominates.** See W1. Everything downstream is cheap.
2. **Context-sensitive coverage.** `__AFL_CTX_SENSITIVE` qualifies *edges* by
   caller context; K-Scheduler's nodes are context-free. Not a correctness
   problem, but `V` will saturate noticeably faster than the edge map, which
   shrinks the horizon earlier than the paper's numbers assume.
3. **The headline gain is small.** vs AFL-based schedulers: 4.21% arithmetic
   mean, **1.91% median**, at 24h over 10 runs (their Table V). That is inside
   the noise band of any single-run comparison. Validate the *math* in
   `tests/support/bandit_env.py`; let the Elo arm earn the *coverage* claim.
4. **The ablations are bigger than the headline.** Loop removal: +21.70%
   (Table XII). Non-uniform β: +24.19% (Table X). Visited-node deletion: +24.13%
   (Table XIII). All three exceed the 4.21% AFL gain — meaning most of the value
   is in the graph transform, not in Katz per se. **None of W3's steps are
   optional polish.** If W3 gets cut down for schedule reasons, the port is not
   worth shipping.
5. **Their own accuracy check is weak.** Kendall tau vs the ideal scheduler is
   0.01–0.09, and *negative* on libjpeg and openthread (Table XIX). Katz is a
   loose approximation that happens to help; treat it as one arm among several
   rather than a replacement for the existing pickers.

---

## 6. Effort estimate

| Phase | Estimate | Unknown |
|---|---|---|
| W1 ICFG | ~1 session | decode wall-time |
| W2 node channel | ~½ session | — |
| W3 horizon graph | ~1 session | connectivity-preserving deletion |
| W4 Katz | ~½ session | — |
| W5 wiring | ~½ session | — |
