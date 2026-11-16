# HLL edge / path sketch — 2026-11-16

Question: can a Flajolet–Martin / HyperLogLog sketch over edge ids, or over the
shim's per-exec `path_hash`, give a signal the fuzzer lacks? Not built.

## 1. Sketch accuracy (fuzzgoat fires vs exact SHM table)

fuzzgoat clang ASAN, `-D__AFL_TRACE_FIRES=1`, 1500 execs (13 seeds + mutants),
median 30 distinct edges/exec. Oracle: unique fires == SHM table on 1500/1500.

| Estimator | median est/true | RSE |
|---|---|---|
| `popcount(OR edge_id)` | stuck at 15–19 bits | — |
| FM 1 word, `fmix32` | 0.86–1.48 | 1.1–1.5 |
| PCSA m=64 | 3.17 | 6.9 (small-range bias) |
| HLL m=16 | 0.96 | 0.17–0.24 |
| HLL m=64 | 1.04 | 0.10 |

Uniform-id A/A matched theory (HLL64 0.11–0.14 vs 0.13; PCSA64 0.09–0.10 vs 0.098).
Raw `ctz(edge_id)` without a mixer: estimate 0.00x on 256-aligned ids — guard ids
must be hashed. Cost, C loop: OR 0.15, FM 0.25, HLL64 1.9 ns/fire (not in-run).

Edges: no gain — the table already holds the exact set (`edge_count`, `dropped_edges`).

## 2. Does distinct-path growth predict new edges?

Plan frozen before runs: fuzzgoat_read_noasan.so, `-D dictionaries/json.dict`,
seeds 1–20 × 30k execs; W = 1000 execs; on edge-plateau windows, AUC of new
paths in (t−W, t] → new edges by t+2W; placebo = another seed's paths;
VOID if < 20 positive plateau windows.

- Plateau windows: 5 (all positive). **VOID** — edges never stalled for 1000 execs.
- Exploratory (post hoc), all windows, per-tick Spearman across seeds vs 2-window edge gain:
  paths +0.045 [−0.03, +0.13]; edges-now +0.17 [+0.10, +0.25]; paths given edges −0.006;
  placebo +0.11 [+0.04, +0.18] — control fails, tick bootstrap is anticonservative.
- Distinct paths ≈ 3.7x edges at 30k (range 2.0–5.0).

Verdict: no evidence path count adds information over edge gain. Not worth wiring.

## Side finding (unverified)

Fresh process per exec, context on: per-exec edge counts equal ctx-off, but the
1500-exec union was 16292 ids vs 315 — the ctx term changes across processes
(likely ASLR on return addresses). Forkserver shares one image; not checked.
