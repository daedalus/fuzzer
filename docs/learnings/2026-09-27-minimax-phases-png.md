# Minimax Phases 3-5 on png — null (wall-order leans negative)

2026-09-27. `tools/lib/bench_paired.py run --set container_signal --targets
png_read_noasan.so`, seeds 0-19, 10k execs, 1 rep, each arm paired against
its `ARM_BASELINES` entry. 3 parallel shards on 4 cores (not
`--lock-single-thread`); eps is contended and only roughly comparable.

| arm (vs baseline) | W | L | McNemar | Wilcoxon | med Δ edges |
|---|---|---|---|---|---|
| `elo-op-minimax` vs `elo` (P4) | 10 | 10 | 1.0 | 0.68 | 0.0 |
| `wall-order` vs `baseline` (P3) | 6 | 14 | 0.115 | 0.17 | -4.5 |
| `minimax-select` vs `minimize-2k` (P5) | 10 | 10 | 1.0 | 0.98 | -0.5 |

| arm | mean edges | sd | median | eps | corpus |
|---|---|---|---|---|---|
| baseline | 121.2 | 25.3 | 128.5 | 43.8 | 58.7 |
| wall-order | 116.2 | 28.3 | 127.5 | 42.6 | 55.4 |
| elo | 122.5 | 32.0 | 132.5 | 37.4 | 62.8 |
| elo-op-minimax | 127.2 | 16.4 | 129.5 | 36.6 | 63.0 |
| minimize-2k | 123.1 | 22.6 | 130.5 | 37.7 | 40.7 |
| minimax-select | 130.5 | 56.3 | 126.5 | 45.8 | 39.1 |

Notes:
- minimax-select seed 17 scored 348 edges; a rerun of that cell gave 126
  (minimize-2k rerun: 139). Not reproducible: one lucky draw. Mean without
  it is 119.1; the paired tests are rank-based and unaffected.
- minimax-select did not inflate the corpus (39.1 vs 40.7 kept seeds).
- wall-order: cmplog is live on png (733 pairs), so `condstmt_solve` is in
  the pool, but per-operator pick counts are not logged; how often the
  wall order actually decided a pick is unmeasured.
- op-minimax: no eps regression visible (36.6 vs 37.4, within contention).
- Phase 2 (risk matrix) needs a heterogeneous target set; only png builds
  here (jpeg/ffmpeg libs absent), so it is unmeasured.

Verdict: no evidence any of the three helps on png. wall-order's 6W/14L is
the only lean, and it is negative. All three are on under `--hail-mary`.
