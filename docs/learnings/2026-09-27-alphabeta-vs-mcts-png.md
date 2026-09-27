# alphabeta vs mcts seed arm on png (E3) — null

2026-09-27. `tools/lib/bench_paired.py run --arms elo-mcts,elo-alphabeta
--set container_signal --targets png_read_noasan.so`, seeds 0-19, 10k execs.
Both arms: `--elo --mc-bandit --lineage` + `--mcts` / `--alphabeta`.

| arm | cells | mean edges | sd | median | eps |
|---|---|---|---|---|---|
| elo-mcts | 20 (9 replicated) | 131.0 | 20.8 | 136.5 | 40.4 |
| elo-alphabeta | 20 | 130.6 | 27.0 | 134.0 | 48.0 |

Paired (alphabeta vs mcts): 12W/8L/0T, McNemar p=0.50, Wilcoxon p=0.67,
median Δ +8 edges.

Noise: per-cell range (max - min) over the replicated mcts cells, 9 cells:
0, 1, 3, 6, 10, 16, 16, 33, 39 (median 10). Seed 0 has 3 runs: an aborted
first launch left an extra rep-0 row. The median Δ (+8) is below the
within-cell spread.

Caveats:
- Not `--lock-single-thread`: 3 shards in parallel on 4 cores, two container
  restarts (harness resumed). Edges are at a fixed exec budget, so contention
  moves eps and wall time, not the budget; eps is not comparable.
- Reps mixed: mcts has 2+ runs on 9 cells (cell median used), alphabeta 1.
- ~100 paired cells are needed to resolve a ~10pt win-rate difference.

Verdict: no evidence alphabeta beats mcts on png. ffmpeg_read still owed.
