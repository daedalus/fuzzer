# `--alloc-sizes` A/B on FFmpeg 9.0.2 ASAN

## Pre-registration (written before any FFmpeg cell ran)

- Arm `alloc-sizes` (`--alloc-sizes`) vs `baseline`; A/A `alloc-sizes-ctl` (baseline flags) vs `baseline`.
- Set `ffmpeg_asan`: `ffmpeg_read_9.0.2_asan.so`, `-D dictionaries/wav.dict`, empty corpus, in-process
  (libasan preloaded by `run_cell`'s wrapper). Only the guarded target thread's allocations mark edges.
- 20 seeds (`eval_set.SEEDS`) × 10k execs, 1 rep; three arms as concurrent processes on 4 cores, as #105.
- Primary metric `real_edges`: cumulative edge ids without bit 31, read from the cell's saved state.
  Raw `edges` is not comparable: the arm's own size-class edges count there.
- Test: McNemar exact on per-seed W/L, α = 0.05. WIN/LOSS only if the A/A does not reject.
- Secondary (reported, not decisive): `corpus` (alloc-only admissions bloat it, cf. #23), `crashes`, `unique`, `eps`.

## Result (ledger #123)

| arm | W/L | McNemar | Wilcoxon | med Δ real edges | arm median |
|---|---|---|---|---|---|
| alloc-sizes | 9/11 | 0.824 | 0.35 | −863.5 | 20,668 |
| alloc-sizes-ctl (A/A) | 11/9 | 0.824 | 0.73 | +67 | 21,501 |
| baseline | | | | | 21,188 |

NULL. The A/A does not reject, so the oracle holds; the arm does not either.

- The channel is live in-process: 22–29 size-class edges per cell.
- No corpus bloat: median 1,590 vs 1,582 (contrast COMPCOV, #23: +70%).
- 0 crashes in every cell of every arm. The bugs the signal targets
  (allocation-size-too-big, OOM) were not reached at 10k execs, so this run
  measures the coverage side effect only.
- `eps` reads 1000.1 in every cell (virtual clock): throughput is not readable
  from this run. User CPU median 268 s vs 259 s (+3.4%), McNemar 0.50: n.s.

Stays opt-in. A crash-oriented design (`tte_allocation-size-too-big` on a
target with a reachable large-allocation bug) is what could show a benefit.
