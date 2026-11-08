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

## Result

Pending.
