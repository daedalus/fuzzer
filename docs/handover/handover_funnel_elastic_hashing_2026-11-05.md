# Funnel / elastic hashing for the SHM edge table: simulated, measured, rejected

**Status (2026-11-05, PRs #293, #294, #295):** no shim change, no layout bump,
resize trigger stays 0.7. Two tools shipped (`tools/sim_funnel_probe.py`,
`tools/edge_hit_spectrum.py`), one bug fixed (`RandPool.randrange_list`), one
false build warning fixed. Rejections are also in `docs/port-backlog.md`
§Rejected.

## Trigger

arXiv 2501.02305 (Farach-Colton, Krapivin, Kuszmaul, "Optimal Bounds for Open
Addressing Without Reordering") → `docs/TODO.md` distance-cap item (c):
"funnel hashing for the edge map's `__AFL_PROBE_MAX` drops". Then: elastic,
a hybrid, a 0.9 resize trigger, Zipf weighting, measured weights on ffmpeg.

## Metric

Cache lines read per lookup (8-byte entries, 64-byte lines, table at byte 32
of the segment: `SHM_METADATA_SIZE`). Hits are the shim's hot path (one
lookup per edge execution); misses are new edges; drops are edges lost when
the 64-slot window is full. Simulation only: no in-run timing, so no
`AB_LEDGER.md` row.

```
linear:  home -> home+1 -> ... (64 slots)            greedy, miss stops at empty
funnel:  A_1 bucket -> ... -> A_alpha -> B -> C      greedy (paper §3)
elastic: A_1 | A_2 | ... halving, phi(i,j) order     non-greedy (paper §2)
hybrid:  elastic arrays of line-aligned 8-entry buckets; an empty slot in a
         probed bucket rules the array out, so misses stop early
```

## Findings

### F1. Uniform hits (load mode), 8192 entries, 5 trials

| load | linear drop | lines/hit lin / fun / ela / hyb | lines/miss lin / hyb |
|---|---|---|---|
| 0.60 | 0 | 1.10 / 2.81 / 3.36 / 1.56 | 1.3 / 5.0 |
| 0.70 | 1.7e-4 | 1.14 / 3.36 / 4.64 / 1.83 | 1.6 / 7.2 |
| 0.77 | 6.7e-4 | 1.20 / 3.84 / 5.15 / 2.06 | 2.1 / 9.0 |
| 0.90 | 9.6e-3 | 1.37 / 5.17 / 7.87 / 2.82 | 4.0 / 16.5 |

Funnel, elastic and hybrid drop nothing to 0.90. Linear's 0.70 rate matches
the earlier ketama handover's simulation (~1 drop per 5,734 inserts).
Funnel at delta = 1/16 is worse (4.06 lines/hit at 0.70). Elastic constant c:
1, 2, 4, 8 give 3.8, 4.4, 5.2, 5.7 lines/hit at 0.77. Elastic misses scan every
probe in use: 30-72 lines.

### F2. Resize trigger 0.7 -> 0.9 (growth mode)

Replays `recommended_map_size` (oracle-tested against `EdgeTracker` at 0.7)
from 8192 entries. Linear, 20 trials, drop events: 7k edges 0.75 -> 2.1, 20k
1.15 -> 3.45, 60k 2.9 -> 5.8. Any drop forces a resize anyway, so memory is
saved only below ~6k edges. Hybrid at 0.9 stays at 8192 for 6-7k edges at
2.0-2.5 lines/hit vs 1.1-1.2. Above ~20k edges the 2x headroom keeps load
under both triggers. The 0.7 threshold only acts on the stall path
(`--resize-map-on-stall`); drops trigger at any load.

### F3. Zipf weighting (hot mode), w_r = r^-s, r = discovery order

| s | linear | hybrid |
|---|---|---|
| 0 | 1.10-1.37 | 1.56-2.82 |
| 1 | 1.02-1.05 | 1.08-1.30 |
| 1.5 | 1.00 | 1.01-1.02 |

Predicted: hybrid viable if s >= 1.5. F4 shows the model's assumption fails.

### F4. Measured spectrum, ffmpeg_read 9.0.2 (noasan), 32k-exec corpus

`tools/edge_hit_spectrum.py`, corpus replayed oldest first, per-edge summed
shim counts. Weights fed to `sim_funnel_probe.py --mode real`, loads 0.6-0.9,
3 trials.

| build | edges | touches | slope | Spearman | hottest 20 at ranks | linear | hybrid |
|---|---|---|---|---|---|---|---|
| ctx off | 16,417 | 2.0417e9 | 1.76 | -0.64 | 174-11,215 | 1.04-1.09 | 1.30-1.45 |
| ctx on | 27,602 | 2.0417e9 | 1.64 | -0.60 | 119-17,811 | 1.03-1.10 | 1.21-1.63 |
| ctx on + ng3 | 37,528 | 2.0432e9 | 1.68 | -0.63 | 5-22,847 | 1.02-1.07 | 1.14-1.60 |

- `fit_zipf`: NOT_POWER_LAW in all three (s_fit 3.57 / 2.94 / 2.85).
- First 1% of discovered edges hold 1% / 6% / 7% of touches; first 25% hold
  ~73%. Hot edges are decoder inner loops reached after a format's seed
  parses, so "hottest first" holds only loosely and the slope overstates it.
- Hybrid and elastic numbers are noisy: a few late hot edges land in a
  different array per trial.
- Same touch total ctx off vs on is the consistency check; ng3 is +0.07%
  (trace-pc instead of trace-pc-guard instrumentation).
- Moot on this target anyway: the auto-sized map is 524,288 entries for both
  builds, so load is 0.03-0.07.

### F5. The table is not cache-line aligned

`SHM_METADATA_SIZE = 32`: slot 0 starts mid-line. Line-aligned buckets
(hybrid) must skip the first 4 slots (`LINE_LEAD`) and a tail. A shim bucket
design would need a layout change (`__AFL_SHM_LAYOUT` bump).

### F6. `_global_edge_hits` is not a traffic count

`EdgeTracker._global_edge_hits` is updated only when an exec finds new
coverage (`fuzz_round._record_edges`), so it cannot give per-edge hit
frequency; `edge_hit_spectrum.py` replays the corpus to get it.

### F7. Bugs fixed

- `RandPool.randrange_list(1 << 32, k)` raised OverflowError: the guard
  `n > 1 << 32` let `n == 2**32` reach `uint32 % n`. Full-span draws now
  return pool words. `tests/test_regression_rand_pool_full_u32_list.py`.
- `warn_if_frame_pointer_omitted` warned on every FFmpeg build: it read only
  `config.mak` and missed make's `ECFLAGS=-fno-omit-frame-pointer`, which
  wins. ffmpeg_read 9.0.2 has rbp frames in 10,775/10,779 functions.
  `tests/test_ctx_collapse_selfcheck.py`.

### F8. Open, documented in `docs/TODO.md`

- `patches/ffmpeg-vpk-divide-by-zero.patch` is already upstream in 9.0.2
  (`vpk.c` guard, `demux.c:2601` ch_layout copy), yet vendoring prints "may
  not apply cleanly": the already-applied test greps a forward `--check` for
  "Check patch", which never matches; a reverse check also fails on drifted
  `demux.c` context.

## Environment (authoring container)

- Fresh clone: no git hooks installed; `pre-commit run --files` by hand. A
  failed manual run is a stop, not advice (one commit went out blocked and
  was fixed with an `impactguard.toml` suppression).
- `impactguard==0.1.21` + `tree-sitter` installed via pip (as in
  `handover_shm_displacement_2026-11-05.md`).
- `libclang-rt-18-dev` was missing: `ffmpeg_read_<ver>_ng3` (executable)
  failed to link `libclang_rt.ubsan_standalone`; the `.so` built.
- `ruff format src/ tests/` touches unrelated files (pre-existing); revert them.
- Builds: `tools/vendor_ffmpeg.sh --top=1`, `FFMPEG_OPTS=noasan
  tools/build_targets.sh --fast`, `FFMPEG_OPTS=ng3 tools/build_targets.sh
  --ngram`. Corpus: `~/fuzzing/corpus_ffmpeg_spectrum` (genseed all, 32k
  execs, ctx off).

## Reproduce

```
python tools/sim_funnel_probe.py --trials 5                       # F1
python tools/sim_funnel_probe.py --mode growth --thresholds 0.7,0.9  # F2
python tools/sim_funnel_probe.py --mode hot --zipf 0,0.5,1,1.5,2     # F3
python tools/edge_hit_spectrum.py ~/fuzzing/builds/ffmpeg_read_9.0.2_noasan \
    -d ~/fuzzing/corpus_ffmpeg_spectrum -o spectrum.pkl [--ctx on]   # F4
python tools/sim_funnel_probe.py --mode real --spectrum spectrum.pkl \
    --loads 0.6,0.7,0.77,0.85,0.9 --trials 3
```

## What would reopen it

A target whose map cannot grow (capped by `AFL_MAP_SIZE_MAX`) and keeps
dropping: hybrid trades ~0.3 lines/hit for zero drops to load 0.9. Try
lowering `__AFL_CTX_BITS` or raising the cap first.
