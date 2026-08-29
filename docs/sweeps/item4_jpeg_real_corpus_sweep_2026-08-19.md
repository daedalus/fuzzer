# Item 4 real-corpus sensitivity sweep, round 11 — `jpeg_read`

Follow-on to `docs/sweeps/item4_png_real_corpus_sweep_2026-08-19.md` (round
10, `png_read`), which identified `jpeg_read` as the best remaining candidate
for exercising the false-negative dead-region path: JPEG has no whole-file
checksum, and `APPn`/`COM` segment payloads beyond what libjpeg parses for
metadata should be genuinely unchecked by any coverage-relevant path.

## Prerequisite: SHM initialization fix

Before collecting data, this run hit the same "0 edges after exec" symptom the
round-10 PNG investigation flagged as a loading-order bug. Root cause:
`ctypes.CDLL()` loads the `.so` and runs its `__afl_auto_init` constructor
before `__AFL_SHM_ID` is set in `os.environ`, so the shim either skips
`shmat()` entirely or attaches to a stale segment. The production
`InProcessRunner` avoids this by setting `__AFL_SHM_ID` / `AFL_MAP_SIZE`
*before* `ctypes.CDLL()` (`adapters/inprocess.py:231-243`); the standalone
sweep collector now follows the same order.

Verified after rebuild: a 4,622-byte seed consistently yields 76 reachable
edges across repeated runs against `targets/jpeg_read.so`.

## Method

- Target: `targets/jpeg_read.so` (`fuzz_jpeg`), rebuilt after the SHM
  initialization fix above. Built with system `libjpeg-turbo` + clang
  `-fsanitize-coverage=trace-pc-guard` via `tools/build_targets.sh`.
- Corpus: 30 real JPEGs generated with Pillow (`tools/corpus_jpeg.py`) —
  varied sizes (522–66,059 bytes), color types, and injected `APPn`/`COM`
  segments with large arbitrary payloads specifically chosen to push seeds
  past `profile_buffer`'s 4,096-byte region window.
- Bounded standalone collector (`tools/sweep_jpeg_liveness.py`) runs seeds
  through `fuzz_jpeg` in-process, applies cheap single-byte flips at random
  offsets, and logs the `(region_idx, diff_bits)` TSV using the same env-gated
  pattern as the round-9/10 sweep instrumentation (zero-cost when
  `FUZZER_LIVENESS_LOG` is unset). This bypasses the fuzzer's
  mutation/selection machinery to target large seeds and specific byte ranges
  directly.
- Raw output: 172 samples across 5,800 mutation attempts (3.0% yield),
  format `region_idx<TAB>comma-separated set-bit positions`. **The TSV was
  never committed** — only the round-9 zlib file
  (`item4_zlib_real_corpus_samples.tsv`) is in the tree. The numbers below are
  therefore not reproducible from this repo; the instrumentation that produced
  them was reverted too. Re-collect before relying on them.
- Replayed per-region, in order, through a fresh `LiveBitMaskEstimator(
  n_bits=65536, switch_after=N)` for `N ∈ {50, 100, 200, 400, 800}`.

## Result

| switch_after | samples to first convergence | final mask popcount | ever converged-dead |
|---|---|---|---|
| 50  | 172  | 78 | no |
| 100 | —  | 78 | no |
| 200 (current default) | —  | 78 | no |
| 400 | —  | 78 | no |
| 800 | —  | 78 | no |

All 172 usable samples fell in **region 0** (bytes 0–4,095). The estimator
reached 78 live bits and achieved 77 consecutive no-growth samples at the end
of the run — just short of the `switch_after=100` threshold, but enough for
`switch_after=50`. At every threshold tested, the final mask is identical
(78 bits), and the estimator never produced a converged-dead verdict.

## What this does and doesn't validate

**Validates:** `switch_after ∈ {50..800}` remains threshold-stable on real
JPEG data — the final live mask is the same at every setting, matching the
round-9/10 finding on `zlib_read` and `png_read`. The SHM initialization
ordering required for in-process coverage collection is now documented and
verified.

**Doesn't validate:** the false-negative dead-region case. Two factors limited
this run:

1. **Single-region data.** Despite generating seeds up to 66 KB with injected
   `APPn`/`COM` payloads, every usable sample attributed to region 0. The
   likely cause is that mutations outside region 0 either produced empty diffs
   (the mutated bytes are structurally irrelevant to libjpeg's control flow)
   or were skipped by the region-attribution gate when the offset fell outside
   any profiled region. Without samples from regions 1+, we cannot exercise
   the per-region estimator behavior the sweep is designed to test.

2. **Low diff yield.** Only 3.0% of mutation attempts produced nonzero
   coverage diffs. Most single-byte flips in `APPn`/`COM` payloads are
   absorbed by libjpeg's error recovery or length-field parsing and do not
   change the instrumented control-flow path. This is consistent with the
   hypothesis that those bytes are *semantically* dead — but it also means
   we collected too few observations to reach convergence at higher
   `switch_after` values.

## Recommendation

The current `_LIVENESS_SWITCH_AFTER=200` default shows no threshold sensitivity
on this target for the data we collected, matching the round-9/10 result. Keep
`_LIVENESS_DEAD_WEIGHT=0.1` unchanged.

To actually exercise the false-negative path on `jpeg_read`, a follow-up run
should either:
- target seeds where mutations in `APPn` trailer payloads are guaranteed to
  fall inside profiled regions (e.g., force mutations into the tail of large
  seeds rather than uniformly random offsets), or
- switch to a richer target with more diverse per-region coverage signal
  before investing more sweep effort here.
