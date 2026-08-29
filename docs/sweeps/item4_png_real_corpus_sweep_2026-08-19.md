# Item 4 real-corpus sensitivity sweep, round 2 (round 10) — `png_read`

Follow-on to `docs/sweeps/item4_real_corpus_sweep_2026-08-19.md` (round 9,
`zlib_read`), which closed with two open scope limits and a recommendation
to re-run against `png_read` next. This run addresses both, plus an
unrelated bug it surfaced along the way.

## Prerequisite: region-attribution bug, found and already fixed upstream

Before any of the results below are meaningful, this run first hit the same
"only region 0 ever appears" symptom the round-9 doc flagged as a scope
limit — except here it persisted even with `--no-deterministic` and even
against seeds with 2–5 profiled regions, which ruled out corpus size as the
explanation. Root cause: `CrashMITracker.weighted_position` and
`MutualInformationTracker.weighted_position` returned `0` (not `None`) when
they had no data yet, and `select_position()` treated that `0` as a valid
candidate alongside the real MI/TE/sensitivity/region candidates — so for a
large fraction of mutation rounds, `f._last_mutation_offset` got pinned to
`0`, and every `record_coverage_diff` call attributed the sample to region
`0` regardless of where the mutation actually landed.

This was independently diagnosed here, then found already fixed upstream in
commit `66bf760` (pulled mid-session): both `weighted_position` methods now
return `None` on empty state, and `select_position` already skips `None`
candidates and falls back to random selection. Re-running the same 60K-exec
`png_read` campaign after the fix confirms it empirically: `offset == 0`
frequency dropped from **97.4%** (pre-fix, 27,518 samples, all region 0) to
**5.7%** (post-fix, 16,063 samples spread across regions 0/1/2) — consistent
with the commit message's own synthetic-diagnostic number (~97% → ~7%).
This sweep's data is the first *real-campaign* confirmation of that fix,
not just the synthetic one.

## Method

- Target: `targets/png_read.so` (`fuzz_png`), built with system `libpng`
  1.6.43 + `zlib` via clang `-fsanitize-coverage=trace-pc-guard` (127
  bitmap slots on the wrapper file per `tools/build_targets.sh`'s own
  measurement — not vendored/recompiled libpng internals).
- Corpus: 76 real PNG files generated with Pillow — varied color types (1/
  L/P/RGB/RGBA/LA/I;16), bit depths, sizes 1×1 to 400×300, Adam7
  interlacing, and ancillary chunks (`tEXt`/`iTXt`/`tRNS`) — including five
  files with this repo's own `.py` source embedded as large (~20 KB)
  uncompressed `tEXt` chunk payloads, specifically to get seeds past
  `profile_buffer`'s 4096-byte window and to have a plausible "coverage-
  irrelevant bytes" region to test the dead-region path against (this was
  the round-9 doc's second open scope limit).
- ~140K total execs across two bounded, resumed campaign rounds
  (`fuzzer-tool fuzz ... --inprocess --inprocess-func fuzz_png -m 65536
  --resume`), saturating at 75/75 reachable edges.
- Same env-gated `record_coverage_diff` instrumentation as round 9 (not
  present in the tree — reverted after collecting samples), extended to
  also log `offset` and region-bounds count for debugging the bug above.
- Raw output: 16,063 samples (post-fix only — the pre-fix 27,518-sample run
  was discarded, it only demonstrates the bug, not the estimator), same
  two-column format as the round-9 file. **The TSV was never committed** —
  only the round-9 zlib file (`item4_zlib_real_corpus_samples.tsv`) is in the
  tree. The numbers below are therefore not reproducible from this repo; the
  instrumentation that produced them was reverted too. Re-collect before
  relying on them.
- Replayed per-region, in order, through a fresh `LiveBitMaskEstimator(
  n_bits=65536, switch_after=N)` for `N ∈ {50, 100, 200, 400, 800}`.

## Result

| region | samples | % nonzero | longest zero-diff run |
|---|---|---|---|
| 0 (header + early IDAT, ≤4096B) | 15,903 | 100.0% | 1 |
| 1 (bytes 4096–8191, incl. large `tEXt` payload) | 121 | 100.0% | 0 |
| 2 (bytes 8192–12287, incl. large `tEXt` payload) | 39 | 100.0% | 0 |

Every region, at every `switch_after` threshold tested, converged to a
stable live mask (74/69/57 bits respectively) and **never once produced a
converged-dead verdict** — including regions 1 and 2, which sit entirely
inside the deliberately-planted large uncompressed `tEXt` comment payloads
and should, semantically, be nearly free of effect on decoded-pixel
control flow.

## Why: this is a structural finding, not another corpus gap

Unlike round 9's zlib run — which found no dead region because
incompressible payloads genuinely have none — this run found no dead
region for a specific, checkable reason: **every PNG chunk is guarded by a
CRC-32 trailer**, including `tEXt`. Mutating any byte inside a chunk's data
(pixel-relevant or not) changes that chunk's CRC, which flips the
CRC-check-pass/fail edge in `png_crc_finish`/`png_crc_error` on every single
mutation. From the fuzzer's edge-coverage point of view this makes the byte
"live" — it does affect *an* edge — even though it has zero effect on the
decoded image once the CRC passes. `LiveBitMaskEstimator` is answering the
question it was built to answer ("does this byte ever move any edge")
correctly; it's just that CRC-guarded formats structurally don't have
coverage-dead bytes by that definition, only *semantically*-dead ones,
which is a different and harder property this estimator was never meant to
capture.

This means the round-9 recommendation to "use `png_read` for its real
reserved/padding fields" was too optimistic: PNG's reserved fields (IHDR's
compression/filter/interlace method bytes, unrecognized ancillary chunks)
are exactly the same size as everything else that's CRC-covered, so they
don't give the estimator a case to fail on either.

## What this does and doesn't validate

**Validates:** the offset-attribution fix (`66bf760`) works on a real
multi-region campaign, not just the synthetic diagnostic — this was worth
confirming independently rather than trusting the commit message alone.
Also reconfirms round 9's finding on a second, structurally different
target: `switch_after ∈ {50..800}` remains threshold-stable (same final
mask at every setting, for all three regions), so there's still no sign of
threshold sensitivity in the shipped default anywhere it's been tested.

**Still doesn't validate:** the false-negative case (a genuinely
coverage-dead region wrongly declared dead too early). Two structurally
different real targets (zlib's incompressible payloads, PNG's CRC-guarded
chunks) have now both failed to produce one, for two different reasons.
The common thread: any format where every byte participates in either the
payload's own semantics or an integrity check leaves nothing for the
estimator to get wrong in this direction.

## Recommendation for the next attempt

Pick a target with bytes that are truly unchecked by anything — no
checksum, no length-derived read boundary that consumes them. Candidates
already in `targets/`:

- `jpeg_read.c` — JPEG has no whole-file checksum. `APPn` segments (e.g. a
  synthetic EXIF/thumbnail-style application segment) are skipped by
  length field alone; bytes inside one that libjpeg doesn't parse for
  metadata should be genuinely coverage-dead as long as mutating them
  doesn't also touch the segment's own length header. This looks like the
  best remaining candidate and is the suggested next step.
- `unrar_read.c` / `lz4_read.c` — worth a quick structural check for
  unchecked trailer/reserved bytes before investing in a corpus for them,
  since both are more likely to be CRC- or length-guarded throughout like
  zlib/PNG.

Recommend keeping `_LIVENESS_DEAD_WEIGHT=0.1` (conservative down-weight,
not exclusion) until one of these produces an actual converged-dead case
to inspect — three real campaigns now, zero false positives *or* the
true-positive case that would let us check for false negatives.
