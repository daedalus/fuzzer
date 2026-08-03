# mi-tracker-oom: per-cell caps don't bound memory — cap the product

**Date:** 2026-08-03
**Context:** fuzzer-new repo; `--mi-guided` / `--elo all` on a real PNG corpus OOM-killed the fuzzer.

## Problem
`--elo all` on a real corpus: peak RSS 7.5GB (OOM kill at ~1100 execs), EPS collapsing 1500->20, and a 796MB `mi.json` on disk that made every subsequent startup worse. Diagnosed first as a suspected regression from the array.array perf work — ruled out by reproducing identically on the pre-change baseline.

## Rejected
- **Blame the recent perf commits** — looked plausible (user saw the regression right after array.array changes landed); dropped by running the exact failing command on the pre-change baseline: identical 7.5GB/11-eps behavior, so the array work was exonerated and the real cause (MI tracker, enabled by `--elo all`) was found.
- **Evict the least-observed position when over budget** — looked plausible (standard bounded-cache idea; evict what's least used); dropped because it THRASHED catastrophically: a freshly-evicted position is immediately re-observed, resets to 1 observation, becomes the next eviction victim, and the joint cycled down to empty. Eviction only works for data that stops being touched; this data is re-touched every iteration.
- **Gate the load but leave live growth** — insufficient alone; the tracker rebuilds the multi-GB structure within a single long run regardless of what was loaded.

## Approach
Three small changes:
1. **Hard cap the total joint cells** (`MAX_JOINT_CELLS` = 2M): `record()` rejects NEW `(position, byte_val, edge)` cells once the budget is exhausted — existing cells keep incrementing. No eviction during recording; the joint freezes at the budget. Memory is bounded by construction, not by a reaction.
2. **Cap tracked positions** (`MI_MAX_POSITIONS` = 4096) independent of the fuzzer's `max_len`, which auto-grows to 65536 — the worst-case product is positions x 256 byte values x 64 edges, so capping the positions factor caps the whole thing.
3. **Resume-gate the `mi.json` load** (like `state.json`), breaking the ratchet where each run saved a huge file and the next run `json.loads`-ed it into a multi-GB object tree at startup. `load()` also recounts cells and trims over-budget state.

## Key insight
A per-cell cap (64 edges per position/byte cell) bounds the *height* of the structure, not the *volume*: positions x byte-values x edges is a product, and three of the four factors were effectively unbounded. The correct invariant is a cap on the product (total cells). And for structures that are written to on every hot iteration, "stop accepting new data at a budget" beats "evict and relearn" — eviction churns on data that keeps coming back.

## Verification
- The exact failing command (`png_read_dist.so`, real corpus incl. the 796MB mi.json, `--map-size 16384 --elo all`): peak RSS **7492MB -> flat 229MB** (no growth over 43k execs; previously OOM at ~1.2k), EPS **11 -> 180+ and rising**, full end-of-run report produced.
- Baseline attribution: same command on the pre-change commit reproduced the blowup, confirming the fix targets the real cause.
- Regression tests pin the invariants: position cap, cell-budget saturation without thrash (joint survives 4000 records at budget), load recount + trim, `max_len` cap at construction, resume-gated load. Full suite 3056 passed.

## Generalizes to
When bounding a nested/accumulating structure, cap the product of dimensions, not one dimension ("per-cell" caps are not memory bounds). For hot-path accumulators, prefer rejecting new entries at the budget over eviction (eviction thrashes on re-observed data; rejection is stable and deterministic). Unconditional state loads are a ratchet: every run makes the next one slower/bigger — gate persisted state on the same flag that gates the rest of the state, and trim oversized state on load.
