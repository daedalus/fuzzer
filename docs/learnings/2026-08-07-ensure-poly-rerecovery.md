# Checksum-learner re-recovery: per-iteration eps collapse

**Date:** 2026-08-07
**Context:** fuzzer-new `src/fuzzer_tool/core/checksum_learner.py`, reported "eps dropped to 4" on a `png_read.so` concolic/cmplog run

## Problem
A heavy fuzz command (persistent `.so`, cmplog, concolic z3, path-negation) collapsed to ~4 executions/sec, but the raw target probe reported 40–55K eps. The SMT/concolic flags were suspected — but the collapse reproduced with cmplog alone.

## Rejected
- **"The concolic z3 solve is the cause"** — looked plausible because that specific command was slow; dropped because profiling showed the SMT/concolic block was only ~10 s of a 103 s profile, and the eps collapse already happened on runs *without* `--enable-smt-z3`/`--mod-solving concolic`/`--path-negation`. Concolic was a real secondary cost, but not the primary.
- **"The SMT flags just cost inherent solver time"** — the real cost was not the solver; it was a per-iteration recovery that re-ran even when it could never succeed.

## Approach
`ChecksumLearner.ensure_poly()` cached `_poly` only on verified success. An unverifiable recovery (PNG CRCs use `init/final_xor=0xFFFFFFFF`; the model verifies with `init=0/final_xor=0`, so the recovered poly never reproduces two observed checksums) leaves `_poly=None`. `ensure_poly()` runs once per fuzz iteration via the `crc_learn` availability gate (`REGISTRY.available → build_ops → mutate`), so every iteration re-ran the full GCD-of-syndromes (reflected + non-reflected) + Berlekamp–Massey + `_verify` over all accumulated pairs (~56 s of a 103 s profile). Fix: `_maybe_recover()` guards recovery on `len(_pairs) != _pairs_attempted_at` — only re-run when new evidence arrives; `_recover()` records the pair count it ran at.

## Key insight
A "cache the result on success" memoization silently turns a failed, *necessarily-failing* computation into a per-call doom loop when the call site is hot. The bug wasn't that recovery was slow — it was that recovery *never caches*, so its (already high) cost was paid every iteration, and the cost grew as the pair set grew, so eps decayed through the session.

## Verification
cProfile before: `_recover` ran 165× across 215 `fuzz_one` calls (~56 s). After: `_maybe_recover` ran 386× but `_recover` only 31× (~92% of calls skipped). avg eps on the exact reported command went 13.5 → 25.8. Full suite 3610 passed. Regression test `test_regression_ensure_poly_no_rerecovery.py` spies on `_recover` with an independent call counter.

## Generalizes to
Any cache keyed only on success, when the cached computation is (a) invoked from a hot per-iteration path and (b) can deterministically fail. Re-Recognize by: does the value re-turn "failed" and get recomputed on the next call even though no input changed? Guard by *memoizing the failure too* — re-run only when the inputs that would change the answer actually changed (here, new evidence). This is a general "cache the last-attempt state, not just the last-success" pattern.
