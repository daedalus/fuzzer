# Handover — the Boltzmann energy A/B, and why it cannot run on `locked`

**Date:** 2026-08-30
**Base:** `3be902d`
**Status: SET UP, NOT RUN.** The harness, the target set and the arms are in
place and smoke-tested. Zero cells of the real matrix are recorded. Everything
needed to start it is in §5.

The debt this pays down is the open item *"A/B the cost-based Boltzmann
energy"* in `docs/TODO.md`, filed when `ab07835` shipped
`SeedPicker._pick_boltzmann_seed` reading `effective_fuzz_count` instead of
`fuzz_count`. That change alters seed selection and was never benchmarked.

---

## 1. The blocking finding: `locked` cannot resolve this arm

The obvious way to run the A/B is `tools/bench_paired.py --set locked`. That
would have produced a number, and the number would have been wrong.

`locked` runs the **executables** in `targets/` — one process spawn per
execution. Measured per-seed mean execution cost across a campaign's corpus on
the same target, same corpus, same seed:

| target | mode | p90/p10 | max/min | CV |
|---|---|---|---|---|
| `targets/png_read` | subprocess | 1.06x | 1.17x | 0.023 |
| `targets/png_read.so` | direct_lite | 4.32x | 48.0x | 1.455 |

About 9 ms of fork/exec dominates about 0.2 ms of actual decode, so in
subprocess mode every seed costs the same to within noise.

This matters because of how the arm is built. `effective_fuzz_count` expresses
the cost ledger in average-cost executions, and under uniform per-execution
cost it reduces to the sample count *exactly* — that identity is deliberate
(`core/cost_ledger.py`). So on `locked` the two arms are not merely similar,
they are arithmetically the same computation. The A/B would return "no
significant difference" whether or not the change has an effect: a false
negative dressed as a null result.

**This generalises past this arm.** `locked` cannot resolve any arm whose
effect depends on per-seed execution cost varying, because process-spawn
overhead flattens the signal before any arm sees it. Worth remembering before
reading a null out of that set.

## 2. What was added

**`tools/eval_set.py`: a `direct_lite` target set.** A new set rather than an
edit to `locked`, per the rule at the top of that file — results are not
comparable across sets, and a number quoted from here has to say so. Six
targets as `.so`: png, jpeg, zlib, lz4, grep, gzip. `.so` targets take the
in-process ctypes path automatically; no flag is needed, and the run log
confirms it with `in-process mode (direct_lite)`.

The format targets carry `-m 65536`. The 4096 default truncates away exactly
the large inputs whose decode cost varies, so it is not a neutral choice for a
cost-sensitive arm — it suppresses the quantity under test. Measured, raising
the cap moves png dispersion from 2.00x to 4.32x.

Direct_lite is also 10x faster — 22 s per cell against 217 s — because process
spawn was most of the old budget. That is what makes a 240-cell run practical.

**`tools/bench_paired.py`: two arms, checkpointing, and a `--set` fix.**

* `boltzmann-count` and `boltzmann-cost`. Both pass `--boltzmann` and differ
  only by a code edit, so they are recorded in `UNWIRED_ARMS` and **must be run
  in separate invocations with the source swapped between them** (§5).
  `--boltzmann` without `--elo` makes `_pick_boltzmann_seed` the sole seed
  strategy; under `--elo` it would be one arbitrated arm among several and most
  picks would never reach the code under test.
* Results are now checkpointed after every cell via a temp file and an atomic
  rename, and a re-run resumes from what is on disk unless `--restart` is
  passed. Previously the file was written once after the last cell of an arm,
  so an interrupted run lost everything it had done — which is exactly what
  happened here: a VM restart discarded 67 completed cells.
* `--set` choices were hardcoded as `("locked", "cmplog")` in the argument
  parser instead of being derived from `TARGET_SETS`, so a set that exists in
  `eval_set.py` was rejected at the CLI. Now derived.

## 3. Environment

`clang` is required — without it `tools/build_targets.sh` aborts before the
trace-cmp targets and leaves holes in the matrix. `apt-get update` first, then
`apt-get install -y clang libjpeg-dev liblzma-dev libbz2-dev zlib1g-dev`, then
`tools/vendor_lz4.sh`, then `tools/build_targets.sh --force`.

`secp256k1` did not vendor in this environment, and `cmplog_exercise` builds
only as `_tcg`. Neither is in the `direct_lite` set. A missing target is
skipped by the harness rather than scored zero, which is correct, but it is
also a silent hole — check the `not built, cells skipped` line before quoting
any result.

## 4. What to expect, and what would falsify the change

The arm should be a **no-op on `zlib_read` and `gzip_read`** and show a
difference, if it has one, on `png_read` and `jpeg_read`. That is not a
prediction about which is better; it is the identity from §1. If the analysis
shows a difference on the flat-cost targets, something is wrong with the
harness or the arms, not with the scheduler.

Report **per-target breakdowns, not just the pooled McNemar.** Two of the six
targets are no-ops by construction and `zlib`/`lz4` come in around 12 edges,
which is close to saturated; pooling lets those cells dilute a real effect on
png/jpeg or manufacture a null.

The risk the TODO item names is still the one to look for: down-weighting
expensive seeds is down-weighting deep paths on targets where depth costs time.
A loss on png/jpeg is a real result and should be recorded as one — the point
of running this is that it can come back against the change.

## 5. To run it

```sh
# arm 1 — the pre-ab07835 energy. Edit services/seed_picker.py, in
# _pick_boltzmann_seed, replacing the effective_fuzz_count line with:
#     n = max(meta.get("fuzz_count", 1), 1)
python3 tools/bench_paired.py run --arms boltzmann-count --set direct_lite \
    --iters 10000 --timeout 600

# arm 2 — restore the shipped line, then:
#     n = max(effective_fuzz_count(meta, mean_exec), 1.0)
python3 tools/bench_paired.py run --arms boltzmann-cost --set direct_lite \
    --iters 10000 --timeout 600

python3 tools/bench_paired.py analyse results/paired/direct_lite_*.json
```

120 cells per arm at roughly 22 s each — about 45 minutes per arm. Both arms
resume if interrupted, so they can be run in slices. Verify the source is in
the intended state before starting each arm: the two arms are
indistinguishable from their flags alone, and a mislabelled arm is a silent
wrong answer rather than an error.

## 6. Also unblocked by this

`docs/TODO.md` carries *"Should cumulative execution cost enter the eviction
ordering?"* (backlog D5's third consumer). It inherits the same A/B
requirement, the same `direct_lite` set, and the same reason `locked` will not
do — and per the correction in
`docs/learnings/2026-08-30-prune-ceiling-and-eviction.md`, the case for it is
stronger than the original write-up assumed: if nearly every eviction candidate
is coverage-equivalent, the tiebreak decides essentially every prune, and
accumulated cost is the signal that varies most across those ties.
