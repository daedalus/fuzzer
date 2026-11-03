# Handover: afl_shim.c guard-path A/B, `ba6cf7e` → `735ea314`

**Date:** 2026-11-02. **Verified against:** `735ea314` (HEAD at the time). **Status:** one change applied (section 6, P1), the rest proposed or rejected.
**Question:** is the current shim slower than `ba6cf7e`, and which commit costs what.
**Answer:** yes on 3 of 6 cases (wall, clang). Patch P1 recovers most of the hit-path loss. The all-miss loss from `8539822a` remains.

## 1. Metric legend

| Metric | Unit | Magnitude here | Better |
|---|---|---|---|
| Ir/run | instructions retired per `run()` (150 guard callbacks), cachegrind | 1.1e4 to 2.2e4 | lower |
| Ir/guard | Ir/run ÷ 150 | 75 to 150 | lower |
| wall | ns per `run()` (150 guards), min of N interleaved rounds | 4.5e2 to 1.6e3 | lower |
| throughput | Mguards/s = 150 000 ÷ wall-min-ns | 94 to 200 | higher |
| Δ% | relative change vs the named base | -10 to +17 | negative is better for Ir, wall; positive is better for throughput |
| static insns | instruction count of `__sanitizer_cov_trace_pc_guard` (objdump) | 128 to 154 | lower |
| noise floor | A/A wall-min difference, HEAD vs an identical HEAD copy | 0.1% to 1.6% | lower (smaller means the A/B is more trustworthy) |
| tests | pytest passed / failed counts | 173 / 4 | passed higher, failed lower |

Ir is deterministic (zero variance). Wall is not: see section 3.

## 2. Verdict

| Case | Meaning | wall HEAD vs old | wall patched vs old | wall patched vs HEAD |
|---|---|---|---|---|
| hit-healthy | one call site, frame pointers kept (realistic: 91% hit on `ffmpeg_read`, per shim comment) | -2.3% | -5.8% | -3.6% |
| hit-const | one call site, `-fomit-frame-pointer` target | **+9.2%** | +1.7% | **-6.8%** |
| hit-junk | one call site, `rbp` clobbered | **+9.3%** | +2.1% | **-6.6%** |
| chain-healthy | 150 distinct call sites, every guard misses the ctx cache (worst case) | **+8.3%** | +6.9% | -1.3% (inside noise) |
| chain-const | as above, no frame pointers | -1.0% | -1.5% | -0.5% (inside noise) |
| chain-junk | as above, `rbp` clobbered | +3.4% | +4.0% | +0.6% (inside noise) |

Negative = faster = better. Source: section 4.1.

## 3. Method

- Host: 1 vCPU container, clang 18.1.3, x86-64. Targets: 150 guarded blocks, 306 guard sites for the hit target.
- Ir: `valgrind --tool=cachegrind --cache-sim=no`; Ir/run = (Ir at 2000 iters - Ir at 0 iters) ÷ 2000.
- Wall: each binary run for 20 000 iterations per round, rounds interleaved across shim versions, min and median kept. N = 25 or 40 rounds.
- **A/A control:** `head` and `head2` are the same source built twice. Their wall-min gap: 0.1% to 1.6% (largest: chain-const).
- **Drift:** between sessions the same old binary moved 602 to 637 ns (hit-healthy) and absolute times rose up to 60%. Compare within one interleaved run only.
- **Trap, fixed:** a harness that never calls `__afl_fast_refresh()` leaves `__afl_fast = 0`, so every guard takes `__afl_guard_slow` (~148 Ir/guard vs ~75 on the fast path). All first-round numbers came from that path and are void (section 5). `run.sh` now fails if `fast != 1`.
- No `perf` counters available (cache misses, branch misses, cycles not measured).

## 4. Results (fast path, clang)

### 4.1 Headline: old `ba6cf7e` vs HEAD `735ea314` vs patched (P1), 40 rounds

Wall min (ns/run, lower is better) / throughput (Mguards/s, higher is better):

| Case | old | HEAD | head2 (A/A) | patched |
|---|---|---|---|---|
| chain-healthy | 1468 / 102.2 | 1590 / 94.3 | 1589 / 94.4 | 1569 / 95.6 |
| chain-const | 1125 / 133.3 | 1114 / 134.6 | 1096 / 136.8 | 1108 / 135.4 |
| chain-junk | 948 / 158.3 | 980 / 153.0 | 982 / 152.7 | 986 / 152.2 |
| hit-healthy | 951 / 157.8 | 929 / 161.5 | 930 / 161.3 | 896 / 167.4 |
| hit-const | 749 / 200.3 | 818 / 183.5 | 821 / 182.7 | 762 / 197.0 |
| hit-junk | 749 / 200.2 | 819 / 183.1 | 815 / 184.0 | 765 / 196.1 |

Ir/run (instructions, lower is better):

| Case | old | HEAD | patched | HEAD vs old | patched vs HEAD |
|---|---|---|---|---|---|
| chain-healthy | 19036 | 22216 | 20975 | +16.7% | -5.6% |
| chain-const | 15778 | 15040 | 14285 | -4.7% | -5.0% |
| chain-junk | 12310 | 12457 | 12447 | +1.2% | -0.1% |
| hit-healthy | 15085 | 14394 | 14023 | -4.6% | -2.6% |
| hit-const | 11344 | 11491 | 11481 | +1.3% | -0.1% |
| hit-junk | 11345 | 11492 | 11482 | +1.3% | -0.1% |

### 4.2 Per-commit Ir/run (instructions, lower is better; 150 guards)

| Commit | Subject | chain-healthy | chain-const | chain-junk | hit-healthy | hit-const | hit-junk |
|---|---|---|---|---|---|---|---|
| `ba6cf7e` | base | 19036 | 15779 | 12310 | 15085 | 11344 | 11345 |
| `04ac45dd` | detect collapsed call-context | 20431 | 16081 | 13953 | 15395 | 12987 | 12988 |
| `2c6db238` | inline bail/miss counters | 18734 | 15477 | 12306 | 14782 | 11339 | 11341 |
| `0fb976c1` | sticky flags in the detector | 19227 | 15477 | 12306 | 14644 | 11339 | 11341 |
| `597d828a` | block-hit bitmap | 19454 | 15644 | 12465 | 15211 | 11499 | 11500 |
| `9bf0afd7` | crash signatures | 19445 | 15644 | 12465 | 15211 | 11499 | 11500 |
| `84439900` | escaping descendants, signal interposition | 19496 | 15644 | 12465 | 15142 | 11498 | 11500 |
| `b66476e6` | plain touched OR, protected hot globals | 19496 | 15644 | 12465 | 14734 | 11499 | 11500 |
| `8539822a` | cfp-keyed ctx-cache hit | 22216 | 15040 | 12457 | 14394 | 11491 | 11492 |
| `735ea314` | exempt shim from coverage | 22216 | 15040 | 12457 | 14394 | 11491 | 11492 |

Attribution (instructions per run, lower is better):
- `04ac45dd`: +1395 chain-healthy, +1643 chain-junk, +1643 hit-const/junk. Reverted by `2c6db238` (back to base ±150).
- `597d828a`: +160 to +230 on const/junk, +567 hit-healthy (+1.1 Ir/guard on const/junk).
- `8539822a`: **+2720** chain-healthy (+18 Ir/guard), -604 chain-const, -340 hit-healthy, ≈0 else.
- `9bf0afd7`, `84439900`, `735ea314`: ≈0.

### 4.3 Per-commit wall min (ns/run, lower is better; single 25-round run, **no A/A, unaligned**)

| Commit | hit-const | hit-healthy | chain-healthy |
|---|---|---|---|
| `ba6cf7e` | 456 | 602 | 930 |
| `04ac45dd` | 672 | 596 | 949 |
| `2c6db238` | 475 | 587 | 987 |
| `0fb976c1` | 458 | 579 | 999 |
| `597d828a` | 485 | 585 | 879 |
| `9bf0afd7` | 479 | 588 | 882 |
| `84439900` | 477 | 625 | 964 |
| `b66476e6` | 478 | 631 | 969 |
| `8539822a` | 502 | 636 | 1053 |
| `735ea314` | 505 | 637 | 1053 |

Read with care: the same old binary moved ±3% between sessions. The `84439900` step (+6% hit-healthy) is not logic (that commit adds signal interposition only, Ir +0.3%) and shrank to 2% under `-falign-functions=64`.

### 4.4 Static instructions in `__sanitizer_cov_trace_pc_guard` (count, lower is better)

| Build | static insns |
|---|---|
| `ba6cf7e` | 128 |
| HEAD `735ea314` | 154 |
| patched (P1) | 144 |

HEAD's hit gate loads, in order: `last_cfp`, `main_lo`, `[caller_fp+8]` (return address), `last_ra`, then `last_ctx`. The old path hashed the return address instead (two multiply-xorshift rounds, ~14 insns). Each global load was `lea r,[rip+X]; mov r,[r]` (2 insns) because `__afl_raw_load` took an `"r"` operand.

### 4.5 Variants tried on top of P1 (wall min ns/run and Ir/run; both lower is better)

| Variant | Ir/run vs P1 | wall min vs P1 | Verdict |
|---|---|---|---|
| V1: drop `cur >= main_lo` from the cfp gate | chain-healthy -2.1%, chain-const -3.2%, hit-healthy -3.2%, else ≈0 (-3.0 Ir/guard) | hit-const +5.8%, hit-junk +3.0%, hit-healthy -4.1%, chain-healthy -1.5%, chain-const -6.3% | mixed, not applied |
| V2: pack `last_ra/last_ctx/last_cfp/main_lo` in one 64-byte-aligned struct | 0 | hit-const +0.5%, hit-junk 0%, hit-healthy +0.7%, chain-healthy +0.1%, chain-const -1.0% | null, not applied |
| V3: `aligned(64)` on the guard entry only | 0 (guard already at 0x3300 in HEAD) | hit-healthy still +5% old→HEAD across 7 commits (615 to 647 ns) | null, reverted |
| `-D__AFL_CTX_CHECK=0` ablation | not rerun on fast path | slow-path run only (void) | open |

V1 to V3 were measured in 25 to 40 round runs on a drifting host and without their own A/A.

## 5. Retracted claims (all from the first, slow-path round unless noted)

1. "Clang junk-`rbp` +18%": noise (min of 3 samples). Fast path: chain-junk +3.4% wall, +1.2% Ir.
2. "The junk early exit is gone": wrong. HEAD junk costs less than HEAD const on every fast-path case (12457 vs 15040 Ir).
3. "Gate the block-bitmap check behind a `-D`": retracted. On the fast path the check sits in the settle path (once per new edge). Measured cost: +1.1 Ir/guard (+1.4%) on const/junk hit.
4. "gcc is unusable / gcc junk +14%": not re-measured; gcc was not rerun on the fast path.
5. "Detector costs ~3% wall on healthy": from the slow-path ablation, void.
6. "Layout/alignment explains the hit-path loss": not supported (V3).

## 6. Proposed changes

| ID | Change | Status | Evidence / expected effect | Risk |
|---|---|---|---|---|
| P1 | `__afl_raw_load` (x86-64): `"m"(*p)` operand instead of `"r"(p)` | **applied** in this patch | wall vs HEAD: hit-const -6.8%, hit-junk -6.6%, hit-healthy -3.6% (A/A ≤0.5%); Ir: chain-healthy -5.6%, chain-const -5.0%; static insns 154 → 144 | low: builds clang + gcc; compiler now sees the read (fixes reorder freedom of the old form) |
| P2 | Same change for aarch64: `"Q"`-style operand for `ldr %0, %1` | proposed | untested (no aarch64 host). Expected: one less address materialisation per global | medium: operand syntax differs between clang and gcc |
| P3 | Backoff for the cfp gate: after N consecutive gate misses, skip the gate for M guards (or `-D__AFL_CFP_CACHE=0` for known miss-heavy builds) | proposed | targets `8539822a`: +18 Ir/guard, +8.3% wall on chain-healthy (still +6.9% with P1). Untested; the miss counter costs a store | medium: adds state |
| P4 | Drop `cur >= main_lo` from the cfp gate (V1) | not applied | -3.0 Ir/guard, wall mixed (section 4.5) | stack-switching code: `cur` could sit below the main stack while `cfp` is on it |
| P5 | Measure with `perf stat` (cycles, branch-misses, L1d-load-misses) on bare metal | proposed | resolves the hit-path wall vs Ir gap (hit-const +9% wall for +1.3% Ir). P1 closes most of it, cause not proven | none |
| P6 | Fix the `docs/DEEP_DIVE.md` claim "+0 Ir per guard hit detached" for the block bitmap | proposed | measured +160 Ir/run (+1.4%, +1.1 Ir/guard) on const/junk hit at `597d828a`; re-measure with P1 applied before editing | none |
| P7 | Benchmark hygiene: assert `__afl_fast == 1`, check cachegrind shows no `__afl_guard_slow`, include an A/A arm, min of ≥25 interleaved rounds | applied in `shim_ab_bench/run.sh` | avoids the section 3 trap | none |
| P8 | `aligned(64)` on the guard; packed hot-globals struct | rejected | V2, V3 null | none |

Hard Rule 58 check for P1: wall vs HEAD is never worse than noise (largest loss +0.6%, chain-junk, noise floor ≤1.6%).

## 7. Tests

`PYTHONPATH=src python3 -m pytest -q` on 13 files: `test_shim_uncovered`, `test_ctx_collapse_selfcheck`, `test_block_bitmap`, `test_ctx_and_map_size`, `test_guard_arena`, `test_regression_ctx_walk_stack_bound`, `test_regression_ctx_use_relative_inline`, `test_regression_ctx_home_clustering`, `test_regression_guard_bind_once`, `test_regression_shim_audit`, `test_edge_id_stability_guard`, `test_ngram_shim`, `test_regression_basic_block_slots`.

| Build | passed (count, higher is better) | failed (count, lower is better) |
|---|---|---|
| HEAD | 173 | 4 |
| patched | 173 | 4 |

The 4 failures are in `test_regression_ctx_use_relative_inline.py` at HEAD too. Full suite and pre-commit hooks were **not** run on this patch (built in a scratch clone).

## 8. Reproduce

```
cd docs/handover/shim_ab_bench
git show ba6cf7e:src/fuzzer_tool/adapters/afl_shim.c > /tmp/old.c
cp ../../../src/fuzzer_tool/adapters/afl_shim.c /tmp/patched.c
OUT=/tmp/shim_ab ROUNDS=40 ./run.sh /tmp/old.c /tmp/patched.c
```

Needs clang, valgrind, python3. Add an identical copy of one shim as an A/A arm. Not measured: gcc wall (only compile-checked), `-D__AFL_CTX_CHECK=0` on the fast path, aarch64, real targets (`ffmpeg_read`), multi-threaded targets.
