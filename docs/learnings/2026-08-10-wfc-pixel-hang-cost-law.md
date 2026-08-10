# wfc-pixel-hang: an unbounded algorithmic cost turned into a "fuzzer hang"

**Date:** 2026-08-10
**Context:** `fuzzer-new`, `src/fuzzer_tool/core/wfc.py` + `core/mutations/bmp.py`, commit `2c5bf21`

## Problem

The fuzzer appeared to hang mid-run. A ^C faulthandler dump showed the main thread inside the WFC constraint solver (`wfc._run_loop → _propagate → _prune_cell → np.any`), invoked from the BMP pixel mutator. The stack looked like a genuine infinite loop, but the wave math showed termination — the real failure was cost, not correctness.

## Rejected

- **"It's an infinite loop — find the missing termination check"** — looked plausible from the faulthandler frame; dropped after tracing the loop structure: every `_observe` collapses exactly one cell permanently, so `_run_loop` iterates at most `n` times, and `_propagate`'s queue is budget-bounded. No unbounded loop exists.
- **"Cap the queue iterations (ac3_budget) harder"** — looked like the obvious knob; dropped because ac3_budget bounds one `_propagate` call, but the loop calls it once per observation and the per-call *initial sweep* is unbounded by it: `n` cells × `n_tiles²` vectorized ops, repeated `n` times — O(n²·n_tiles²) where the budget never fires.

## Approach

1. **Benchmark to find the cost law, not the bug.** Timed runs showed 0.07 s at 64 cells/16 tiles → 9 s at 512/128 → minutes at BMP widths — clean O(n²·n_tiles²) scaling, worst case dominated by a *satisfiable* wave (contradictory ones reject fast, so they were never the problem).
2. **Bound work where the cost is incurred (wfc.py):** `WaveGrid.run` now takes a hard `work_budget` counted in `_prune_cell` calls; exhausting it stops the collapse and returns the partial grid (callers map uncollapsed cells to a fallback) — a termination guarantee independent of input. Propagation also became incremental: one full sweep per collapse attempt to enforce caller pins, then only the observed cell's neighbours re-pruned plus the AC-3 cascade (was: full sweep every observation, the n× multiplier).
3. **Bound the problem at the caller (bmp.py):** `_wfc_pixels` refuses rows whose width (>512 tiles), alphabet (>64 distinct tiles), or total cells (>200k) makes collapse unaffordable and falls back to the existing byte-flip path. This is what actually protects the fuzz loop — the wfc budget is the safety net for every caller (PNG/JPEG have tiny tables and are unaffected).

## Key insight

A hang backtrace inside a loop doesn't mean the loop never terminates — it means the loop's per-iteration cost is unbounded relative to the input, and the "budget" that exists (ac3_budget) bounds the wrong loop. Here the real driver was a fixable algorithmic defect (full-grid sweep per observation: O(n·n_tiles²) per step) combined with a caller that can feed arbitrarily large inputs (the natural size of a BMP row). Both had to be fixed: the algorithm (incremental AC-3) and the input size (caller affordability guard), plus a hard global budget as the guarantee that no future caller can reintroduce the stall.

## Verification

- Benchmark before/after on the identical adversarial waves: 64/16: 0.067→0.030 s; 256/64: 1.447→0.433 s; 512/128: 9.009→1.726 s; and the budget fires deterministically on `work_budget=500` over a 4096×256 wave, returning a partial grid in well under a second (pre-fix: minutes).
- Regression suite (5 tests): work budget honoured + grid shape preserved on a 1024×256 wave (wall < 30 s); tight budget returns a partial grid; an all-distinct 24-bit row falls back to the ≤8-byte flip; an affordable row still runs WFC; the exact reported hang shape (1024-wide 24-bit BMP) returns in < 5 s.
- Full suite: 4121 passed, 0 failed.

## Generalizes to

- **When a "hang" has a stack but no cycle, measure the cost law.** Bench the function across input scales; if time scales at n² (or worse) the fix is bounding work or input size, not finding a missing exit condition. The loop was correct-and-terminating — just exponentially too slow for legitimate input sizes.
- **A budget that bounds the wrong loop is no budget.** ac3_budget capped one `_propagate` call; the blowup lived in the outer loop's per-step sweep. Any cap must be a global work counter across the whole run, checked at the hot primitive (here: `_prune_cell`).
- **Fix algorithmic cost and input size together.** The incremental AC-3 cut n×; the caller guard cut tiles²; only both made worst-case rows affordable. The hard work budget covers everyone else who might grow the input later.
- **Satisfiable ≠ cheap; contradictory ≠ the danger.** The pathological input converged successfully (no contradiction, no restart) — it was the *valid* wave that cost minutes. Optimization work should target the converging path, not the backtracking one.
