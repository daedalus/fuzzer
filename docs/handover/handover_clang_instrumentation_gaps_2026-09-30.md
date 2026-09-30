# Handover: missing clang instrumentation (sancov)

> **Status (2026-09-30): G1, G2, G3 fixed; G4-G6 open.** Baseline: `fa7c9fc3`.
>
> **G1 (implemented):** `__sanitizer_cov_trace_pc_indir()` in `afl_shim.c`: FNV-1a over `(site, callee)`, base-relative, into `__AFL_SYNTH_ID` via `__afl_map_id`. Callees outside the module image are dropped (ASLR-dependent). Build: `--indir-cov` / `--sancov=...,indirect-calls` (`validate_sancov_modes`, `add_indir_mode`). Tests: `tests/test_shim_indir_cov.py` (8, incl. direct-call, no-mode, libc-callee, ASAN), `tests/test_sancov_modes.py`. Not done: CTX-hash mixing (kept separate to avoid renaming real edges), a Python-side runtime gate, Rust (G6), effect on discovery and map pressure (unmeasured).
>
> **G2 (implemented):** `__afl_note_stack()` in `afl_shim.c`, called from `__afl_map_loc`. First sample after reset pins the base frame; deepest later frame is published live to SHM offset 0. Samples above the base or over `__AFL_STACK_WINDOW` (64 MiB) below it are other threads' stacks and ignored. Tests: `tests/test_shim_stack_depth.py` (5: deep, monotonic, shallow falsification, second-thread adversarial, ASAN); the 3 positive ones failed before the fix (depth 0). Per-edge cost: within noise on a 60M-iteration loop (min 0.505s old vs 0.491s new; run-to-run variance was larger than the difference). Effect of the boost on discovery is untested.

> **G3 (implemented):** `validate_sancov_modes` groups modes as edge / extra / cmp; `sancov_cmp_modes` (`tools/build_targets.sh`) is the one list, also feeding `sancov_trace_flags` for both `--tracecmp` vendored builds (flag string unchanged byte for byte). `--sancov=trace-pc-guard,trace-cmp,trace-div,trace-gep` now validates and sets `WITH_CMPLOG=1`. `WITH_CMPLOG` already defaults to 1 with no off switch, so that line only guards a future one. `SANCOV_FLAG` reaches library objects only, never the wrapper, so the G5 concern does not arise. `stack-depth`, `trace-pc`, `no-prune` stay rejected. MSAN/TSAN: `build_wants_cmplog` adds the cmplog layer to those executables only when `--sancov` names a cmp mode; the default (no cmp mode) is unchanged. Measured with clang 18: without it the link still succeeds, because the sanitizer runtimes ship weak no-op `__sanitizer_cov_trace_cmp*/div*` stubs (`nm` shows `W`), so requested compare tracing was silently dead; with it the shim's own callbacks link in (`t`). Tests: `tests/test_sancov_modes.py` (86 pass with clang, including `TestLinks`). Not done: no fuzzing run with cmp modes under MSAN/TSAN, so the shim's cmp layer there is verified to link and execute a small target, not measured for overhead or false reports.
>
## Goal
List `-fsanitize-coverage` features the build scripts or `afl_shim.c` do not support, ranked by expected value.

## Current coverage
| Feature | Build | Shim | Gate |
|---|---|---|---|
| `trace-pc-guard` | default (`SANCOV_MODES`) | yes | `--clang-scov` |
| `inline-8bit-counters`, `inline-bool-flag` | `--sancov=` | yes | `validate_sancov_modes` |
| `trace-cmp` (+`const_cmp`, `trace_switch`) | vendored libs only | yes | `--tracecmp` (implies cmplog) |
| `trace-div`, `trace-gep` | `--tracecmp` path | yes (`div4/8`, `gep`) | `--tracecmp` |
| `trace-loads`, `trace-stores` | `--sancov=` | yes (`__AFL_DATAFLOW_CB`) | `validate_sancov_modes` |
| `pc-table` | `--sancov=` | stub (`__sanitizer_cov_pcs_init`) | `validate_sancov_modes` |

## Gaps

### G1. `indirect-calls` (`__sanitizer_cov_trace_pc_indir`) — P1 (FIXED 2026-09-30)
- No callback in `afl_shim.c`; mode absent from `validate_sancov_modes` (`tools/build_targets.sh:364`).
- Callback gets `(callee)` at each indirect call site. Edge coverage sees only the callee entry block, not which site reached it.
- Value: C++ vtables, function-pointer tables (FFmpeg `AVInputFormat`/codec tables), callback-heavy parsers.
- Fit: mix `(caller_pc, callee)` into the existing call-stack-sensitive 3-term hash; synthetic channel like COMPCOV/DATAFLOW (bit 31, `__AFL_SYNTH_ID`).
- Callback must be `hidden` visibility and `__AFL_NO_COV` (libasan weak-stub interposition, same as guard callbacks).
- Open: map-pressure cost; gate behind a flag like `--indir-cov`.

### G2. `stack-depth` never fed — P1 (FIXED 2026-09-30)
- No `stack-depth` flag anywhere in `tools/`.
- `afl_shim.c`: `__afl_max_stack_depth` (l.569) is only ever reset to 0 (l.1735); no assignment found. Line 1716 writes it to SHM offset 0 (`SHM_STACK_DEPTH_OFFSET`), so the field appears to always be 0.
- Consumer: `core/schedules.py:116` stack-depth factor. Check whether Python fills SHM offset 0 elsewhere before concluding it is dead.
- Fix options: (a) enable `-fsanitize-coverage=stack-depth`, read `__sancov_lowest_stack` (TLS, runtimes own the definition per shim header l.24, so needs a decision on linking); (b) sample frame address in the guard callback via `__builtin_frame_address(0)` vs a per-exec base.
- Confirmed: SHM value was 0 on a 400-frame recursive target. Fixed via option (b). Not covered: `inline-8bit-counters` / `inline-bool-flag` builds have no per-edge callback and still report 0.

### G3. `--sancov` mode gating — P2 (FIXED 2026-09-30)
- `validate_sancov_modes` rejects `trace-cmp`, `trace-div`, `trace-gep`, `trace-pc`, `trace-pc-indir`, `stack-depth`, `no-prune`.
- No single-call build of `trace-pc-guard,trace-cmp,trace-div,trace-gep`; compare tracing requires `--tracecmp` (and thus cmplog).
- Fix: one place, per AGENTS.md rule 1. Split "needs cmplog" modes from "shim-only" modes in the validator.

### G4. No allowlist / ignorelist — P2
- `-fsanitize-coverage-allowlist` / `-ignorelist` unused; whole vendored lib is instrumented.
- Value: restrict FFmpeg to demuxers/decoders, drop `libavutil` noise. Reduces map pressure and the id-union growth seen in P1-4.
- Cost: allowlist file format + a `tools/` switch; ids shift, so invalidates saved `edge_tracker.json`.

### G5. `trace-cmp` scope — P3 (documented, deliberate)
- Applied to vendored libs only (`build_targets.sh:997`); wrapper has almost no compares.
- `memcmp` lowered after the sancov pass is invisible to trace-cmp; `$NOBUILTIN_CMP` covers this (comment at l.266-281). No action.

### G6. Rust builds — P3
- `tools/build_rust_target.sh:121,233`: `trace-pc-guard` + `trace-compares` at `level=3` only.
- Missing: `trace-divs`, `trace-geps`, `trace-loads`/`trace-stores`, `indirect-calls`.
- Depends on G1 for the last.

### Not gaps
- `trace-div1/2`: clang has only `div4`/`div8`.
- LLVM has no float-compare tracing.

## Suggested order
1. ~~G2~~ done.
2. ~~G1~~ done.
3. ~~G3~~ done.
4. G4 allowlist.

## Files
`tools/build_targets.sh` (l.340-379, 795-927, 997-1010, 1299), `tools/build_rust_target.sh`, `src/fuzzer_tool/adapters/afl_shim.c` (l.415, 569, 1316-1364, 1418-1510, 1652-1735, 3032), `src/fuzzer_tool/core/schedules.py:116`, `src/fuzzer_tool/core/elf.py:659-844`.
