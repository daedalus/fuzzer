# Handover: gcc `trace-pc` support in the AFL shim (clang-compatible)

> **Status (2026-10-07): implemented**, one commit on top of `d3a95ef7` (tests were first run on `7d3e15e0`, then re-run after rebasing). Supersedes the plan-only version of this document. Verified on gcc 13.3.0 and clang 18.1.3 (Ubuntu 24.04, x86-64). Not verified: gcc 12, non-x86 hosts, any clang other than 18, a full `pytest` run (see "Verification").

## What and why

gcc cannot do `-fsanitize-coverage=trace-pc-guard` (it only accepts `trace-pc` and `trace-cmp`). The shim already had a `__sanitizer_cov_trace_pc` callback, but a gcc trace-pc target **segfaulted at startup**: gcc does not skip functions named `__sanitizer_cov_*` like clang does, and the shim left most of its own code unmarked, so the callback instrumented itself and recursed (63 self-calls in `targets/test_target.c`, rc 139).

This work makes gcc trace-pc builds work, side by side with clang, without changing anything clang emits.

```
clang  trace-pc-guard ─▶ __sanitizer_cov_trace_pc_guard(guard*) ─┐
clang  trace-pc       ─▶ __sanitizer_cov_trace_pc()  (ngram/dist)├▶ __afl_map_loc(id) ─▶ edge map
gcc    trace-pc       ─▶ __sanitizer_cov_trace_pc()  (NEW)       ┘
```

## What changed

### `src/fuzzer_tool/adapters/afl_shim.c`
- **`__AFL_GCC_NO_COV`**: expands to `__AFL_NO_COV` under gcc and to nothing under clang. Applied to every shim function the `-include` drags into the target TU (≈55 definitions, plus the `__AFL_DATAFLOW_CB` macro). It is deliberately *not* plain `__AFL_NO_COV`: clang instruments those functions today, so marking them would shift guard numbering and therefore every edge id. Needs gcc ≥ 12 (`no_sanitize_coverage`).
  - Why helpers appeared after the first pass: gcc will not inline an instrumented helper into a `no_sanitize_coverage` caller, so helpers previously folded into annotated functions (`__afl_timer_ready`, `__afl_put_*`, `__afl_fb_*`, ...) became separate instrumented symbols. All are annotated now. `always_inline` helpers are left alone on purpose: they take on the caller's setting.
- **`__sanitizer_cov_trace_pc` un-gated** from `__AFL_DISTANCE_MODE`. `__afl_base` and `__afl_pc_key` moved with it; only `__afl_probe_distance(key)` stays behind the gate.
- **Edge id**: clang keeps `__afl_map_loc((uint32_t)(key >> 1))` unchanged. Non-clang uses `splitmix(key ^ SALT) & ((1 << __AFL_TRACEPC_BITS) - 1)`, default 20 bits (must be in [8, 24], matching `VIRGIN_DENSE_MAX`). The mask is private to the callback and never touches `__afl_loc_mask`, so hand-written `__afl_map_edge` ids keep their width.

### `src/fuzzer_tool/core/elf.py`
- `trace_pc_call_sites(target)`: counts `E8 rel32` calls to `__sanitizer_cov_trace_pc` in executable sections. x86-64 little-endian only; returns `None` (not 0) for unsupported arch, stripped binary or missing symbol.
- `sancov_guard_status` falls back to it, so a gcc trace-pc binary is "present" instead of "absent".
- `estimate_map_size_detail` gets a `"trace_pc_calls"` tier before the profile/branch-density guesses. It is **not** `exact`: clang also instruments the shim, whose sites are counted.

### `src/fuzzer_tool/services/fuzzer.py`
- `_detect_distance` now keys on `__afl_dist_flush` only. The bare `__sanitizer_cov_trace_pc` fallback is gone because that symbol is now in every build.

### `tools/build_targets.sh`
- `cov_flag_for_cc CC`: clang → `${SANCOV_FLAG:-trace-pc-guard}` (unchanged); gcc ≥ 12 → `-fsanitize-coverage=trace-pc` (+`,trace-cmp` with cmplog); otherwise empty. `GCC_TRACE_PC=0` forces empty (rollback switch). Note the script always sets `SANCOV_FLAG` to the guard flag, so gcc must not read it.
- Replaced the hard-coded flag logic at the secp256k1, grep-family, sqlite and fuzzgoat object compiles.
- New opt-in **`--gcc-scov`** pass (after the clang-scov pass): simple targets, simple `.so`, standalone `.so`, and UBSAN `.so` variants with `gcc` + the flag. Narrower than clang-scov on purpose: vendored libs, fgrep and vendored `.so` helpers hard-code clang flags and were left to the clang pass.
- `verify_sancov` accepts trace-pc call sites (via `objdump`) for gcc builds, and runs for either scov flag.
- Feature-matrix line, usage header and the stale `_pick_cc` comment updated.

### Docs / tests
- `README.md` (matrix cell + the "gcc edge-coverage limitation" section), `CHANGELOG.md` (Added, Fixed).
- New `tests/test_gcc_trace_pc.py` (28 tests): runs at -O0..-O3; structural check that no shim function is instrumented across configs (default, distance off, cmplog, ngram, ASAN); edges depend on input; ids stable across runs and within the dense-map bound; detection and map-size provenance; stripped → `None`; clang unchanged (guard build, trace-pc with distance off links, unmixed ids); `cov_flag_for_cc` per compiler.
- Two existing tests that cut a build-script function out in isolation now also load `cov_flag_for_cc` (`test_sancov_modes.py::test_object_follows_modes`, `test_regression_fuzzgoat_sancov.py`).

## Clang compatibility: what was checked

Built `targets/test_target.c` with clang 18 against the **baseline shim** and the **patched shim**, six configs, compared:

| config | guard section bytes (before/after) | trace-pc/guard call sites (before/after) |
|---|---|---|
| trace-pc-guard | 0x480 / 0x480 | 289 / 289 |
| guard + trace-cmp (cmplog) | 0x58c / 0x58c | 356 / 356 |
| trace-pc | n/a | 288 / 288 |
| trace-pc + ngram | n/a | 288 / 288 |
| guard, distance off | 0x424 / 0x424 | 266 / 266 |
| trace-pc, distance off | failed to link / links and runs | 0 / 265 |

All runs exited 0. Identical guard counts mean guard numbering, hence edge ids, did not move. The last row is the one intended behaviour change on clang.

## Verification

- gcc: 9-config matrix (O0/O1/O2/O3, distance off, cmplog, ngram, frame pointer, ASAN) all rc 0 with zero instrumented shim functions.
- `tools/build_targets.sh --gcc-scov` ran end to end: `verify_sancov` reports 26 `.so` targets instrumented; `test_target_nosan` runs; `png_read_nosan` has 1175 call sites.
- Tests run: the new file (28 pass); `test_sancov_modes`, `test_regression_fuzzgoat_sancov`, `test_regression_scov_target_coverage`, `test_regression_compiler_coverage_detection`, `test_ctx_and_map_size`, `test_ngram_shim`, `test_shim_indir_cov`, `test_edge_id_stability_guard` (159 pass), plus 55 more shim/distance/icfg/elf/cmplog-related files (831 pass).
- **Pre-existing failure, unrelated:** `tests/test_regression_shim_ctx_asan.py::test_regression_ctx_walk_adds_no_asan_checks` (16 vs 15 ASAN checks) fails identically on upstream `7d3e15e0` with clang 18.
- **Not run:** the full suite, gcc 12, any real fuzzing campaign on a gcc build, performance of trace-pc vs guards.

## Known limits and follow-ups

1. **Edge ids differ between a gcc and a clang build of the same source.** Share corpora by seed bytes, not edge maps.
2. gcc has no `trace-div`/`trace-gep`, trace-loads/stores or inline counters; `--sancov=` remains clang-only.
3. `--gcc-scov` does not rebuild vendored libs or fgrep (their helpers hard-code clang flags). On a gcc-only box, library objects do get trace-pc from `cov_flag_for_cc`, but the vendored-lib pass `compile_vendored_libs` still uses a hard-coded clang flag string and was not touched.
4. Rerunning `--gcc-scov` printed "checksum changed" for the UBSAN `.so` targets (the default pass and the gcc pass both write them). Not compared against `--clang-scov` reruns (too slow to run here), so unknown whether it predates this work.
5. `trace_pc_call_sites` is x86-64 only; elsewhere gcc trace-pc binaries still read as "absent". An `objdump`/capstone path would cover other arches.
6. Hot-path cost of the callback (lazy `dladdr` once, then a subtract and a mix per block) is unmeasured.
7. gcc ≥ 12 is required; the helper refuses older gcc. gcc 11 and earlier would recurse because `no_sanitize_coverage` does not exist there.

## Rollback

One commit. `GCC_TRACE_PC=0` disables the gcc flag without a revert. Reverting restores the old behaviour: gcc targets get no coverage flag and a hand-built gcc trace-pc target crashes at startup again.
