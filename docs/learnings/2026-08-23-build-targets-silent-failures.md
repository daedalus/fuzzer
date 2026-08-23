# Two silent failures in build_targets.sh

Date: 2026-08-23

Changed: `tools/build_targets.sh`.

Found by installing clang and trying to build targets in order to validate the
`inprocess.py` SHM fixes. Neither is subtle once seen; both had been invisible
because the failure mode was "looks like it worked".

## 1. `set -e` plus an optional-vendor guard returning 1

`compile_fuzzgoat_object` opened with:

```bash
[ -f "$VENDOR/fuzzgoat/fuzzgoat.c" ] || return 1
```

Every other optional-vendor guard in the script returns 0. This one returned 1,
and under `set -e` that aborted the entire build — on any machine without
`vendor/fuzzgoat`, which is optional and not fetched by default. The abort
landed after `grep_read` and before:

- all `.so` targets (`build_simple_so_targets`, `build_standalone_so_targets`)
- `compile_perf_shim`
- the whole verify pass: `verify_afl`, `verify_shm_run`, `verify_cmplog`,
  `verify_target_md5`
- `build_tracecmp_targets`

The last line printed was a green `OK: grep_read_nosan`, and the usual
invocation pipes through `tee` or `tail`, which discards the exit code. The
script's own header promises the opposite: *"all are optional — the build warns
and skips when the tree is absent"*.

Fixed to the `HAS_FGREP` pattern already in the file: a `HAS_FUZZGOAT` flag, a
matrix `SKIP` row, `return 0`, and `if` blocks at the three call sites.

The `if` matters. Writing `[ "$HAS_FUZZGOAT" -eq 1 ] && build_target ...` as a
function's last statement returns 1 when the test fails and recreates exactly
the bug being fixed. An `if` with no taken branch returns 0.

Also added an `ERR` trap with `set -E` so the next one of these announces
itself with file, line and command instead of impersonating success.

## 2. `--clang-scov` never instrumented the `.so` targets

The flag whose entire purpose is compiler-inserted edge coverage did not apply
it to any `.so` target carrying `fuzz_shm_run` — that is, to everything
`--inprocess` and `--inprocess-direct`/direct_lite actually run. Two causes:

- `build_simple_so_targets` was not called in the `--clang-scov` branch at all,
  so `test_target.so`, `png_read.so`, `zlib_read.so` and the rest kept whatever
  the uninstrumented no-ASAN pass produced.
- `build_standalone_so_targets` took only `suffix, flags, label` — no `cc` or
  `extra_cflags` — so it had no way to receive `$SCOV_FLAGS` even when called.

Measured before the fix: every in-process campaign printed
`AFL instrumentation: detected` and then `shm: 0 | map: 0.0%` with
`Edges discovered: 0`, for the whole run. A coverage-guided fuzzer with no
coverage, reporting success. Confirmed against `HEAD~1` that this predated the
`inprocess.py` work rather than being caused by it.

After threading `cc`/`extra_cflags` through both builders and adding the
missing calls: `--inprocess-direct` 8 edges, `--inprocess` 5 edges on
`test_target.so`. The UBSAN `.so` variants needed the same treatment and were
the last nine offenders.

## The verify function that certified the bug

The reason nobody noticed: `verify_afl` checks that `__afl_*` symbols are
present. They always are — the shim is `-include`'d into every target. It says
nothing about whether the compiler emitted any *calls* to them.

I then wrote `verify_sancov` with the same flaw, and it reported 29/29 OK while
`test_target.so` was demonstrably uninstrumented. `__sanitizer_cov_trace_pc_guard`
is *defined by the shim*, so `nm` finds it either way. Measured on a
deliberately uninstrumented build versus an instrumented one: 12 matching
symbols each.

The discriminator is the `__sancov_guards` **section**, emitted only by
`-fsanitize-coverage=trace-pc-guard`, and it is the array the instrumented call
sites index into: 0 sections versus 1. `verify_sancov` uses `readelf -S` now.

Corrected, it immediately found nine real offenders I would otherwise have
shipped.

## Carrying forward

A verification step that checks for a symbol the shim itself provides can only
ever confirm the shim was linked. It cannot distinguish an instrumented target
from an uninstrumented one, which is the only question worth asking. Check for
the artifact the *compiler* produces — a section, a guard array, a relocation —
not for a name the runtime supplies.

And test the check against a known-bad input before trusting a green result.
Mine passed 29/29 on a build I already knew was broken.
