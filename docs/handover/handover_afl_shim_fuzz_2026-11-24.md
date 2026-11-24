# Fuzzing afl_shim.c with AFL++ (2026-11-24)

`tools/shim_fuzz/` fuzzes the shim's cmplog layer: the libc interposers (`memcmp` family, `str*`, `wcs*`,
`wmemcmp`), the `__sanitizer_cov_trace_{cmp,const_cmp,switch,div,gep}` callbacks and the
`__sanitizer_weak_hook_*` entry points. Regression test: `tests/test_regression_shim_fuzz_harness.py`.

## How it works

- The shim is built as a `-D__AFL_PRELOAD_ONLY` preload library with the no-coverage pragma stripped, so
  `afl-clang-fast` instruments the shim itself (the shim cannot supply its own coverage runtime here).
- `harness.c` decodes each input into (function, operand placement, lengths, `n`, data) and builds operands
  flush with a `PROT_NONE` page. Legality is a reference scan, so only calls libc itself may legally make are issued.
- Oracles: any fault; result differs from libc; `errno` changed; and, in records-only mode, a non-equal libc
  compare must raise `byte_used + byte_dropped` of the records segment.
- Records-only mode (`mkrec.c` creates the segment) also scribbles the segment header and site table, since the
  shim documents that the fuzzer owns them.
- `run.sh text|rec [seconds]` runs a campaign; needs `afl-fuzz`/`afl-clang-fast` (apt `afl++ clang` works).

## Results (upstream 533bf0f = 93cca99d + the channel-gate and F1/F2/F4/F5/F6 fixes)

| run | execs | crashes | hangs | mismatches |
|-----|------:|--------:|------:|-----------:|
| text mode, 270 s, 93cca99d | 22.6M | 0 | 0 | 0 |
| records-only, 170 s, patched | 19.1M | 0 | 0 | 0 |
| text mode + callbacks, 170 s, patched | 15.1M | 0 | 0 | 0 |

One core each, no ASAN/TSAN (the interposers collide with the sanitizer's own).

## Controls (why a clean run means something)

- Shim from `a55cdd4d^` (before bounded string scans): AFL++ re-finds the unterminated-operand SEGV in
  about 1,000 execs; the test pins it with `seeds/ctrl_unterminated_strcmp`.
- Shim from `93cca99d` (before the record-gate fix): the records oracle trips on `strcmp`, `strcasecmp`,
  `wmemcmp`, `wcscmp`, `wcsncmp`, `wcscasecmp`; the patched shim passes.
- Oracle caveat found while building it: `__afl_on_strcmp` / `__afl_on_strcasecmp` record nothing when either
  string is empty, by design; the harness expects that.

## Not covered

The guard-frame / crash-handler / tick / `vfork` / persistent-loop code (where the 20 audit bugs live) needs
scenario fuzzing, not input fuzzing. The dist-table, hits and testcase SHM headers are not scribbled yet
(only the records segment is). Callback cases check faults only, not what they record. The F1 `wcscasecmp`
arity bug is reachable only when `dlsym` fails and the fallback runs; `test_regression_shim_review_findings.py` covers it.
`test_regression_shim_builds_warning_free_gcc[private-heap]` fails on Ubuntu gcc 13: `__afl_ph_die` ignores
`write()`'s result (`-Werror=unused-result`); one-line fix not applied here.
