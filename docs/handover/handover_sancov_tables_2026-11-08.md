# Sanitizer instrumentation gaps: shipped and next

**Status:** PR #320. Shipped `trace_cmpf`/`cmpd`, the pc-table and
control-flow dumps, the `funcs:` stats field and the hybrid CFG parse. Items
below are open, ranked by value per unit of work.

## Shipped

| Item | Where | Tests |
|------|-------|-------|
| gcc float/double compares: IEEE bits logged as 4/8-byte `CMP` | `afl_shim.c` layer 2 | `tests/test_trace_cmp_float.py` |
| pc-table + CFG dumped to `$__AFL_SANCOV_OUT` (O_EXCL, init-time only, load bias removed) | `afl_shim.c` (`__afl_sancov_*`), `adapters/sancov_tables.py` | `tests/test_sancov_tables.py` |
| CFG table → `FunctionCFG` (native < 4096 words, hybrid above) | `core/sancov_cfg.py` | same |
| Stats `funcs: hit/total` | `services/stats.py::_blocks_str`, `BlockBitmapShm.covered_at` | same |
| `--sancov=...,control-flow` accepted | `tools/build_targets.sh` | `tests/test_sancov_modes.py` |
| Sanitizer death callback: crash flush + distance tail before the runtime's exit (edge and preload-only builds) | `afl_shim.c` (`__afl_san_death`, `__afl_crash_flush`) | `tests/test_sanitizer_death_callback.py` |
| `__sanitizer_weak_hook_*` into the layer-1 recorders; override calls muted | `afl_shim.c` (`__afl_on_*`, `__afl_hook_mute`) | `tests/test_sanitizer_weak_hooks.py` |
| Allocation-size edges (`--alloc-sizes`) via `__sanitizer_install_malloc_and_free_hooks` | `afl_shim.c` (`__afl_alloc_*`), `core/shim_health.py` | `tests/test_alloc_size_signal.py` |

```
module ctor ── guard_init(G) ── pcs_init(P) ── cfs_init(C)
                   │               │               │
             guard region     first = G.first   (no pairing)
                   │               └──────┬────────┘
                   ▼                      ▼
           block bitmap SHM      $__AFL_SANCOV_OUT  (first process, O_EXCL)
                   │                      │
                   └──── stats funcs: ────┘      core/sancov_cfg ──> FunctionCFG (unwired)
```

## Next, ranked

### 1. Sanitizer `*_default_options` parity (MSAN / UBSAN / TSAN / LSAN)
- Why: only ASAN gets `halt_on_error=0:abort_on_error=0`, from a temp shim
  compiled at run time (`services/fuzzer.py` ~L5059, in-process ASAN only).
  MSAN/TSAN/UBSAN abort on the first report unless the env sets options, so
  in-process mode loses the process instead of recovering.
- Sketch, one generated shim exporting every runtime's hook:
  ```c
  const char *__asan_default_options(void)  { return "halt_on_error=0:abort_on_error=0:verify_asan_link_order=0"; }
  const char *__ubsan_default_options(void) { return "halt_on_error=0:print_stacktrace=1"; }
  const char *__msan_default_options(void)  { return "halt_on_error=0"; }  /* needs -fsanitize-recover=memory */
  const char *__tsan_default_options(void)  { return "halt_on_error=0:report_signal_unsafe=0"; }
  ```
- Wiring: move the inline C string from `services/fuzzer.py` into
  `adapters/shim_factory.py` (one builder, rule 35). Call it for every
  sanitizer `detect_elf_type` reports, not only ASAN.
- Tests: falsification (a UBSAN `.so` with two overflows in one input reports
  both); adversarial (MSAN built without `-fsanitize-recover`: the process
  still ends, and it is detected as a crash, not a hang).

### 2. ~~`__sanitizer_set_death_callback`~~ shipped (2026-11-08)
- Open: escaping a fatal report through the guard (`siglongjmp` from the
  callback) would make UBSAN/MSAN fatal errors recoverable in direct_lite.
  Unsafe as is: `Die()` runs inside the runtime's report scope, which holds
  its report lock, so the next report deadlocks. Needs the runtime's
  cooperation (or `halt_on_error=0`, item 1).
- Open: the runtime keeps one user callback. A target registering its own
  replaces the shim's. Detect it with a health counter (the callback bumps it;
  zero after a sanitizer death means it was displaced).

### 3. ~~`__sanitizer_weak_hook_*`~~ shipped (2026-11-08)
- Open: hooks in a `dlopen(RTLD_LOCAL)` target stay unbound under a shared
  runtime (libasan.so resolves them from the global scope at its own load).
- Why: libasan calls `__sanitizer_weak_hook_{memcmp,strcmp,strncmp,strcasecmp,strncasecmp,strstr,strcasestr,memmem}`
  from its interceptors. Code that reaches libc through the interceptor
  (other DSOs, `-fno-builtin` gaps) bypasses the shim's strong overrides.
- Sketch: thin forwarders into the existing layer-1 recorders, e.g.
  ```c
  void __sanitizer_weak_hook_memcmp(void *pc, const void *a, const void *b, size_t n, int r) {
      (void)pc;
      __AFL_CMP_GATE();
      __afl_cmplog_bytes(a, b, n, r);     /* existing layer-1 writer (~L4360) */
  }
  ```
- Dedup: an overridden memcmp reaching the hook too logs once:
  `__afl_cmplog_bytes` dedups on an operand fingerprint (`__afl_bytes_fp`,
  no pc). Per-call counters (`__AFL_CMP_COUNT`) would double-count; count only
  in the override, or only in the hook.
- Wiring: shim only; the records reuse the existing `CMP` format.
- Tests: memcmp from a second, uninstrumented `.so` linked against ASAN: the
  pair appears (fails today). Adversarial: one call yields exactly one record.

### 4. Compiler CFG into `analyzer_distance`
- Why: `_build_cfgs` / `_decode_cfgs` (`core/analyzers/analyzer_distance.py`
  ~L674) decode x86 per target function. The dump has exact successors and
  flags indirect calls, and costs no decode.
- Sketch:
  ```python
  def _build_cfgs(self):
      tables = self._sancov_tables  # passed in, see wiring
      cfs = [r for r in tables.records() if r.kind == SancovKind.CFS] if tables else []
      if cfs:
          self._cfgs.update(cfs_functions(cfs[0].words, cfs[0].bias, self._name_at))
          return
      self._decode_cfgs(sorted(target_funcs))  # unchanged fallback
  ```
- Block ends: the table holds starts only. Take `end` = next block start in
  address order within the function, or ask the decoder for the last block.
  Distance only needs block membership, so check `_addr_to_block` first.
- Wiring: `Fuzzer` owns `_sancov_tables` (services). Hand it to the distance
  analyzer through its constructor; core must not import adapters (rule 36b).
  The dump appears after the first exec, so distance setup must run after
  calibration, or re-run once.
- Tests: on one binary, distance with the dump equals distance with the
  decoder (rule 46 control: decoder vs decoder). A/B ledger row on a directed
  fuzzgoat target.

### 5. pc-table consumers beyond `funcs:`
- (a) `--report`: list functions never entered (pc-table entry vaddr →
  symbol). That makes the cold-function list actionable.
- (b) inline-8bit-counters builds get `first = NO_FIRST` (only guard regions
  pair). Pair with `__afl_sancov_regions` the same way, and add a counters
  bitmap or reuse the fold, if those builds should show `funcs:`.
- (c) 64-module cap (`__AFL_SANCOV_MAX_SEEN`): count the drops into a
  health counter so a silent cap shows up.

### 6. Own counters for `cmpf`/`cmpd`
- Today they count as `trace_cmp4`/`trace_cmp8`. Separate names need two new
  `__afl_cmp_names` slots, `cmp_shm.py` `_NAMES`, and a `CMPSHM02` magic so
  old fuzzer/new shim pairs fall back to the sidecars. Low value; do it with
  the next counter-segment change.

### 7. Rebuild and measure
- Rule 56: targets were not rebuilt (none in the authoring container).
  Rebuild fuzzgoat with `--sancov=trace-pc-guard,pc-table,control-flow`.
  Record binary-size and exec-speed deltas in the ledger (the pc-table A/B
  from 2026-09-27 was null; control-flow adds a data section only).
- Rule 52: calibrate on clang-built fuzzgoat.

## Environment gaps hit
- No `libclang-rt-18-dev`: 7 tests in `test_gcc_trace_pc`,
  `test_regression_shim_audit` and `test_regression_tracecmp_visibility` fail
  at link on master too (existing TODO "Vendor scripts: name a missing clang
  runtime").
- PyPI `impactguard` crashes without `tree-sitter` (`NameError: _log` in
  `languages/lib/shared.py`). `pip install tree-sitter tree-sitter-c
  tree-sitter-python` works around it.
- Hooks are not installed in a fresh clone: run `pre-commit install`.
