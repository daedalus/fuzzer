# TODO — fuzzer-tool Roadmap

> **Status note**: This roadmap tracks aspirational and in-progress work. Items without [x] are still pending.

## Coverage & Instrumentation
- [ ] **Forkserver on default execution path** — `ForkserverRunner` exists but is commented out in the default subprocess path; every run still pays `posix_spawn` + full ELF load. Re-enable for standalone executables to recover the 2–10× throughput gap.
- [ ] **Bounded probe window for `__afl_map_edge`** — linear-probe cost is O(map_size) on saturated tables. Bound to 8–16 slots and count drops via the SHM header so the trade is observable.
- [ ] **Hit-count bucketing on the SHM path** — `count_class.py` is already implemented but only used by the ptrace fallback. Wire it into `ShmCoverage._check_new_coverage` so loop-count-guarded branches become visible.
- [ ] **`favored` / `cull_queue` minimal-set-cover** — power schedules always run in unfavored mode because `favored` is never computed. Implement `cull_queue` over `EdgeTracker.seed_edges` and pass `favored` through the scheduler.
- [ ] **Deterministic stages + SkipDet** — the effector map is dead because no code walks the deterministic operators systematically across a seed. Add a per-seed deterministic pass gated by `SkipDetector`.
- [ ] **Fast path empty-edge-set bug** — `_check_new_coverage` returns `(False, set())` on unchanged input, but callers cache that as the real edge set, making the next diff report all edges as new.
- [ ] **Call stack coverage** — distinguish `f()→g()` from `h()→g()` by encoding caller context into the edge ID, not just `prev_loc ^ cur_loc`. Would improve edge resolution for shared-library targets.
- [ ] **Sanitizer coverage** — `-fsanitize-coverage=trace-pc-guard` support via `--clang-scov`, with auto-detection of sancov counters in `.so` targets.
- [ ] **Cmplog/comparison coverage** — symbol-based (`cmplog_shim.c`, LD_PRELOAD) and compiler-IR (`trace-cmp`, `tracecmp_shim.c`) both implemented.

## Mutation
- [ ] **Per-format tuning of the regularity band** — the operators are currently offered unconditionally (except `invariant_break`). Several are format-shaped in practice: `spectral_peak` matters for DCT codecs, `degenerate_geometry` for vector/mesh parsers, `rank_deficient` for erasure coders. A sniffer gate like `_FORMAT_SNIFFERS` would stop them burning budget on targets that cannot use them.

## Scheduling
- [x] **Fluctuation theorems for fuzzing** (IMPLEMENTED 2026-08-13) — Jarzynski/Crooks relations implemented for mutation trajectories. Work functional `w_i = -log(max(p_i, ε))` over operator-selection probabilities; free-energy estimator `ΔF̂` printed via `StatsReporter.print_stats`. Phase 1 is diagnostics-only: estimates are not fed back into scheduling. CLI flag: `--fluctuation-theorems`. See `src/fuzzer_tool/core/fluctuation.py`, `services/fuzzer.py`, `services/stats.py`.
- [ ] **Collaborative scheduling across parallel workers** — parallel workers currently sync corpus but don't coordinate scheduling decisions. Could share exploration/exploitation state.

## Crash Analysis
- [ ] **Root cause diff** — show minimal byte diff from nearest non-crashing input to root-cause bytes.

## Performance
- [ ] **`_apply_single_mutation` havoc `max_len` enforcement** — havoc doesn't enforce `max_len` strictly (allows +1 byte per insert, up to +8 total).

## Infrastructure
- [ ] **Dockerfile** for reproducible builds and CI
- [ ] **Structured logging** (e.g. `--log-json`) for machine-parseable output
- [ ] **`fuzzer-tool-asan` wrapper** — CLI wrapper that sets `LD_PRELOAD=libasan.so.8` and exec's into the real fuzzer (mentioned in ASAN-LIMITATION.md but not yet generated as a installable entry point)
- [ ] **Persist `invocation` into `state.json`** — `fuzzer.invocation` (`sys.argv`, captured in `cmd_fuzz` for the report exec lines) is not saved on shutdown; a `--resume` run therefore reports only the resumed command, not the original one. Saving it into `state.json` would let reports on resumed sessions carry the original invocation.

## Integer-Modulus Checksum Recovery (follow-ons)
- [ ] **Weighted-sum multiplier sweep is a fixed candidate list** (`_MULTIPLIER_CANDIDATES`) — a target using an unlisted multiplier is missed entirely. Recovering `k` properly means root-finding mod `N`; Coppersmith's bound (`N^(1/deg)`) is useless at realistic data lengths, so the list is the pragmatic answer for now. Consider deriving candidates from cmplog constants instead of hardcoding.
- [ ] **`_extract_zlib_adler_pairs` only fires on valid streams** — `decompressobj` raises on an Adler mismatch, so mutated PNGs yield no pair. Pairs therefore come only from corpus seeds and successful recompressions. Reading the trailer without validating would widen the source but needs a raw-deflate path.
- [ ] **Fletcher-32 word endianness is swept, not detected** — both LE and BE are tried and verification arbitrates. Fine, but it doubles the general-path work for that family.
- [ ] **No format-aware patcher for integer checksums** — `_op_crc_learn` patches only the generic trailing field when an integer model is active. A real zlib/IDAT Adler patcher belongs in the `recompress_zlib` mutator, not here.
- [ ] **`field_constraints.py` bounded-integer pre-pass** (handover §1, deprioritized) — z3 is already fast on these small bitwidth systems, so the win is thin. Revisit only if the integer-checksum pattern proves out.

## Pending Bugs
- [ ] `_apply_single_mutation` havoc doesn't enforce `max_len` strictly (allows +1 byte per insert, up to +8 total)
- [ ] `parse_dict_line` triple-encode chain fragile for bytes > 0x7F
