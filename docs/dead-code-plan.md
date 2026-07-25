# Dead Code Plan — fuzzer-tool

Generated from codebase audit. All candidates verified by grepping entire `src/` directory for zero callers.

---

## Core modules — 22 items

### Unused functions

| File:Line | Name | Notes |
|-----------|------|-------|
| `src/fuzzer_tool/core/similarity.py:54` | `hamming_distance_padded` | Zero callers |
| `src/fuzzer_tool/core/similarity.py:157` | `normalize_frame` | Zero callers (only used by `crash_signature_similarity` in same file, which is itself unused) |
| `src/fuzzer_tool/core/similarity.py:366` | `edit_script_summary` | Zero callers outside tests |
| `src/fuzzer_tool/core/similarity.py:503` | `find_nearest_bytes` | Zero callers outside tests |
| `src/fuzzer_tool/core/mutations.py:498` | `could_be_bitflip` | AFL++ duplicate-elimination helper, never imported |
| `src/fuzzer_tool/core/mutations.py:535` | `could_be_arith` | AFL++ duplicate-elimination helper, never imported |
| `src/fuzzer_tool/core/mutations.py:599` | `could_be_interest` | AFL++ duplicate-elimination helper, never imported |
| `src/fuzzer_tool/core/mutations.py:762` | `splice_diff_located` | Improved splice variant, never called |
| `src/fuzzer_tool/core/mutations.py:884` | `radamsa_mutate_num` | Radamsa-style number mutation, never called |
| `src/fuzzer_tool/core/elf.py:81` | `find_load_segment` | ELF helper, never called from services |
| `src/fuzzer_tool/core/schedules.py:305` | `compute_mean_log_n_fuzz` | COE scheduling utility, never called |
| `src/fuzzer_tool/core/rq_encodings.py:459` | `encoders_summary` | Never called (only `generate_mutations` in same file is used) |

### Unused constants/data

| File:Line | Name | Notes |
|-----------|------|-------|
| `src/fuzzer_tool/core/grammar.py:260` | `GRAMMARS` | Only used by `load_grammar()` in same file; never imported externally |
| `src/fuzzer_tool/core/grammar.py:725` | `_IHDR_TYPES` | Leftover from PNG format-aware grammar, never used |
| `src/fuzzer_tool/core/grammar.py:748` | `_IHDR_BIT_DEPTHS` | Leftover, never used |
| `src/fuzzer_tool/core/grammar.py:757` | `_IHDR_COLOR_TYPES` | Leftover, never used |
| `src/fuzzer_tool/core/bmp_mutations.py:29` | `DIB_V5HEADER` | Defined but never referenced |
| `src/fuzzer_tool/core/gzip_mutations.py:21` | `FTEXT` | Defined but never referenced |

### Unused entire modules/classes

| File:Line | Name | Notes |
|-----------|------|-------|
| `src/fuzzer_tool/core/qea.py:98` | `QEAIndividual` | Entire QEA module never imported anywhere |
| `src/fuzzer_tool/core/qea.py:349` | `QEALifecycle` | Entire QEA module never imported anywhere |
| `src/fuzzer_tool/core/colorizer.py:41` | `Colorizer` | Only `CmplogColorizer` is used; `Colorizer` itself is dead |
| `src/fuzzer_tool/core/renyi.py:225` | `CoverageSpectrumAnalyzer` | Exported in `__init__` but never instantiated |

---

## Services/Adapters — 17 items

### Dead adapter methods

| File:Line | Name | Notes |
|-----------|------|-------|
| `src/fuzzer_tool/adapters/shm.py:51` | `CoverageEntry` | NamedTuple only used by dead `read_entries()` |
| `src/fuzzer_tool/adapters/shm.py:120` | `ShmCoverage.read_bitmap()` | Zero callers (only `InProcessRunner.read_bitmap` is used) |
| `src/fuzzer_tool/adapters/shm.py:128` | `ShmCoverage.read_entries()` | Zero callers |
| `src/fuzzer_tool/adapters/shm.py:165` | `ShmCoverage.get_edge_bitmap_view()` | Dead call chain end-to-end |
| `src/fuzzer_tool/adapters/shm.py:191` | `ShmCoverage.reset()` | Zero callers (only `reset_edge_map` is used) |
| `src/fuzzer_tool/adapters/shm.py:267` | `ShmCoverage.commit_snapshot()` | Zero callers |
| `src/fuzzer_tool/adapters/shm.py:284` | `ShmCoverage.record_edge()` | Test-only, zero production callers |

### Dead service methods

| File:Line | Name | Notes |
|-----------|------|-------|
| `src/fuzzer_tool/services/fuzzer.py:93` | `_write_and_close()` | Module-level function, zero callers (separate one in runner.py is used) |
| `src/fuzzer_tool/services/fuzzer.py:1433` | `Fuzzer._invalidate_agg_cache()` | Zero callers; callers set `_agg_cache_valid = False` directly |
| `src/fuzzer_tool/services/fuzzer.py:2120` | `Fuzzer._get_current_edge_bitmap()` | Zero callers |
| `src/fuzzer_tool/services/fuzzer.py:2131` | `Fuzzer._get_edge_bitmap_view()` | Zero callers |
| `src/fuzzer_tool/services/stats.py:409` | `StatsReporter.get_current_edge_bitmap()` | Only caller is dead `_get_current_edge_bitmap()` |
| `src/fuzzer_tool/services/stats.py:443` | `StatsReporter.get_edge_bitmap_view()` | Only caller is dead `_get_edge_bitmap_view()` |
| `src/fuzzer_tool/services/ptrace_coverage.py:425` | `PtraceCoverage.remove_breakpoints()` | Zero callers |
| `src/fuzzer_tool/services/forkserver.py:225` | `ForkserverRunner.stderr_output()` | Zero callers |

### Dead module-level code

| File:Line | Name | Notes |
|-----------|------|-------|
| `src/fuzzer_tool/services/report.py:42` | `_format_ci()` | Zero callers (only `_format_ci_inline` is used) |
| `src/fuzzer_tool/services/report.py:335` | unreachable `return` | Duplicate return statement, unreachable |

### Dead module (entire file)

| File | Name | Notes |
|------|------|-------|
| `src/fuzzer_tool/services/differential.py` | `diff_run()`, `DifferentialTracker` | Never imported by anything |

---

## Priority ranking (most actionable first)

### P0 — Remove entirely (never used, no risk)

1. `qea.py` — entire module dead
2. `differential.py` — entire module dead
3. `_get_current_edge_bitmap()` + `StatsReporter.get_current_edge_bitmap()` — dead call chain
4. `_get_edge_bitmap_view()` + `StatsReporter.get_edge_bitmap_view()` — dead call chain
5. `_invalidate_agg_cache()` — defined but callers bypass it
6. `could_be_bitflip` / `could_be_arith` / `could_be_interest` — never imported
7. `splice_diff_located` — never called
8. `radamsa_mutate_num` — never called
9. `hash_data_crypto()` — never called
10. `ShmCoverage.read_entries()` + `CoverageEntry` — never called
11. `ShmCoverage.get_edge_bitmap_view()` — dead call chain
12. `ShmCoverage.reset()` — never called (only `reset_edge_map` used)
13. `ShmCoverage.commit_snapshot()` — never called
14. `ShmCoverage.read_bitmap()` — never called on ShmCoverage
15. `_write_and_close()` in fuzzer.py — dead duplicate
16. `remove_breakpoints()` — never called
17. `ForkserverRunner.stderr_output()` — never called
18. `_format_ci()` — never called
19. `encoders_summary()` — never called
20. `find_load_segment()` — never called
21. `compute_mean_log_n_fuzz()` — never called
22. `find_nearest_bytes()` — never called outside tests
23. `hamming_distance_padded()` — never called outside tests
24. `edit_script_summary()` — never called outside tests
25. `normalize_frame()` — only used by dead `crash_signature_similarity` chain

### P1 — Remove unused constants

26. `_IHDR_TYPES`, `_IHDR_BIT_DEPTHS`, `_IHDR_COLOR_TYPES` in grammar.py
27. `DIB_V5HEADER` in bmp_mutations.py
28. `FTEXT` in gzip_mutations.py

### P2 — Investigate before removing (may have test coverage or be API surface)

29. `Colorizer` class — test-only? Check test files
30. `CoverageSpectrumAnalyzer` — exported in `__init__`, might be public API
31. `GRAMMARS` dict — used by `load_grammar()` internally
32. `ShmCoverage.record_edge()` — docstring says "for tests only"
33. `report.py:335` unreachable return — trivial fix

---

## Impact estimate

- **~300-400 lines** of dead code across core, services, adapters
- **2 entire modules** (`qea.py`, `differential.py`) never imported
- **2 dead call chains** in stats/fuzzer (4 methods total)
- **5 unused AFL++ helper functions** in mutations.py
- Removing all P0 items would reduce `src/` by ~3% with zero risk
