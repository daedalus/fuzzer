# Dead Code Plan — fuzzer-tool

Generated from codebase audit. All candidates verified by grepping entire `src/` directory for zero callers.

**Last updated: 2026-07-27** — Most items resolved (wired up or deleted).

---

## Status

| Status | Count |
|--------|-------|
| ⚡ Wired up (now live) | 10 items |
| ❌ Deleted (truly dead) | 15 items |
| ❓ Document corrections (were alive all along) | 3 items |
| **Remaining** | **0 items** |

---

## ⚡ Wired up (code kept, connected to callers)

| Item | Where wired |
|------|-------------|
| `compute_mean_log_n_fuzz()` | → COE scheduling: `SeedScorer.score()` + `_refresh_agg_cache()` |
| `differential.py` module | → `fuzz --differential TARGET_B` CLI flag |
| `CoverageSpectrumAnalyzer` | → `StatsReporter` report dict (dominance_ratio, hot_edge_fraction) |
| `splice_diff_located` | → New mutation operator in `MUTATIONS` + dispatch table |
| `radamsa_mutate_num` | → New mutation operator in `MUTATIONS` + dispatch table |
| `could_be_bitflip/arith/interest` | → `_is_deterministically_redundant()` in `_op_havoc` dedup check |
| `encoders_summary` | → Cmplog init debug output |
| `hamming_distance_padded` | → `EdgeTracker.compute_byte_level_diversity()` |
| `edit_script_summary` | → `find_nearest_corpus()` return tuple |
| `find_nearest_bytes` | → `SeedPicker._pick_by_similarity()` |

## ❌ Deleted (truly dead)

### Source deletions

| File | Items |
|------|-------|
| `src/fuzzer_tool/core/gzip_mutations.py` | `FTEXT` constant |
| `src/fuzzer_tool/core/bmp_mutations.py` | `DIB_V5HEADER` constant |
| `src/fuzzer_tool/core/grammar.py` | `_IHDR_TYPES`, `_IHDR_BIT_DEPTHS`, `_IHDR_COLOR_TYPES` |
| `src/fuzzer_tool/services/report.py` | `_format_ci()` (replaced by `_format_ci_inline`) |
| `src/fuzzer_tool/services/fuzzer.py` | `_write_and_close()`, `_invalidate_agg_cache()`, `_get_current_edge_bitmap()`, `_get_edge_bitmap_view()` |
| `src/fuzzer_tool/services/stats.py` | `get_current_edge_bitmap()`, `get_edge_bitmap_view()` |
| `src/fuzzer_tool/services/ptrace_coverage.py` | `remove_breakpoints()` |
| `src/fuzzer_tool/adapters/forkserver.py` | `stderr_output()` |
| `src/fuzzer_tool/adapters/shm.py` | `CoverageEntry`, `read_bitmap()`, `read_entries()`, `get_edge_bitmap_view()`, `reset()`, `commit_snapshot()` |

### Test deletions

| Test file | Tests removed |
|-----------|---------------|
| `tests/test_shm.py` | `test_read_bitmap_returns_entry_bytes`, `test_read_entries_empty_after_reset`, `test_commit_snapshot`, `test_read_bitmap_still_returns_only_edge_bytes` |
| `tests/test_forkserver.py` | `test_stderr_output_empty`, `test_stderr_output_capped` |

## ❓ Document corrections (were alive all along)

| Claimed dead | Actual status |
|--------------|---------------|
| `qea.py` module | Alive — wired behind `--qea` flag |
| `crash_signature_similarity` | Alive — used by `crash_metadata.py`, `filesystem.py` |
| `normalize_frame` | Alive — used within `similarity.py` by live callers |
