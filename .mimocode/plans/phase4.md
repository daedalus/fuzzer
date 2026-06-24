# Plan: Items 4.1–4.5

Five remaining features from the original request, ordered by impact:

## 4.1 Per-seed edge bitmask for coverage-guided scheduling
**Goal**: Track which coverage edges each seed contributes. Deprioritize seeds fully subsumed by others.

**Files**:
- `services/fuzzer.py` — `seed_meta`, `_init_seed_metadata`, `_pick_seed`, `fuzz_one`
- New: `core/edge_tracker.py` — `EdgeTracker` class

**Approach**:
- After each execution that produces new coverage, record which edges (from SHM bitmap or ptrace edge_map) are now hit
- Store as a `set[int]` per seed in `seed_meta["edge_set"]`
- In `_pick_seed()`, compute subsumption: seed A is subsumed if all its edges are covered by other seeds. Subsumed seeds get 0.1x weight
- This is purely additive — doesn't change existing power scheduling, just adds a subsumption penalty

## 4.2 Full ASAN output parsing enhancement
**Goal**: Parse shadow memory descriptions, allocation/deallocation stacks, access size/type from raw ASAN output.

**Files**:
- `core/sanitizer.py` — already has regex patterns, verify they work correctly
- `core/crash_metadata.py` — `format_sidecar()` already outputs these fields

**Approach**: The regex patterns were added in Phase 1. Verify they capture real ASAN output correctly with a test using a sample ASAN report. Fix any regex gaps.

## 4.3 Coverage-corpus minimization integration
**Goal**: Auto-minimize corpus when it exceeds --max-corpus N.

**Files**:
- `services/fuzzer.py` — `save_to_corpus`, `run()`
- CLI: `--max-corpus N` flag

**Approach**:
- Add `--max-corpus N` parameter to Fuzzer and CLI
- In `save_to_corpus()`, if `len(self.corpus) > max_corpus`, trigger `_minimize_corpus_inline()`
- Inline minimization: deduplicate by content hash (cheap) + remove seeds whose edge_set is subsumed
- Don't call the full `minimize_corpus` (which replays targets) — just do hash dedup + subsumption check

## 4.4 Stats aggregation in parallel mode
**Goal**: Aggregate per-worker stats into unified summary.

**Files**:
- `services/parallel.py` — already has result_queue, add live aggregation

**Approach**:
- In the parent loop, periodically drain `result_queue` and accumulate totals
- Print aggregated stats alongside per-worker stats every N seconds
- Final summary already works; just needs live accumulation

## 4.5 Sancov shim coverage logging integration
**Goal**: Wire sancov LD_PRELOAD shim into --coverage-log for Clang-instrumented binaries.

**Files**:
- `services/fuzzer.py` — `_run_target`, `_append_coverage_log`
- `adapters/shim_factory.py` — `build_sancov_shim()`

**Approach**:
- When sancov shim is active and coverage_log is set, read the bitmap file after each execution
- Count new edges (bytes > 0 in bitmap) and append to coverage_log
- This extends the existing coverage_log to work with Clang-instrumented targets

## Verification
1. `pytest tests/ -q` — all existing tests pass
2. `ruff check src/ tests/` — lint clean
3. Integration test: compile a test target, fuzz with --max-corpus, verify corpus stays bounded
4. Parallel mode test: run with --jobs 2, verify aggregated stats
