# AGENTS.md — fuzzer-tool

Coverage-guided binary fuzzer: ASAN/MSAN/TSAN/UBSAN detection, dictionary mutations,
Markov chain generation, Monte Carlo optimization, format-aware grammar mutations, and
state persistence. CLI tool that fuzzes arbitrary binaries (see `fuzzer-tool --help`).

The fuzzer executes attacker-controlled input against instrumented targets and parses
the targets' own binaries — any bug in this tool's parsing/process code is a bug in the
fuzzer, not just the target.

## References (read on demand)

| File | Open when |
|------|-----------|
| `docs/refs/bug-classes.md` | Touching process/signal handling, timeouts, ptrace, concurrency, resource cleanup, hashing/identity, caching, ELF/low-level parsing, numeric edge cases, state persistence, dispatch tables, error swallowing, .so symbol visibility, widely-used return-value APIs, or test mocks. Also carries the regression-testing rules. |
| `docs/refs/architecture.md` | Working inside coverage/SHM internals, the AFL shim, `--no-shm`/`--deep-coverage`, the Elo meta-scheduler, or state persistence (`state.json` / `edge_tracker.json` / `markov.json`). |

## Hard Rules

1. Never bypass the pre-commit hooks (`--no-verify`). Fix the warnings, then recommit.
2. Always fix impactguard breaking changes.
3. Always use clang, never gcc (build scripts prefer clang automatically).
4. Stop suggesting `use_direct_lite = False` to work around ASAN in `direct_lite` mode — debug the root cause instead.
5. Never commit binary files or corpus directories — build targets from source, keep corpus data local.
6. Before deleting code, find where it should be wired first; if not found, clean up. When removing code, stay strictly within the scope of the removal — do not remove unrelated code.
7. Do not nuke the repo.
8. All fuzz targets: compile with ASAN (`-fsanitize=address`) and AFL edge coverage via `afl_shim.c` (`-include src/fuzzer_tool/adapters/afl_shim.c`). Pre-compile library sources as `.o` files; link the shim only into the target wrapper.
9. Always create TODOs. Always commit and push after finishing a task.
10. Update `docs/DEEP_DIVE.md` with new features (the comprehensive reference). Update `README.md` only when adding or changing high-level capabilities visible in the quick-start or feature overview.

## Corpus Rules

- **Always improve the corpus, never delete it.** Corpus files represent discovered coverage and crash triggers. Only add new inputs, never remove existing ones. Use `fuzzer-tool minimize` to prune redundancies — removed inputs are moved to `corpus/pruned/`, not deleted.
- **Do not clean the corpus between runs.** The corpus directory accumulates discovered inputs across sessions. `rm -rf corpus/*` destroys coverage history and forces rediscovery from scratch. Always use `--resume` to continue. When generating a new corpus (e.g. `corpus_png.py`), write to a fresh directory, not an existing one.

## Workflows

### Finish a task

1. Create a TODO tracking the work.
2. Make surgical changes; every fixed bug ships with a regression test (`test_regression_<brief_description>`).
3. Run the full suite: `pytest` — must be green.
4. `ruff format src/ tests/` and `ruff check src/ tests/`.
5. Update docs per Hard Rule 10.
6. `git commit` — pre-commit hooks run ruff (with `--fix`), the full pytest suite, and impactguard. Fix any warnings and recommit; never `--no-verify`. Then `git push`.

### Add a new fuzz target

1. Write `targets/<name>_read.c` following the pattern of an existing target (e.g. `targets/png_read.c`): read the input from stdin/file, call the library's parse/decode entry point.
2. Compile with clang: `-fsanitize=address -include src/fuzzer_tool/adapters/afl_shim.c`; pre-compile library sources as `.o` files and link the shim only into the wrapper.
3. For in-process mode, also build `<name>_read.so` (`-shared -fPIC`; link `-lasan` explicitly and use `-Wl,-Bsymbolic` when cmplog is on — see `tools/build_targets.sh`, and prefer adding the target there over hand-rolling the flags).
4. Verify with `nm`: `__afl` symbols present in the executable, `fuzz_shm_run` present in the `.so`; then run `tools/build_targets.sh` and confirm the target appears in the feature matrix.
5. Add a `dictionaries/` token file if the format has meaningful tokens.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format src/ tests/` / `ruff check src/ tests/` | Format / lint |
| `fuzzer-tool --help` | Show CLI help |
| `tools/build_targets.sh` | Build all fuzz targets (ASAN + cmplog by default; see the script's flag list) |
| `python tools/corpus_png.py --out corpus --download` | Generate PNG corpus |
| `tools/bench.sh` / `tools/bench_sweep.sh` | Config comparison / feature sweep |
| `lizard --CCN 15 -w .` | Cyclomatic complexity violations |
| `vulture --min-confidence 80 .` | Find duplicated code |

## Layout

```
src/fuzzer_tool/
├── core/         # Domain logic: markov, schedulers/, shapley, ga, sanitizer, edge_tracker,
│                 #   cmplog, grammar, bloom, elf, target_profiler, fast_json,
│                 #   chi_squared, rand_pool, mutations/<format>.py (structure-aware
│                 #   per-format mutators: png, jpeg, gif, webp, webm, zip, protobuf, …)
├── adapters/     # Process execution, filesystem ops, afl_shim.c / cmplog_shim.c / perf_shim.c
├── services/     # Orchestration: fuzzer.py, operators.py, seed_picker.py, runner.py,
│                 #   stats.py, corpus_manager.py, parallel.py, report.py
└── cli/          # CLI entry point (commands.py, __main__.py)

tools/            # build_targets.sh, corpus_png.py, bench.sh, bench_sweep.sh, release.sh
targets/          # Fuzz target sources (*.c) — compiled binaries are never committed
dictionaries/     # Format token dicts (png.dict)
vendor/           # Vendored FFmpeg 7.1.3 (+ zlib/libpng/libjpeg-turbo for trace-cmp builds)
docs/             # DEEP_DIVE.md (comprehensive reference), TODO.md, refs/ (agent reference files), per-feature docs
```

## Code Style

- Format: `ruff format`; lint: `ruff check`
- Docstrings: Google style
- Type hints: strict mypy
- Verify claims against code: before acting on behavior, type, or API shape, read the source — don't infer from names.
- Prefer array.array over Python lists for homogeneous numeric data to minimize memory overhead, and only use lists when arrays are unsuitable.
- Prefer DP over recursive functions.

## Testing

- **Run the full test suite after changes** — `pytest` must pass before a change is complete.
- **No hardcoded counts in tests.** Use `>=` for minimum bounds, not `==` — operators and features are added frequently and `assert len(X) == N` breaks on every addition.
- **Regression tests are mandatory** for every fixed bug (`test_regression_<description>`); equivalence assertions must derive one side independently of the code under test (see `docs/refs/bug-classes.md` §Testing).
- **Hash functions must be consistent.** When matching filenames against content (corpus eviction, dedup), use `hash_data()` from `fuzzer_tool.adapters.filesystem` — not `hashlib.sha256()` directly. `hash_data()` prefers xxhash when installed; hardcoding SHA-256 causes silent data loss.
- **Cache invalidation on method renames.** When renaming a method with side-effect calls (e.g. `_invalidate_*_cache()`), grep for all call sites — a renamed method silently drops its callers' invalidation hooks.

## Development

```bash
# Setup
pip install -e ".[test]"

# Test
pytest

# Format / lint
ruff format src/ tests/
ruff check src/ tests/
```
