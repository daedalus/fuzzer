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

1. **Always follow existing conventions.** Before adding anything — a target, a script, a scheduler, a test fixture — find the closest existing example and match it: directory layout, file naming, function shape, flag names, error handling, comment style. Read the surrounding code first; do not invent a parallel way of doing something the repo already does. Concretely: vendored library sources go in `vendor/<lib>/` (gitignored) fetched by a `tools/vendor_<lib>.sh` script — never committed and never under `targets/`; new fuzz targets are wired into `tools/build_targets.sh` rather than built by hand; new schedulers register in `_OPERATOR_STRATEGY_NAMES` and follow the `select_op`/`record`/`bandit_stats` interface. If a convention appears wrong, fix it in one place for everything rather than working around it locally, and say so.
2. Never bypass the pre-commit hooks (`--no-verify`). Fix the warnings, then recommit.
3. Always fix impactguard breaking changes.
4. Always use clang, never gcc (build scripts prefer clang automatically).
5. Stop suggesting `use_direct_lite = False` to work around ASAN in `direct_lite` mode — debug the root cause instead.
6. Never commit binary files or corpus directories — build targets from source, keep corpus data local.
7. Before deleting code, find where it should be wired first; if not found, clean up. When removing code, stay strictly within the scope of the removal — do not remove unrelated code.
8. Do not nuke the repo.
9. All fuzz targets: compile with ASAN (`-fsanitize=address`) and AFL edge coverage via `afl_shim.c` (`-include src/fuzzer_tool/adapters/afl_shim.c`). Pre-compile library sources as `.o` files; link the shim only into the target wrapper.
10. Always create TODOs. Always commit and push after finishing a task.
11. Update `docs/DEEP_DIVE.md` with new features (the comprehensive reference). Update `README.md` only when adding or changing high-level capabilities visible in the quick-start or feature overview.
12. Op mutators have a single source of truth: `src/fuzzer_tool/core/operator_registry.py`'s `REGISTRY`. Register new operators there only — the dispatch table (`build_dispatch`), the per-input op list (`build_ops`), scheduler arming (`_register_arms`), and `OPERATOR_CATEGORIES` all derive from it. Never add operator names to the legacy `MUTATIONS`/`FORMAT_MUTATIONS`/`DICT_MUTATIONS` lists or hand-edit `OPERATOR_CATEGORIES`; schedulers discover ops through the services layer and never hardcode op lists.

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

0. If the target needs a library that isn't a system package, vendor it first: add `tools/vendor_<lib>.sh` modelled on `tools/vendor_lz4.sh` (download → extract to `vendor/<lib>/` → verify). `vendor/` is gitignored — never commit the sources, and never put them under `targets/`.
1. Write `targets/<name>_read.c` following the pattern of an existing target (e.g. `targets/png_read.c`): read the input from stdin/file, call the library's parse/decode entry point. Expose `fuzz_shm_run(const unsigned char *, size_t)` for in-process/`direct_lite` mode.
2. Compile with clang: `-fsanitize=address -include src/fuzzer_tool/adapters/afl_shim.c`; pre-compile library sources as `.o` files and link the shim only into the wrapper. `-include` applies to *every* `.c` on a command line, so passing library sources alongside the wrapper emits `__afl_map_shm`/`__afl_area`/`__afl_guarded_call` into each object and fails the link with multiple-definition errors — compile them in a separate pass (see `compile_lz4_objects` in `tools/build_targets.sh`).
3. For in-process mode, also build `<name>_read.so` (`-shared -fPIC`; link `-lasan` explicitly and use `-Wl,-Bsymbolic` when cmplog is on — see `tools/build_targets.sh`, and prefer adding the target there over hand-rolling the flags).
4. Verify with `nm`: `__afl` symbols present in the executable, `fuzz_shm_run` present in the `.so`; then run `tools/build_targets.sh` and confirm the target appears in the feature matrix.
5. Add a `dictionaries/` token file if the format has meaningful tokens.

### Add a new op mutator

1. Register the operator in `src/fuzzer_tool/core/operator_registry.py`: add its name to the
   category band in `_CATEGORIES` and, if it is gated (dictionary / markov / cem / grammar /
   cmplog / flag / per-input), an availability predicate in `_AVAILABLE`. Nothing else —
   `build_dispatch()`, `build_ops()`, `_register_arms()`, and `OPERATOR_CATEGORIES` derive
   from `REGISTRY`.
2. Add the `_op_<name>` handler on `OperatorEngine` (`services/operators.py`) — a registration
   without a handler raises at dispatch-build time.
3. Add a regression test in `tests/test_regression_operator_registry.py` (category placement,
   availability gating, dispatch coverage) if the operator is not already covered by the
   smoke tests.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format src/ tests/` / `ruff check src/ tests/` | Format / lint |
| `fuzzer-tool --help` | Show CLI help |
| `tools/build_targets.sh` | Build all fuzz targets (ASAN + cmplog by default; see the script's flag list) |
| `tools/vendor_lz4.sh` / `vendor_grep.sh` / `vendor_ffmpeg.sh` | Fetch vendored library sources into `vendor/` (required before building the matching targets) |
| `python tools/corpus_png.py --out corpus --download` | Generate PNG corpus |
| `tools/bench.sh` / `tools/bench_sweep.sh` | Config comparison / feature sweep |
| `lizard --CCN 15 -w .` | Cyclomatic complexity violations |
| `vulture --min-confidence 80 .` | Find duplicated code |
| `fuzzer-tool fuzz <target> -c -d <corpus> -n <iters> --profile-hotpath [--profile-out PATH]` | cProfile hotpath profile of the fuzz run (tottime/cumtime/ncalls tables; dump defaults to `/tmp/fuzzer_hotpath.prof`) |



## Layout

```
src/fuzzer_tool/
├── core/         # Domain logic: markov, schedulers/, shapley, ga, sanitizer, edge_tracker,
│                 #   operator_registry (canonical op-mutator registration + dispatcher),
│                 #   operator_categories (taxonomy derived from the registry), cmplog, grammar,
│                 #   bloom, elf, target_profiler, fast_json, chi_squared, rand_pool,
│                 #   mutations/<format>.py (structure-aware per-format mutators: png, jpeg,
│                 #   gif, webp, webm, zip, protobuf, …)
├── adapters/     # Process execution, filesystem ops, afl_shim.c / cmplog_shim.c / perf_shim.c
├── services/     # Orchestration: fuzzer.py, operators.py, seed_picker.py, runner.py,
│                 #   stats.py, corpus_manager.py, parallel.py, report.py
└── cli/          # CLI entry point (commands.py, __main__.py)

tools/            # build_targets.sh, vendor_<lib>.sh (ffmpeg/grep/lz4), corpus_png.py,
                  #   bench.sh, bench_sweep.sh, release.sh
targets/          # Fuzz target sources (*.c) — compiled binaries are never committed
dictionaries/     # Format token dicts (png.dict)
vendor/           # Vendored library sources — gitignored, fetched by tools/vendor_<lib>.sh.
                  #   FFmpeg 7.1.3, lz4 (+ zlib/libpng/libjpeg-turbo for trace-cmp builds)
docs/             # DEEP_DIVE.md (comprehensive reference), TODO.md, refs/ (agent reference files), per-feature docs
```

## Code Style

- Format: `ruff format`; lint: `ruff check`
- Docstrings: Google style
- Type hints: strict mypy
- Verify claims against code: before acting on behavior, type, or API shape, read the source — don't infer from names.
- Prefer array.array over Python lists for homogeneous numeric data to minimize memory overhead, and only use lists when arrays are unsuitable.
- Prefer DP over recursive functions.
- Prefer hoist `len(data)` out of scan loops like `n = len(data); while i < n` instead of `while i < len(data)`; the same logic applies to `for` loops

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
