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

0. Always make surgical changes.
1. **Always follow existing conventions.** Before adding anything — a target, a script, a scheduler, a test fixture — find the closest existing example and match it: directory layout, file naming, function shape, flag names, error handling, comment style. Read the surrounding code first; do not invent a parallel way of doing something the repo already does. Concretely: vendored library sources go in `$FUZZ_VENDOR_ROOT/<lib>/` (default `~/fuzzing/vendoring/<lib>/`, legacy `vendor/<lib>/` via `--in-tree-vendor`), fetched by a `tools/vendor_<lib>.sh` script — never committed and never under the build root (`$FUZZ_BUILD_ROOT/<lib>/`, default `~/fuzzing/targets/`, legacy `targets/` via `--in-tree-targets`); new fuzz targets are wired into `tools/build_targets.sh` rather than built by hand; new schedulers register in `_OPERATOR_STRATEGY_NAMES` and follow the `select_op`/`record`/`bandit_stats` interface. If a convention appears wrong, fix it in one place for everything rather than working around it locally, and say so.
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
13. Only run the full pytest suite if a file in the codebase was modified.
14. Always after developing a new function and verify its correctness try to vectorize it after the fact, keep the fastest version.
15. Always use the existing pickle machinery, avoid json.
16. For random always use the prng in `src/fuzzer_tool/core/rand_pool.py`.
17. Do not create artifacts in the source codebase dir.
18. Always create corpus on ~/.
19. Always make sure when creating a new functionality that is wired-up where needed.
20. try except pass is a bad pattern.
21. Always excersice higiene: every test must clean up their mess.
22. If you solved the halting problem you are allowed to run tests that the user is saying they are hanging otherwise read the code to see what it does.
23. For every new functionality always add one falsification test and one adversarial test.
24. Always use subagents for multiple file exploring or long tasks.
25. Without any question when the user instructs you to git and commit now, you obey, create the commit message, commit and push. no questions asked and no deliberation.
26. When writing something intended for human consumption, (comment, commit message, reply to prompt) use as few words as possible. Pick every word meticulously to reduce the volume to a strict minimum. Be down to the point. Less is more.
27. Avoid superlatives and praise. Stop telling me I am absolutely right. Give me the cold hard truth.
28. Avoid magic numbers and strings by extracting recurring or meaningful values into descriptive constants (const) or enums. Keep self-explanatory, one-off values inline to avoid clutter. If a value comes from a spec (e.g. HTTP 200 OK), use a constant regardless.
29. Reduce code indentation. Avoid Arrow Anti-Pattern. Leverage early return and continue.
30. Keep function names short. Less than 30 characters.
31. Use enums instead of booleans for function parameters.
32. Let the reader of the code breathe. Add empty lines between logical blocks of code.
33. Add a small, to the point, comment to explain *what* the block does and *why*. Use examples when possible. Propose ASCII drawings to explain complete systems.
34. Treat member visibility changes as a breaking design shift. Keep all fields and functions private unless external access is strictly required by the design. Prompt the user for explicit approval before changing any access modifier from private to internal or public.
35. Program to levels of abstraction. Lower-level mechanics must be encapsulated in a dedicated driver/abstraction layer. Expose clean, high-level APIs to the rest of the application so calling code works with domain concepts, not raw implementation details.
36. Don't touch blocks of code unrelated to the feature you implement. e.g. Don't add comments to a block of code if you did not create it or modify it. As much as possible try to minimize the number of changed lines when implementing a feature.
36. Strictly adhere to the layered boundary hierarchy: each layer may only communicate with its immediate neighbor directly below it. Never "punch holes" through layers (e.g., controllers or UI components must never directly call database queries, raw hardware drivers, or low-level network clients; always route through the intermediate service/abstraction layer).
37. If the prompt indicates that a bug is being fixed, don't write the fix right away. First write the test. Observe it failing. Then write the fix. And observe the test passing.
38. When implementing a new feature don't write it right away. First write the test. Observe it failing. Then write the feature. And observe the test passing (Test driven development).
39. No retry-until-random-hit loops in tests (`for _ in range(N): if cond: break/found=True`). This tests luck, not behavior — it can pass while the code is broken and fails unreproducibly when it doesn't. Inject a scripted/fake RNG that deterministically drives the exact call sequence and assert the exact output. See `docs/refs/bug-classes.md` §Testing.
40. Every scheduler armed through `_register_arms` (`src/fuzzer_tool/services/fuzzer.py`) must declare an explicit class-level `supports_priors` bool: `True` only when its `init_arm()` accepts an informative `(prior_alpha, prior_beta)` override. `_register_arms` gates priors behind `getattr(scheduler, "supports_priors", False)`, so a scheduler that omits the flag silently discards format-operator priors instead of failing loudly. Declare it directly after the class docstring with a one-line reason, following `monte_carlo.py` / `exp3.py`.
41. Always check that the new features implemented or bug fixes do not introduce speed penalty regressions.
42. When implementing new features always think of edge cases for the tests.
43. Always maintain succinct, brief, down to the point and updated documentation.
44. When adding, removing or wiring a new subsystem update `architecture.png`.
45. Every time a new source `file.py` is created run `lizard --CCN=15 file.py` and fix warnings.

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

0. If the target needs a library that isn't a system package, vendor it first: add `tools/vendor_<lib>.sh` modelled on `tools/vendor_lz4.sh` (download → extract to `$FUZZ_VENDOR_ROOT/<lib>/`, default `~/fuzzing/vendoring/<lib>/`; legacy `vendor/<lib>/` with `--in-tree-vendor` → verify). The vendored source root is gitignored — never commit the sources, and never put them under the build root (`$FUZZ_BUILD_ROOT/`, default `~/fuzzing/targets/`; legacy `targets/` with `--in-tree-targets`).
1. Write `targets/<name>_read.c` following the pattern of an existing target (e.g. `targets/png_read.c`): read the input from stdin/file, call the library's parse/decode entry point. Expose `fuzz_shm_run(const unsigned char *, size_t)` for in-process/`direct_lite` mode.
2. Compile with clang: `-fsanitize=address -include src/fuzzer_tool/adapters/afl_shim.c`; pre-compile library sources as `.o` files and link the shim only into the wrapper. `-include` applies to *every* `.c` on a command line, so passing library sources alongside the wrapper emits `__afl_map_shm`/`__afl_area`/`__afl_guarded_call` into each object and fails the link with multiple-definition errors — compile them in a separate pass (see `compile_lz4_objects` in `tools/build_targets.sh`).
3. For in-process mode, also build `<name>_read.so` (`-shared -fPIC`; link `-lasan` explicitly and use `-Wl,-Bsymbolic` when cmplog is on — see `tools/build_targets.sh`, and prefer adding the target there over hand-rolling the flags).
4. Verify with `nm`: `__afl` symbols present in the executable, `fuzz_shm_run` present in the `.so`; then run `tools/build_targets.sh` and confirm the target appears in the feature matrix.
5. Add a `dictionaries/` token file if the format has meaningful tokens.
6. If the format already has a structure-aware mutator, check its sniffer predicate in `core/operator_registry.py` before choosing an input layout. A mode-selector prefix byte (as in `lz4_read.c`) shifts the magic off offset 0, the sniffer stops firing, and the corpus gets flat-byte mutated on a structured format with no visible symptom — see `targets/sqlite_read.c` and `tests/test_regression_sqlite_target.py`.

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
| `tools/vendor_lz4.sh` / `vendor_grep.sh` / `vendor_ffmpeg.sh` / `vendor_secp256k1.sh` / `vendor_sqlite.sh` | Fetch vendored library sources into `$FUZZ_VENDOR_ROOT/<lib>/` (default `~/fuzzing/vendoring/`; legacy `vendor/` via `--in-tree-vendor`); required before building the matching targets |
| `FUZZ_VENDOR_ROOT=...` / `FUZZ_BUILD_ROOT=...` | Env-var knobs; defaults are `~/fuzzing/vendoring/` (sources) and `~/fuzzing/targets/` (build artifacts). Both have legacy `--in-tree-*` flags. |
| `python tools/corpus_png.py --out corpus --download` | Generate PNG corpus |
| `tools/bench.sh` / `tools/bench_sweep.sh` | Config comparison / feature sweep |
| `lizard --CCN 15 -w .` | Cyclomatic complexity violations |
| `vulture --min-confidence 80 .` | Find duplicated code |
| `fuzzer-tool fuzz <target> -d <corpus> -n <iters> --profile-hotpath [--profile-out PATH]` | cProfile hotpath profile of the fuzz run (tottime/cumtime/ncalls tables; dump defaults to `/tmp/fuzzer_hotpath.prof`) |
| `dot -Tpng -Gdpi=130 docs/architecture.dot -o docs/images/architecture.png; dot -Tsvg            docs/architecture.dot -o docs/images/architecture.svg` |  To rebuild the architecture png |



## Layout

```
src/fuzzer_tool/
├── core/         # Domain logic: markov, schedulers/, shapley, ga, sanitizer, edge_tracker,
│                 #   operator_registry (canonical op-mutator registration + dispatcher),
│                 #   operator_categories (taxonomy derived from the registry), cmplog, grammar,
│                 #   bloom, elf, target_profiler, fast_json, chi_squared, rand_pool,
│                 #   mutations/<format>.py (structure-aware per-format mutators: png, jpeg,
│                 #   gif, webp, webm, zip, protobuf, …)
├── adapters/     # Process execution, filesystem ops, afl_shim.c (edge + cmplog) / perf_shim.c
├── services/     # Orchestration: fuzzer.py, operators.py, seed_picker.py, runner.py,
│                 #   stats.py, corpus_manager.py, parallel.py, report.py
└── cli/          # CLI entry point (commands.py, __main__.py)

tools/            # build_targets.sh, vendor_<lib>.sh (ffmpeg/grep/lz4/secp256k1/sqlite), corpus_png.py,
                  #   bench.sh, bench_sweep.sh, release.sh
targets/          # Fuzz target sources (*.c) + built artifacts (.so/.bin) under legacy layout.
                  #   Default build output is $FUZZ_BUILD_ROOT (~/fuzzing/targets/); this in-tree
                  #   `targets/` is the legacy path (use --in-tree-targets for back-compat).
dictionaries/     # Format token dicts (png.dict)
vendor/           # Vendored library sources — gitignored, fetched by tools/vendor_<lib>.sh.
                  #   Default location is $FUZZ_VENDOR_ROOT (~/fuzzing/vendoring/); this in-tree
                  #   `vendor/` is the legacy path (use --in-tree-vendor for back-compat).
                  #   FFmpeg 7.1.3, lz4, secp256k1, sqlite (+ zlib/libpng/libjpeg-turbo for trace-cmp builds)
docs/             # DEEP_DIVE.md (comprehensive reference), TODO.md, refs/ (agent reference files), per-feature docs

~/fuzzing/        # OUT-OF-TREE build + cache roots (default layout; override via env vars)
├── vendoring/    #   $FUZZ_VENDOR_ROOT  — vendored library sources (read-only, per-lib dirs)
└── targets/      #   $FUZZ_BUILD_ROOT  — built target artifacts (.so/.bin) and per-lib build trees
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
- **No retry-until-random-hit loops** (Hard Rule 39). Inject a scripted RNG — `tests/support/scripted_rng.py` for `random`-API operators, same class for the `RandPool`-API `_rand_pool` seam — drive the exact draw sequence, and assert the exact output, deriving expected values from helpers/arithmetic rather than echoing literals back.
- **No hardcoded counts in tests.** Use `>=` for minimum bounds, not `==` — operators and features are added frequently and `assert len(X) == N` breaks on every addition.
- **Regression tests are mandatory** for every fixed bug (`test_regression_<description>`); equivalence assertions must derive one side independently of the code under test (see `docs/refs/bug-classes.md` §Testing).
- **Hash functions must be consistent.** When matching filenames against content (corpus eviction, dedup), use `hash_data()` from `fuzzer_tool.adapters.filesystem` — not `hashlib.sha256()` directly. `hash_data()` prefers xxhash when installed; hardcoding SHA-256 causes silent data loss.
- **Cache invalidation on method renames.** When renaming a method with side-effect calls (e.g. `_invalidate_*_cache()`), grep for all call sites — a renamed method silently drops its callers' invalidation hooks.

## Development

```bash
# Setup
pip install -e ".[test]"

# Test
pytest --timeout=15 --timeout-method=signal

# Format / lint
ruff format src/ tests/
ruff check src/ tests/
```
