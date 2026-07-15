# AGENTS.md — fuzzer-tool

RULES:
Always create TODOs.
Always update the README.md with the new features added.
Always git commit and push after finish a task.


## Overview

Coverage-guided binary fuzzer with ASAN/MSAN/TSAN/UBSAN detection, dictionary mutations, Markov chain generation, Monte Carlo optimization, kernel crash verification, and state persistence. CLI tool for fuzzing arbitrary binaries.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format src/ tests/` | Format code |
| `ruff check src/ tests/` | Lint code |
| `fuzzer-tool --help` | Show CLI help |
| `python tools/corpus_png.py --out corpus --download` | Generate PNG corpus |

## Development

```bash
# Setup
pip install -e ".[test]"

# Test
pytest

# Format
ruff format src/ tests/

# Lint
ruff check src/ tests/
```

## Project Structure

```
src/fuzzer_tool/
├── core/           # Domain logic
│   ├── markov.py       # Byte-level Markov chain (with save/load persistence)
│   ├── montecarlo.py   # Thompson sampling + CEM
│   ├── mutations.py    # Mutation operators
│   ├── ga.py           # Genetic algorithm lifecycle (fitness, speciation, population)
│   ├── sanitizer.py    # ASAN/MSAN/TSAN output parsing
│   ├── edge_tracker.py # Per-seed coverage tracking (with save/load)
│   ├── dmesg.py        # Kernel crash verification via dmesg
│   ├── cmplog.py       # Comparison tracing via LD_PRELOAD
│   ├── grammar.py      # Grammar-aware mutations
│   ├── bloom.py        # Bloom filter for dedup
│   ├── crash_metadata.py # Crash enrichment
│   ├── elf.py          # ELF parsing utilities
│   └── target_profiler.py # Static analysis for fuzzing guidance
├── adapters/       # Process execution, filesystem operations
├── services/       # Fuzzer orchestration (fuzzer.py, parallel.py, etc.)
└── cli/            # CLI entry point

tools/
├── corpus_png.py      # PNG corpus generator for libpng fuzzing
└── release.sh         # Release automation

dictionaries/
└── png.dict           # PNG format tokens

targets/
├── png_read.c         # libpng fuzz target
├── png_read           # Compiled target
├── test_target.c      # Minimal crash target
└── test_target        # Compiled target
```

## Key Concepts

### State Persistence
Fuzzer state is saved to `{corpus_dir}/state.json` on shutdown. Use `--resume` to continue:
- `state.json` — exec counts, crash sigs, op stats, seed metadata
- `edge_tracker.json` — per-seed edge coverage
- `markov.json` — Markov chain transitions

### Coverage Modes
- `--no-shm` — forces ptrace for uninstrumented binaries
- `--deep-coverage` — capstone disassembly for basic block discovery
- Default SHM — for AFL-instrumented targets

### Kernel Crash Verification
- Historical poll: `dmesg -l err,warn,info --json` (one JSON document: `{"dmesg": [...]}`, not NDJSON)
- Live streaming: `dmesg -l err,warn,info -w` in **text** format, not `--json` (JSON streaming is bursty and expensive to parse line-by-line; text is reliable for real-time line-by-line reads)
- `info` level is required, not just `err,warn` — Linux logs segfaults at priority 6 (INFO)
- PID-filtered crash attribution (PID is embedded in the `msg` field as `comm[pid]:`, not a separate field)
- Requires root or CAP_SYSLOG
- Three-layer detection: async stream → sleep+re-drain → synchronous `_poll_text(since=0)` fallback

### Markov Persistence
- Markov chain saved to `markov.json` on exit
- Loaded on init; skip retrain if loaded to avoid double-counting
- Transitions accumulate across sessions

### Meta-Scheduler (Elo Arbitration)
- `--elo` alone now enables Elo-based arbitration between operator strategies (bandit/MOpt/replicator) and seed strategies (ga/weighted/pareto/format); the separate `--meta-elo` flag was consolidated into `--elo` (see `_use_elo` in `services/fuzzer.py`)
- Enable `--mc-bandit`/`--mopt`/`--replicator` alongside `--elo` to add those strategies to the arbitration pool
- All available strategies run in shadow; Elo picks which one to trust each iteration
- Strategy ratings tracked in `elo.json` under `strategy_ratings` / `strategy_match_count`
- Probabilistic selection via softmax over Elo gap (temperature=400)

## Code Style

- Format: ruff format
- Lint: ruff check
- Docstrings: Google style
- Type hints: strict mypy

## Rules

- **Always improve the corpus, never delete it.** Corpus files represent discovered coverage and crash triggers. Only add new inputs, never remove existing ones. Use `fuzzer-tool minimize` to prune redundancies — removed inputs are moved to `corpus/pruned/`, not deleted. The active corpus keeps only inputs that produce the most edges.
- **Do not clean the corpus between runs.** The corpus directory accumulates discovered inputs across sessions. Running `rm -rf corpus/*` destroys coverage history and forces the fuzzer to rediscover everything from scratch. Always use `--resume` to continue. When generating a new corpus (e.g. `corpus_png.py`), write to a fresh directory, not an existing one.
- **Verify claims against code.** Before acting on behavior, type, or API shape, read the source. Don't infer from names.
- **Run the full test suite after changes.** `pytest` must pass before considering any change complete.
- **Hash functions must be consistent.** When matching filenames against content (corpus eviction, dedup), use `hash_data()` from `fuzzer_tool.adapters.filesystem` — not `hashlib.sha256()` directly. `hash_data()` prefers xxhash when installed; hardcoding SHA-256 causes silent data loss.
- **Cache invalidation on method renames.** When renaming a method that has side-effect calls (e.g. `_invalidate_*_cache()`), grep for all call sites. A renamed method silently drops its callers' invalidation hooks.
- **No hardcoded counts in tests.** Use `>=` for minimum bounds, not `==`. Operators and features are added frequently; `assert len(X) == N` breaks on every addition.

## Bug Classes

These rules are extracted from ~85 `fix:` commits in project history. Each names the recurring bug *class*, not just the single instance that surfaced it — recognize the pattern before it reappears in a new file.

### dmesg / kernel crash detection

- **Never assume an external tool's output schema — capture and read real output before writing the parser.** `dmesg --json` is one JSON document (`{"dmesg": [...]}`), not one-object-per-line; the PID lives inside the `msg` string, not a separate field; the default priority filter misses INFO-level segfaults entirely. Bugs like this are silent no-ops — the fallback path swallows the parse failure, so nothing looks broken until checked against real output.
- **When multiple code paths parse the same data, every path must extract the same fields.** `_poll_text`, `_poll_json`, `_stream_reader`, and `_process_entry` all parse dmesg output — if one path skips PID extraction, crashes parsed through that path have `pid=None` and get silently dropped by PID filtering. Audit every path that calls `_match_crash()` and verify it passes `pid` and `proc_name`.
- **The initial seed replay loop is a separate crash path from `fuzz_one()`.** `run()` runs each corpus seed as-is before mutating. If this loop detects a crash but skips kernel verification, those crashes are never dmesg-verified. Every crash detection site must include the same verification logic.

### Signals, processes, and timeouts

- **Check syscall return values under signal-based timeouts — don't assume an interrupted syscall failed cleanly.** A `SIGALRM` handler racing `waitpid`/`os.wait` can leave `status` unset on `EINTR`, which then reads as `WIFEXITED(0)` — a false "success" instead of a timeout. Branch explicitly on the wait call's return value; if interrupted, force-kill and re-reap before deciding the outcome. This recurred independently in both the C loader and the Python persistent runner.
- **Put child processes in their own process group before ever using `killpg` on them.** `preexec_fn=os.setsid` (Python) / `setsid()` (C) must run before any code path can `SIGKILL` via `killpg`, or the signal lands on the caller's own group and can kill the fuzzer itself.
- **A timed-out or killed parent can leave an orphaned grandchild running forever.** If a subprocess itself forks/execs (e.g. a loader dlopen-ing and calling a target function), killing the immediate child on timeout isn't enough — track the grandchild's PID explicitly (e.g. via a PID file) and kill its process group too.
- **Guard `kill()`/`os.kill()` against `ESRCH` / `ProcessLookupError` races, and don't let a broad `except` swallow the real result.** A process can exit between a status check and a cleanup `kill()`; if the broad exception handler around that pattern also wraps the actual crash-detection logic, a benign race turns "a real crash" into "no crash detected." Catch the specific race exception close to the call that raises it, not several frames up.
- **`ChildProcessError` in a waitpid path means "already reaped" — not "success."** Returning `rc=0` on `ChildProcessError` silently masks crashes. Return `rc=-2` (unknown) so the crash detection pipeline treats it as suspicious, not clean.
- **Stale loop flags cause redundant waitpid on already-reaped children.** In ptrace mode, `last_action`/`last_sig` record "we last resumed the child after a breakpoint," but the post-loop code treats them as "the child might still be alive." A target that crashes after hitting even one coverage breakpoint (normal for guided fuzzing) gets its correctly-captured crash discarded by a redundant `waitpid` → `ChildProcessError` → `return 0`. Track explicitly whether the loop already reaped the child (`child_reaped` flag) and skip the redundant wait/kill entirely if so.

### Concurrency & resource cleanup

- **Release threads, fds, and temp resources in `finally` — but don't assume setup succeeded before scheduling cleanup.** A `thread.join()` in `finally` raises `NameError` if `fork()`/thread creation failed before the variable was assigned. Guard the cleanup call, or initialize the variable to `None` first.
- **Kill processes before detaching their shared memory.** Detaching SHM while the target is still writing to it can cause SEGV in the target. Always SIGKILL → waitpid → detach SHM, not the reverse.
- **Never use `tempfile.mktemp()`.** It's a TOCTOU/symlink race by construction. Use `mkstemp()`/`mkdtemp()`.
- **Namespace any filesystem path shared across parallel workers by PID.** Compiled shim/loader binaries and other on-disk artifacts written under `-j N` must embed `os.getpid()` (or equivalent), or concurrent workers race to compile/clean up the same file.
- **Return the actual PID on exception, not 0.** `run_target_stdin`/`run_target_file` callers use the returned PID for dmesg filtering. Returning `pid=0` on exception matches the swapper/idle process, silently discarding real kernel crashes.

### Hashing & identity

- **Never use Python's builtin `hash()` for anything persisted or shared across processes.** `hash(bytes)` is randomized per-process via `PYTHONHASHSEED`. Using it for a seed/edge-tracker key orphans every entry on fuzzer restart and produces divergent keys between `-j N` parallel workers. Always go through `hash_data()`/`hashlib`. Grep for `str(hash(` before adding any new keying scheme.

### Caching

- **Cache invalidation must key off the actual dependency, not a proxy.** Invalidating on every `exec_count` tick makes the cache useless (recomputes on almost every call — cost 80 eps once). Invalidating on the wrong signal, or never, serves stale data instead. Key strictly off the values the cache depends on (e.g. `corpus_version`, `edge_version`), and check both directions: does it recompute on every real change, and skip recomputing when nothing relevant changed?

### Low-level parsing (ELF, ptrace, dmesg)

- **Use exact bitmask equality for "all these bits must be set" checks, never bare truthiness.** `flags & (PF_R | PF_X)` is truthy if *either* bit is set; a read-only RELRO segment (`PF_R` only) then satisfies a check meant to require both, silently selecting the wrong ELF segment. Write `(flags & mask) == mask` when the intent is "all of these bits."
- **This tool parses attacker-controlled binaries (the fuzz target's own ELF headers) and user-supplied grammar files as part of its own operation.** Bounds-check every offset/count read from a section header, program header, or symbol table before indexing with it. Clamp any grammar-controlled repeat/recursion count to a fixed MAX — an unbounded count is a resource-exhaustion bug in the fuzzer itself, not just in whatever it's fuzzing.

### Numeric & mutation edge cases

- **Clamp any input-derived range before calling `random.randint`/`randrange`.** If bounds come from input length (e.g. `len(raw) // stride - 1`), a small or degenerate input can make `hi < lo`, raising `ValueError` at fuzz time. Guard with `max(0, ...)` or an early return for the degenerate case.
- **Clamp arithmetic-mutation results to their valid range before packing.** `val * 2` or `val ^ (1 << 31)` can overflow the target field width and crash `struct.pack_into` — clamp to `[lo, hi]` for the field width in use.

### State & double-counting

- **Persisted state must have exactly one source of truth.** If a value can either be reloaded from disk (`markov.json`, `edge_tracker.json`) or freshly re-derived by a normal-startup code path, the reload path must skip re-derivation — otherwise transition counts / edge stats double up silently across restarts.
- **Reduction and minimization must re-verify the specific property of interest, not just "did something happen."** Crash minimization (`tmin`) that only checks "does it still crash" can drift onto a different bug on a multi-bug target mid-delta-debugging. Pin the original crash signature up front and require every candidate to match it exactly.

### Testing

- **Assert on behavior, not on stale or accidentally-inverted expectations.** Some past bugs shipped *with* a passing test because the assertion checked the wrong direction (e.g. asserting a timeout counts as a crash) or referenced an error string that had since changed. When fixing a bug, re-read the failing test's assertion and confirm it actually encodes the intended behavior, not just "no exception was raised."

### Dispatch table & half-shipped features

- **Every entry in `_build_dispatch()` must have a corresponding module and class.** If an operator name is registered in `MUTATIONS` / `FORMAT_MUTATIONS` and wired into the dispatch table, but the module it imports doesn't exist, the fuzzer crashes with `ModuleNotFoundError` the moment the scheduler picks that operator. This is invisible to unit tests because they never exercise the live dispatch path. The integration smoke test (`test_operator_smoke.py::test_all_ops_fire`) catches this by calling every handler once — it must pass before any release.

### Silent error swallowing

- **Never `except Exception: pass` in production code paths.** A broad except that swallows errors hides real failures (disk full, permission denied, EMFILE). At minimum, log at `warning` level. Reserve `log.debug` for genuinely expected/recoverable situations only.
- **`except ChildProcessError` in waitpid does NOT mean success.** It means the child was already reaped — return `-2` (unknown), not `0` (success), to avoid masking crashes.
