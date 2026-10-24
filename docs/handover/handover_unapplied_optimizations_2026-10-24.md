# Unapplied optimizations: audit

**Date:** 2026-10-24
**Scope:** per-exec hot path (`services/`, `adapters/`, `afl_shim.c`) and `docs/port-backlog.md` against live code.
**Status:** audit only, no code changed. Gains are estimates from code reading and `python -c` microbenchmarks; none measured end to end.

## 1. Throughput (execs/s)

Budget (TODO.md): ~6.8 ms/exec fuzzer-side vs ~1.8 ms in target. Ranked by expected gain.

| # | Where | Now | Proposal | Est. gain/exec | Trajectory |
|---|---|---|---|---|---|
| T1 | `adapters/forkserver.py:297-308`, `persistent_subprocess.py:432-437`, `forkserver.set_timeout` 397-404 | New `threading.Thread` per exec to `readline()` with timeout (~190 µs spawn+join) | `select.poll` + `os.read` on raw fd, deadline loop. Precedent: `adapters/process.py:180-191` | 50-190 µs | identical |
| T2 | `adapters/shm.py:1367-1381` `_scan_new` | Fast path skips only if path equals the *previous* exec; new seed every exec, so ~20 numpy calls run nearly always | Bounded set of fully-folded `path_hash`es (AFL known-checksum). Mix edge count into key (`h*31^id` is linear). Full mode needs ids cached per hash (`_region_liveness`, good-turing) | 50 µs-0.7 ms | identical if ids cached |
| T3 | `adapters/inprocess.py:646` | `reset_bitmap()` memsets map each exec (4 MB at 2^19), redundant with generation bump in `reset_edge_map()`; forces cold `__afl_slot_claim`, bumps `edge_count`, kills T2 fast path | Delete call | 100-400 µs | identical |
| T4 | `core/cmplog.py:1489-1542` `collect_counts`, `collect_sites` 1566-1573; `setup_env_for_run` 1009-1027 | Text files: stat/open/seek/`readlines` per exec; ~1000 `CNS` lines on ffmpeg (fold ~330 µs); 4-6 `putenv` per exec | Shim writes fixed counter arrays to SHM tail (bump `__AFL_SHM_LAYOUT`, Hard Rule 53); numpy view. Env once per run | 30 µs->0.4 ms | identical |
| T5 | `services/fuzzer.py:9913-10024` | `pick_seed` + `_seed_scorer.score(~25 args)` every exec; DEEP_DIVE: pick+mutate ~60% of LITE profile | AFL stage: pick once, k mutations, k from `_last_perf_score` | up to ~1.5x | changes: paired A/B |
| T6 | `afl_shim.c:6139` `__afl_start_forkserver`, `fuzz_loader.c:203-215` `stage_input` | No `__AFL_LOOP`, no `__AFL_INIT`, testcase via lseek/write/ftruncate | Persistent loop, deferred init, SHM testcase delivery | 2-20x (AFL++ figures) | harness opt-in |
| T7 | `adapters/inprocess.py:733-737, 764-776`; `_run_c_direct` 655-704 | ASAN stderr capture: ~10 syscalls/exec; SIGALRM handler swapped per exec | Persistent O_NONBLOCK pipe or memfd on fd 2, one read/exec | 10-20 µs | identical |
| T8 | `adapters/process.py:600-615` `run_target_stdin` | Popen + writer thread + watchdog thread per exec. Hit by `--clock virtual`, `--file-mode`, `--target-args` (forkserver excluded, `fuzzer.py:5100`) | Route via `run_target_fast` poll design, or forkserver with file sink | 200-400 µs | identical |
| T9 | `core/bloom.py:200-225`, loops 146-184; called `fuzzer.py:5891` | SHA-256 + 256-bit int (~3.4 µs) + two Python k-loops, up to `EXEC_DEDUP_RETRIES+1`/exec | xxh3 64-bit key into numpy bit array. Check `xxhash` is a declared dependency | 3-8 µs | dedup set changes |
| T10 | `adapters/inprocess.py:341-354` `_call_guarded` | `restype`/`argtypes` reassigned, `ctypes.cast` per exec; `setitimer` armed twice (748, 763) even with C-side timeout | Bind once; skip Python itimer when `__afl_guarded_call_timeout` exists | 2-3 µs | identical |
| T11 | `services/runner.py:215-332`, `inprocess.py:603-615` | ~6 backend checks + multi-target lookup + cmplog env check per exec | Bind one run callable at setup (Hard Rule 57) | 1-2 µs | identical |

Other open hot-path items, already in `docs/TODO.md`:
- Colorization rebuilds Aho-Corasick ~190 times.
- MinHash `find_similar`: single numpy pass.
- Map 2^16 vs 2^19 on ffmpeg_read (~20% exec/s, unverified).
- Non-main threads still pay `__tls_get_addr` per edge.
- `crash_hashes` and `KatzChannel._masks` unbounded (Hard Rule 54).

## 2. Search effectiveness: port backlog still open

Verified against code. Found implemented despite the backlog: A1, A4, B2, C2-C4, D2, F9, E1 (`core/crash_metadata.py:509-667`), E2 (`services/tmin.py` `_GrammarPass`), F8 (`operators.py:7203-7207`).

| Id | Item | Effort | Status / evidence |
|---|---|---|---|
| F7 | Check havoc sub-op availability before select | S | Not impl. `operators.py:5817` selects then guards; `_HAVOC_EXPLORE=0.15` |
| D5 rem. | Exec cost in `_maybe_prune` eviction order | S-M | Not impl. |
| F1 rem. | Invariant checks for derived caches | S | Partial. None of six caches in `tests/test_regression_hotpath_invariants.py` |
| F2 | Distribution assertions in tests | S | Partial. `test_operator_smoke.py:38` single 32-byte buffer |
| F3 | Random-method-order subsystem harnesses | S-M | Not impl. |
| B1 | Identifier renaming second corpus | ½ d | Not impl. Needs text target |
| J1-J7 | FFmpeg harness: `fuzz_params.h`, decoder knobs, `av_parser_parse2`, NULL drain, `AVERROR_BUG`, seek callback (`ffmpeg_read.c:267` NULL) | steps 1-4 | Mostly not impl. Cheapest new FFmpeg coverage per backlog |
| C1 | Cmplog operand-range invariant violation | ~1 wk | Not impl. |
| A2 | Gramatron (grammar automata) | M | Not impl. |
| B3 | WindRanger deviation blocks | M | Partial: dynamic `pos_effector.py` only |
| B4 | IDFuzz field gradients | M | Not impl. |
| B5 | DFSan DataFlowTrace + focus_function | M-H | Partial: `adapters/track_parser.py` only |
| B6 | NestFuzz input-processing tree | M-H | Partial: hand-written `weizz_tags.py` |
| D1 | PrescientFuzz reachable-uncovered weighting | L | Partial: `--frontier-bin`; A/B null png, harmful FFmpeg |
| D3 | SelectFuzz reaching probability | L-M | Not impl. |
| E3 | Exact MaxSAT minimization | L-M | Not impl. Must be in-house (Hard Rule 51) |
| E4 | Igor WL-kernel crash clustering | M-H | Partial: stack clusters + ddmin |
| F4 | `--performance` fixed-profile mode | M | Not impl. |
| F5 | Component-set bisect | 1-2 d | Not impl. Gated on F6 |
| F6 | Deterministic replay under fixed seed | ? | Not holding: 0/20 identical (TODO.md:25) |
| G1 | SAND: only novel exec patterns to sanitizer build | M | Partial: crash replay only (`fuzz_round.py:1973`) |
| G2 / G5 | userfaultfd / Nyx snapshots | M / H | Not impl. |
| G3 / G4 | E9Patch / Fuzztruction | M / H | Not impl. |
| H1 / H2 | SymCC / JIGSAW | M / M-H | Partial: cmplog concolic, `gradient_descent.py` (A/B null p=0.29) |
| A3 | Tree-sitter + subtree bank | ~1 wk | Not impl. Needs text target |
| I1-I3 | LLM generators / plateau / evolution | L-M | Not impl. |

Port candidates in TODO.md not started: honggfuzz (per-cmp matching bits, instruction count, crash verifier, self-restart), Revizor (relational oracle, pair minimizer), wtf (`--cov-file`, ptrace breakpoints, drcov).

Rejected: see `docs/port-backlog.md` §Rejected. Do not re-propose.

## 3. Blockers

- **F6.** Same `-s` does not reproduce a run, so every rep=1 paired result is noise-capped. Fix before trusting small A/B deltas. `--clock virtual` reproduces decisions but forces the slow spawn path (T8).
- **~100 shipped features unmeasured** (`docs/AB_PENDING.md`). Measuring may beat adding.

## 4. Recommended order

1. **T3:** delete one line. Re-enables T2's fast path.
2. **T1:** self-contained adapter change; trajectory identical.
3. **T2:** LITE first; full mode after per-hash id cache.
4. **T4 + T8:** T8 unblocks fast `--clock virtual`, which helps F6.
5. **T5, T6:** structural; paired A/B via `tools/lib/bench_paired.py`, A/A control (Hard Rule 46).

Per item: test first (Hard Rules 37/38), regression test, benchmark before/after (Hard Rule 41), fuzzgoat calibration with clang (Hard Rule 52), rebuild targets if `afl_shim.c` changes (Hard Rule 56).

### Wiring sketches

**T1:**
```python
# forkserver.run_one: replace thread + readline
deadline = time.monotonic() + self.timeout + _LOADER_GRACE
line = self._read_line(deadline)  # poll(fd) + os.read into self._rx buffer
if line is None:
    log.warning("Forkserver timed out after %.1fs, restarting", self.timeout)
```
Drop `bufsize` buffering on `stdout` (raw fd), or the BufferedReader holds bytes that `poll` cannot see.

**T2:**
```python
# shm._scan_new, after the existing previous-exec check
key = (path_hash, edge_count)
if path_hash and key in self._folded:  # bounded: clear at cap
    self.last_new_ids = frozenset()
    self.last_old_bucket_novel = False
    return False
...  # slow path
self._folded.add(key)
```
Clear `_folded` on any virgin/mask import that can re-add novelty.

**T3:** delete `inprocess.py:646-647`. Regression test: two execs of the same input, the second keeps `edge_count` unchanged and takes the fast path.
