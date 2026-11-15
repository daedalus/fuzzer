# AFL++ port candidates — 2026-11-15

Source: AFL++ `d798677` (5.04a dev), cloned to `~/fuzzing/AFLplusplus`.
Method: Changelog 5.00c–5.04a, `TODO.md`, `src/`, `utils/`, `custom_mutators/`
read; each candidate grepped and opened in our tree. Every claim below is
grep/read evidence, not measurement. Nothing here has an A/B.

Six planned subagent audits died on the account rate limit before writing;
this pass is single-handed. **Not audited:** op-by-op libFuzzer/honggfuzz
mutator parity, redqueen transform parity (`afl-fuzz-redqueen.c` vs
`core/rq_encodings.py`), `env_variables.md` knob-by-knob, frida/qemu/nyx/unicorn.

## Already present (don't re-port)

| AFL++ | ours |
|---|---|
| afl_untracer (int3 coverage) | `services/ptrace_coverage.py` (`--ptrace`) |
| libtokencap | `afl_shim.c` `__sanitizer_weak_hook_{memcmp,strcmp,strncmp,strcasecmp,strstr}` |
| libdislocator | `adapters/guard_arena.py`, `inprocess.py` guard-overread |
| MOpt (PSO) | `core/schedulers/op_mopt.py` (`--mopt`) |
| frameshift | `core/analyzers/analyzer_frameshift.py`, `f._frameshift` in `services/operators.py` |
| IJON | `--ijon`, `services/ijon_channel.py` |
| power schedules fast/coe/rare/lin/quad | `--schedule` |
| ctx-sensitive edges | `--ctx-sensitive` |
| CompCov / split-switch via trace-cmp | `--compcov-level`, `core/cmplog.py` |
| afl-merge | `import` subcommand |
| afl-tmin / afl-cmin | `tmin`, `minimize` |
| crash exit codes (`AFL_CRASH_EXITCODE`) | `--crash-codes` |
| autotokens | `--autotokens` |
| skip-det / effector map | `--pos-effector`, `--no-deterministic` |
| Grimoire, FormatFuzzer, EcoFuzz, LBR, PT | see `docs/port-backlog.md` |

## Inherited bug

**B1. Trimming throws away its finds** (AFL++ fixed in 5.03c: `op:trim` saves).
`services/corpus_manager.py:1527` `trim_new_coverage`: one extra exec on
`data[:len//2]`; a crash (rc not in `(-2,-1)`) or a different edge set →
`return`. Crash never saved, new edges never admitted.
Repro: target crashing on inputs ≤ N/2 bytes whose N-byte parent has new
edges → no crash file. Fix per Rule 37: test first; route the trim exec through
the normal crash/novelty path (`FuzzRound` admission), then keep-or-drop.

## Gaps, ranked (value / effort)

| # | Item | AFL++ loc | Ours | Value | Effort |
|---|---|---|---|---|---|
| 1 | Multi-instance sync (`-M/-S/-F`) | `afl-fuzz-run.c:sync_fuzzers` | absent; one-shot `import` only | high: only way a Python fuzzer scales past one core | M |
| 2 | AFL `trim_case` (len/16…len/1024 chunk removal) | `afl-fuzz-one.c:trim_case` | only one halving try | high: smaller seeds → faster execs, cleaner splices | S |
| 3 | Starve mode, full | `afl-main.c:afl_advance_queue_cycle`, `afl-fuzz-queue.c:1426` | partial: `--stall` → random-mode mutations | med–high | M |
| 4 | Value-profile frontier | `afl-fuzz-valprof.c`, `docs/value_profiling.md` | partial: cmp-distance *operators* (`gradient_cmp`, `mb_cbh`), no distance *feedback* | high on magic-heavy targets | M |
| 5 | C11 local-variable seed weight (ICSE'26) | `afl-fuzz-queue.c:368,519` | absent | med; cheap via DWARF | S |
| 6 | SAND sanitizer forwarding | `afl-fuzz-sanfuzz.c`, `docs/SAND.md` | partial: `--asan-target` replays crashes only (backlog G1) | high on ASAN-slow targets | M |
| 7 | BUG_SCALAR max-per-arith-site | `afl-llvm-bug-pass.so.cc` | absent (IJON max is manual) | med | M (sancov can't see arith; needs pass or IJON auto-annotation) |
| 8 | Dynamic coverage filter | `utils/dynamic_covfilter` | absent | med for big targets (FFmpeg) | S |
| 9 | Adaptive MOpt context (input_mode × explore/exploit) | `afl-fuzz-mopt-adaptive.c` | partial: many bandits, no text/binary context | low–med | S |
| 10 | CRASHLIST (abort on entering denylisted fn) | `AFL_LLVM_CRASHLIST` | absent | low–med | S |
| 11 | cmplog: retire site once both outcomes seen | AFL++ `TODO.md` Ideas | not found by grep, verify first | med | S |
| 12 | cmplog loop-count resolving | AFL++ `TODO.md` Should | absent | med | M |
| 13 | Desock / argv / defork preload shims | `utils/{libaflppdesock,argv_fuzzing,defork}` | absent | low for file parsers; needed for net/CLI targets | S each |
| 14 | StatsD/Prometheus export | `afl-fuzz-statsd.c` | absent (`--stats-file`, `--log-json` exist) | low | S |
| 15 | Ball-Larus path coverage | `AFL_LLVM_PATH` | absent | low–med; needs LLVM pass | L |

## Sketches

**1. Sync.** `--sync-dir DIR --sync-id NAME`. Each instance writes admitted
seeds to `DIR/NAME/queue/` (existing seed writer). Every `SYNC_INTERVAL_EXECS`,
scan peers' `queue/` past a pickled per-peer cursor (Hard Rule 15), run each
file once, admit on new coverage (same path as `_calibrate_seed_baselines`).
Lands: `services/sync.py` (new class), hook in `Fuzzer.run` maintenance tick,
CLI in `commands.py`. Interop with stock AFL++ falls out (`-F` reads
`queue/id:*`). Falsification test: a peer seed with no new edges is not admitted.

**2. trim_case.** Port AFL++ loop: `remove_len = next_pow2(len)/16` down to
`/1024`, slide over input, keep cut when edge set unchanged; budget
`TRIM_MAX_EXECS`. Replace the halving body of `trim_new_coverage`; fix B1 in
the same change. Rule 58 in-run check: execs/s and corpus bytes.

**3. Starve mode.** On entering stall: (a) prefer unfuzzed seeds before going
random; (b) flatten seed weights (`sqrt`, `afl-fuzz-queue.c:566`); (c) pick a
random power schedule; (d) after `2×` the stall budget, cmin-style minimize
(`services/full_cover.py`) and hand pruned seeds' unique edges back to the
virgin map so they are rediscoverable ("Novelty Not Found" restarts). Lands:
`Fuzzer._stall_recovery_enter`, `seed_picker.py`.

**4. Value-profile frontier.** Shim already gets every `trace_cmp`. Per
`(site, width)` keep best `popcount(a^b)` in a small SHM slot table; Python
treats an improved slot as soft novelty (no edge) and favors its owner seed.
Gate: `--value-profile {off,on,stall}`; `stall` matches `-r -1`. Bump
`__AFL_SHM_LAYOUT` (Rule 53), rebuild targets (Rule 56). Risk: frontier churn
floods the corpus — cap admissions per site.

**5. C11.** `core/dwarf.py` counts `DW_TAG_variable` per `DW_TAG_subprogram`;
map a seed's edges → functions via pc-table (`adapters/sancov_tables.py`);
seed score = max locals over touched fns; weight `*= 1.1 + 0.9·t²`, `t`
min-max normalized, off while starved. New `SeedScorer` term in
`services/seed_picker.py`.

**6. SAND.** `build_targets.sh --san` builds ASAN-without-coverage sibling;
forward an input to it only when its execution-pattern hash (existing exec
dedup bloom) is new. Plain build runs the hot loop. Backlog G1 has the rest.

**8. Dynamic covfilter.** `--cov-filter FILE` of function names → pc-table →
edge-id mask applied in Python novelty check (no rebuild). Lands:
`adapters/sancov_tables.py`, `services/edge_tracker` novelty gate.

**11. Retire solved cmp sites.** When both branch edges after a cmp site are
covered, drop the site from I2S candidates. Needs edge↔site mapping from the
`control-flow` dump (`core/sancov_cfg.py`).

## Not worth porting

qemu/frida/unicorn/nyx/coresight modes (binary-only engines; `--ptrace`,
`--lbr`, `--intel-pt` cover our binary-only need), afl-gotcpu, macOS/wine utils,
radamsa (removed upstream as pointless), GUIFuzz (X11 GUIs), afl-gcc passes
(Rule 4).
