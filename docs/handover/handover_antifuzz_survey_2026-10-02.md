# Anti-fuzzing literature survey

Date: 2026-10-02. Scope: published anti-fuzzing techniques and the
countermeasures each paper discusses. No code changes.

Caveat: usenix.org, sciencedirect.com, github.io blocked by egress proxy;
summaries come from abstracts, search snippets and the AntiFuzz README.
Numbers below are as reported by the authors, unverified.

## Taxonomy

Anti-fuzzing attacks four fuzzer assumptions:

| Assumption | Attack class |
|---|---|
| High exec/s | Anti-fast-execution (delays) |
| Coverage feedback is truthful | Anti-feedback (fake / hidden coverage) |
| Crash = signal | Crash masking |
| Comparisons are solvable | Anti-hybrid (hash compares, implicit flows) |

Two activation modes: **always-on** (Fuzzification) or **detect-then-act**
(AntiFuzz sleep, No-Fuzz, CatchFuzz).

## Papers

| Work | Venue | Techniques | Reported effect |
|---|---|---|---|
| AntiFuzz (Güler et al.) | USENIX Sec '19 | Input-hash-selected fake blocks (default 10k); crash -> `exit`/timeout; 750 ms sleep on malformed input; SHA-512 compare of constants; input encrypt/decrypt | Hid all baseline bugs |
| Fuzzification (Jung et al.) | USENIX Sec '19 | SpeedBump (delays in cold paths), BranchTrap (fake input-dependent branches, bitmap pollution), AntiHybrid (anti-taint / anti-symex) | -70.3 % paths, -93.0 % crashes |
| No-Fuzz (Zhou et al.) | SecureComm '22 | Timing-based env detection; calls jump into random-byte "landing space" -> fake new coverage, bitmap saturation; cheap arithmetic instead of hashes | -92.2 % storage cost vs prior |
| SAFTE | Comput. Electr. Eng. '23 | Self-injection: program stored as data, loaded at runtime, evades compile-time instrumentation | n/a |
| CatchFuzz (Kim & Lee) | Computers & Security '24 | Detect fuzzer (99.6 % acc, 0.5 % FPR), then disorder feedback so interesting inputs look uninteresting | -95.39 % unique crashes |
| Practical Anti-Fuzzing w/ Perf. Opt. | IEEE '23 | Taxonomy + overhead optimisation | n/a |
| obfFuzz | 2024 | Counter side: selective/dynamic deobfuscation to fuzz obfuscated binaries | n/a |

### AntiFuzz switches (from README)

| Flag | Effect |
|---|---|
| `--enable-anti-coverage [n]` | n fake blocks |
| `--signal --crash-action exit\|timeout` | Mask crashes |
| `--enable-sleep --sleep ms` | Delay on error paths |
| `--hash-cmp --enable-encrypt-decrypt` | Anti-symex |

API: `antifuzz_init()`, `antifuzz_onerror()`, `antifuzz_str_equal()`.

## Fuzzer-detection signals used

- Timing gap native vs instrumented.
- Env vars (`__AFL_SHM_ID`, `AFL_MAP_SIZE`), fork-server fds, `LD_PRELOAD`,
  tracer present.
- High rate of malformed / error-path inputs.

## Countermeasures discussed in the literature

| Attack | Countermeasure |
|---|---|
| Any detect-then-act | Detection code is static; locate and patch. Papers concede bypass is always possible, cost is manual effort. |
| Crash masking | Sanitizer report / hook as crash oracle, not exit signal. |
| Fake coverage | Flag inputs adding many edges with no behaviour change; flag edges that flip on semantically null byte changes; track stability. |
| Delays | Tight timeouts; strip sleep calls; persistent mode cuts startup only. |
| Hash compares | cmplog sees digests only. Hook the hash function; brute-force short constants. |

## Relevance to fuzzer-tool

Already shipped (see `docs/TODO.md` "Validate the AntiFuzz defeats"):
hardened crash oracle (`stderr_crash_marker`), coverage-noise probe
(`core/coverage_noise.py`), `--antifuzz-evade` shim, and
`targets/antifuzz_demo.c` in `build_targets.sh`.

`afl_shim.c` still exposes the detection signals above (`__AFL_SHM_ID`,
`AFL_MAP_SIZE`, fork-server fds).

## Implementation details (shipped defeats)

Four countermeasures, one per AntiFuzz technique, plus a benchmark target.
Each maps to the attack it answers.

### 1. Hardened crash oracle — `adapters/process.py`

`stderr_crash_marker(returncode, stderr)` answers crash masking (crash
rewritten to `exit 0` or a timeout). A marker in stderr only corroborates a
*failing* exit:

- Returns `None` when `returncode == 0` or is a sentinel (`-1`, `-2`), so a
  target that prints "Segmentation fault" then exits 0 is treated as **not**
  crashed — it is claiming a crash, not having one.
- Otherwise returns the first of `CRASH_STDERR_MARKERS` (`SIGSEGV`,
  `SIGABRT`, `SIGFPE`, `SIGBUS`, `Segmentation fault`, `Aborted`) found.

Callers: `services/runner.py` (`_is_crash`), `root_cause.py`, `tmin.py`.
ASAN's own report remains the primary oracle; this closes the exit-code gap.

### 2. Coverage-noise probe — `core/coverage_noise.py`

Answers hash-keyed fake coverage (AntiFuzz §4.1 / Fuzzification BranchTrap).
Pure module, no I/O; the caller runs the target. Three pieces:

- **`classify_noise(base_runs, variant_sets)`** — offline probe. Appends one
  byte (values `1..NOISE_PROBE_VARIANTS`, default 8; `tail_variants`) to a
  seed. If the base seed's repeated runs disagree the verdict is
  `UNMEASURED` (nondeterminism is not the input's fault). If all 8 tail
  variants land on **distinct** edge sets → `SUSPECTED` (byte-value dispatch
  on trailing garbage or a hash); else `CLEAN`. Wired at
  `fuzzer.py:_report_coverage_noise`, warn-only.
- **`AdmissionMonitor`** — runtime sibling. Fires once when admissions reach
  `FLOOD_RATE` (0.5) per execution after `MIN_EXECS` (5000) — a campaign
  admitting most executions is being fed noise. Healthy campaigns admit
  well under 1% past the first seconds.
- **`fake_novelty_factor(fuzz_count, children_admitted)`** — per-seed weight.
  After `FAKE_NOVELTY_MIN_FUZZ` (256) mutants, a seed whose children flood
  the corpus gets weight `FAKE_NOVELTY_PENALTY` (0.1), else 1.0. Applied in
  `seed_picker.py:_weight_fake_novelty`.

### 3. `--antifuzz-evade` LD_PRELOAD shim — `adapters/antifuzz_evade.c` + `evade_shim.py`

Answers self-ptrace anti-debug (§4.2) and delay-on-malformed (§4.3) from
outside the target, no source access, so binary-only modes (ptrace coverage,
Intel-PT, `--no-shm`) keep working:

- `ptrace(PTRACE_TRACEME, …)` → returns 0 (pretend nobody is tracing). Every
  other ptrace request forwards to real libc, so the fuzzer's own
  ptrace-based coverage is untouched.
- `sleep` / `usleep` / `nanosleep` / `clock_nanosleep` → return success
  immediately.
- Per-behaviour opt-out env: `ANTIFUZZ_EVADE_PTRACE=0`, `ANTIFUZZ_EVADE_SLEEP=0`.

`evade_shim.py` is the driver (layer boundary: callers work in "evade" terms,
not compiler flags): compiles the `.so` once per process (`clang -shared
-fPIC -O2 … -ldl`, scrubbing `ASAN_OPTIONS`/`LSAN_OPTIONS`/`LD_PRELOAD` from
the compile env) and composes the `LD_PRELOAD` value. CLI flag
`--antifuzz-evade` (OFF by default, `commands.py:4072`) →
`fuzzer.py:_install_antifuzz_evade`; best-effort, prints a notice and
continues if the shim will not build.

### 4. Benchmark target — `targets/antifuzz_demo.c`

Wraps a single real ASAN bug in all four techniques, each env-gated
(default on) so a benchmark can isolate them:

| Env | Technique | Defeat exercised |
|---|---|---|
| `AF_COVERAGE=0` | hash-keyed fake edges (table of 64, bounded per Hard Rule 54) | coverage-noise probe |
| `AF_CRASH=0` | crash masking | hardened oracle + ASAN |
| `AF_SPEED=0` | delay on malformed input | `--antifuzz-evade` sleep |
| `AF_PTRACE=0` | self-ptrace anti-debug | `--antifuzz-evade` ptrace |
| `AF_HASHCMP=0` | hashed magic compare -> `memcmp` | cmplog |

`AF_CRASH`/`AF_PTRACE` run once in `main()`; the `.so` skips them (in
`direct_lite` they traced and re-handled the fuzzer's own process).
`--antifuzz-evade` is ignored in in-process mode: `LD_PRELOAD` cannot reach
an already-running host.

Bug: input starting with 4-byte magic `crsh` overflows a stack buffer; the
magic is checked via a byte hash (§4.4-style) unless `AF_HASHCMP=0`. Wired in
`tools/build_targets.sh` (ASAN + `afl_shim`, plus a `fuzz_shm_run` `.so` for
`direct_lite`), modelled on `asan_target.c`.

## Open / TODO

- [ ] ASAN-build validation of the shipped defeats — tracked in `docs/TODO.md`.
- [ ] Hash-compare constants: no counter beyond cmplog digests.
- [ ] Re-fetch full papers when egress allows; verify numbers above.

## Sources

- AntiFuzz: https://www.usenix.org/system/files/sec19-guler.pdf ·
  https://github.com/RUB-SysSec/antifuzz
- Fuzzification: https://www.usenix.org/system/files/sec19fall_jung_prepub.pdf
- No-Fuzz: https://link.springer.com/chapter/10.1007/978-3-031-25538-0_38
- CatchFuzz: https://www.sciencedirect.com/science/article/abs/pii/S0167404824002062
- SAFTE: https://www.sciencedirect.com/science/article/pii/S0045790623004044
- Practical Anti-Fuzzing: https://ieeexplore.ieee.org/document/10209185/
- obfFuzz: https://www.researchgate.net/publication/386869631_obfFuzz_Empirical_Study_to_Boost_Fuzzing_the_Obfuscated_Software
- Escaping the Fuzz: https://publications.lib.chalmers.se/records/fulltext/238600/238600.pdf
