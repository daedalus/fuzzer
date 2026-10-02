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
