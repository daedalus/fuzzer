# TODO — fuzzer-tool Roadmap

## Done
- [x] Bloom filter for corpus dedup (128KB for 100K entries)
- [x] Thompson sampling bandit for mutation selection
- [x] CEM byte distribution learning
- [x] Ptrace coverage (function-entry breakpoints)
- [x] Crash signature dedup (ASAN/MSAN/TSAN/UBSAN)
- [x] Stats JSON export
- [x] CEM engagement fix (refit on elite count, not just interval)
- [x] Sanitizer regex groups closed (ASAN/TSAN)
- [x] Timeout crash detection fix
- [x] Ptrace initial SIGTRAP crash detection

## Bugs Fixed
- [x] `--stats-file` eaten by `-A` (REMAINDER) — user must place `-A` last
- [x] CEM never engaging — refit now triggers at elite_set >= 10
- [x] dict_insert/dict_replace 0/0 — was missing `-D` flag, now works

## Pending Bugs
- [ ] `_apply_single_mutation` havoc doesn't enforce max_len strictly (allows +1 byte per insert, up to +8 total)
- [ ] `parse_dict_line` triple-encode chain fragile for bytes > 0x7F

## Performance (Priority Order)
- [ ] **Forkserver mode** — fork() from pre-initialized copy, 2-5x throughput
- [ ] **Cmplog/comparison coverage** — intercept memcmp/strcmp for magic-byte discovery
- [ ] **Corpus distillation on-the-fly** — evict subset-seeds during fuzzing
- [ ] **Exploitability scoring** — tag crashes by ASAN error type

## Coverage
- [ ] Sanitizer coverage (-fsanitize-coverage) via LD_PRELOAD
- [ ] Call stack coverage (distinguish f()→g() from h()→g())
- [ ] Deep coverage with capstone BB discovery (already partially implemented)

## Mutation
- [ ] Radamsa-style structural mutations (line/field repetition, truncation)
- [ ] Token-level mutations for text protocols (grammar integration)
- [ ] Havoc stage weighting by per-operator success history

## Scheduling
- [ ] Seed energy burst on discovery, decay over time
- [ ] Collaborative scheduling across parallel workers

## Crash Analysis
- [ ] Automated crash bucketing (cluster by ASAN report similarity)
- [ ] Root cause diff (show bytes diff from nearest non-crashing input)
- [ ] Exploitability scoring (critical/high/medium/low)

## Infrastructure
- [ ] Dockerfile for reproducible builds
- [ ] Individual entry points (fuzzer-fuzz, fuzzer-tmin, etc.)
- [ ] Structured logging (--log-json)
