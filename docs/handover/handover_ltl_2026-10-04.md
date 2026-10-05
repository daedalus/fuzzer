# `--ltl`: LTL property monitor (2026-10-04)

Port of LTL-Fuzzer (ltlfuzzer/LTL-Fuzzer, ICSE '22), items 1-5 of the survey.

## Built
| # | Item | Where |
|---|------|-------|
| 1 | `__fuzz_event[_at]` hook; monitor over a precomputed HOA automaton; (event order) folded into the edge map | `afl_shim.c`, `core/ltl.py` |
| 2 | Frontier: shortest prefix per automaton state, unfired-transition test, `1/(1+dist)` draw | `LtlFrontier` |
| 3 | `ltl_prefix` op: stored prefix + 1..32 fresh bytes | `services/operators.py`, registry (block band) |
| 4 | Directed run: NOT built (reuse the ICFG distance channel on the event call site) | - |
| 5 | Lasso: accepting state revisited with same non-zero state-var digest and a closed walk | `Monitor._lasso` |

Violation = crash with signature `ltl:trap` | `ltl:lasso` (`classify_crash`). New transition admits.

## Limits
- Item 4 absent: no distance-directed scheduling toward event sites.
- Input: `ltl2tgba -B -H` only (state acc, one Inf set, explicit labels, APs `eN`/`N`). Generation is offline; spot is not a dependency.
- Digest covers only `state_vars.py` enums: `ltl:lasso` is a candidate, `ltl:trap` is sound for the observed prefix.
- Prefix pinning needs `__fuzz_event_at`; events without an offset are never stored.
- Bounds: 65536 events/run, 4096 states, 1024 prefixes, 32 lasso replays/run.

## Measured
- Unit/shim/wiring tests: 71. E2E on a toy target (`F e1`): 63 `ltl:trap` crashes in 3000 execs.
- Not measured: a real target, spot-generated automata, A/B vs plain coverage, speed cost (one `write(2)` per event).

## Open
- Lasso with a real spot HOA; directed runs (item 4); `architecture.dot` not updated.
