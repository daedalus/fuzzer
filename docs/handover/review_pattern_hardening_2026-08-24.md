# Handover: Test-Determinism & Capability-Contract Hardening Pass

**Source:** analysis of all 8 closed/merged PRs on `daedalus/fuzzer` (2026-08-24)
**Trigger:** PRs #1, #7, #8 each got sent back for review at least once, and all three
corrections cluster into two reusable patterns. This doc is the plan to apply those
patterns proactively across the rest of the codebase instead of waiting for each
instance to get caught in review one at a time.

## Patterns identified

### Pattern A — non-deterministic "retry until random hit" tests
PR #7 (`f00927a`) and PR #8 (`f5d599a`) were both bounced for tests that looped up to
200x hoping an unseeded/weakly-seeded RNG would eventually hit the code path under
test, then asserted on a loosely-checked or hardcoded-magic-number outcome. Reviewer's
fix pattern:
1. Replace the real RNG with a fake/injected RNG object whose `random()`/`randint()`
   sequence is fully scripted, so the exact code path is deterministic.
2. Derive the expected value **from the function under test's own helper** (e.g.
   `n = _log2_ceil(9)` then assert `== 100 - n`), not a hardcoded literal — so the test
   still catches a regression in that helper instead of just echoing it back.

### Pattern B — duck-typed capability checks
PR #1 round 1 (`6fe966a`) replaced `hasattr(scheduler, "arm_alpha")` with an explicit
`supports_priors = True/False` class attribute declared on every scheduler. Good news:
grep confirms this was already rolled out consistently to all current scheduler classes
(`monte_carlo.py`, `cmaes.py`, `exp3.py`, `hierarchical.py`, `contextual.py`,
`gp_ucb.py`, `epsilon_greedy.py`, `seed_quality.py`) — **no action needed there**, but
new schedulers must declare `supports_priors` going forward (add to AGENTS.md).

The `hasattr(f, ...)` / `hasattr(f._elo, ...)` / `hasattr(csd, ...)` calls in
`report.py` / `stats.py` / `seed_picker.py` are a *different* case — feature-detecting
optional instrumentation attached dynamically to the live `Fuzzer` instance, not an
interface-capability question — so they're **out of scope** for Pattern B unless
someone hits a concrete bug there.

## Scope: Pattern A candidate inventory

Grepped for `for _ in range(N): ... (break|assert)` loops built on randomness. Triage
below — confirm each before touching, some may already be fine.

### Tier 1 — likely flaky, needs determinizing (same shape as PR #7/#8)
| File:line | Function under test | Notes |
|---|---|---|
| `tests/test_honggfuzz_features.py:45` | `tlv_mutate` | loop-until-`result != data` |
| `tests/test_mcts_seed_scheduler.py:139` | `sched.select` | loop-until-`== "b0"`, `break` |
| `tests/test_mutations.py:108` | `splice` | loop-until-prefix/suffix match |
| `tests/test_new_operators.py:190` | `_op_line_mutate` | loop-until-exact-output |
| `tests/test_new_operators.py:302` | `_op_fuse_old` | loop-until-changed |
| `tests/test_new_operators.py:472` | `ascii_num_arithmetic` | loop-until-non-None |
| `tests/test_new_operators.py:524` | `ascii_num_arithmetic` | loop-until-substring-in-result |
| `tests/test_new_operators.py:564` | `chunk_shuffle` | `for/else`, loop-until-changed |
| `tests/test_new_operators.py:624` | `_op_dict_compound` | loop-until-shape-check |

### Tier 2 — verify RNG is actually seeded before assuming flaky
| File:line | Notes |
|---|---|
| `tests/test_structured_mutations.py:429` | uses a `seeded` rng var — confirm it's a fixed-seed instance, not just named that |
| `tests/test_structured_mutations.py:459` | same |
| `tests/test_subtree_population_crossover.py:129` | already uses `_FixedOpRng()` — likely already deterministic, just confirm |
| `tests/test_mcts_seed_scheduler.py:122` | loop with no visible break/assert in-window — check what it actually asserts after the loop |

### Tier 3 — probable false positives, spot-check only
| File:line | Why probably fine |
|---|---|
| `tests/test_regression_forkserver_shm.py:387` | polling `_pid_alive`, not randomness |
| `tests/test_regression_operator_registry.py:340` | iterating N random *inputs* through every registered op as a smoke test, not retry-until-hit; only a problem if some op can legitimately return `None` |
| `tests/test_new_operators.py:68` | fixed 30x loop, deterministic op, no conditional skip — false positive |
| `tests/test_new_operators.py:735` | general havoc smoke test, no visible assert-on-first-hit in the excerpt — check full body |

## Plan

1. **Confirm Tier 1 list is exhaustive.** The grep was pattern-based (`for _ in
   range(` + `break`/`assert` within 15 lines) — re-run with a wider window and check
   `tests/test_new_operators.py` fully by hand since it accounts for most hits.
2. **Fix Tier 1 tests one file at a time**, in the same style as `f00927a`/`f5d599a`:
   - Inject a scripted fake RNG (reuse `_FakeRng`/`_FakeRand` helper classes already
     established in `tests/test_regression_bugreport_easy_fixes.py` — promote to a
     shared `tests/_fake_rng.py` helper if reused 3+ times).
   - Replace "loop N times hoping to see X" with a single deterministic call whose
     expected output is asserted exactly.
   - Where the expected value is itself computable from a helper function, derive it
     via that helper rather than a literal, per the `f5d599a` fix.
3. **Resolve Tier 2** by reading each site; downgrade to Tier 1 or drop as appropriate.
4. **Spot-check Tier 3**; drop from scope if confirmed non-issues.
5. **Update AGENTS.md**: add a rule under the testing section — "no retry-until-random-hit
   loops; inject a scripted RNG and assert exact output" — and a rule under the
   scheduler section — "every `Scheduler` subclass must declare `supports_priors`."
6. **Deliver as usual**: regression tests first, then a git format-patch-compatible
   zip per logical group (Tier 1 fixes can likely land as one PR; AGENTS.md update
   can ride along or go separately).

## Explicitly out of scope
- The `hasattr(f, ...)` dynamic-instrumentation checks in `report.py`/`stats.py`/
  `seed_picker.py` (different pattern, see above).
- PRs #3/#4/#5, which merged without reviewer pushback — no action implied.

## Open questions
- Is there a reason some Tier 1 ops (e.g. `chunk_shuffle`, `ascii_num_arithmetic`)
  are inherently probabilistic in production and the test is *intentionally*
  loop-based? If so, the fix is to inject a seed that's known to hit the branch
  deterministically (as PR #7/#8 did) rather than removing the loop's intent —
  confirm this reads the same before batch-editing.
