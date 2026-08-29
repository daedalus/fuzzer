# Porting "A Tale Of Four Fuzzers" (matklad / TigerBeetle, 2025-11-28) into `daedalus/fuzzer`

Repo analysed: `github.com/daedalus/fuzzer` @ HEAD (shallow clone, 2026-08-21).
Scale: 136k LOC Python, 215 test files / 59.5k LOC of tests, 135 `_op_*` mutation
operators, 10 schedulers.

## Status

Landed and pruned from this file: P1-3 scheduler convergence (five defects
found, four fixed — `docs/learnings/2026-08-21-scheduler-convergence.md`), P1-5
exhaustive enumeration (`core/exhaustive_pool.py`, 70 of 134 operators fully
enumerable — `docs/learnings/2026-08-22-exhaustive-pool-p1-5.md`), and P2-6
negative space (`tests/test_count_class_exhaustive.py` —
`docs/learnings/2026-08-22-count-class-exhaustive.md`). What each of them found
is in those learnings notes, which is where to look rather than here.

What is still open:

| item | state |
|---|---|
| P0-1 seed discipline | **partly done** — plugin landed; all 14 bare `Random()` migrated and guarded by `tests/test_seed_discipline.py`. The ~250 hardcoded seed literals have *not* been migrated, deliberately: see the note in that commit. |
| P0-2 distribution assertions | **pattern established** in `tests/test_scheduler_convergence.py::TestHarnessCoverage`, not yet applied to the operator registry or cmplog. |
| P1-4 minimal interface | **prerequisite done** — `MutationContext` replaces the `Fuzzer` in `core/mutator_interface.py`'s `mutate()` and `is_available()`, closing the `**ctx` leak while that interface still had no implementors. The `operators.py` extraction itself (365 `self.f` attribute reads, 29 distinct, 249 of them `max_len`/`_rand_pool`) is not started. |
| P2-7 swarm harness | not started |
| P2-8 `--performance` mode | not started |
| `parallel.py` measure-don't-model | not started |

Three of the flakes this document predicted have since been confirmed and fixed
(`test_bloom_exec_dedup` at 0.33%, `test_mb_cbh_reanchor` at 1.0%,
`test_inserts_magic_value` at 0.658%), all unseeded RNGs in tests asserting
statistical properties.

**Open follow-up from P1-5, not unfinished P1-5.** 20 operators are unenumerable
*only* because a coin flip is written `rng.random() < 0.5` rather than
`rng.randint(0, 1)`. Converting them is mechanical, does not change the
distribution, and would roughly double the enumerable set. Recount 2026-08-22:
20 sites remain (of 31 `rng.random()` uses in `core/`), across `arm.py`,
`der.py`, `webm.py`, `isobmff.py`, `protobuf.py`, `gif.py`, `webp.py`,
`structured.py`, `wfc.py` and `schedulers/cmaes.py`. The original count of 21
predates the `_op_regex_bomb` / `_op_utf8_widen` fixes.

## The framing that matters

The post is not about fuzzing *targets*. It is about **fuzzing your own
subsystems**, and the recipe is: minimal data interfaces → drive them through a
PRNG *interface* → then reuse that same interface four different ways
(exhaustive, negative-space, idealized-lab, hammer-a-single-instance).

`daedalus/fuzzer` is a fuzzer whose own internals are barely fuzzed. The tests
are overwhelmingly example-based with hardcoded seeds. That is the gap, and the
post is an unusually good map of how to close it.

Ranked by value-per-effort:

---

## P0-1 — Seed discipline: generate the seed *outside* the test process

**Current state.** `tests/conftest.py` has no seeding infrastructure at all.
Across the suite: 34× `Random(1)`, 29× `seed=42`, 26× `seed=1`, 14× `Random(7)`,
13× `Random(42)`, 12× `Random(5)`, 11× `seed=7`, 11× `Random(3)`, 14× bare
`Random()`, and a long tail. Roughly 250 seed literals. Every run explores the
same points forever; the 14 bare `Random()` calls are the worst case — random
but unrecoverable.

**Why this repo specifically.** matklad's argument for out-of-process seed
generation is "the test process can explode completely and the parent still
prints the seed." You have already been bitten by exactly that:
`docs/handover/suite_segfault_z3_finalization_2026-08-16.md` and
`docs/handover/test_shm_hang_2026-08-14.md`. A suite that segfaults during
finalization is precisely the case where an in-test `print(seed)` is lost. This
is not a theoretical benefit here.

**Port.** In `tests/conftest.py`:

```python
def pytest_addoption(parser):
    parser.addoption("--fuzz-seed", type=lambda s: int(s, 0), default=None)

def pytest_configure(config):
    seed = config.getoption("--fuzz-seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little")
    config._fuzz_seed = seed

def pytest_report_header(config):
    # printed at session start, before any test runs -> survives SIGSEGV
    return f"fuzz-seed: reproduce with --fuzz-seed=0x{config._fuzz_seed:016x}"

@pytest.fixture
def random_seed(request):
    # per-test derived seed, so test order does not perturb reproduction
    return zlib.crc32(request.node.nodeid.encode()) ^ request.config._fuzz_seed
```

The `pytest_report_header` placement is the whole trick — the seed hits stdout
before collection, so a hard crash or a hang leaves it in the CI log.

**Then the dual-run pattern.** matklad's `check(92)` + `check(random_seed)`:
statistics asserted against a fixed seed, coverage accumulated with a random
one. Rewrite the ~60 highest-value randomised tests to that shape; leave the
genuinely deterministic example tests alone.

Do *not* port Zig's `std.testing.random_seed` literally — the Python-side
equivalent is the plugin above.

---

## P0-2 — Assert that your generators actually reach both sides of the boundary

matklad's `route_decode` test looked fine and generated **zero** valid codes in
100 million attempts. His fix was counters plus `assert(counts.valid > 50)`.

**Current state.** Nothing in the suite asserts anything about the distribution
of its own generated inputs. The clearest case is
`tests/test_operator_smoke.py:36` — every one of the 135 operators is exercised
against a *single* 32-byte buffer, `b"\x00\x01...\x07" * 4`. Operators gated on
structure (`_op_der_*`, `nal`, `isobmff`, `protobuf`, `zip`, the 13 regularity
operators) will decline on that buffer, and the test still passes. It is
measuring "the dispatch table is wired up," not "the operators work" — and the
docstring is honest about that, but the coverage assertion is missing.

**Port.** Wherever a test generates inputs and branches on validity, count both
sides and assert both are non-negligible:

- `operator_registry` availability predicates / `MutatorBase.is_available` — assert
  each operator both fires and declines at least N times across a generated
  corpus, so a permanently-declining operator is a test failure rather than a
  silent pass.
- cmplog pair matching (`core/cmplog.py`) — assert both hit and miss.
- `core/structural_constraints.py`, `core/field_constraints.py` — assert both
  satisfiable and violating instances get generated.

matklad's quick-and-dirty alternative is worth adopting as a review habit: drop
an `assert False` into a branch you believe is reached and confirm the suite
goes red.

---

## P1-4 — The minimal-interface argument, with numbers

matklad: pass `op: u64`, not the whole `Prepare`. Injecting the whole `Replica`
is the banana-gorilla-jungle pattern.

**Current state — this is the clearest structural finding in the repo.**
`services/operators.py` (3,511 lines) references `self.f` — the `Fuzzer` instance
(`services/fuzzer.py`, 5,239 lines) — **423 times**. But only **29 distinct
attributes** are ever touched, and the distribution is extremely skewed:

| attribute | refs |
|---|---:|
| `self.f.max_len` | 133 |
| `self.f._rand_pool` | 116 |
| `self.f.corpus` | 17 |
| `self.f._cmplog` | 15 |
| `self.f._crash_mi` | 9 |
| everything else (24 attrs) | ≤ 5 each |

**249 of 423 references — 59% — are an `int` and a PRNG.** The essential
interface of the mutation layer is roughly `(max_len: int, rng: RandPool)` plus
read-only views of the corpus and cmplog tables. Everything else is accidental
dependency, mostly lazily-constructed format-mutator singletons
(`_png_mutator`, `_zip_mutator`, `_webm_mutator`, … 3 refs each) that could be
owned by the operators themselves.

**Cost you are already paying.** `tests/test_operator_smoke.py:14` has to build a
full `Fuzzer` — temp corpus dir, temp crashes dir, a compiled `targets/test_target`
binary — to call `_op_bit_flip` on a bytearray. That construction cost is why
that test can only afford one buffer instead of thousands. Fix the interface and
that test becomes a real fuzzer for free.

**Port.**

1. Introduce a `MutationContext` dataclass (`max_len`, `rng`, `corpus_view`,
   `cmplog`, `dictionary`) and thread that instead of `self.f`. Mechanical, and
   most of it is a rename of `self.f.max_len` → `ctx.max_len`.
2. **Fix `mutator_interface.py` before it ossifies.** The new class-based
   interface is nearly right — `mutate(self, data, rng, max_len=0, **ctx)` passes
   `rng` and `max_len` as data, exactly as matklad advocates — but its docstring
   states `**ctx` currently carries `fuzzer`, the `Fuzzer` instance. That is the
   banana-gorilla-jungle pattern re-entering through the new door. Since this
   interface is young and low-churn, this is the cheapest moment in the repo's
   life to replace `fuzzer` with the narrow context object. matklad's point that
   *"the best time to capture an interface is before the first line of code is
   written"* applies with unusual force here — you are one commit past that
   moment, not a hundred.

**Where you're already ahead.** matklad's fix for `Routing` keeping a private
`view: u32` copy is an `invariants()` method asserting
`self.view == self.routing.view`. You have the equivalent already —
`self._invariants` in `operators.py` and
`tests/test_regression_hotpath_invariants.py`, which pins the `RandPool`
numpy-array / Python-list mirror equivalence and the `ExecutionTimeTracker`
divisor cache. That is exactly the right pattern. Extend it: `_havoc_table`,
`_havoc_trials`, `_elite_pool_corpus_len`, `_redqueen_sorted_version`,
`_region_cache`, and `_invariants_corpus_len` are all derived caches of corpus
state whose staleness would be silent. Each deserves a line in `_invariants()`.

---

## P2-7 — Fuzzer #4: hammer one instance in a radioactive room

matklad's fourth fuzzer drops the multi-replica model entirely, instantiates a
single `Routing`, and calls every public method in random order, obeying only the
documented preconditions — checking a weak but always-true invariant (the
next-hop chain visits each replica exactly once).

**Current state.** Your subsystem tests call methods in the order the `Fuzzer`
happens to call them, which restricts you to executions the real campaign
produces — precisely the limitation matklad identifies.

**Port.** Random-method-order harnesses with one always-true invariant each:

- `core/edge_tracker.py` — every tracked edge index stays within the map size;
  owner counts never go negative. (You fixed an owner-count bug on the SHM path
  once already.)
- `core/bloom.py` — no false negatives, ever, under any interleaving of
  `add`/`query`/resize.
- `services/corpus_manager.py` — corpus entry count matches on-disk file count;
  every entry's edge set is a subset of the global map.
- `core/state_store.py` — `get`/`set`/`save`/`load`/`cleanup_legacy` in random
  order; `__len__` always equals the number of sections that survive a
  save/load cycle.

Cross-reference his earlier post *Swarm Testing Data Structures*, which is the
detailed version of this technique.

---

## P2-8 — Fuzzer #5: `--performance` mode, which you have most of

VOPR's performance mode fixes fault parameters to realistic values, controls
drastic faults explicitly (`--replica-missing=2`), and reports **message counts**
— then ARR was validated by running with and without it and checking the counts
improved across the board.

**Current state.** You are closer to this than to anything else in the post.
`tools/gen_synthetic_target.py` is the right primitive and its docstring already
articulates matklad's core principle better than he does: *"ground truth is known
by construction rather than inferred from the target's behaviour, which is what
makes a false-negative rate measurable at all."* `--blocks`, `--unstable N`, and
the provably-dead byte region are three known-by-construction knobs.
`tools/bench_havoc_subop.py`, `docs/sweeps/`, and
`tests/test_bench_paired_stats.py` are the beginnings of the harness.

**Port.** Formalise a fixed-profile campaign mode: pinned seed, pinned synthetic
target, pinned fault profile, N execs, reporting a counter table (execs, unique
edges, corpus size, operator fire counts, cmplog hits) the way VOPR reports its
message table. Then A/B features against it. matklad's caveat is worth
inheriting verbatim: this is hard to turn into a test that fails *only* on bugs,
so run it as an experiment harness, not a CI gate.

---

## The one genuine *algorithmic* port: measure, don't model, in `parallel.py`

Everything above ports the *testing* methodology. There is exactly one place
where the ARR idea itself transfers.

`services/parallel.py` syncs corpora between workers on a **fixed
`sync_interval: int = 30`** (:248) via `_sync_corpus_in` (:189), which scans all
sibling directories with a per-sibling file-count cursor. That is a static,
modelled topology and a static, modelled cadence — matklad's V1 ring, and subject
to the same fallacy: it assumes the cost/benefit of syncing is uniform across
siblings and constant over time. It is not; late in a campaign most synced seeds
are redundant.

The PCC-style alternative he describes — *don't model, do something and measure
the outcome directly* — maps cleanly: track, per sibling, the fraction of imported
seeds that produced new coverage; periodically run an experiment with a different
interval or a different subset of siblings; keep the change if measured
edges-per-import-second improves. You already have the measurement machinery
(`core/seed_quality.py`, `core/elo.py`, `services/stats.py`) and, ironically, ten
bandits sitting in `core/schedulers/` that solve exactly this problem and are
currently only pointed at operator selection.

And per P1-3, the ground-truth convergence harness would be the thing that tells
you whether the adaptive version actually beats the fixed 30s — which is the same
order matklad followed: build the fuzzer that can judge the algorithm, then build
the algorithm.

---

## Not worth porting

- **ARR itself.** No distributed consensus, no replication. Only the "measure,
  don't model" kernel transfers, as above.
- **`std.testing.random_seed` verbatim.** Language-specific; the pytest plugin in
  P0-1 is the real port.
- **Exhaustive enumeration over `RandPool`'s continuous methods.** Not possible;
  scope it out explicitly rather than fudging it.
- **The u64-permutation encoding trick.** Cute, but you have no six-element
  permutation to serialize.

---

## Suggested sequence

1. P1-4 `MutationContext` extraction in `operators.py` — the `mutator_interface.py`
   `**ctx` fix is already done and was deliberately done first, before it
   accreted users.
2. P0-2 distribution assertions, retrofitted onto the operator registry and
   cmplog as other work touches them.
3. P0-1's remaining ~250 hardcoded seed literals, if and when they cost
   something.
4. P2-7 / P2-8 as follow-on.
5. `parallel.py` measure-don't-model — the one genuine algorithmic port here,
   and independent of the rest.
