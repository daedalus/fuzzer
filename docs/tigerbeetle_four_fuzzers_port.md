# Porting "A Tale Of Four Fuzzers" (matklad / TigerBeetle, 2025-11-28) into `daedalus/fuzzer`

Repo analysed: `github.com/daedalus/fuzzer` @ HEAD (shallow clone, 2026-08-21).
Scale: 136k LOC Python, 215 test files / 59.5k LOC of tests, 135 `_op_*` mutation
operators, 10 schedulers.

## Status

| item | state |
|---|---|
| P0-1 seed discipline | **partly done** — `--fuzz-seed` option and `random_seed` fixture landed in `tests/conftest.py`. The ~250 hardcoded seed literals across the suite have *not* been migrated. |
| P0-2 distribution assertions | **pattern established** in `tests/test_scheduler_convergence.py::TestHarnessCoverage`, not yet applied to the operator registry or cmplog. |
| P1-3 scheduler convergence | **done.** See `docs/learnings/2026-08-21-scheduler-convergence.md`. Found five defects; four fixed. Section below is preserved as written, before any of it was known. |
| P1-4 minimal interface | not started |
| P1-5 exhaustive enumeration | not started |
| P2-6 negative space | not started |
| P2-7 swarm harness | not started |
| P2-8 `--performance` mode | not started |
| `parallel.py` measure-don't-model | not started |

Two of the flakes this document predicted have since been confirmed and fixed
(`test_bloom_exec_dedup` at 0.33%, `test_mb_cbh_reanchor` at 1.0%), both
unseeded RNGs in tests asserting statistical properties.

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

## P1-3 — The idealized-lab convergence fuzzer for the 10 schedulers (DONE)

> Landed. Everything below is preserved as originally written, before the
> harness was built — including the guess about what it would find, which is
> worth comparing against what it actually found in
> `docs/learnings/2026-08-21-scheduler-convergence.md`.

This is Fuzzer #3, and it maps onto your codebase almost one-for-one.

**Current state.** `core/schedulers/` has ten schedulers behind an already-clean
shared interface: `select_op(ops) -> str` and
`record(name, success: bool, weight: float)`. Existing tests
(`test_regression_scheduler_independence.py`, `test_regression_scheduler_fallback_precedence.py`,
`test_cmaes.py`, `test_montecarlo.py`, `test_contextual.py`) check import-graph
independence, fallback ordering, and structural properties. **None of them
asserts that any scheduler converges to the best arm.**

You have a decade's worth of bandit theory in the repo and no ground-truth test
that the bandits bandit.

**Port.** Build the analogue of matklad's ring-of-replicas: a synthetic
environment where the optimal answer is known *by construction*, and then be
strict about it.

```python
# Ground truth: arm k has Bernoulli reward p_k, best arm is argmax p_k.
# Not realistic — deliberately. Realism is the *other* fuzzer's job.
def converges(scheduler_cls, seed, n=20_000):
    ops = [f"op{i}" for i in range(12)]
    p = {...}                       # one clearly-best arm
    best = max(p, key=p.get)
    sched = scheduler_cls()
    picks = Counter()
    for _ in range(n):
        op = sched.select_op(ops)
        picks[op] += 1
        sched.record(op, success=rng.random() < p[op])
    return picks[best] / n
```

Assertions worth making strict, in ascending difficulty:

1. Best-arm selection frequency in the final 10% of rounds exceeds a threshold.
2. Cumulative regret is sublinear (fit `log(regret)` vs `log(t)`, slope < 1).
3. **Non-stationary variant** — decay `p_best` toward zero partway through
   (coverage saturation, which is the actual regime) and assert the scheduler
   re-converges to the new best arm within a bounded number of rounds. Most
   bandits fail here, and this is where the Elo arbitration layer's behaviour
   becomes testable.
4. Run all ten under the same environment and assert their ranking is stable
   across seeds — otherwise the Elo arbitration is ranking noise.

**The reason this matters more than the assertion itself.** matklad's ARR fuzzer
did not find a coding bug; it found that his *cost function* was wrong (median +
maximum, missing the sum term), and then that his mental model of what the
latencies even measured was wrong. Your equivalent latent risk is the reward
definition: `record(name, success: bool, weight: float)` attributes a coverage
find to a single operator name, but havoc stacks operators, and
`test_regression_elo_op_attribution.py` suggests attribution has already been
contested once. A ground-truth harness is the only way to discover that the
reward signal is measuring something other than what the scheduler assumes.

Quote worth pinning above this work: *"Don't write fuzzers to find bugs in the
code, write fuzzers to find bugs in your understanding of the problem."*

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

## P1-5 — Exhaustive enumeration through the PRNG interface ("Generate All The Things")

**Current state.** `core/rand_pool.py` is already the required abstraction — a
single `RandPool` class with `randrange(n)`, `randint(a, b)`, `choice(seq)`,
`weighted_choice(seq, w)`, `shuffle`, `sample`. Every draw has an explicit bound.
And `MutatorBase.mutate(data, rng, ...)` already takes it as a parameter. You are
one class away from matklad's trick.

**Port.** Implement `ExhaustivePool` with the same method names, backed by his
`Gen` state machine (`v[32]` of `(value, bound)`, `done()`, `p`/`p_max`). Then:

```python
gen = ExhaustivePool()
while not gen.done():
    out = engine._op_bit_offset_span(bytearray(b"\x00\x01\x02\x03"), 0, data)
    assert out is None or len(out) <= max_len
```

...and you have enumerated *every* output that operator can produce on a 4-byte
buffer, with no recursion written by hand.

**Two honest caveats specific to your `RandPool`:**

1. Its API is much larger than Zig's `int_inclusive`. `random()`, `gauss`,
   `expovariate`, `betavariate`, `gammavariate`, `lognormvariate` are continuous
   and **cannot** be enumerated. Split the interface: a *discrete core* (the six
   bounded methods) that `ExhaustivePool` implements, and a continuous extension
   it raises on. Operators drawing continuous values are simply out of scope for
   exhaustive mode — that is fine and worth stating in the docstring rather than
   papering over.
2. The `_list` bulk variants (`randrange_list`, `randbytes`, `randint_list`,
   `choice_list`) blow up the enumeration tree combinatorially. Cap them, or
   restrict exhaustive mode to operators that only draw scalars.

**Bonus, cheap and immediate.** `core/count_class.py` is a small finite space
that is currently under-tested. `tests/test_count_class.py` asserts
`len(table) == 65536` three times but never verifies the table's *contents*
against the scalar `_classify_byte` reference. A four-line exhaustive sweep
subsumes ~20 hand-written range tests:

```python
for v in range(65536):
    lo, hi = v & 0xFF, v >> 8
    assert LOOKUP_U16[v] == _classify_byte(lo) | (_classify_byte(hi) << 8)
```

Same for `bucket_bit` over 0..255, and a differential exhaustive check of the
numpy `classify_counts` path against the pure-Python path across all 256 byte
values and both parities of buffer length. That numpy/scalar divergence is
exactly the failure mode `test_regression_hotpath_invariants.py` was written to
guard against elsewhere.

---

## P2-6 — Negative space for the deserializers: `serialize ∘ deserialize`

matklad's split — `deserialize ∘ serialize == id` is positive space,
`serialize ∘ deserialize` is negative space — with the crucial refinement that
*purely random inputs bounce off the edges*. You must generate mostly-valid data
with a corrupted bit.

**Highest-value target: `core/state_store.py`.** `_SafeUnpickler.find_class` +
`_safe_loads` + `UnsafeStateError` is a restricted unpickler over on-disk state,
with a legacy JSON fallback path. It is the one deserializer in the repo with a
security-relevant allowlist and the one where "silently misinterpret valid data"
is worst. matklad's argument for offensive programming applies directly: the
call sites should assert loudly rather than fall back quietly.

Port his three-tier generator shape verbatim:

```python
raw = valid_pickle_bytes(state)           # start from a real serialization
if prng.chance(1, 20):
    raw = flip_one_bit(raw)               # hug the valid/invalid boundary
if prng.chance(1, 20):
    raw = prng.randbytes(len(raw))        # ...but keep some pure noise
# Contract: loads, or raises UnsafeStateError. Nothing else. Never executes.
```

Then count and assert both outcomes occur, per P0-2.

Same treatment, descending priority: `field_constraints._pack/_unpack` (:115,
:119), `structural_constraints.serialize_tlv` (:407), the nine `encode` methods
in `rq_encodings.py`, `grammar.serialize` (:396), `elf._decode_x86_64` (:128).
`docs/learnings/2026-08-10-structured-roundtrip-determinism.md` shows you have
been round-tripping already — this is the missing inverse direction.

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

1. P0-1 seed plugin (half a day, unblocks everything else).
2. P2-6 `count_class` exhaustive sweep (an hour, proves the pattern, deletes ~20 tests).
3. P1-3 scheduler convergence harness (the highest-value item; likely to change
   your mind about something, which is the point).
4. P1-4 `MutationContext` extraction — but do the `mutator_interface.py` `**ctx`
   fix *first* and separately, before it accretes users.
5. P1-5 `ExhaustivePool` on the discrete core, pointed at the byte-level operators.
6. P0-2 distribution assertions retrofitted as each of the above lands.
7. P2-7 / P2-8 as follow-on.
