# Handover: Persistence Mechanics — what ports, what does not

**Source.** Kate Lenore Meyer, *"Persistence Mechanics: Architecture, Dynamics,
and the Entropic Action"*, 26 August 2026 (11pp, independent, unpublished), plus
a two-script matplotlib demo shipped alongside it. Brought in for evaluation on
2026-08-29 against head `ad7dbb0`. Not our work and nothing here is vendored.

**Verdict.** One item ports. Four are already in the tree under other names and
are recorded below so they are not re-proposed. Nothing ports from the demo
scripts. The port that survives (§1) is not really the paper's physics — it is
one methodological claim that happens to name a gap we have.

This document is analysis, not a plan. Effort estimates are absent on purpose:
§1 is three independent changes to three subsystems and each wants its own
measurement before it is worth writing.

**Status (round 17).** The gating measurement was run and the ledger is not
clustered, so §1 survives — see
`docs/learnings/2026-08-29-per-seed-cost-ledger.md`. §1a and §1b are
implemented on top of a new `core/cost_ledger.py`. §1c is **not** written and
should not be: its host function no longer runs (see the note under §1c). Two
factual corrections to what follows are marked inline.

---

## What the paper claims

Two halves. The **architecture**: three existential problems (Finiteness,
Conservation, Causality) crossed with two modes of engagement (provision,
defense) yield six irreducible pillars — Capacity, Map, Protocol, Governor,
Toll, Margin. Necessity is argued by counterfactual elimination (remove one,
lifetime goes to zero); sufficiency by claiming the taxonomy is exhaustive.

The **dynamics**: any state transition has a payload, a cost, and a rate, giving
a dissipation density `D = xi * K * f` — irreducible toll per operation, local
Kolmogorov complexity, coordinated transition flux. Integrating `D` over the
system's spatial footprint gives an instantaneous impedance `Z_net(t)`, and
integrating that over the lifetime gives the **Entropic Action**
`S_phi = integral of Z_net dt`. A multiplicative survival argument then yields
`P(survival | trajectory) = exp(-beta * S_phi)`, from which least-action
behaviour falls out as survivorship rather than optimization. Appendix A derives
a contention cross-term with a `rho / (1 - rho)` waiting form.

---

## 1. The one port: read the ledger we already keep

The paper's §III.G argues that viability cannot be judged pointwise — that what
accumulates across a system's whole history is what determines whether it
survives, and that instantaneous measures systematically mislead. That claim
names a gap we have, independently of anything else in the paper.

`meta["total_time"]` is a per-seed accumulator. It is written in
`Fuzzer.fuzz_one` (`services/fuzzer.py`, under the comment *"Per-seed wall-clock
cost"*) and it is clean: the timing-contamination fix moved `t_start` below the
mutation call, so it measures target execution and nothing else.

**Correction (round 17): it did not survive resume.** This paragraph originally
claimed it did. `total_time` was absent from `CorpusManager.save_state` and
`load_state` while `fuzz_count` was persisted, so a resumed seed carried a
large restored count against a zero numerator, hit the `max(1.0, ...)` floor in
every reader, and read as the cheapest seed in the corpus — permanently, since
the numerator restarts at zero and the count does not. Measured on `png_read`:
after a 200-execution resume, 116 of 216 fuzzed seeds carried
`total_time == 0.0` against 407 restored fuzzes.

**Second correction: `fuzz_count` was the wrong denominator even within one
run.** The initial seed replay in `Fuzzer.run` increments the count without
crediting any time; the two disagreed on 126 of 147 timed seeds in the same
campaign. `core/cost_ledger.py` introduces `cost_samples`, a denominator
counting exactly the executions whose time is in the numerator, persisted
alongside `total_time`. It also makes *unmeasured* and *measured as free*
distinguishable — before, a zero numerator and the 1 microsecond floor were the
same value, so an untimed seed won every edge in the favored set.

It has exactly three readers. **All three divide it by `fuzz_count`** to recover
a mean `exec_us`:

| reader | symbol | what it computes |
|---|---|---|
| `services/fuzzer.py` | `Fuzzer._cull_queue` | cheapest-seed-per-edge for the favored set |
| `services/fuzzer.py` | `Fuzzer.run` | `exec_us` fed to `SeedScorer.score` |
| `services/corpus_manager.py` | `CorpusManager.auto_minimize_corpus` | set-cover tie-break above 5000 edges |

The accumulated quantity is never consumed as an accumulated quantity. We keep a
cost ledger and only ever read its average. Three places where the cumulative
form is the better question:

### 1a. Stale-seed detection is count-based, and the cost data is already there

`StatsReporter._print_summary_seeds` (`services/stats.py`) flags a seed stale on:

```python
m.get("fuzz_count", 0) >= 50 and m.get("coverage_edges", 0) == 0
```

A seed costing 200 ms per execution and one costing 0.2 ms are declared equally
exhausted at 50 fuzzes, though one burned a thousand times the budget to earn
that verdict. `total_time` is in the same dict. Cost-based futility —
*"spent more than X seconds of target time and found nothing"* — is the question
a scheduler actually wants answered, and needs no new plumbing.

**Measured, and this guess was backwards.** `png_read` at a realistic
`max_len` is the *spread* target (p90/p10 4.32x, CV 1.455) and the flat
decompressor `gzip_read` is the clustered one (1.32x, CV 0.106): cost disperses
where the input controls how much work the target does and concentrates where a
fixed overhead dominates. The count criterion and an equal-sized cost criterion
disagreed on 10 of 13 flagged seeds on `png_read`, and the seeds it called
exhausted had burned between 6.9 ms and 116.9 ms for the same verdict. Full
numbers in `docs/learnings/2026-08-29-per-seed-cost-ledger.md`.

**Implemented** in `StatsReporter._print_summary_seeds` against
`effective_fuzz_count`, with the threshold named `STALE_SEED_EXEC_EQUIVALENTS`
and carried over at 50 verbatim — it was never calibrated and is not calibrated
here.

### 1b. The Boltzmann arm's energy term

`SeedPicker._pick_boltzmann_seed` (`services/seed_picker.py`) uses
`E = log(fuzz_count + 1)`, giving `P(seed) proportional to (fuzz_count + 1)^(-1/T)`.
Substituting accumulated cost for the count makes the arm literally the paper's
`exp(-beta * S_phi)` with `T = 1 / beta`, and gives it a semantics it currently
lacks: seeds decay in proportion to what they actually cost, not to how often
they happened to be picked.

**Caveat that decides this.** `E = log(n+1)` and `E = log(total_time+1)` differ
only where per-seed exec cost varies — which is the same precondition as §1a. If
1a's measurement says exec times are clustered, this changes nothing and should
not be written. If it says they are not, this is a one-line change that needs an
A/B through `tools/bench_paired.py` like everything else in the backlog, because
it changes seed selection and could plausibly make things worse: down-weighting
expensive seeds is down-weighting deep paths on targets where depth costs time.

**Implemented, with one correction to the shape above.** The energy must not be
`log(total_time + 1)` on raw seconds: for any campaign shorter than a few
seconds of target time per seed that puts the whole corpus at `E ~ 0` and turns
the arm uniform without `T` changing — a silent behaviour change disguised as a
one-line substitution. `effective_fuzz_count` converts the ledger to
average-cost executions instead, which keeps `T` meaning what it meant and
makes the substitution *exactly* the identity under uniform cost rather than
approximately so. Measured on `png_read`, `fuzz_count` and `total_time` order
the corpus only weakly alike (Kendall tau 0.46), so this does change selection
there.

**The A/B is still owed.** It was not run here. This changes seed selection and
the argument above for why it could make things worse still stands untested.

### 1c. `_maybe_prune` has no cost term at all

`EdgeTracker._maybe_prune` (`core/edge_tracker.py`) evicts by subsumption first
(seeds owning no singleton edge) and falls back to insertion order. Neither phase
consults cost. The subsumption phase is correct as it stands and should not be
touched — it is protecting coverage, which outranks cost. The **age fallback** is
the arbitrary one: when every candidate owns a unique edge, "oldest first" is a
tiebreak with no defence behind it, and cumulative cost per retained edge is at
least an argued one.

Lowest confidence of the three. Eviction changes which seeds exist, so it is the
hardest to A/B cleanly, and the age fallback rarely fires.

**Not written, and the reason is stronger than low confidence: the host
function no longer runs at all.** `_maybe_prune` returns immediately unless
`len(seed_edges) > max_tracked_seeds`, and that default went from 200 to
200,000 in `fe8fd42` ("perf: skip Katz ICFG when no targets and reduce
seed-picker/calibration overhead", 2026-08-27) — a 1000x change inside a
performance commit whose message does not mention it. Verified: 500 seeds
recorded, zero pruned. Campaigns in this measurement reached 221-403 seeds, so
neither phase fires any more. Putting a cost term in the age fallback would be
adding a heuristic to dead code, which is precisely the bug family §2 of this
document catalogues.

That cap change has consequences past §1c and is tracked as its own open item
in `docs/TODO.md`: the memory bound `_maybe_prune` provided is gone, and the
`_edge_owner_count` rebuild that lives in the same function (added because
stale owner counts degrade the rarity signal that drives the schedule) no
longer runs either. Whether 200,000 was intended is a separate question from
this port and is not decided here.

**Update (round 18): the blocker is fixed and two of this section's premises
were wrong.** The ceiling is 1,000 with batched pruning, so `_maybe_prune`
runs again; details in `docs/learnings/2026-08-30-prune-ceiling-and-eviction.md`.
Correcting what is written above, because a later reader would otherwise plan
against it:

* *"The subsumption phase is correct as it stands and should not be touched."*
  It was not. Its protection test was a snapshot taken before any eviction,
  which is only sound when exactly one seed is evicted — the situation batching
  ends. Two seeds jointly owning an edge each see an owner count of 2, so the
  snapshot protects neither and evicting both drops the edge the phase exists
  to protect. Reproduced, then fixed by revalidating at the moment of eviction.
* *"The age fallback rarely fires."* It is the only path that fires. Measured
  on a corpus-shaped workload, subsumption evicted 0 seeds and the fallback
  evicted all 420: every seed owns a unique edge precisely because owning one
  is what got it admitted, so subsumption almost never has a candidate.

The second correction promotes §1c rather than closing it. The fallback is not
a corner case to be tidied — it is the eviction policy. It now orders by how
much unique coverage an eviction costs, which is a defensible tiebreak where
insertion order was not, so the specific complaint in this section ("oldest
first is a tiebreak with no defence behind it") is answered. Whether *cumulative
execution cost* should enter that ordering as well is now a live, testable
question rather than a blocked one, and it inherits the A/B requirement from
§1b since it changes which seeds exist.

---

## 2. Already in the tree — do not re-propose

Recorded because "absent from the doc" and "considered and rejected" are
different states, and only one of them should be re-proposable.

**`D = xi * K * f` is AFL's `perf_score` with new letters.** `core/schedules.py`
is already a product of speed, bitmap and depth factors, and both
`Fuzzer._cull_queue` and `CorpusManager.auto_minimize_corpus` already compute
`cost = exec_us * input_size` — that is `xi * K` with the same multiplicative
structure and the same extensivity argument. Adding `f` would be a category
error: in a fuzzer, transition rate is the thing we *control*, not a property we
observe. The paper's ceiling `xi * K * f <= D_max` is a constraint on a system
that cannot choose its own flux; we choose ours every pick.

**The exponential filter is the `--boltzmann` arm.** Shipped, wired into
`_OPERATOR_STRATEGY_NAMES`, reachable via `--boltzmann`. The paper supplies an
interpretation of it, not a mechanism we lack. (Its one useful consequence is
§1b, which is about the energy term, not the filter.)

**Appendix A's `rho / (1 - rho)` is weaker than what we measured.** The obvious
target is the bounded linear probe in `adapters/afl_shim.c` (`__AFL_PROBE_MAX`),
which is a genuine load-divergence phenomenon. But that comment already carries
*measured* drop rates against `ffmpeg_read` at load 0.77 — window 8 → 4.43%,
16 → 1.60%, 32 → 0.40%, 64 → 0.04% — and the correct closed form for linear
probing is Knuth's, not a memoryless queue. The paper's model has the wrong
failure geometry and less evidence behind it. Nothing to take.

**The six pillars are a relabeling.** Mapping Capacity/Map/Protocol/Governor/
Toll/Margin onto fuzzer subsystems produces a diagram and no decisions. The one
part with content is not the taxonomy but its *test*: a component whose removal
changes nothing was never load-bearing. That is the falsification discipline this
repo already applies to every fix — and it is the shape of a bug family we keep
finding, where a component exists, is documented, and is inert:

- the honggfuzz entropy branch, present since the `power.c` port with no producer for `input_entropy` (fixed `ad7dbb0`)
- `_hf_entropy_penalties`, declared, summed and formatted as `ent:N`, never incremented (same commit)
- `cmaes` missing from the Elo ballot in `services/operators.py`, making its dispatch arm dead code
- `SeedPicker._pick_by_similarity`, `Fuzzer._weighted_pick_seed`, `_havoc_mutate` and four `MonteCarlo` methods, still listed as open cleanup in `docs/TODO.md`

No port. Noted because it is a useful sentence to keep, not a framework to adopt.

**Nothing ports from the demo scripts.** `persistence_mechanics_demo.py` (3
figures) is a strict subset of `persistence_mechanics_full_demo.py` (6 figures);
both are matplotlib illustration with no reusable numerics. They also do not
faithfully implement the paper — the trajectory figure selects survivors on an
absolute threshold (`P > 0.05`) where the paper's equation 8 is a *ratio* against
the minimum-action trajectory, and it samples `(xi, K, f)` independently with no
`D_max` rejection, so half its candidate pool is inadmissible under the paper's
own Proposition 1. Two of the six figures underflow to a solid block on a linear
colormap. Reported upstream, not our problem, and mentioned here only so nobody
mines the demo for an implementation of the equations.

---

## Where the rest of this lives

`docs/port-backlog.md` §D5 carries §1 as a backlog item and its Rejected section
carries §2, so this file does not become a fifth orphan survey — the round-16
merge that produced `port-backlog.md` existed to prevent exactly that. The open
work is tracked in `docs/TODO.md` under *Scheduling*.

If §1a's measurement comes back saying exec times are clustered on our targets,
all three of §1a/1b/1c collapse and this document should be deleted, with the
measurement recorded in `docs/learnings/`. That outcome is worth as much as the
port and costs less to find out.

**It did not come back that way** — see
`docs/learnings/2026-08-29-per-seed-cost-ledger.md`. The measurement is
recorded there regardless, because the *shape* of the result is the reusable
part: clustering is a joint property of target and corpus, not of the target,
and the same target moved from 2.00x to 4.32x purely by raising `max_len`. Any
future consumer of this ledger should be written against
`effective_fuzz_count` for the same reason §1a and §1b were — it collapses to
the count form on a clustered target by construction, so the gate does not have
to be re-argued per target.
