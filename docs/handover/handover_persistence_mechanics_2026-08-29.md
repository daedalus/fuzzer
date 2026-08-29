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
mutation call, so it measures target execution and nothing else. It also
survives resume, and `_maybe_prune` rebuilds `seed_meta` from survivors, so it
cannot outlive its seed.

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

**Before writing this**: measure the spread of `total_time / fuzz_count` across a
real corpus on a campaign that has run long enough to have stale seeds. If exec
times are tightly clustered on our targets the two criteria agree and this is not
worth the churn. On `ffmpeg_read` they will not be clustered; on `png_read` they
may well be. This is display-only today, so the measurement is cheap and the
change is low risk — do this one first.

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
