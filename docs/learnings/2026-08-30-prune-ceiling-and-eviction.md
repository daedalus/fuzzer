# A bound raised to make a cost go away, and the two bugs that hid behind it

**Date.** 2026-08-30. **Base.** `b9100a5`. **Why.** `EdgeTracker._maybe_prune`
had not run since `fe8fd42` (2026-08-27), which raised `max_tracked_seeds` from
200 to 200,000 inside a performance commit whose message does not mention it.
That blocked the third consumer in backlog D5 and, more importantly, silently
removed a memory bound and the `_edge_owner_count` rebuild.

## The raise was a real fix for a real cost

Reverting the number alone would have reintroduced an 18x regression. Measured,
600 `record_edges` calls of 800 edges each:

| ceiling | wall time | tracked at end |
|---|---|---|
| 200 (prunes) | 14.78 s | 200 |
| 200,000 (never prunes) | 0.81 s | 600 |

The cause is that pruning back to the ceiling *exactly* leaves `excess == 1` in
steady state. Every subsequent insertion is one over budget again, so the
O(tracked seeds x edges per seed) owner-count pass ran on **every**
`record_edges` call in order to evict a single seed. The ceiling was not
raised because 200 was the wrong number; it was raised because pruning at all
was unaffordable.

This is worth naming as a pattern: a bound that is expensive to enforce gets
raised until it stops being enforced, and the commit reads as a speedup because
it is one. The mechanism disappears without a single line of it being deleted,
and nothing fails.

## 200,000 was never a memory bound

The ceiling exists to bound `seed_edges` and its eight companion maps. Measured
RSS per tracked seed, and what the ceiling implies:

| edges per seed | per seed | at 1,000 | at 200,000 |
|---|---|---|---|
| 500 (png_read-shaped) | 95.3 KiB | 93 MiB | 18.2 GiB |
| 2,000 | 148.9 KiB | 145 MiB | 28.4 GiB |
| 8,189 (ffmpeg_read-shaped) | 592.2 KiB | 578 MiB | 113.0 GiB |

A ceiling of 113 GiB is not a ceiling. Note also that a *seed count* is the
wrong unit for a memory bound — the same count spans 6x in cost across our own
targets — but a count is what the field is, and changing the unit is a larger
change than unblocking this one.

## The fix

Prune in batches: overflow drops to `max_tracked_seeds - int(max_tracked_seeds
* 0.1)`, so the owner-count pass amortises over the next ~10% of the ceiling in
insertions instead of running per insertion. 14.78 s → 1.15 s at ceiling 200,
against 0.81 s for never pruning. The batch is floored by truncation rather
than `max(1, ...)`, so a ceiling under 10 gets a batch of zero and keeps the
exact old behaviour; that is what lets every existing small-ceiling test keep
its semantics unmodified rather than being rewritten to fit.

Ceiling set to 1,000 on the memory figures above: it binds at the 221-403
tracked seeds our measured campaigns reach, and worst case costs 578 MiB.

## Two correctness bugs that batching makes reachable

Both lose coverage silently, both were unreachable while exactly one seed was
evicted at a time, and both would have become routine the moment pruning
resumed. Neither is a regression introduced by this work — they were waiting.

**1. The protection snapshot.** Seeds owning a singleton edge were marked
protected once, up front. Two seeds jointly owning one edge each see an owner
count of 2, so neither is protected, and evicting both drops the edge that the
protection logic exists to preserve. Reproduced directly.

**2. The two phases working against each other.** Evicting a subsumed seed can
make another seed's edges unique. The subsumption phase would evict A because B
covered for it, then the age phase would evict B — dropping an edge that
neither eviction loses on its own.

Both are fixed by the same change: one pass ordered cheapest-first by how much
unique coverage goes with the seed, revalidating that figure at the moment of
eviction and re-queueing the entry if it went stale. Fully-subsumed seeds have
a loss of zero, so the old subsumption-before-age behaviour falls out of the
ordering rather than needing a separate phase, and ties keep insertion order so
age remains the tiebreak it always was.

## The age fallback is not a corner case

The `_maybe_prune` docstring, and the §1c analysis in
`docs/handover/handover_persistence_mechanics_2026-08-29.md`, both treat
age-based eviction as a rarely-taken last resort. It is not: which seed gets
evicted is settled by the tiebreak in essentially every prune.

**Correction, 2026-08-30.** The first version of this section reached that
conclusion from the wrong evidence and gave the wrong reason, and both are
worth recording because the mistake is an easy one to repeat.

It reported that subsumption evicted 0 of 420 seeds while the fallback evicted
all of them, measured on a synthetic workload — and explained it with a
structural argument: a seed is admitted because it contributed coverage no
other seed had, so owning a unique edge must be the normal state. The argument
is plausible and the number was real. But the synthetic workload gave every
seed a guaranteed-unique edge *by construction*, so a loss of at least 1 for
every candidate was an artifact of the generator, not a finding. The reasoning
was invented to explain data that had been arranged to produce it.

Instrumenting real campaigns instead — every `_maybe_prune` call over png_read
(ceiling 150) and gzip_read (ceiling 60) — gives the opposite distribution:

| target | candidates with loss 0 | frontier loss | cost spread inside the tie |
|---|---|---|---|
| png_read | 99.9% | 0 at every prune | 148x – 864x (median 352x) |
| gzip_read | 99.9% | 0 at every prune | 8x – 710x (median 69x) |

Nearly every tracked seed is *fully subsumed*, not uniquely-covering. The
missing half of the structural argument is that uniqueness decays: a seed is
admitted for coverage no other seed had, and then later seeds cover it too.
By the time the ceiling binds, the edge space has saturated and redundancy is
the normal state.

So the original conclusion survives and its reason does not. The coverage-loss
ordering is right to have, because it is exactly what protects the 0.1% of
candidates that still hold unique coverage — but it is a tie for everything
else, and the tiebreak carries the decision. That is what makes §1c worth
doing: among coverage-equivalent candidates, accumulated cost is the only
signal that varies, and it varies by two to three orders of magnitude.

The general lesson is the same one twice over: **a mechanism that never runs
and a branch that always runs both look, from the outside, like the thing you
expected.** The ceiling made the first invisible and the docstring made the
second invisible. And the follow-on: a synthetic workload will confirm whatever
its generator was written to contain, so a measurement that produces a
satisfying structural explanation is exactly the one to re-run against real
data.
