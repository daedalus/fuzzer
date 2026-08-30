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
age-based eviction as a rarely-taken last resort. Measured on a corpus-shaped
workload — overlapping edges plus the one unique edge each seed was admitted
for — subsumption evicted **0** seeds and the fallback evicted **all 420**.

The reason is structural and should have been predictable: a seed is admitted
to the corpus *because* it contributed coverage no other seed had. Owning a
unique edge is therefore the normal state of a tracked seed, not the exception,
so the subsumption phase almost never has a candidate. Everything that matters
about eviction quality lives in the path that was assumed not to matter.

That is the general lesson here, and it is the same one twice: **a mechanism
that never runs, and a branch that always runs, both look like the thing you
expected from the outside.** The ceiling made the first invisible; the docstring
made the second invisible. Only counting which branch fired distinguished them.
