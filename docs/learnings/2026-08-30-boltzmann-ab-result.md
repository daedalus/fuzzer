# The cost-based Boltzmann energy: measured, and what the measurement bounds

**Date:** 2026-08-30
**Base:** `3b2c6c0`
**Closes:** the TODO item *"A/B the cost-based Boltzmann energy"*, opened when
`ab07835` shipped `SeedPicker._pick_boltzmann_seed` reading
`effective_fuzz_count` instead of `fuzz_count`.
**Verdict: no significant difference, with a bound.** The change is not
detectably better and not detectably worse on any of the three targets that
can carry it. What is new here is that the matrix can say *how large* an
effect it would have caught, so this is a bounded null rather than a bare
one.

---

## 1. The result

Three replicates per cell, both arms paired in time, `direct_lite` at
`-m 65536`, 10k execs, in-process. Ten seeds on png and jpeg; twenty on
grep, for the reason in §1.

| target | seeds | cost wins | loses | ties | median Δ | McNemar p |
|---|---|---|---|---|---|---|
| `png_read.so` | 10 | 3 | 6 | 1 | −2.0 edges | 0.508 |
| `jpeg_read.so` | 10 | 2 | 5 | 3 | −1.5 edges | 0.453 |
| `grep_read.so` | 20 | 12 | 7 | 1 | +3.0 edges | 0.359 |

No target reaches significance and the directions do not agree: png and
jpeg lean slightly against the change, grep slightly for it. Nothing here
supports either sign.

The grep half is worth recording in full because it is a lesson about
reading interim results. The first ten grep seeds came back 7 wins to 2
losses, median +4.5, p = 0.180 — and grep is the one target measured to
have strongly varying per-seed cost (CV 0.922, p90/p10 7.18x), which is
precisely where the cost-based energy is supposed to help. That is a
mechanism-consistent story and it was tempting. Seeds 10-19 were then run
to settle it and came back **5 wins to 5 losses**: an exact coin flip. The
combined 12/7 with p = 0.359 is the honest number, and the apparent signal
in the first half was noise that happened to line up with the theory.

The general point: an interim split on half a matrix is not a weak version
of the result, it is a different quantity. With a per-cell standard
deviation of 12.3 edges, a 7/2 split over ten cells is entirely ordinary
under the null.

## 2. What the matrix could have resolved

A null is only a finding if the design had power. Computed from the measured
within-cell spread (mean sd 4.6 edges on png, 4.7 on jpeg; per-cell
comparison se ≈ 3.7 after three replicates), for the paired sign test at
α = 0.05 over 10 cells:

| true effect | png / jpeg (10 cells, sd ~4.6) | grep (20 cells, sd 12.3) |
|---|---|---|
| 2 edges | 15% | 9% |
| 5 edges | ~77% | 38% |
| 10 edges | ~100% | 91% |
| 20 edges | ~100% | 100% |

grep needed twice the seeds to reach comparable power because its
within-cell spread is ~12 edges in absolute terms against png's ~4.6. Low
*relative* noise (CV 0.007 on a base of ~800) is not low noise for this
purpose: what competes with the effect size is the absolute spread.

So an effect of 10 edges or more would almost certainly have shown, and a
5-edge effect probably would have. A 2-edge effect would not. The observed
median Δ is −2, sitting exactly in the band the design cannot resolve.

**The usable statement:** on png and jpeg the cost-based energy changes
edge discovery by under about 5 edges in either direction (under ~8% of
png's ~60, ~2.5% of jpeg's ~205); on grep, by under about 10 edges (~1.2%
of ~800). Resolving smaller than that needs more replicates on png and
jpeg, where the noise is within-cell, and more of both on grep.

## 3. Why the original matrix could not have produced this

Two problems with `direct_lite` as specified in the Boltzmann handover, both
found by measuring rather than reasoning (`tools/noise_probe.py`, same arm,
same seed, repeated).

**Half the set carries no information.** `zlib_read.so` and `lz4_read.so`
return 12 edges and `gzip_read.so` returns 36, bit-for-bit, on every
replicate of every seed. They are saturated. They cannot produce a
discordant pair for any arm, so their 60 cells per arm are budget spent on a
guaranteed tie. The handover predicted zlib and gzip would be no-ops from
the flat-cost identity; the real reason is stronger, and it also covers lz4,
which that argument missed.

**A cell is not a fixed function of its seed.** Same arm, same seed, five
replicates: png spans 45–63 and 39–66 (CV 0.118 / 0.189); jpeg spans
201–206 and 196–209. Across replicate pairs of the same arm, **46% differ** —
each one a pair McNemar would have scored a win or a loss had the two
replicates carried different arm labels.

That does not invalidate McNemar. Arm assignment is independent of the
noise, so discordant pairs still split 50/50 under the null and type I error
is controlled. What it destroys is power. One replicate per cell puts a
~7-edge standard deviation against an effect now bounded under 5.

**And the noise cannot be engineered away.** The obvious repair is to
virtualise the clock so cells reproduce exactly. It is unavailable:
`effective_fuzz_count` is `total_time / mean_exec`, which under constant
per-execution cost reduces to `fuzz_count` *exactly* — the same identity
that disqualifies the `locked` set. A deterministic clock would collapse the
two arms into the same computation. The quantity under test and the noise
are the same phenomenon. The only lever is repetition.

## 4. Two harness changes this forced

**Arms paired in time** (`tools/bench_replicated.py`). `bench_paired.py`
runs one arm to completion, then the other. With hours between them any
drift in machine state is confounded with the arm. Here the two arms run
back to back inside each replicate, and each arm is selected by source tree
via `PYTHONPATH` rather than by editing a file between runs — which also
removes the mislabelling risk the handover warns about, since the tree an
arm ran from is what defines it.

**A single-process lock** (`tools/bench_lock.py`, `--lock-single-thread`).
This one was forced by an incident. Two campaigns were started in this
container by two operators unaware of each other; on one core they halved
each other and cells went from ~22 s to ~50 s. That is not merely slow.
`mean_exec` is a measured quantity and the arm under test reads its energy
from it, so a second process perturbs the input to the thing being measured —
silently, because every cell still completes and still records a plausible
number. The contaminated cells are in `results/paired/contaminated/` and
were not used.

The flag refuses to start and names the current holder rather than queueing:
a run that waited would still be launched by someone who believed they had
the machine alone. It is `flock` rather than a pid file so that a holder
killed by a VM restart — which happened twice during this run — leaves the
lock free rather than wedging the machine. `tests/test_bench_lock.py` covers
exclusion, the refusal code, unclean holder death, and the thread caps;
falsified by switching `LOCK_EX` to `LOCK_SH`, at which point 2 of the 6
fail on behaviour.

## 5. A cost estimate that was wrong by 5x

`grep_read.so` was first deferred on the grounds that it cost ~250 s per
cell, putting a paired 10x3 run at about four hours. That number was stale:
it predates the in-process grep port in `7eb1c2a`. Measured, a grep cell is
**~40 s**, and the full 20-seed paired run took about 80 minutes.

The item was filed as reasonably deprioritised when it was in fact the
cheapest remaining thing to do, and the deferral reasoning was written up
in enough detail to look considered. Worth a habit: when a cost estimate is
load-bearing for a decision to *not* do something, measure it rather than
carrying it forward from an earlier write-up, especially across a port that
changed the execution path.

`tools/cost_dispersion.py` was added on the way. The identity from §1 of the
Boltzmann handover -- that under uniform per-execution cost the two arms are
arithmetically the same computation -- is a property of the *target*, not of
the harness, so it needs checking per target before cells are spent on one.
Measured on grep: CV 0.922, p90/p10 7.18x, max/min 32.5x, over 810 seeds
carrying cost samples. That is by far the widest dispersion of any target in
the set, which is why grep was the interesting one to run even though it did
not produce a result.

## 6. An operational trap worth not repeating

The lock did its job in production: a duplicate launch of the grep run was
refused with rc 3 and named the holder. But the refused process still ran
the shell's trailing `; echo DONE > stamp`, so a completion stamp appeared
while the real run had 17 cells left. Anything wrapping a harness in
`cmd; echo DONE > stamp` inherits this -- the stamp records that *a*
process exited, not that the work finished.

Check the results file, or the holder pid, not the stamp. The run log is
also unreliable for this: two processes redirecting to the same path
interleave and the file reads as binary, so `grep -c` under-counts. The
checkpointed JSON is the only honest progress record.

## 7. What is still open


Resolving the sign on grep, if anyone wants it. The 12/7 split leaves a
possible effect of up to ~10 edges unresolved, and grep is the target where
the arm's input genuinely varies. Closing that would take roughly 40 seeds
at 3 replicates, or 20 seeds at 6 -- two to three hours either way. Whether
that is worth spending on an effect bounded under 1.2% is a judgement call,
and the honest default is no.

The eviction-ordering item (`docs/TODO.md`) inherits all of this: the same
replicated design, the same lock, and the same ~5-edge resolution floor.
