# The cost-based Boltzmann energy: measured, and what the measurement bounds

**Date:** 2026-08-30
**Base:** `3b2c6c0`
**Closes:** the TODO item *"A/B the cost-based Boltzmann energy"*, opened when
`ab07835` shipped `SeedPicker._pick_boltzmann_seed` reading
`effective_fuzz_count` instead of `fuzz_count`.
**Verdict: no significant difference, with a bound.** The change is not
detectably better and not detectably worse. What is new here is that the
matrix can say *how large* an effect it would have caught, so this is a
bounded null rather than a bare one.

---

## 1. The result

Ten seeds per target, three replicates per cell, both arms, `direct_lite`
at `-m 65536`, 10k execs, in-process.

| target | cost wins | loses | ties | median Δ | McNemar p |
|---|---|---|---|---|---|
| `png_read.so` | 3 | 6 | 1 | −2.0 edges | 0.508 |
| `jpeg_read.so` | 2 | 5 | 3 | −1.5 edges | 0.453 |
| pooled *(reference only)* | 5 | 11 | 4 | −2.0 | 0.210 |

The direction is consistently against the shipped change on both targets —
11 losses to 5 wins pooled — which is the direction `docs/TODO.md` warned
about: down-weighting expensive seeds is down-weighting deep paths on
targets where depth costs time. It is not significant and should not be
reported as a loss. It is a lean, and it is the lean the risk predicted,
which is worth remembering if this is ever revisited.

## 2. What the matrix could have resolved

A null is only a finding if the design had power. Computed from the measured
within-cell spread (mean sd 4.6 edges on png, 4.7 on jpeg; per-cell
comparison se ≈ 3.7 after three replicates), for the paired sign test at
α = 0.05 over 10 cells:

| true effect | power |
|---|---|
| 2 edges | 15% |
| 5 edges | ~77% |
| 10 edges | ~100% |
| 20 edges | ~100% |

So an effect of 10 edges or more would almost certainly have shown, and a
5-edge effect probably would have. A 2-edge effect would not. The observed
median Δ is −2, sitting exactly in the band the design cannot resolve.

**The usable statement:** on png and jpeg, the cost-based energy changes
edge discovery by less than about 5 edges in either direction — under ~8%
of png's ~60 and ~2.5% of jpeg's ~205. Anyone wanting to resolve smaller
than that needs more replicates, not more seeds: the noise is within-cell.

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

## 5. What is still open

`grep_read.so` was not run. It is the strongest remaining candidate and the
reason is specific: its per-seed cost varies with DFA construction, so it
carries more of the signal the arm consumes than any format target, and it
now runs in-process. It was excluded on budget — ~250 s per cell against
~22 s, so a 10-seed 3-replicate pair is about 4 hours. Measured noise is
sd 5.5 on a base of ~800 edges over 3 replicates.

Note that the low relative noise does not by itself buy power: what matters
is sd against effect size, and 5.5 is comparable to png's 4.6 in absolute
edges. A grep run at 3 replicates and 10 seeds would resolve roughly the
same ~5-edge floor, at ten times the cost. Worth doing only if the question
is specifically whether grep's wider cost dispersion changes the answer —
which is a real question, not a formality, since it is the one target where
the arm's input actually varies strongly.

The eviction-ordering item (`docs/TODO.md`) inherits all of this: the same
replicated design, the same lock, and the same ~5-edge resolution floor.
