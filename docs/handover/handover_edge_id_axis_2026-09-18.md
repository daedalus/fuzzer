# The (edge_id, hit_count) matrix: what the id axis is, which analyses are defined on it

**Status:** analysis plus one new standalone diagnostic,
`tools/edge_matrix_analysis.py`. No production code changed. Two findings
below (F1, F2) are defects with reproductions and no fix yet. Every number
here was measured at `c26f0a1`; the patch is rebased onto `a45a63a`, whose two
intervening commits add operator schedulers and touch neither the shim, the
SHM path, nor any file cited below.

## Trigger

"Map edge id to x and hit count to y -- what analysis can we apply to that
2-D structure?" The short answer is that most of the 2-D toolbox is not
*defined* here, and the interesting part is why, so this documents what the
axis is, what survives, and what the measurement traps are.

## What the x axis actually is

`src/fuzzer_tool/adapters/afl_shim.c`:

    edge_id = (caller_ctx ^ prev_loc ^ cur_loc) | 1        (__afl_map_edge)
    caller_ctx = splitmix64(return_address_two_frames_up) & __AFL_CTX_MASK
    guards = 1, 2, 3, ...                (__sanitizer_cov_trace_pc_guard_init)

Three consequences, in increasing order of how much they change what you may
compute:

1. **XOR is an ultrametric.** `|a - b|` is not defined on the id axis at all;
   the only notion of nearness is "shares a high-bit prefix". Intervals,
   gradients and anything integrating along x are undefined. This is the same
   conclusion the Wasserstein/CRPS family reached the expensive way -- those
   metrics were computed over the edge index, measured nothing, and were
   moved onto the `log2(1 + hit count)` axis. That precedent is the reason
   this doc exists rather than another round of it.
2. **The axis is blocked, not nominal.** The context term only perturbs the
   low `__AFL_CTX_BITS`, so `id >> ctx_bits` is a context-free edge family and
   `id & mask` is the caller tag. Measured below: the two builds of the same
   target produce *identical* sets of `id >> 8` values (22 of them), which is
   the direct confirmation that the high bits are context-invariant.
3. **`|= 1` kills tag bit 0.** The OR exists so that a valid edge cannot hash
   to the "empty slot" sentinel, and it is correct for that. Its side effect
   is that `__AFL_CTX_BITS=8` yields **7 usable bits**: only 128 of the 256
   tags are reachable, and two call sites whose context hashes differ only in
   bit 0 are merged. Neither the `__AFL_CTX_BITS` comment nor the `|= 1`
   comment mentions the other.

## Measurement setup

Target: `targets/fuzzgoat_read.c` + vendored fuzzgoat, clang 18,
`-fsanitize-coverage=trace-pc-guard -fno-omit-frame-pointer -O2`, 1712 guards.
Two builds: default (`__AFL_CTX_SENSITIVE=1`) and `-D__AFL_CTX_SENSITIVE=0`.
250 inputs (120 generated JSON documents, 120 byte-level mutations of those,
10 hand-written edge cases), one execution each, 65536-entry map.

    python3 tools/edge_matrix_analysis.py --target <exe> --corpus <dir> \
        [--ctx-bits 0] [--keep-aslr]

Caveat on the target: `fuzzgoat_read.c` calls `__afl_map_edge(0x1000 + ...)`
by hand for AST-walk structure, so ids above the natural `2 * guard_count`
range are partly harness-made. Family *counts* and the correlations are
unaffected; the raw id range is not worth reading closely on this target.

## Findings

### F1 (defect). Edge ids are not reproducible across processes under ASLR

`__afl_get_caller_ctx` hashes a raw return address. Its comment says
"ASLR/PIE base differences across runs don't matter here: we only need
identical call chains WITHIN one process/session to hash identically" -- which
is true for one process and false for the subprocess execution model, where
every input is a new process with a new PIE base.

Six executions of one input, `__AFL_CTX_SENSITIVE=1`:

| | sizes | union | intersection | Jaccard |
|---|---|---|---|---|
| ASLR off | 106 x6 | 106 | 106 | **1.000** |
| ASLR on | 98, 98, 99, 96, 93, 105 | 295 | 2 | **0.007** |

Over the 250-input corpus the union of ids goes from 445 (ASLR off) to 1513
(ASLR on) -- a 3.4x phantom inflation, all of it noise, every phantom id
owned by exactly one seed and therefore maximally "rare" to the seed picker.
With `__AFL_CTX_SENSITIVE=0` both regimes are exactly stable (318 ids), so
the context term is the entire cause.

Production is protected: `services/fuzzer` calls `adapters/process.disable_aslr`
once at startup and children inherit the personality. The gap is the
documented escape hatch. `disable_aslr`'s own docstring says
`FUZZER_KEEP_ASLR=1` is there for "an ASAN target on a kernel whose fixed
mmap layout collides with ASAN's shadow range" -- i.e. the one variable a
user is told to set when ASAN targets won't start is also the variable that
silently destroys context-sensitive edge identity. Suggested guard (sketch,
`services/fuzzer.py`, next to the existing `self._aslr_disabled` assignment):

```python
self._aslr_disabled = disable_aslr()
if not self._aslr_disabled and detect_ctx_bits(self.target) > 0:
    # elf.detect_ctx_bits already reads the __afl_ctx_bits_N marker before
    # the first execution, for map sizing.
    log.warning(
        "ASLR is on and the target is a __AFL_CTX_SENSITIVE build: edge ids "
        "are per-process and coverage will be noise. Rebuild with "
        "-D__AFL_CTX_SENSITIVE=0 or unset FUZZER_KEEP_ASLR."
    )
```

A cheaper and more general check is to run the stability probe itself:
`_calibrate_seed_baselines` already re-executes seeds, and
`measure_stability()` in the new tool is the whole test in twelve lines.
That would also catch ASLR-independent instability, which is what
`mask_edges` currently discovers one edge at a time.

**F1 is fixed in the shim as of dd834d1**, which resolves `ra` to a
load-base-relative offset before hashing when `FUZZER_KEEP_ASLR=1` is set in
the target's environment. Everything above still describes what a target
built before that commit does, and the numbers are unchanged for it. See
P0-2 for the residue: the shim keys the mode off the variable rather than off
whether ASLR is actually on, which leaves the refused-`personality()` case in
raw mode, and an old-shim target cannot be told from a current one by any
symbol it exports.

### F2 (defect, mechanism unresolved). The first execution against a clean table disagrees with every later one

Same input, same process image, ASLR off, one SHM segment reused with
`reset_edge_map()` between executions -- exactly the production arrangement:

    run 0 (clean table)   n=105  sum_counts=756  path_hash=0xfd4499b0c623864e
    run 1 (dirty table)   n=106  sum_counts=756  path_hash=0x7679a69957647d7c
    run 2 (dirty table)   n=106  sum_counts=756  path_hash=0x7679a69957647d7c

Run 0 carries 3 ids that never reappear; runs 1+ carry 4 that run 0 lacks.
Steady state from run 1 onward is exact. With `__AFL_CTX_SENSITIVE=0` the id
*set* and the per-edge counts are identical across all runs, but the path
hash still differs between run 0 and runs 1+ (`0xbf5dbcc2...` vs
`0x1f911d2c...`), i.e. the same edges fire the same number of times in a
different order.

Two hypotheses falsified:

- *The generation value matters.* No. A clean table at generations 0, 1, 2
  and 5 gives byte-identical results (n=90, same path hash, same counts).
  The discriminator is table contents, not the tag.
- *Edges are being dropped from a crowded probe window.* No.
  `read_dropped_edges()` is 0 in every run.

Not yet explained: how table contents can change the order in which the
target's blocks fire at all. `__afl_map_edge` is `always_inline` into
`__sanitizer_cov_trace_pc_guard`, which SanitizerCoverage skips by name, so
the claim-vs-reclaim branch should not be self-instrumented. Worth pinning
down before trusting any first-execution measurement: the first execution of
a campaign is the one that calibrates seed baselines, and an edge that
appears only there is exactly the shape `mask_edges` interprets as
nondeterministic. Reproduction is four lines with the tool's `_run_one`.

Mitigation in the tool meanwhile: one discarded warm-up execution before
collecting, and `first_exec_matches` reported separately from the
steady-state Jaccard so the two never get averaged together.

### F3. The id axis carries block structure, and only that

`__AFL_CTX_SENSITIVE=1`, ASLR off, 445 edges, 22 families:

- variance of `log2(1 + count)` explained by family (ICC): **0.490**
- lag-1 autocorrelation of count along id order: **+0.106**, permutation null
  `-0.003 +/- 0.037`, **z = +2.94**
- control, counts shuffled *within* each family: **+0.054**
- `__AFL_CTX_SENSITIVE=0`: lag-1 **+0.037**, z = +0.82 -- indistinguishable
  from the null. The context-free axis carries nothing at all.

Read: there is a real but weak block effect under the default build, roughly
half of it surviving the within-family control, and none of it a trend. The
defined statistic is the nested decomposition, not a fit.

**The trap, stated so it is not walked into twice:** run the same analysis
with ASLR on and you get ICC 0.876 and lag-1 +0.856 at z = +33.8. Those
numbers are pure F1 noise -- ASLR scatters each real edge across the whole
tag space, which manufactures exactly the block structure the statistic is
looking for. A strong result here is evidence of a broken setup before it is
evidence of anything about the program.

### F4. What context sensitivity costs and buys on this target

1.40x more distinct ids (445 vs 318) for an ICC of 0.490. Largest observed
family is 102 distinct tags of the 128 reachable, on a JSON parser of 1712
guards -- so the tag space is already close to saturation on a toy target,
and `__AFL_CTX_BITS` has no feedback loop anywhere: it is a compile-time 8
with a comment telling the reader to check the drop counter by hand. ICC and
family occupancy together are the measurement that would let it be chosen.

### F5. The y marginal is bimodal; do not quote a power law

445 edges: Shannon 6.02 bits of a possible 8.80, i.e. **65 effective edges**
(`2^H`) carrying the execution volume; Gini 0.871; top 1% of edges take 28.9%
of all hits. AFL count classes are U-shaped -- 69 edges in class 1, 129 in
class 128. The Zipf fit is poor (log-log slope -2.31, residual sd 0.88), as
expected for a mixture; the tool prints the residual next to the slope so the
slope cannot be quoted alone.

`2^H` is the one scalar here with an obvious unused consumer:
`edge_hit_distribution()` has had no caller since it was written, and
"effective edges collapsing while edge count is flat" is a saturation signal
that the stall machinery currently has no equivalent of.

### F6. Axes that do carry meaning

Spearman against total hit count, over the same 445 edges:

| substituted x | rho (CTX on) | rho (CTX off) |
|---|---|---|
| owner count (`_edge_owner_count`) | +0.957 | +0.945 |
| first-seen index (`_edge_first_seen`) | -0.854 | -0.864 |
| per-edge max count (`_max_counts`) | +0.747 | +0.767 |

The first row quantifies the old `_weight_edge_penalties` defect: volume was
used where incidence was meant, and rho = +0.96 is why it produced
plausible-looking numbers for as long as it did. The fourth meaningful axis,
the distance table's `node_idx`, is the only one that is program geometry
rather than a hash of it, and needs a `__AFL_DISTANCE_MODE` build to collect;
the tool says so rather than pretending otherwise.

### F7. SVD is the most ASLR-fragile statistic we have

Asked as a follow-up: what does SVD give on this? Nothing on the
`(edge_id, count)` object itself -- two incommensurable columns, so its SVD is
a rescaled covariance of a label against a count. The matrix SVD is defined on
is **seed x edge**, which is `core/schedulers/seed_tang.py` territory
(`--tang`, shipped opt-in 2026-09-07; findings in
`handover_done_2026-09-06.md` §14, open A/B as E6 in
`handover_pending_2026-09-06.md`). Nothing here reopens that verdict.

What is new is how badly F1 damages the spectrum. Same 250 inputs, same
target, only ASLR toggled:

| matrix | sigma1^2/‖A‖_F^2 | effective rank | rho at k=10 |
|---|---|---|---|
| clean ids, raw counts | **0.952** | 1.1 | 0.042 |
| clean ids, log1p | 0.820 | 1.5 | 0.195 |
| clean ids, binary | 0.643 | 2.3 | 0.317 |
| ASLR on, raw counts | **0.104** | 28.0 | 0.718 |
| ASLR on, log1p | 0.325 | 8.5 | 0.673 |
| ASLR on, binary | 0.276 | 12.3 | 0.770 |

(effective rank = participation ratio of the squared spectrum; rho(k) =
‖A − A_k‖_F / ‖A‖_F, the same quantity §14 reports for png and zlib.)

Every phantom id is owned by exactly one seed, so ASLR pushes the matrix
toward block-identity. The edge count inflated 3.4x; rank-1 dominance fell by
a factor of nine and effective rank rose 25x. A campaign run with
`FUZZER_KEEP_ASLR=1` would report a target as irreducibly high-rank when it
is essentially rank-1 -- which is exactly the measurement E6 turns on, and
ASAN targets are both the reason that variable exists and a plausible way to
run ffmpeg. **E6's ffmpeg re-measurement must assert ASLR is off before
trusting a spectrum.**

Contrast with the failure signature §14 already recorded: there, binary,
log1p and raw counts gave *identical* spectra, which was a harness bug
upstream of all three. A spectrum that does not move when the cell semantics
change is broken; a spectrum that moves this much when ASLR changes is also
broken. Both checks are one line.

### F8. GF(2) reduction of the coverage matrix: sound, incomplete, and outclassed

Asked as the next follow-up: binarise the matrix and row-reduce over GF(2).
Measured on the same two matrices, binary cells:

| matrix | GF(2) rank | real rank | distinct rows | XOR-dependent | union-redundant |
|---|---|---|---|---|---|
| clean ids | 133 | 133 | 162 | 117 | **235** |
| ASLR on | 250 (full) | 250 | 250 | 0 | 151 |

**What the reduction detects is redundancy, soundly but partially.** A row
reduces to zero exactly when its coverage is the symmetric difference of
earlier rows, and XOR can only set bits present in an operand, so its edges
are necessarily contained in their union. That is a theorem, not a
coincidence, and the data agrees: 117 of 117 XOR-dependent seeds are union-
redundant. Incompleteness is equally structural -- dependence *requires*
cancellation, and a seed whose edges are a subset of another's produces none,
so it stays independent. The reduction found 117 of the 235 seeds that are
actually redundant.

**Two by-products are real artefacts.** The 133 pivot columns are injective
on the row space -- verified, all 162 distinct coverage vectors stay distinct
when restricted to those 133 edges, a 445 -> 133 bit separating fingerprint.
And the 133 basis seeds necessarily cover every edge in the corpus (same
support argument), so the reduction hands back a valid corpus-minimisation
solution with no greedy loop.

**But as a minimiser it loses outright.** Greedy set cover on the same matrix:
**31 seeds in 15.6 ms**, against the GF(2) basis's **133 seeds in 15.3 ms**.
Same cost, 4x worse cover -- and not from implementation slop: the basis must
keep every independent row, including rows independent only because of a
single edge nobody else has.

The reason is algebraic and worth stating once so it is not re-derived.
Coverage lives in a union-closed lattice; elimination needs a field, and the
operation that makes GF(2) a field is the one with no meaning here -- "seed A
hits edge e, seed B hits edge e, so together they do not" is not a statement
about coverage. The right analogue is Boolean rank (biclique cover number),
which is NP-hard, which is why the minimiser is greedy in the first place.

**Third F1 canary, and the sharpest.** ASLR takes GF(2) rank from 133 to a
fully degenerate 250: every seed acquires private singleton columns, so no
dependency can exist. More fragile than union redundancy (235 -> 151) and
than the spectrum.

Where GF(2) elimination does earn its place in this tree, it already has one,
and in every case because the underlying relation genuinely is linear:
`core/xor_map_solver.py` for exact byte-to-map relations,
`invert_bitmask_map` in `core/gf2_common.py`, `LiveBitMaskEstimator`'s
OR-accumulator over XOR diffs, Berlekamp-Massey for CRC recovery, and the
GF(2)-linear PRNG recovery at `21e2621`. Checksums, LFSRs and bit-permutation
maps are linear over GF(2). Coverage is not, and 133-vs-31 is what that costs.

The one variant not tested: rank as an *incremental* scalar rather than a
batch decomposition -- each admitted seed either raises the GF(2) rank or does
not, at O(rank * n/64) per insert, giving a "structurally novel" test distinct
from `_check_new_coverage`, which only ever asks about new edges. Speculative,
and subject to the same pre-registered A/B bar as P3-2 below.

### F9. LLL over the raw-count matrix: real structure, none of it sparse

Asked next: reduce the lattice. Three results.

**The literal (edge_id, count) lattice is all of Z^2.** Hermite normal form of
the 445 generators is the identity, index 1, so LLL returns the standard basis.

**The raw-count seed x edge matrix is rationally dependent, and not by
accident.** 250x445, rank over Q **108** -- counts are *more* dependent than
incidence, whose rank was 133. 178 distinct rows, 72 exact duplicates. LLL on
a 100-row slice via the standard `[I | N*A]` construction, where a reduced row
whose `A`-part vanishes carries an exact relation in its `I`-part:

| | relations | support | max abs coeff | L1 | time |
|---|---|---|---|---|---|
| real slice, 100x359, rank 88 | 12 | 52 of 100 | 47 | 670 | 111 s |
| column-shuffled null | 0 (rank 100) | -- | -- | -- | 114 s |
| generic rank-88 integer matrix, same shape | 0 | -- | -- | -- | 291 s |

Both controls matter. Shuffling each column independently -- preserving every
column's marginal, destroying co-occurrence within a run -- returns a **full
rank** matrix, so the deficiency is structure rather than an artefact of the
value distribution. And a synthetic rank-88 matrix of the same shape yields
LLL nothing, so the real kernel vectors are genuinely shorter than a generic
lattice's.

**But the relations are dense, and that is not a limitation of LLL.**
Exhaustive search over the 178 distinct rows finds **0** scalar-multiple pairs
and **0** exact `A = B + C` triples. No seed's count profile decomposes as a
small integer combination of others, so the only short vectors in this lattice
are duplicate-row differences -- which a hash finds in O(m) rather than LLL's
111 s. A 52-seed relation with coefficients reaching 47 is numerically exact
and says nothing about the corpus.

The honest limit: LLL's approximation factor is 2^((n-1)/2), vacuous at
n = 100. Absence of short relations is proven only for support <= 3, by the
exhaustive search, not by the reduction.

Cost ranking across everything now tried on this matrix, same data: hashing
microseconds, SVD ~1 ms, GF(2) 15 ms, LLL 111 s and superlinear in rows. Most
expensive, least returned, and unlike GF(2) it does not even leave a valid
cover behind. Section [6] of the tool is opt-in (`--lll`) for that reason.

**Where LLL would belong in this tree, one module over.**
`core/prng_state_recovery.py` is GF(2)-linear by construction -- its `Opcode`
set is MOV/XOR/SHL/SHR/AND and its docstring states that nothing else is
allowed, because that is what makes a step symbolically evaluable. That covers
xorshift, LFSR and Taus and structurally excludes anything multiplicative.
Recovering a linear congruential generator's state from truncated output is
the textbook lattice application (Frieze, Hastad, Kannan, Lagarias, Shamir),
`java.util.Random`, the `rand48` family and PCG's LCG core are all plausible
in a target that salts a format, and the dimension there is 5-20 rather than
100. See P4-1 below.

### F10. The transpose: two invariants, and one verdict that flips

Every matrix analysis above re-run on `A^T` (445 edges x 250 seeds).

| | seed x edge | edge x seed |
|---|---|---|
| singular spectrum | -- | identical to **9e-13**, all three cell semantics |
| GF(2) rank / rank over Q | 133 / 108 | 133 / 108 |
| XOR-dependent rows | 117 | 312 |
| union-redundant rows | 235 of 250 | **445 of 445** |
| greedy cover | 31 | **1** |
| duplicate rows | 72 of 250 | **126 of 445** |
| sparse relations, exhaustive | 0 multiples, 0 triples | **4 multiples, 34 triples** |
| LLL, 100 rows | 12 relations, support 52, max coeff 47, 111 s | **45 relations, support 5, max coeff 1**, 42 s |

The first two rows are theorems -- `sigma(A) = sigma(A^T)` and rank is
orientation-free -- and the tool prints them as controls, because a
difference there means the pipeline is wrong rather than the target
interesting.

**Two metrics are seed-side only and degenerate transposed.** Union redundancy
goes to 445 of 445 (no seed fires exactly one edge, so every edge sits inside
the union of the others) and greedy cover goes to 1: a single edge every input
reaches. That last number is section [4]'s 0.952 rank-1 dominance restated
combinatorially -- the parser prologue. Section [5] now prints a note rather
than reporting them as if they meant something.

**126 duplicate edges, and most of them are ours.** Edges whose count profile
is identical across all 250 executions -- same information, nothing in this
corpus told them apart. Split by `id >> 8`: **79 edges in 14 classes sit
inside a single family**, i.e. context tags that never once differed, map slots
`__AFL_CTX_SENSITIVE` bought and did not use. The largest is 45 edges, all
family 17, consecutive odd ids from 4481. The other 47 edges in 21 classes span
families and are straight-line block chains -- a property of the target. This
is a better input to P3-1 than the ICC: exact and countable rather than a
variance fraction, and it separates instrumentation waste from target
structure. Section [7] of the tool reports it.

**LLL flips from vacuous to informative.** F9's relations over seeds were dense
(support 52 of 100, coefficients to 47). Over edges they are sparse with unit
coefficients (support 5, max coeff 1), and the exhaustive search that found
nothing among seeds finds 4 scalar-multiple pairs and 34 `A = B + C` triples
among edges. That is flow conservation: edge counts on a CFG satisfy
Kirchhoff's law at every join and loop, so the kernel of `A^T` is essentially
the cycle space, and this is Ball-Larus optimal-profiling territory -- only a
spanning tree's complement needs instrumenting, the rest is derivable. It also
explains the rank: 445 edges carry at most 108 independent count coordinates.

**The caveat that gates acting on it.** These relations hold across 250 runs of
one target, which does not distinguish structural from coincidental. The
separation is available: `core/icfg.py` already builds the CFG, so an
empirical relation can be checked against it. See P1-2.

## Not defined on the id axis -- do not re-propose

Linear regression or slope of count against id; autocorrelation or FFT along
id; kernel density over id; clustering in the (id, count) plane; convex hull,
fractal or box-counting dimension; Wasserstein, CRPS or KS with id as the
ground metric; treating `Spearman(id, count)` as a trend (the tool prints it
at -0.34 precisely so that it is visibly non-zero and visibly meaningless).
Every one of these needs an ordered or metric x. Bit-prefix (trie) bucketing
is the defined replacement when a positional statistic is genuinely wanted.

## The tool

`tools/edge_matrix_analysis.py`, standalone, numpy only (scipy is not a
dependency, so Spearman is rank + Pearson in-file). Reuses
`adapters.shm.ShmCoverage`, `adapters.process.disable_aslr` and
`core.count_class.classify_single` rather than reimplementing any of them.

    # collect and analyse
    python3 tools/edge_matrix_analysis.py --target targets/fuzzgoat_read \
        --corpus ~/fuzzing/corpus/json --save /tmp/edges.npz

    # re-analyse without the target
    python3 tools/edge_matrix_analysis.py --load /tmp/edges.npz --json out.json

    # reproduce F1
    python3 tools/edge_matrix_analysis.py --target ... --corpus ... --keep-aslr

Eight sections: [0] cross-process id stability, [1] x-axis structure, [2] the
permutation-invariant y marginal, [3] substituted axes, [4] the singular
spectrum, [5] the GF(2) structure, [6] integer relations (opt-in, `--lll`, the
only section needing sympy) and [7] edge equivalence classes. `--transpose`
runs [4] to [6] on the edge x seed matrix and enables [7]. Per Hard Rule 46 the
lag-1 statistic ships with both of its controls -- a global permutation null
*and* a within-family shuffle that must leave the effect standing if the
effect really is the blocking -- because the within-family control is what
separates F3 from F1's artefact.

## Follow-up items

Tiered by what the item *is*, matching `handover_pending_2026-09-06.md`: P0 a
defect in shipped code a live campaign can hit, P1 measurement with the
harness already in hand, P2 shipped-but-unwired, P3 design work gated on a
question answered on paper first, E an evaluation run.

### P0-1. Diagnose F2 (first-execution divergence)

Gates every first-execution number, and the first execution is what
calibrates seed baselines. The path hash proves the fire *sequence* differs;
it does not say where. Decisive experiment, one run:

```c
/* afl_shim.c, inside __afl_map_edge, after edge_id is final */
#if __AFL_TRACE_FIRES                 /* env-gated like _CMPLOG_COUNTS: the fd is the flag */
    if (__afl_fire_fd >= 0) {
        char b[16]; int n = snprintf(b, sizeof b, "%u\n", edge_id);
        if (n > 0) { ssize_t w = write(__afl_fire_fd, b, (size_t)n); (void)w; }
    }
#endif
```

Run the same input against a clean table and against a generation-reset one,
diff the two id streams, and read off the first divergence. Deliverable is a
diagnosis; then either a fix or a stated invariant ("the first execution after
a fresh segment is not comparable"), plus a regression test pinning whichever
it is. Do not build anything on top of first-execution measurements until
this closes.

### P0-2. Guard F1 (per-process edge ids under ASLR) -- DONE, rescoped

Written against the pre-dd834d1 tree, then re-aimed after the shim fix landed.
What follows is the second version; the first is only interesting for the two
sketch bugs it found, kept below.

dd834d1 does not close the item, it moves it. The shim's base-relative mode
switches on `FUZZER_KEEP_ASLR=1`, read by `getenv()` **in the target
process** -- and that is not the same condition as "ASLR is on".
`disable_aslr()` also returns False when `personality()` is refused (seccomp,
container runtimes, non-Linux), and there the variable is unset, so a
context-sensitive target runs in raw mode under live ASLR with nothing
reporting it. Asking the user to fix that by hand means asking them to set a
variable named for *keeping* ASLR on a host that never turned it off.

Two pieces landed:

1. `Fuzzer._ensure_ctx_ids_are_exec_stable()`, after the SHM setup block in
   `__init__`: if ASLR survived startup, coverage is on the SHM path and any
   target carries a positive `__AFL_CTX_BITS`, set `FUZZER_KEEP_ASLR=1` in
   `os.environ` -- which `services/runner` copies into every child -- and say
   so in one line. Silent when there is nothing to do. `disable_aslr()` has
   already run and cached its answer by then, so writing the variable cannot
   change what it decides; a target built before dd834d1 ignores it.
2. `Fuzzer._report_edge_id_stability()`, three executions of one seed at the
   end of `_calibrate_seed_baselines`, modelled on `_report_comparison_reach`:
   same warn-only shape, same call site, no masking. This is the part that
   survives the shim fix, because **an old-shim target cannot be detected
   statically** -- dd834d1 adds no marker symbol, so `detect_ctx_bits` cannot
   distinguish a binary that honours the variable from one that ignores it.
   Only a measurement can. It also catches the causes the id axis knows
   nothing about (threads, time, uninitialised memory), which is what
   `mask_edges` currently discovers one edge at a time without reporting a
   cause.

Details worth keeping:

* The probe measures the *widest* seed of the calibration pass, tracked for
  free in the loop already there: more edges, more chances for a moving id,
  identical cost. It runs after the seeds have, so the table is warm and F2's
  first-execution regime cannot masquerade as F1.
* No threshold was invented. Any Jaccard below 1.0 is reported; the
  attribution is printed only when the diagnosable cause is present. Since
  relative mode is already on by the time the probe runs, that diagnosis is
  now specific: a target that still drifts is one that predates dd834d1.
* Drops during the probe abstain, for the same reason
  `_calibrate_seed_stability` abstains: a saturated table discards by arrival
  order, so set divergence is not evidence of drift.
* The collection loop is shared (`_repeat_edge_sets`) with
  `_calibrate_seed_stability`, so the drop-counter drain cannot drift apart
  between the consumer that masks and the one that only reports.

Measured on a gcc build of the `test_ctx_and_map_size.py` driver -- the shim's
callback is called directly, so no clang and no real target is needed. Six
executions of the identical input, 40 guards, PIE, current shim:

| | sizes | union | intersection | Jaccard |
|---|---|---|---|---|
| ctx=8, ASLR off | 22 x6 | 22 | 22 | **1.000** |
| ctx=8, ASLR on, raw mode | 22 x6 | 76 | 0 | **0.000** |
| ctx=8, ASLR on, relative mode | 22 x6 | 22 | 22 | **1.000** |
| ctx=0, ASLR on | 22 x6 | 22 | 22 | **1.000** |

Row 2 is the gap piece 1 closes and row 3 is what it closes it to, so the two
rows together are also an end-to-end regression test for dd834d1 through the
fuzzer's own probe rather than a standalone harness. Row 4 is the control
pinning the context term rather than ASLR as the cause. Not one id survived in
row 2, sharper than the 0.007 on fuzzgoat and the same phenomenon with less
overlap to dilute it.

Two things found on the way, both fixed here:

* `tools/edge_matrix_analysis.py` claimed `--keep-aslr` "reproduces what a run
  with `FUZZER_KEEP_ASLR=1` actually sees". After dd834d1 that is false in
  both directions: the flag skips `disable_aslr()` and sets nothing, so it
  reproduces the *raw* regime, while a real `FUZZER_KEEP_ASLR=1` run is now
  base-relative and stable. Docstring and `--help` corrected; the flag's
  behaviour is unchanged and is still the sharpest canary the tool has, it is
  just no longer described as the thing it is not.
* `tests/test_aslr.py::test_aslr_still_randomizes_without_the_call` fails
  whenever it runs after any test that constructs a `Fuzzer`: `disable_aslr()`
  sets the persona on the pytest process itself and children inherit it, so
  the negative control has nothing left to randomize. Pre-existing, left
  alone. The new probe launcher clears `ADDR_NO_RANDOMIZE` itself rather than
  trusting the parent, and skips where the host cannot randomize at all;
  `test_aslr.py` could take the same treatment.

Still open, and cheap: **dd834d1 emits no marker symbol.** `__afl_ctx_bits_N`,
`__AFL_NGRAM_K` and the SHM layout tag all exist precisely so the Python side
can read a build contract before executing anything, and relative mode is the
one contract it cannot read. A `__afl_ctx_relative_capable` marker would turn
the probe's diagnosis into a startup check and would let
`_ensure_ctx_ids_are_exec_stable` say "this target will ignore the variable,
rebuild it" instead of setting it and hoping. One symbol, one `_symbol_names`
scan.

Sketch bugs from the first version, still worth recording:

* The startup hook does **not** belong beside the `disable_aslr()` call as
  sketched. That line runs before `self.use_coverage` is assigned and before
  the coverage backend is chosen, so acting there touches `--no-coverage` and
  `--no-shm` runs, neither of which reads a shim edge id.
* `disable_aslr()` returning False does not imply `FUZZER_KEEP_ASLR=1`. The
  sketch's message told the user to unset it, which for the
  refused-`personality()` case is advice to unset something that was never
  set -- and, after dd834d1, is advice that makes things strictly worse.

Original sketch, for the record:

Two forms, both worth landing; the second subsumes the first and is the one to
write if only one gets done.

1. Startup warning, `services/fuzzer.py`, next to the existing
   `self._aslr_disabled = disable_aslr()`: if it returned False and
   `elf.detect_ctx_bits(target)` is non-zero, warn that edge ids will be
   per-process and name both escapes (`-D__AFL_CTX_SENSITIVE=0`, unset
   `FUZZER_KEEP_ASLR`). `detect_ctx_bits` already runs before the first
   execution for map sizing, so this costs nothing.
2. Stability probe in `_calibrate_seed_baselines`, modelled on
   `_report_comparison_reach()` -- same warn-only shape, same
   end-of-calibration call site. `measure_stability()` in
   `tools/edge_matrix_analysis.py` is the whole test in twelve lines. It also
   catches ASLR-independent instability, which `mask_edges` currently
   discovers one edge at a time without ever reporting a cause.

Regression test: build a target both ways, assert the probe fires under
`FUZZER_KEEP_ASLR=1` and is silent without it.

### P1-1. Three targets, not one

Every number in this document is one small JSON parser. Re-run the tool on
`png_read`, `zlib_read` and ffmpeg and record ICC, family occupancy, the drop
counter and (per F7) the spectrum. No code change; machine time and a table.
ffmpeg is the one that matters -- 8189 edges, genuinely multimodal, and the
target §14's measurements never reached because it would not build in that
container. This gates P3-1 and feeds E6.

### P1-2. Check the edge-count relations against the real CFG

F10's 45 sparse relations either are Kirchhoff identities on the CFG or are
artefacts of 250 runs of one target, and nothing measured so far separates the
two. `core/icfg.py` builds the graph; the test is whether each empirical
relation corresponds to a join or a loop in it. A positive opens Ball-Larus
(instrument a spanning tree's complement, derive the rest, and stop paying for
337 of 445 count coordinates); a negative closes the whole integer-relation
line, which is worth as much. Blocked on nothing but machine time.

### P2-1. Wire `2^H` into the stall reason

`edge_hit_distribution()` has zero callers in `src/` and zero in `tests/`.
Effective edges collapsing while the raw edge count is flat is a saturation
signal the stall machinery has no equivalent of; it belongs in the same
reason string as the Allan noise type and `wall_summary()`. Self-contained,
one commit.

### P2-2. Spectrum and GF(2) sections in the tool -- DONE

Landed as sections [4] and [5]: F7's three spectrum rows and F8's GF(2) rank,
union-redundancy count and greedy-cover comparison. Both are one-command
checks now rather than paragraphs in this file, and both double as F1 canaries
(`--keep-aslr` reproduces the degenerate reading). Two guards ride along: the
dense SVD is skipped above `SVD_CELL_BUDGET` cells, and the report warns if
the raw and binary spectra agree to 1e-6, which is §14's harness-bug signature
rather than a result.

Not covered and still open: `node_idx` needs a `__AFL_DISTANCE_MODE` build, so
section [3] names the axis and declines to collect it.

### P3-1. `__AFL_CTX_BITS` feedback

Blocked on P1-1, and on paper first: write the decision rule before touching
code. The inputs exist (ICC, family occupancy against the 2^(bits-1)
reachable tags, `read_dropped_edges()`, and -- better than the ICC -- F10's
count of edges duplicated *within* a family, which is unused context width
measured exactly rather than inferred); what does not exist is a stated rule
for stepping the width down or up, or evidence that the ICC threshold means
the same thing on a multimodal target as on a JSON parser.

### P3-2. Column leverage as a seed score

The one SVD-derived score §14 did not test. Leverage measures distance from
the dominant subspace, which is the direction Tang's l2 sampling gets
backwards, and it is *not* the algebraic complement of covered mass -- so it
is not the sign-flip that already failed. Given that record, the only
acceptable form is `bench_paired.py` with a pre-registered threshold,
designed before implementation. Observational correlation has been wrong
twice on this question.

### E6 (existing, amended)

`handover_pending_2026-09-06.md` E6 already asks for the ffmpeg re-measurement
of low-rank structure. Amendment from F7: assert ASLR is off and the
stability probe is clean *before* reading the spectrum, because the ASAN
kernel workaround that may be needed to run ffmpeg is the same variable that
inflates effective rank 25x.

### P4-1. LCG state recovery by lattice reduction

`prng_state_recovery.py` cannot represent a multiplicative step, so the whole
LCG family is out of its reach; F9 argues lattice reduction is the standard
tool for exactly that gap, at dimensions where LLL is cheap. Genuine but
blocking nothing -- and it needs a dependency decision first, since sympy is
not in `pyproject.toml` and the LLL section of the tool degrades gracefully
without it.

### Docs nit

The `|= 1` comment and the `__AFL_CTX_BITS` comment in `afl_shim.c` should
each mention the other: the OR costs one bit of context width, so `N` bits
gives `N-1` usable ones and merges call sites differing only in tag bit 0.
One line each, ride it along with whichever patch lands first.
