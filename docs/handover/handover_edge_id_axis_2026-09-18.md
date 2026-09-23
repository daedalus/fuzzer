# The (edge_id, hit_count) matrix: what the id axis is, which analyses are defined on it

**Status (refreshed 2026-09-23 on `5e18ccef`):** analysis plus one
standalone diagnostic, now `tools/edge_diagnostic.py matrix` (the former
`tools/edge_matrix_analysis.py` was folded into it in 9dad75b6). One
production-code addition: `adapters.shm.ShmCoverage._scan_with_positions()` (a
read-only, un-memoized analysis helper; the hot path is untouched) added with
F14/F15 below. F1 is fixed (dd834d1, guarded by P0-2 and P0-3); F2 is half
closed (P0-1). **F16 (2026-09-23) retracts F3 and part of F4 and re-measures
F7-F10: the shim's id function was merging 58% of fuzzgoat's edges until the
hashed-location shim change. Read F16 before quoting any number above it.**
F16's "after" column was reproduced independently on `5e18ccef` (see
"Reference run" under Follow-up items). Numbers in F1-F15 were measured at
`c26f0a1` or later as each section states. **The follow-up items are
re-prioritized below; read "Priority order" first.**

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
F14/F15 additionally sweep `--map-size` over 512/1024/8192/65536 to force
probe displacement (the target's max id 7069 cannot collide at native 8192+);
the same 250 inputs, one execution each, ASLR off, clang rebuild with 1712
guards as above.

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
The in-file reduction is about ten times faster than the sympy call it
replaced (111 s -> 10.8 s on the seed orientation, 42 s -> 4.2 s on the edge
one) and returns the same relations.

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

### F11. Two build defects that made everything above unreachable in practice

Both are fixed; recorded because the series F1-F10 was measured on
hand-built targets and the repo's own build path was producing something
different, which took three sessions to notice.

**`dd834d1` did not survive a real instrumented build.** The commit that
fixed F1 added `__afl_ctx_resolve_base()` and `__afl_ctx_use_relative()`
without `__AFL_NO_COV`. Both are `static inline`, and at -O2 clang leaves
them out of line -- `nm` shows them as local text symbols -- so in a build
carrying `-fsanitize-coverage=trace-pc-guard` they are instrumented like
anything else in the translation unit, and they are called from inside the
callback. Measured on fuzzgoat (clang 18, -O2, PIE): SIGSEGV on the first
input, 2 edges recorded of 59, in both raw and base-relative mode, with
`__AFL_CTX_SENSITIVE=0` unaffected. Fixed in `ad23878`.

The existing coverage could not see it. `test_edge_id_stability_guard.py`
and `test_ctx_and_map_size.py` both drive `__sanitizer_cov_trace_pc_guard`
directly from a **gcc** build, and gcc has no trace-pc-guard support at all
-- its valid `-fsanitize-coverage=` arguments are `trace-cmp` and `trace-pc`
-- so no part of the shim is ever instrumented in either.
`test_shim_ctx_instrumented.py` adds the clang case.

**`--clang-scov` never reached three targets.** The flag is
`build_simple_targets`' fifth argument, forwarded to each build call as
`$extra_cflags`. Three calls put something else in that slot:
`fuzzgoat_read` passed `-I$VENDOR/fuzzgoat` in both the executable and .so
passes (and `compile_fuzzgoat_object` took the include *instead of* the
flags, leaving the parser itself dark), and the `ffmpeg_read` executable
passed `$FFMPEG_INC`. Measured before the fix: `readelf` found no
`__sancov_guards` in the built fuzzgoat, and a 250-input run recorded 3
edges per execution -- the harness's own `__afl_map_edge` calls, ids all
above 0x1000 -- against 73 after. Fixed in `e4e947f`.

**And nothing said so at runtime.** `afl_instrumentation_status` looked for
`__afl_area`/`__afl_map_shm`/`__sanitizer_cov`, all of which are the shim's
own definitions, and the shim is `-include`'d into every target. A default
`build_targets.sh` run produced 20 binaries, 4 carrying a guard section, and
all 20 classified "present". 300 execs against `png_read_noasan.so` under
`--inprocess-direct` reported

    [*] AFL instrumentation: detected
    [*] execs: 301 | shm: 2 max: 2 sat: 100% | map: 0.0%
        Edges discovered:  2
        Total richness:    2 - 2 (95% CI, Chao2)
        P(new code next):  0.00%

The schedulers, the rarity machinery and Chao2 all ran on a two-element
universe and reported saturation -- indistinguishable from a target that
really is exhausted. `core/elf.sancov_guard_status()` and
`Fuzzer._warn_no_compiler_coverage()` fix that in `055ada2`; the tri-state
has to be decided on `.symtab` alone, because `.dynsym` survives stripping
and a merged check would false-alarm on a stripped working target.

**Replication.** Re-measured on the repo-built clang target once all three
were fixed: 412 distinct ids, ICC 0.518, lag-1 +0.134 at z = 3.33 with the
within-family control at +0.107, dominance 0.949, effective rank 1.1, 120
duplicate edges of which 79 in 15 within-family classes -- the same 79 as
F10, from a different build. F2 still reproduces (5 first-execution ids that
never return). Stability is 1.000 both with ASLR off and under
`FUZZER_KEEP_ASLR=1`, so F1's fix holds on a production-flag build.

### F12. Folding the edge vector into a square adds nothing

Asked: lay the edges out on a grid of side `floor(sqrt(N)) + 1`, position
from the index or the id, value the hit count. Measured on the repo-built
target (N = 412, s = 21):

| layout | row eta^2 | col eta^2 |
|---|---|---|
| rank order, 21x21 | **0.532** (null 0.045, z = +36.6) | 0.019 (null 0.045, z = -1.9) |

Rows carry half the log-count variance and columns carry none -- below the
null. The cause is direct: **10 of the 20 rows lie entirely inside one
`id >> 8` family**, so the banding is F3's blocking drawn with a line break
every 21 elements. A fold is a bijection on a 1-D vector; it cannot add
information, only re-render it.

Positioning by the raw id surfaces an artefact instead. At side 85 (odd) all
85 columns are touched; fold at an **even** width and exactly half the
columns are structurally empty -- 45 of 90 -- because `edge_id |= 1` makes
every id odd, and column eta^2 rises from 0.006 to 0.047 purely from that
comb. Vertical striping in such a picture is the OR in the shim, not the
program.

No periodic component exists to find: folding at 256, aligned exactly with
the context-family period, gives row eta^2 0.202 against 0.200 at width 257.
The large z-scores against a shuffled null at every width are occupancy
clustering, not periodicity, and aligned-vs-misaligned is the control that
settles it.

One layout does beat the raster, if a bitmap is ever wanted for looking at:
neighbour smoothness on a 128x128 grid, mean |difference| between
4-neighbours of log counts, gives raster 0.151 (shuffled 0.175, z = -28.1)
against **Morton / Z-order 0.136** (z = -54.9). Z-order interleaves the id
bits, so cells sharing a high-bit prefix land in one square block -- F3's
ultrametric made geometric, families as blocks instead of stripes. Use an
odd stride or the parity comb comes back.

### F13. The eigenvalues restate the SVD; the eigenvectors do not

Eigen-decomposition needs a square matrix, and the three candidates here
behave very differently.

**The folded grid: no information.** The 21x21 image has 14 complex
eigenvalues and |lambda|_1 = 8938; refolding the same 412 numbers at 22x22
gives 8 complex pairs, |lambda|_1 = 4657, and a trace moving from 2882 to
8686. Every quantity is a function of the fold width, because the matrix is
not an operator on anything.

**The Gram matrices: the SVD again.** Eigenvalues of `AA^T` are the squared
singular values -- verified, max |lambda - sigma^2| = 7.5e-9 -- so lambda_1
/ sum = 0.949 is F7's dominance, participation ratio 1.11, and 62 nonzero
eigenvalues for 250 seeds. There is no second opinion available here: eigen
of the Gram and SVD of the matrix are one computation.

**The eigenvectors are where the content is.**

- PC1 is volume: rho = +0.989 against total hits, +0.925 against live edge
  count. The parser-prologue mass, again.
- **PC2 separates valid from malformed input.** Orthogonal to volume by
  construction and empirically uncorrelated with it (rho = -0.026), it gives
  rho = +0.312 against a parse-success flag, mean +0.0109 over the 129
  inputs that parse as JSON against -0.0129 over the 121 that do not. This
  is the first quantity in the whole series that recovers a semantic
  property of the inputs without being told about it. See P1-3.
- PC3 is length: rho = -0.453 against input size.

**The co-occurrence Laplacian: one component.** `W = B^T B` with a zeroed
diagonal gives 412 nodes, 53301 edges, no isolated nodes, and exactly one
zero eigenvalue, so no subset of edges avoids co-occurring with the rest.
On a multi-format target that multiplicity would count independent code
regions and is worth watching; on a single-format JSON parser, one is the
expected answer. Fiedler value 19.05, and the Fiedler vector splits 5/407
with rho = +0.930 against owner count -- popularity again, the same
direction that sank Tang's sampling.

Neither F12 nor F13 is in the tool. F12 is strictly weaker than section [1],
and F13's eigenvalues are section [4] under another name. The one candidate
for a section is PC2, gated on P1-3.

### F14. The (edge_pos, edge_id) count matrix is a stable bijection, so the fold adds nothing

Asked: the same matrix families exist in another coordinate. `edge_pos` is the
SHM slot index a live edge occupies -- `home = edge_id % map_size` plus a
linear-probe displacement, read straight out of the occupied slot indices. What
does the 2-D projection (x = edge_pos, y = edge_id, value = count) carry that
the (id, count) columns do not?

Nothing, and that is now measured rather than asserted. Four map sizes, the
same 250-input corpus, fuzzgoat rebuilt with clang (ASLR off), every row
reproduced exactly on a second collection:

| map_size | triples | distinct ids | ids at one position | slots with >1 id | mean displacement | home hits |
|---|---|---|---|---|---|---|
| 65536 | 16250 | 317 | 317 (100%) | 0 | 0.000 | 100% |
| 8192  | 16250 | 317 | 317 (100%) | 0 | 0.000 | 100% |
| 1024  | 17116 | 327 | 327 (100%) | 0 | 0.054 | 94.7% |
| 512   | 17216 | 328 | 328 (100%) | 0 | 0.227 | 85.6% |

1. **Placement is a bijection and it is stable.** Every id sits at exactly one
   position for the whole campaign, and no slot ever hosts two distinct ids --
   the shim claims a slot on first fire and never reclaims it, and this is that
   invariant measurable end to end.
2. **The per-run matrix is a partial permutation**, so its singular spectrum is
   exactly the sorted per-edge counts: max|rel err| = 0.00e+00 and rank = the
   live edge count on a measured run. F12's theorem -- a fold is a bijection
   onto its image; it cannot add information, only re-render it -- is now
   quantitative, and section [8] verifies it per run rather than assuming it.
3. The only content the position axis can carry is the probe displacement, and
   at native occupancy it is zero: fuzzgoat's max id (7069) cannot collide in an
   8192+ table, so every edge lands exactly at home. Displacement only appears
   under artificial map pressure (512 slots: mean 0.227, 85.6% home hits).

Unresolved, one item: the union *grows* as the map shrinks (317 -> 328 ids,
triples 16250 -> 17216), stably per map across re-collections. Resolved in
P1-4: not a placement artefact -- a CTX-derived id-value shift for one logical
edge, keyed on the advertised map size, fire-side (path_hash diverges), layout-
invariant, target-driven impossible (no getenv in the target). The exact shim
line is unlocated, but the practical rule is safe: pin the map size.

### F15. The 3-D tensor (pos, id, count) factors through id; count-position independence is a load artefact

The 3-D binary tensor T[pos, id, count] asks whether any co-occurrence
structure survives the placement. It does not.

* **T factors through id.** Placement is a stable bijection (F14), so every id
  is observed at its single position in 100% of runs at every map size:
  T[pos, id, c] = 1 iff (id, c) fired and pos = f(id). No residual triple, and
  the 2-D matrix is the count histogram re-rendered through a permutation.
* **Count is independent of position whenever the question is defined, and the
  residual is a load artefact, not structure.** At 65536/8192 every
  displacement is 0, so Spearman is degenerate (0.000) -- the two axes are
  literally decoupled. Under map pressure a small sign-consistent negative
  association appears (Spearman -0.059 at 512, -0.108 at 1024; z = -7.5 / -14.5
  against a count-preserving permutation null whose mean is 0.0000 and sd 0.0075
  -- 1/sqrt(n) to three digits, so the Spearman machinery is sound under this
  null), but it does **not** resolve into the insertion-order confound:
  displacement tracks first-seen run at 512 (+0.300) and almost not at 1024
  (+0.041), while first-seen tracks count *positively* (+0.228) and displacement
  tracks it negatively. The home-vs-displacement correlation even flips sign
  between maps (-0.208 at 1024 vs +0.054 at 512). These are insensitive
  load-regime quantities, not properties of the axis, and none of them survives
  contact with F14.

Section [8] in the tool reports all of the above with their permutation
controls (Hard Rule 46) and the per-run spectrum check. Nothing in F14/F15
changes a scheduler design: the position axis is F12's fold with a named,
measurable cause instead of a hand-wave.

### F16 (defect, fixed by hashing guard and wrapper locations). The id function merged most edges; F3-F10 were measured through it

Measured 2026-09-23 on fuzzgoat (clang 18, trace-pc-guard, the 250-input
`corpus_fuzzgoat.py` corpus) against `tools/ground_truth_tracer.c`, which logs
the real (prev location, cur location, call site) of every coverage event and
computes no ids. Reproduce with `edge_diagnostic.py matrix --ground-truth`:

    context-free build, shim before the fix    344 real edges -> 145 ids  (57.8% merged)
    context-free build, shim after             344 real edges -> 344 ids
    context build, after                       408 (edge, call site) -> 408 ids

Two causes, both in the "What the x axis actually is" formula above:
`edge_id |= 1` erased bit 0 of `cur_loc` on every edge (80 of the 344 alone:
successors 2k and 2k+1 of one block, typically the two sides of one branch),
and sequential guards pinned every id below ~2N (fuzzgoat: 256 odd values for
344 edges, 8 of them on id 95). Consequence 3 above understated it: `|= 1`
merged edges, not just context tags. Section [8] could not see any of this --
every id sat at its home slot with zero probing, because the merging happens
before the table.

**What this does to the findings above.** Same corpus, same fuzzgoat object,
context build, old shim vs new shim (`--transpose --lll`):

| | before | after |
|---|---|---|
| distinct ids (union) | 188 | 408 |
| effective edges, 2^H | 62 | 110 |
| lag-1 autocorr along id, z | +0.2225, **+3.63** | -0.0503, -1.22 |
| binary effective rank | 2.6 | 3.9 |
| GF(2) rank | 85 | 89 |
| exact duplicate edge profiles | 26 of 188 | **178 of 408** |
| LLL relations: support / max coeff | 16 / 1 | **5 / 1** |
| [7] same-family duplicates ("unused ctx tags") | 5 | **0** |

* **F3 is retracted.** The block structure on the id axis was the structure of
  the aliasing: all ~280 guard edges sat in families 0 and 1 (`id >> 8`), so the
  "family" ICC contrasted parser code against the wrapper's hand-written ids.
  With hashed locations lag-1 is indistinguishable from the null. The "do not
  re-propose" list below stands, now for a stronger reason.
* **F4's saturation reading is retracted.** "Largest family 102 of 128 tags"
  was family 0 -- about 93 distinct context-free edges, not one edge in 102
  contexts. Context now costs 408/344 = 1.19x ids on this corpus. P3-1 needs
  re-measuring before anyone sizes `__AFL_CTX_BITS` from these numbers.
* **F7-F9 hold in shape, not in number.** GF(2) rank barely moves (85 -> 89)
  while ids double: real coverage on this corpus is low-dimensional, and the
  merges had been hiding that as fewer columns rather than more dependence.
* **F10's flow-conservation reading gets stronger, not weaker.** Aliasing summed
  unrelated edges into one column, which is what spread the old relations; with
  that gone LLL returns support-5 unit relations and 178 of 408 edges duplicate
  another's profile exactly (straight-line chains). The "79 edges in one family
  = unused context tags" split does not survive: after the fix no within-family
  duplicates remain. P1-2 is still the gate.
* **F1** is fixed (base-relative context under `FUZZER_KEEP_ASLR=1`; Jaccard
  1.000 on the three richest inputs). **F2**'s id-set divergence no longer
  reproduces (`first_exec_matches` true on the same three inputs, context and
  context-free builds, already on the old shim, before the id change); its
  path-hash half was not rechecked. See P0-1.

Absolute numbers here differ from the c26f0a1 ones (445 ids then, 188 now on
the old shim) because the corpus and wrapper moved since; the before/after
columns are the comparable pair.

Follow-up: a context-free family is now the hashed location's bits above
`ctx_bits`, so edges sharing them count as one family (323 families for 344
edges). Section [1] says so; exact counts need a `__AFL_CTX_SENSITIVE=0` build
or `--ground-truth`.

## Not defined on the id axis -- do not re-propose

Linear regression or slope of count against id; autocorrelation or FFT along
id; kernel density over id; clustering in the (id, count) plane; convex hull,
fractal or box-counting dimension; Wasserstein, CRPS or KS with id as the
ground metric; treating `Spearman(id, count)` as a trend (the tool prints it
at -0.34 precisely so that it is visibly non-zero and visibly meaningless).
Every one of these needs an ordered or metric x. Bit-prefix (trie) bucketing
is the defined replacement when a positional statistic is genuinely wanted.

`edge_pos` *is* ordered where `edge_id` is not -- it is a slot index, so
|a - b| means "probe distance". That buy is void: pos is a stable bijection
onto the ids (F14), so any ordered statistic along pos is the same statistic
along a derangement of the id axis, i.e. F3's permutation null by
construction. Plain (pos, count) bands and the displacement histogram are
meaningful; correlated structure between pos and count is not.

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

    # placement / tensor structure (F14/F15), opt-in like --lll
    python3 tools/edge_matrix_analysis.py --load /tmp/edges.npz --positions

    # reproduce F1
    python3 tools/edge_matrix_analysis.py --target ... --corpus ... --keep-aslr

Nine sections: [0] cross-process id stability, [1] x-axis structure, [2] the
permutation-invariant y marginal, [3] substituted axes, [4] the singular
spectrum, [5] the GF(2) structure, [6] integer relations (opt-in, `--lll`),
[7] edge equivalence classes and [8] placement structure (opt-in,
`--positions`). `--transpose`
runs [4] to [6] on the edge x seed matrix and enables [7]. Per Hard Rule 46 the
lag-1 statistic ships with both of its controls -- a global permutation null
*and* a within-family shuffle that must leave the effect standing if the
effect really is the blocking -- because the within-family control is what
separates F3 from F1's artefact. The `--positions` section carries the same
discipline per statistic: every Spearman it prints is ranked against a
count-preserving permutation null, and it verifies F14's claim per run by
checking the spectrum of the first-run (pos x id) matrix against the count
histogram rather than assuming the fold is a bijection.

## A scheduler built on all of this

Asked what a seed scheduler and an operator scheduler would look like if they
used everything above. The honest answer is thin, and the shape is the
result: most of F1-F13 is negative, so the design is one substrate that is
unconditionally right, two small arms gated on measurements not yet made, and
a long exclusion list. There are already 5 seed schedulers and 34 operator
schedulers in `core/schedulers/`; a 6th and a 34th are not where the value
is.

### Shared substrate

**Preflight guards.** Score nothing until two conditions hold, because every
statistic below is meaningless otherwise. `sancov_guard_status(target)` must
not be `"absent"` (F11 -- a campaign on an uninstrumented target reports
Chao2 richness 2-2 and 100% saturation, which reads exactly like success),
and the stability probe must be at Jaccard 1.0 (F1 -- under per-process ids
every edge is a singleton owned by one seed, i.e. maximally rare to any
rarity-weighted scheduler). Both checks exist now; neither is consulted by a
scheduler.

**Canonical edge space.** Collapse edges whose count profile is identical
over the last W executions (F10: 126 of 445, 28%). The motive is correctness
before performance: 79 of those were context tags that never differed, so
scoring raw ids counts one branch up to 45 times and distorts every rarity
and novelty weight by that multiplicity. Recompute at refit by hashing each
column; do not maintain it incrementally, because classes split as the corpus
grows and a merge-only structure would be wrong.

**Independent-coordinate mask.** The count vector has rank 108 of 445 (F9),
and on the edge orientation the relations are sparse with unit coefficients
(F10), consistent with Kirchhoff conservation. If P1-2 confirms them against
`core/icfg.py`, mark the derived edges: a spanning tree's complement carries
the information and the remaining coordinates are determined. Off until then
-- a relation holding over one corpus is not a CFG identity.

### Seed arm: score what volume does not explain

The most repeated result in this repo is that a derived seed score collapses
to a row sum: Tang's partial correlation controlling for total hits was
+0.006 with Wilcoxon p = 1.0 over ten matrices, degree beat every low-rank
score at +0.27..+0.77, and PC1 here is volume at rho = +0.989.

So make the orthogonalisation the estimator rather than the audit. Score a
seed by its mass on the canonical edge space *after* regressing out
rank(total hits), weighted by `1/owner_count` -- incidence, not volume, since
rho(owners, total) = +0.957 means the two look alike and are not, which was
a live defect once already. Add PC2 and PC3 as features only if P1-3
replicates.

The module carries its own falsification, the way `seed_tang` exports
`modfkv_sample_complexity`: recompute the partial correlation against total
hits and degree at every refit and log it. Converging to zero means the arm
has become a row sum again and should be pulled. It enters as one arm under
the existing Elo dispatch, never as a replacement, because the arbiter
degrades a weak arm rather than letting it do damage. Expected gain: small.

### Operator side: reward shaping, not another bandit

Nothing in F1-F13 touched operator selection, and the bandit family is
already 34 modules deep, so the contribution is to the reward every one of
them consumes.

**Credit independent coordinates only.** An operator that reaches a derived
edge learned nothing new, and a duplicate class should pay once rather than
45 times. Because this changes the signal rather than the selector, it is
testable against the current arms without replacing any of them -- the
cheapest A/B available here.

**Re-temper on `2^H`.** The measured reason learners fail in this codebase is
drift, not capacity: the MLP fell to uniform (1.03 early, 0.98 late) while
LinUCB held 1.07-1.14, and with drift switched off everything worked at 4x.
Effective edges collapsing while the raw edge count is flat (F5: 65 of 445)
is a drift signal the stall machinery has no equivalent of, and
`edge_hit_distribution()` still has no caller. Use it to raise exploration
and decay stale arm statistics, not to reset.

### Excluded by measurement, not by taste

Low-rank seed scoring in any form; l2-magnitude sampling, which returns the
most crowded edges at 1.3-1.9x above uniform; GF(2) or LLL as a minimiser
(133 seeds against greedy's 31, same 15 ms); every id-axis statistic in the
"Not defined on the id axis" list above, including the square-fold picture
whose banding is the sort order and whose vertical striping is `edge_id |= 1`
(F12); and a neural scorer over aggregate metrics, which has 3.27 effective
dimensions to work with and loses under drift.

### Sequencing

Substrate first and alone: it is the only part that corrects numbers already
being computed, and it is gated on nothing. Operator reward shaping next,
gated on P1-2. The seed arm last, gated on P1-3, with `bench_paired.py` and a
pre-registered threshold, because observational correlation has been wrong
twice on precisely this question. Tiered as P2-3, P3-3 and P3-4 below.

## Follow-up items

Tiered by what the item *is*, matching `handover_pending_2026-09-06.md`: P0 a
defect in shipped code a live campaign can hit, P1 measurement with the
harness already in hand, P2 shipped-but-unwired, P3 design work gated on a
question answered on paper first, E an evaluation run.

### Priority order (refreshed 2026-09-23 on `5e18ccef`)

The tiers above say what an item *is*; this says what to do *next*. The order
is by what each result unblocks, not by tier: P1-2 decides whether an entire
line (Ball-Larus, `op_credit`'s derived half, the integer-relation sections of
the tool) continues or closes, and nothing else on the list has that leverage.

| # | item | cost | gates / unblocks |
|---|---|---|---|
| 1 | **P1-2** edge-count relations vs the real CFG | an afternoon, no production code | P3-3 derived half; Ball-Larus; closes or keeps tool [6] |
| 2 | **P0-1** close F2 (path-hash half) | one env-gated `write()`, one run | every first-execution number |
| 3 | **P2-4** `edge_diagnostic.py` runs only on the maintainer's machine | one line + a smoke test | anyone else reproducing anything in this file |
| 4 | **P1-1 + P1-3** png, zlib, ffmpeg (absorbs E6) | machine time, a table | P3-1, P3-4, E6 |
| 5 | **P2-1** 2^H in the stall reason | one commit | nothing; self-contained |
| 6 | **P3-1** `__AFL_CTX_BITS` feedback | paper first | blocked on 4; its best input is gone (see item) |
| 7 | **P3-4** `seed_residual` A/B | bench_paired run | blocked on 4 (PC2/PC3 features) |
| 8 | **P3-3** `op_credit` derived-edge credit | re-run the existing A/B | blocked on 1 |
| 9 | **P3-2** column leverage | bench design first | nothing; lowest expected value |
| 10 | **P4-1** LCG recovery by lattice reduction | new feature | nothing; blocks nothing |
| 11 | **P1-4 residual** fuzzgoat crash triage | ordinary triage | nothing; the id effect is gone |

Closed and kept for the record (below the open items): P0-2, P0-3, P2-2,
P2-3, and the Docs nit (superseded by the hashed-location shim).

**Standing precondition for items 1, 4, 6-9:** ASLR off (or a
`__afl_ctx_relative_capable` target), stability probe at Jaccard 1.000, and a
compiler-instrumented build (`coverage_trust()` passes). F1, F7 and F8 each
show what a violation does to the numbers.

### Reference run (2026-09-23, `5e18ccef`)

Reproduces F16's "after" column exactly, so it is the baseline the items below
compare against. fuzzgoat, clang 18, 250-input `corpus_fuzzgoat.py` corpus,
ASLR pinned, 65536-entry map. Three builds (vendored tree at
`$FUZZ_VENDOR_ROOT/fuzzgoat`, default `~/fuzzing/vendoring/fuzzgoat`):

```sh
V=~/fuzzing/vendoring/fuzzgoat; SHIM=src/fuzzer_tool/adapters/afl_shim.c
clang -fsanitize-coverage=trace-pc-guard -O2 -g -fno-omit-frame-pointer -I$V \
    -c $V/fuzzgoat.c -o /tmp/fg.o
clang -O2 -g -fno-omit-frame-pointer -I$V -include $SHIM \
    -o /tmp/fg_ctx targets/fuzzgoat_read.c /tmp/fg.o -lm
clang -O2 -g -fno-omit-frame-pointer -D__AFL_CTX_SENSITIVE=0 -I$V -include $SHIM \
    -o /tmp/fg_noctx targets/fuzzgoat_read.c /tmp/fg.o -lm
clang -O2 -fno-omit-frame-pointer -I$V -include tools/ground_truth_tracer.c \
    -o /tmp/fg_gt targets/fuzzgoat_read.c /tmp/fg.o -lm -ldl
python3 tools/corpus_fuzzgoat.py --out /tmp/corpus
python3 tools/edge_diagnostic.py matrix --target /tmp/fg_ctx --corpus /tmp/corpus \
    --ground-truth /tmp/fg_gt --positions --lll --save /tmp/ctx.npz
python3 tools/edge_diagnostic.py matrix --load /tmp/ctx.npz --transpose --lll
```

| | ctx build | context-free build |
|---|---|---|
| ids (union) / real edges [9] | 408 / 408 (edge, site) | 344 / 344, **0 merged** |
| [0] Jaccard, 6 processes | 1.000 | 1.000 |
| [0] Jaccard, `--keep-aslr` + `FUZZER_KEEP_ASLR=1` | 1.000 (F1 fixed) | -- |
| 2^H effective edges | 110 | 95 |
| lag-1 along id | -0.0503, z = -1.22 | -0.0679, z = -1.40 |
| binary effective rank | 3.9 | 3.3 |
| GF(2) rank | 89 | 89 |
| greedy cover vs GF(2) basis (seeds) | 44 vs 89 | -- |
| exact duplicate edge profiles [7] | 178 in 102 classes, all cross-family | 121 in 79 classes |
| transposed: scalar pairs / A=B+C triples | 5 / 81 | -- |
| LLL, 100 rows: relations, support, max coeff, time | 29, 5, 1, 2.0 s | -- |
| [8] ids at one slot / beyond PROBE_MAX | 100% / 0 | -- |

Context costs 408/344 = **1.19x** ids. The ctx build's family ICC (0.892) is
near-degenerate and not a finding: under hashed locations a family is the
location's upper bits, and 408 ids fall in 323 families, mostly singletons, so
it mostly measures edge identity. It is not F3 coming back.

### P1-2. Check the edge-count relations against the real CFG -- PRIORITY 1

**Refreshed 2026-09-23.** The counts this item was written against ("F10's 45
sparse relations", "337 of 445 count coordinates") were measured through the
merging id function (F16) and are void. Current numbers, transposed ctx build:
29 LLL relations of support 5 and unit coefficients (2.0 s on 100x242, rank 71,
kernel 29), exhaustive 5 scalar-multiple pairs and 81 A=B+C triples, 178 of
408 edges duplicating another's profile. F16 already notes the relations got
*sparser* when the aliasing went away, which is the direction the Kirchhoff
reading predicts, but that is still one corpus of one target.

**Method change: the ground-truth tracer makes `core/icfg.py` unnecessary for
the mapping.** On the context-free build [9] shows ids and real (prev, cur)
pairs are in bijection (0 of 344 merged), so the graph comes straight from
`GT_OUT`: nodes are locations (guard indices, plus the wrapper's raw
`__afl_map_edge` values -- 0x1000+, above fuzzgoat's 1712 guards, so the two
ranges do not collide), edges are (prev, cur) pairs, per-run counts from the
same log. Three steps, cheapest first:

1. **Kirchhoff per node, per run.** In-flow equals out-flow at every node of
   every one of the 250 runs, except the entry node, sinks, and runs that
   abort. fuzzgoat aborts on its planted bugs; the tracer is unbuffered so the
   tail survives, but an aborted run ends at a node with no out-edge -- add a
   virtual exit rather than dropping the run. Any violation at an interior
   node is a tracer or mapping bug; stop and fix it before step 3.
2. **The Ball-Larus bound, measured.** Cycle rank E - V + c of the observed
   graph against the Q-rank of the full count matrix (not the 100-row LLL
   sample). If they agree, the rank *is* the cycle space and instrumenting a
   spanning tree's complement is exact; the gap is the saving on this target.
3. **Each empirical relation against the incidence matrix.** A relation r
   over edge counts holds for every flow iff r is in the row space of the
   node-edge incidence matrix B (each row is one node's conservation law).
   Test all 29 LLL relations and all 81 triples for membership over Q;
   relations outside rowspan(B) are coincidences of this corpus. The A=B+C
   triples should be almost entirely one-in/two-out or two-in/one-out nodes.

Then, and only then, `core/icfg.py`: the tracer sees executed edges only, so
the static graph is what says whether a relation that held on 250 runs holds
on unexecuted paths too. Harness-made ids are not CFG edges; report them
separately.

**Decision.** Positive (most relations in rowspan(B), rank = cycle rank):
Ball-Larus opens, `MatrixSubstrate.derived` gets populated, and P3-3's A/B is
re-run -- that is exactly the case `--shaped-reward-floor` was kept for.
Negative: close the integer-relation line and demote tool section [6] to
diagnostic-only. Either is worth the same afternoon.

Original text (numbers void after F16, kept for the trail):

F10's 45 sparse relations either are Kirchhoff identities on the CFG or are
artefacts of 250 runs of one target, and nothing measured so far separates the
two. `core/icfg.py` builds the graph; the test is whether each empirical
relation corresponds to a join or a loop in it. A positive opens Ball-Larus
(instrument a spanning tree's complement, derive the rest, and stop paying for
337 of 445 count coordinates); a negative closes the whole integer-relation
line, which is worth as much. Blocked on nothing but machine time.

### P0-1. Diagnose F2 (first-execution divergence) -- PRIORITY 2

**Refreshed 2026-09-23:** still one run from closed. The fire log has to come
from the shim, not from `tools/ground_truth_tracer.c`: F2 is about the shim's
table state (clean vs generation-reset), which the tracer does not have. The
snippet below still applies unchanged under the hashed-location shim --
`edge_id` is final at the same point. If the path hash matches, close F2 as
fixed-by-F16 with the run attached; if not, the first diverging id names the
location.

**Status 2026-09-23:** the id-set half does not reproduce on the old shim either (see
F16); the path-hash half was not rechecked. Close after one run of the fire
log below confirms the path hash too, or re-open with the input that diverges.

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

### P2-4. `tools/edge_diagnostic.py` runs only on the maintainer's machine -- PRIORITY 3

New 2026-09-23. Line 74 hardcodes
`SRCDIR = "/home/dclavijo/my_code/fuzzer-new/src"` and prepends it to
`sys.path`, so from a fresh clone every mode dies with
`ModuleNotFoundError: No module named 'fuzzer_tool'` unless the package happens
to be installed (`pip install -e .` works around it). Every reproduction command
in this file goes through that tool. Fix with the idiom the neighbouring tools
already use (`tools/find_hidden_edges.py`:
`Path(__file__).resolve().parent.parent / "src"`), plus a smoke test that runs
`edge_diagnostic.py matrix --help` from a clean interpreter. Same defect class
as `tools/profile_hotpath.py`'s hardcoded `os.chdir`. While there: the module
docstring still says the originals "are preserved untouched at
/tmp/opencode/*.py", which is true of one machine and one boot; and several
memory/edge modes need binaries under `/tmp/opencode/` or `/tmp/libpad.so`
with no build recipe -- either add the recipe or say the mode is local-only.

### P1-1. Three targets, not one -- PRIORITY 4 (with P1-3, absorbs E6)

**Refreshed 2026-09-23.** Still the gate for P3-1 and P3-4. Two changes to
what gets recorded, both from F16: drop the family ICC (near-degenerate under
hashed locations, see the reference run) and record instead the ctx /
context-free id ratio, [9]'s merge count, [7]'s duplicate classes, 2^H, the
[4] spectrum, and the drop counter. Run P1-3's PC2-vs-validity correlation in
the same pass -- png and zlib carry the validity label for free. ffmpeg is the
one that matters; `tools/vendor_ffmpeg.sh --nosan --minimal` is about four
minutes on one core. E6's precondition (ASLR off, probe clean) is the standing
precondition above, so E6 is done when ffmpeg's row is.

Every number in this document is one small JSON parser. Re-run the tool on
`png_read`, `zlib_read` and ffmpeg and record ICC, family occupancy, the drop
counter and (per F7) the spectrum. No code change; machine time and a table.
ffmpeg is the one that matters -- 8189 edges, genuinely multimodal, and the
target §14's measurements never reached because it would not build in that
container. This gates P3-1 and feeds E6.

### P1-3. Does PC2 separate valid from invalid on other targets? -- run with P1-1

F13 found the second principal component of the seed Gram matrix tracking
parse success on fuzzgoat (rho = +0.312, means +0.0109 vs -0.0129). If that
holds on png and zlib -- both have a well-defined notion of a valid file, so
the label is free -- it is a validity signal derived from coverage alone.
`ValidityChannel` already tracks the same property, but as a verdict the
target reports per execution (`Validity.VALID/INVALID/UNKNOWN`, recorded
into `seed_meta['valid']`), so a coverage-side estimate would be a second,
independent read on it rather than a replacement. One afternoon of machine time, no code change to
measure. If it replicates, section [4] of the tool should report the top
eigenvector correlations and not just the spectrum.

### P2-1. Wire `2^H` into the stall reason -- PRIORITY 5

**Refreshed 2026-09-23.** Half landed via P2-3: `EdgeTracker.effective_edges()`
exists and the end-of-run summary prints it ("Effective edges: N of M").
Still not in the stall reason: `Fuzzer`'s stall path builds `reason` from the
noise type, the entropy gate and `coverage_growth_model()` (the
`" + near-saturation"` suffix) and never reads 2^H. The signal is the *trend* --
effective edges collapsing while the raw count is flat -- so it needs the value
at the last new edge, not just the current one. `edge_hit_distribution()`
still has no caller in `src/`; `scheduler_substrate.effective_edges` documents
why it should stay that way (O(edges x seeds) vs one pass).

`edge_hit_distribution()` has zero callers in `src/` and zero in `tests/`.
Effective edges collapsing while the raw edge count is flat is a saturation
signal the stall machinery has no equivalent of; it belongs in the same
reason string as the Allan noise type and `wall_summary()`. Self-contained,
one commit.

### P3-1. `__AFL_CTX_BITS` feedback -- PRIORITY 6, blocked on P1-1

**Refreshed 2026-09-23: its best input is gone.** The text below names F10's
count of edges duplicated *within* a family as the exact measure of unused
context width. After F16 that count is **0** on fuzzgoat (79 edges in 14
classes at `c26f0a1`, 5 just before the fix): they were the aliasing. The ICC is near-degenerate too. What is
left to build the rule on: the ctx / context-free id ratio (1.19x here), the
drop counter, and [9]'s (edge, site) triples against ids. Write the rule
against those, on paper, after P1-1 has a multimodal target in the table.

Blocked on P1-1, and on paper first: write the decision rule before touching
code. The inputs exist (ICC, family occupancy against the 2^(bits-1)
reachable tags, `read_dropped_edges()`, and -- better than the ICC -- F10's
count of edges duplicated *within* a family, which is unused context width
measured exactly rather than inferred); what does not exist is a stated rule
for stepping the width down or up, or evidence that the ICC threshold means
the same thing on a multimodal target as on a JSON parser.

### P3-4. `seed_residual` arm -- PRIORITY 7, blocked on P1-1/P1-3

**Built, off by default, no bench result**
(`handover_matrix_schedulers_2026-09-19.md`); the per-refit partial-correlation
log is in. PC2/PC3 features are not, pending P1-3. Gated on P1-3 and on P2-3's canonical space. Mass orthogonal to
rank(total hits), weighted by `1/owner_count`, with the partial-correlation
falsification recomputed at every refit and logged. `bench_paired.py` with a
pre-registered threshold, no exceptions: the same question has produced two
wrong answers from observational correlation already.

### P3-3. Operator reward on independent coordinates -- PRIORITY 8, blocked on P1-2

**Refreshed 2026-09-23: there is a bench result.** `--shaped-reward` was run
paired against `--elo` on fuzzgoat (seeds 0-11, 2,000 execs): 5W/7L, median
-7.0 edges, CI [-13.5, +1.6], and with floor 0.25 5W/7L, -2.5; not adopted
(`docs/learnings/2026-09-20-shaped-reward-ab-result.md`,
`handover_matrix_schedulers_2026-09-19.md` "A/B result"). That measured the
duplicate-class half only, with `derived` empty. Re-run the same A/B once P1-2
populates `derived`; until then there is nothing new to measure. `op_credit`'s
own selector is still unmeasured.

**Built as `op_credit`, off by default, no bench result** (see
`handover_matrix_schedulers_2026-09-19.md`). The duplicate-class half is in;
the derived-edge half waits on P1-2 (`MatrixSubstrate.derived` is empty). The
paper question is answered there: credit is a function of the current partition,
so a split raises it. Gated on P1-2. Credit a duplicate class once instead of per member, and give
a derived edge no novelty credit at all. Changes the reward rather than the
selector, so it is measurable against the 34 existing operator schedulers
without replacing any of them. Paper question first: what happens to credit
when a class splits mid-campaign.

### P3-2. Column leverage as a seed score -- PRIORITY 9

The one SVD-derived score §14 did not test. Leverage measures distance from
the dominant subspace, which is the direction Tang's l2 sampling gets
backwards, and it is *not* the algebraic complement of covered mass -- so it
is not the sign-flip that already failed. Given that record, the only
acceptable form is `bench_paired.py` with a pre-registered threshold,
designed before implementation. Observational correlation has been wrong
twice on this question.

### P4-1. LCG state recovery by lattice reduction -- PRIORITY 10

`prng_state_recovery.py` cannot represent a multiplicative step, so the whole
LCG family is out of its reach; F9 argues lattice reduction is the standard
tool for exactly that gap, at dimensions where LLL is cheap. Genuine but
blocking nothing. No dependency decision to make: the tool's LLL is written
out in-file under Hard Rule 51, so the same routine is available to any
caller in the tree.

### E6 (existing, amended) -- folded into P1-1

Done when P1-1's ffmpeg row is recorded under the standing precondition.
Original amendment kept:

`handover_pending_2026-09-06.md` E6 already asks for the ffmpeg re-measurement
of low-rank structure. Amendment from F7: assert ASLR is off and the
stability probe is clean *before* reading the spectrum, because the ASAN
kernel workaround that may be needed to run ffmpeg is the same variable that
inflates effective rank 25x.

### Closed items

Kept verbatim below each note: a closed item still needs a pointer to where
its artefacts live, or the next reader re-proposes it.

### P1-4. Why does the union grow as the map shrinks -- NO LONGER REPRODUCES after F16; residual is PRIORITY 11

**Re-measured 2026-09-23 on `5e18ccef`, same corpus:** the growth is gone.
ctx build: union 403 at map 512, 408 at 1024, 8192 and 65536; context-free
build: 344 at both 512 and 8192. The union now *shrinks* at 512 (5 ids and 8
hits fewer), which is the ordinary saturation direction at load 408/512 = 0.80
(mean probe displacement 0.386, 83.7% home hits); the tool does not print the
drop counter, so that attribution is by direction, not by count. The
"unlocated ctx line" is moot. What remains is the fuzzgoat heap-buffer-overflow
triage described at the end -- but fuzzgoat plants memory bugs on purpose, so
check the report against fuzzgoat's list of planted bugs before treating it as
a finding.

F14's table: union 317 ids / 16250 triples at 65536/8192, 327/328 ids /
17116/17216 at 1024/512. The 3.5% inflation is now characterized end to end.
It is **not** a placement artefact; it is the child computing a *different
edge_id stream* when the advertised `AFL_MAP_SIZE` changes, for one logical
edge. The experiment trail on fuzzgoat (clang, ASLR pinned, `~/fuzzing/builds/fuzzgoat_read`):

* **Segment size is irrelevant.** Allocate a 65536-entry segment and pass
  `AFL_MAP_SIZE=512`: the extra id (209) appears. Same 65536 segment with
  view 8192: it does not. The driver is the advertised map value, not the
  backing store.
* **It is fire-side, not storage-side.** Native `path_hash` (placement-
  independent; `hash = hash * 31 ^ edge_id` per fire) differs between views
  (7615122267079587265 vs 1946655851876716116) at equal `edge_count` (5) and
  dropped=0. The same logical edge is stored as 209/223 at map 512 and 219 at
  map 8192 -- mutually exclusive, tracked by the *current* exec (mixed-
  history runs confirmed: 209 iff the recording exec ran at 512, 219 iff at
  8192).
* **It is a CTX artifact.** A rebuild with `-D__AFL_CTX_SENSITIVE=0` stores
  byte-identical tables at 512 and 8192 (same 13 ids, same path_hash).
* **It is layout-invariant.** Env-length padding (0..4096 B), `MALLOC_*`
  tunables, and a 64 MiB anonymous LD_PRELOAD pad all leave 209 fixed at
  map 512.
* **The target cannot be the driver.** Neither `targets/fuzzgoat_read.c` nor
  the vendored `fuzzgoat.c` calls `getenv` or reads the segment, so the value
  shift is produced inside the shim's own CTX/fire path -- yet no `__afl_map_size`
  read exists in `__afl_get_caller_ctx()`/the probe loop, only
  placement/`window`/tail-offset/wrap-wipe reads. The exact line is still
  unlocated; the storage semantics themselves are exonerated by the
  path_hash divergence (placement cannot alter it).

Consequence for F14/F15: the bijection and the tensor factorization hold per
*fixed* map size; on a CTX-sensitive target, cross-map union comparison is not
apples-to-apples because one logical edge can carry a different id value at
512 vs 8192. Standing recommendation (unchanged, and now load-bearing): pin
`AFL_MAP_SIZE` for any comparison and evaluate all positional statistics on a
single map size.

**Addendum: the minimal direct-callback driver does not reproduce this, which
narrows the search.** Attempted the cheapest possible reproduction of the
CTX/id shift with `tests/test_ctx_and_map_size.py`'s own driver (gcc,
`-D__AFL_CTX_SENSITIVE=1 -fno-omit-frame-pointer`, `__sanitizer_cov_trace_pc_guard`
called directly, no clang, no real target) rather than fuzzgoat: same 65536-entry
backing segment, two `subprocess.run` calls differing only in the
`AFL_MAP_SIZE` view (512 vs 8192), edge_id logged per fire via a temporary
env-gated write right after `edge_id |= 1` (the same hook P0-1 sketches,
built as a scratch, uncommitted local patch to afl_shim.c for this
investigation only). First attempt: all 40 fires differed between views --
looked like an instant reproduction, until the same-view rerun (8192 vs 8192)
*also* differed, which pins the cause on the harness rather than the shim:
unsetting `FUZZER_KEEP_ASLR` disables the shim's own base-relative addressing,
it does not touch the kernel's ASLR, so the two bare `subprocess.run` calls
each got a fresh randomized PIE base and every `caller_ctx` naturally differed
-- the exact class of artefact F1 already named, just at the harness level
instead of the shim's. Re-run with the process actually pinned
(`setarch x86_64 -R`, the same effect `disable_aslr()` achieves in
`edge_diagnostic.py`'s `assert disable_aslr()` at line 132) and confirmed
first via a same-view/same-pinning control (8192 vs 8192, byte-identical):
0/40 fires differ between the 512 and 8192 views. `__afl_get_caller_ctx`'s
own math -- fixed compile-time `__AFL_CTX_MASK`, no read of `__afl_map_size`
anywhere in it or in `__afl_map_edge` up to the `|= 1` line -- is confirmed
clean by direct measurement, not just by the grep the main writeup did.

That the effect requires the real fuzzgoat/clang build to appear at all (a
flat 40-guard single-TU loop, one call site, cannot produce it) says the
missing line is not in the hashing math itself but in something only a
multi-TU, dynamically-linked binary exercises around it -- `dladdr()`/PLT
stub resolution, `.init_array` constructor ordering across translation
units, or a second call site whose frame layout differs from the driver's.
Next cheapest step, still no clang required: extend the direct-callback
driver to two translation units with an indirect (function-pointer) call
between them, the simplest structural difference from the single-TU loop
that a real binary has and this driver does not.

**Second addendum, with clang now available, on the real target: the CTX
attribution above does not survive direct measurement, and the actual
mechanism looks like a bug in fuzzgoat, not in the shim.** Built
`targets/fuzzgoat_read` for real (`tools/vendor_fuzzgoat.sh`, clang
`-fsanitize-coverage=trace-pc-guard`, exactly `build_targets.sh`'s recipe:
`fuzzgoat.c` compiled separately without the shim, linked into the
`-include afl_shim.c` wrapper). `tools/corpus_fuzzgoat.py`'s 120-file
corpus reproduces the union growth directly (197 ids at map 8192/65536,
203 at map 512, ASLR pinned via the real `disable_aslr()`); 6 of the 120
inputs individually diverge. Traced one (`mut_insert_0047.json`) fire by
fire with the same env-gated log used in the first addendum: divergence
starts at fire #39, and the field that differs there is **`cur_loc`
itself** (416 vs 420), not just the derived `edge_id` -- the target calls
`__afl_map_edge` with a different value, meaning the *actual sequence of
instrumentation points reached* differs (151 vs 147 total fires), not
merely which id that sequence hashes to. That is a stronger and different
claim than "CTX id-value shift": it survives a rebuild of the exact same
driver with `__AFL_CTX_SENSITIVE` left off entirely (no `caller_ctx` term
in `edge_id` at all) -- 155 vs 147 fires, same split. Whatever this is, it
is not in `__afl_get_caller_ctx()`, confirming that function is a dead end
for this line of investigation. A same-map-size/same-binary control run
five times each was byte-identical both ways, and padding `AFL_MAP_SIZE`'s
own env-string length while holding its parsed value at 8192 reproduced
nothing (0/9 pad lengths shifted the fire count) -- ruling out both
non-determinism and the stack-layout-via-env-length hypothesis the first
addendum's team already tested, now specifically against a confirmed
reproducer rather than in the abstract.

The fire count changing from 155 to 151 after an unrelated rebuild (one
extra `fprintf` argument logging `__afl_map_size` itself, which read back
correctly and uncorrupted at every single fire in both views) is the tell:
a *deterministic-per-binary but layout-fragile-across-rebuilds* effect is
the signature of memory corruption, not of a hash computation. An ASan
build of the same target (`fuzzgoat.c` ASan-instrumented separately per
the same no-shim-in-that-TU rule, then linked) confirms it directly:
`mut_insert_0047.json` trips a real
**heap-buffer-overflow READ of 8 bytes, 7 bytes past a 33-byte allocation,
in `json_value_free_ex` (vendor/fuzzgoat/fuzzgoat.c:258)** -- identically
at both AFL_MAP_SIZE=512 and 8192 (same address, same stack, ASan does not
care about the shim's env var). 13 of the 120 corpus inputs trip the same
report; 3 of the 5 fire-count-diverging inputs are among them (the other
2 diverge without tripping this particular redzone, consistent with a type
confusion bug in the value union that only sometimes reads far enough to
hit a poisoned byte). This crash fires *after* the traversal that produces
the diverging fire count (`json_value_free` runs once `process_value` has
already completed in `fuzz_shm_run`), so it cannot be the literal
mechanism -- but it is very likely the same family of bug: fuzzgoat's
`json_value` union being read under the wrong member/size assumption
somewhere upstream, silently in-bounds often enough that ASan does not
always catch it during the parse/traversal phase, occasionally out-of-
bounds enough that it does during the free phase. This reframes P1-4: the
union-growth-under-a-smaller-map is likely a real, ASan-confirmed bug in
the *vendored fuzzgoat target* surfacing as an id-count artifact, not a
defect in the shim's CTX or hashing path, both of which are now measured
clean. Next step: minimize `mut_insert_0047.json` under the ASan build and
locate the exact union member fuzzgoat reads with the wrong type -- ordinary
crash triage, no more shim archaeology needed.

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

### P0-3. `__afl_ctx_relative_capable` marker -- DONE

dd834d1 shipped relative mode with no way for the Python side to know whether
a given binary has it. `__afl_ctx_bits_N`, `__afl_ngram_k_N` and
`__afl_shm_layout_N` all exist precisely so a build contract can be read
before anything executes, and relative mode was the one contract that could
only be discovered by running the target three times and comparing edge sets.

The shim now emits `__afl_ctx_relative_capable` (bare name, no value encoded:
there is no value, only presence), unconditionally rather than under
`#if __AFL_CTX_SENSITIVE`, for the same reason `__afl_ctx_bits_0` is emitted
for context-free builds -- a missing symbol has to mean "old shim" and
nothing else. `elf.detect_ctx_relative_capable()` reads it with the same
`_symbol_names` scan the other three use, returning True / False / None,
where None is "could not read" and is deliberately not warned about. False is
only meaningful once `detect_ctx_bits` has established there is a
context-hashing shim at all, since an uninstrumented binary has no marker
either.

Three consumers, in the order a run hits them:

* `_ensure_ctx_ids_are_exec_stable` now partitions the targets. Capable ones
  get `FUZZER_KEEP_ASLR=1`; stale ones get a named warning with the rebuild
  as the escape, instead of being handed a variable they ignore and told it
  was fixed. A mixed multi-target run gets both halves -- one stale target
  must not cost the others their fix.
* The probe's attribution is now three-way rather than assumed: no marker ->
  rebuild; marker but the variable unset -> the startup hook did not reach
  this target, which in a real campaign is our own bug and now says so;
  marker and variable set -> not the F1 shape, look at threads, time and
  uninitialised memory.
* `docs/refs/architecture.md` gains the marker alongside `__afl_ctx_bits_N`.

The regression test builds the stale arm from the real pre-dd834d1 source
(`git show dd834d1^:src/fuzzer_tool/adapters/afl_shim.c`, compiled with gcc)
rather than simulating it: the question the marker answers is literally "was
this built before that commit", and a hand-written stand-in would beg it.
Both current widths carry the marker, the pre-dd834d1 build does not and
still reads `__AFL_CTX_BITS=8`, and end to end that stale binary drifts
(Jaccard 0.000) with `FUZZER_KEEP_ASLR=1` set, which is the one case no
static check could reach before.

Shim diagnostics are unchanged: identical gcc `-Wall -Wextra` output to
dd834d1 under default, `-D__AFL_DISTANCE_MODE=0`, `-D__AFL_CTX_SENSITIVE=0`
and `-D__AFL_NGRAM_K=4`.

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

### P2-3. Scheduler substrate: guards and the canonical edge space -- DONE

Landed as 4af8a07c (`core/scheduler_substrate.py`): `coverage_trust()`
consults `sancov_guard_status` and the stability probe and is the single
preflight used by `Fuzzer` and `MatrixSubstrate`; `EdgeCanonicalizer` collapses
duplicate-profile classes; `effective_edges()` is exposed (and printed in the
summary). Applying them to weights was left to P3-3/P3-4 on purpose. The 2^H
half of P2-1 that it overlapped is noted there.

Original text:

Gated on nothing, and the only part of the design above that fixes numbers
already being computed rather than adding a new one. Three pieces: consult
`sancov_guard_status` and the stability probe before any scheduler scores
anything (both checks exist, neither is consulted); collapse duplicate-profile
edge classes at refit so rarity and novelty weights stop counting one branch
up to 45 times; and expose `2^H` from `edge_hit_distribution()`, which still
has no caller. Overlaps P2-1, which wires the same scalar into the stall
reason -- do them together.

### Docs nit -- SUPERSEDED by F16

The `|= 1` this nit asks to cross-reference no longer exists: the
hashed-location shim remaps zero instead of forcing bit 0 (`afl_shim.c`,
"Not `edge_id |= 1`"), so `N` context bits give `N` usable ones. Considered and
overtaken, not done -- recorded so it is not re-applied to the new code.

Original text:

The `|= 1` comment and the `__AFL_CTX_BITS` comment in `afl_shim.c` should
each mention the other: the OR costs one bit of context width, so `N` bits
gives `N-1` usable ones and merges call sites differing only in tag bit 0.
One line each, ride it along with whichever patch lands first.
