# Handover — Combinatorics and Permutations Across `fuzzer-tool`

**Date:** 2026-09-02. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_combinatorics_permutations_2026-09-02.md`.
**Verified against:** `4c021daa`.

Litmus test for any item (original §0): a combinatorics addition must serve one
of (1) reaching worst-case program paths, (2) allocating budget to productive
operators, or (3) falsifying operator reachability via `ExhaustivePool`.

---

## 1. `_swap_tuple` (m>2) — wired at byte level only; yield unmeasured (orig. §10a.1)

**State:** `_swap_tuple(domain, rng, m, *, start=0)` (`core/mutations/generic.py`)
has one caller: `OperatorEngine._op_swap_bytes` (`services/operators.py`), which
takes m∈{4,5} with p=0.15. The per-format element swaps (`core/mutations/<format>.py`,
32 `_swap_pair` sites) are still m=2 only.

**Open:**
- No A/B of m>2 vs m=2 edge yield. The only positive signal (4/4 new-edge on
  ffmpeg isobmff) was m=2.
- Validity-cliff hypothesis (m permuted elements break m offsets, so parsers
  reject earlier) was *not* supported on `libsqlite3` / `libavformat`
  top-level box order. Untested: ZIP central directory, and a harness that
  dereferences `stco`/`stsz` entries deep enough to hit a stale offset.
- Whether to extend m>2 to structural element swaps.

**Constraints if extended:** draw m relative to n (n ranges ~4 to thousands);
keep m=2 as its own arm; use `m = 2+k`, not `2k` (odd m=3,5 minimise Cayley
diameter); `ExhaustivePool` enumerates `n!/(n−m)!` — n=20, m=5 exceeds
`DEFAULT_MAX_RUNS` and leaves `exhausted=False`.

**Acceptance:** fuzzgoat + one offset-table target, paired runs, edges/exec for
m>2-enabled vs m=2-only, control-vs-self per Hard Rule 46.

## 2. `ExhaustivePool` bulk gate is all-or-nothing (orig. §10b)

**Where:** `core/exhaustive_pool.py` `_bulk_guard` / `randbytes`. Any
`randbytes(n>0)` raises `BulkDrawError` unless `allow_bulk=True`.

**Proposal:** per-call budget — `randbytes(n)` enumerable for `n <= 2` by
default (65,536 paths), larger `n` only with explicit opt-in. Lets
`spectral_peak`, `de_bruijn_fill` (`core/mutations/structured.py`) be
exhaustively tested.

**Risk:** several `randbytes` calls in one path multiply; the cap must be per
path, not per call, or `max_runs` truncates silently.

## 3. Coin-flip idiom blocks enumeration (orig. §10c)

**Where:** `rng.random() < p` — 94 sites under `src/fuzzer_tool/` (incl.
`_op_swap_bytes` above). `tests/test_exhaustive_pool.py::test_continuous_error_names_the_cheap_fix`
records 21 operators unenumerable only for this reason.

**Proposal:** rewrite as bounded draws (`rng.randint(0, 1)`, or
`randint(0, N-1) < p*N`). One line each; each touched operator needs a test
asserting `ExhaustivePool.exhausted`.

**Open:** produce the census list (which operators, which sites); check that
the bounded-draw rewrite keeps the RandPool fast path (Hard Rule 41).

## 4. Operator Markov chain is first-order only (orig. §10d)

**Where:** `core/schedulers/op_monte_carlo.py` — `transition_counts[prev][next]`,
`transition_total`, `select_op` pairwise blend.

**Proposal:** sparse second-order table `(prev2, prev) -> next`, observed
triples only (dense is ~150³ ≈ 3.4M counters; Hard Rule 54).

**Open:** does second order add signal? Needs an A/B with `--pairwise-blend > 0`
before any wiring; persisted via existing state machinery if adopted.

## 5. Grammar derivation space: no skeleton set (orig. §10e)

**Where:** `core/grammar.py` `Grammar.generate` / `generate_boltzmann`.
Boltzmann sampling (size-uniform) has since landed; a precomputed set of
minimum representative derivations (one per derivation shape) for bootstrap
seeding has not.

**Open:** needs a finite canonical-form projection, which not every grammar
has. Decide whether bootstrap coverage justifies it given Boltzmann sampling.
