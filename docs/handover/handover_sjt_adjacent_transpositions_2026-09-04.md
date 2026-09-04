# Handover — Steinhaus–Johnson–Trotter Adjacent Transpositions

**Date:** 2026-09-04
**Base:** `be05bd7` ("build: exit non-zero when targets failed to build")
**Status:** Analysis + implementation plan. No code landed. Positions SJT
against the existing combinatorial surface surveyed in
`handover_combinatorics_permutations_2026-09-02.md` (§1–§2, §10a).

---

## 0. Rule

Same litmus as the combinatorics handover:

1. Reach worst-case program paths (adversarial input construction).
2. Allocate execution budget to operators that have produced coverage.
3. Falsify that a mutation operator is reachable in its parameter space
   (`ExhaustivePool`).

SJT only belongs if it does one of those three jobs better than the
primitives already present (`_swap_pair` C(n,2), Fisher-Yates via
`ExhaustivePool.shuffle` / `byte_shuffle` / `token_shuffle`, random
window shuffles). The claim below is that it does (1) and (3) for the
*adjacent-only* subspace that the current random C(n,2) and full n!
generators do not systematically cover.

---

## 1. What exists today (permutation surface)

From `handover_combinatorics_permutations_2026-09-02.md` and current tree:

| Primitive | Location | Distance metric | Exhaustive? |
|---|---|---|---|
| C(n,2) element swap | `core/mutations/generic.py:_swap_pair` + 15 call sites in format mutators | arbitrary pair | via `ExhaustivePool.sample(..., 2)` |
| Fisher-Yates n! | `ExhaustivePool.shuffle`, `byte_shuffle`, `token_shuffle` | full permutation | yes (odometer) |
| Random window shuffle | various structural ops | full | no |
| de Bruijn / perm_lock / cycle_lock | `core/mutations/structured.py` | adversarial shapes | no |

Nothing currently generates the **adjacent-transposition Gray code**
(Hamiltonian path on the permutohedron). Consecutive mutants under SJT
differ by exactly one adjacent swap; the current generators either jump
to an arbitrary pair or to a uniformly random full permutation.

---

## 2. What SJT provides

Steinhaus–Johnson–Trotter (Even’s directed form) produces the sequence
of all n! permutations in which each successive pair differs by a single
adjacent transposition. Average time per permutation is constant.

Consequences for this codebase:

- **Locality.** An adjacent swap changes fewer structural invariants
  (length fields, checksums, relative offsets) than a random C(n,2) or
  Fisher-Yates. Useful when the goal is “reorder while preserving as
  much of the already-discovered edge set as possible.”
- **Gray-code coverage of the adjacent subspace.** `ExhaustivePool`
  already falsifies the full n! space. It does not, by itself, give a
  generator whose successive draws are adjacent. SJT is the missing
  ordered generator for that subspace.
- **Small-n exhaustive campaigns.** For windows of length ≤ 8 the full
  SJT sequence is cheap and can be materialised into the corpus once
  (ordering-sensitive parsers, instruction reordering, token streams).

It does **not** replace `_swap_pair` or Fisher-Yates. It is an additional
generator with a different distance metric.

---

## 3. Proposed landing (minimal)

### 3.1 Core primitive

**New file:** `src/fuzzer_tool/core/sjt.py`

- `SJT(n)` — Even’s directed generator over indices `0..n-1`.
- `next_permutation() -> list[int] | None`
- `adjacent_swap_indices(n) -> Iterator[tuple[int,int]]` — yields only
  the (i, i+1) pairs that realise the SJT path (so callers can apply the
  swap in-place on an existing buffer without materialising the full
  permutation).
- Pure, no I/O, no project imports except typing. Must accept an optional
  `rng` only if a future randomised starting point is needed; default
  path is deterministic from the identity.

Hard constraints (AGENTS.md):
- Use `core/rand_pool.py` if any randomness is introduced later.
- Cyclomatic complexity ≤ 15 (`lizard`).
- Function names short.

### 3.2 Registration

**`core/operator_registry.py`** — structural category:

```text
sjt_adjacent_swap   # single next adjacent transposition (cheap, local)
sjt_permute         # replace a small window with the next SJT permutation
```

Availability:
- `sjt_adjacent_swap`: length ≥ 2.
- `sjt_permute`: 2 ≤ length ≤ 10 (hard cap; beyond that factorial cost
  is not interesting for a mutator that runs every few execs).

Both must respect format-lock protected prefixes and FrameShift length
repair exactly as the existing structural operators do.

### 3.3 Operator handlers

**`services/operators.py`** — `_op_sjt_adjacent_swap`, `_op_sjt_permute`
on `OperatorEngine`.

- Window selection: same heuristics already used by block / structural
  operators (random contiguous slice, or whole buffer when small).
- State: per-seed or weak-key LRU of `SJT` instances keyed by
  `(id(seed_buf), window_start, window_len)` so successive calls on the
  same window walk the Gray code instead of restarting at the identity.
  Do not store generators in the global process state without a bound.

### 3.4 ExhaustivePool interaction

`ExhaustivePool` already enumerates n! via Fisher-Yates. For falsification
of the *adjacent-only* path it is enough to drive `adjacent_swap_indices`
through a fixed number of steps and assert that every yielded pair is
truly adjacent and that the induced permutations are unique until the
cycle ends. No change to `ExhaustivePool` itself is required for the
first landing.

### 3.5 Tests (mandatory)

- `tests/test_sjt.py`
  - n = 0..6: all n! permutations appear exactly once.
  - Every consecutive pair differs by exactly one adjacent swap
    (falsification: inject a non-adjacent swap and assert detection).
  - `adjacent_swap_indices` produces the same permutation sequence.
  - Adversarial: empty, n=1, n just above the operator length cap.
- `tests/test_regression_operator_registry.py` — category, availability,
  dispatch coverage for the two new names.
- `tests/test_regression_sjt_operator.py` — end-to-end on a buffer:
  length invariant, format-lock respect, successive calls on the same
  window advance the Gray code.

TDD: write the failing tests first (Hard Rule 38/39).

### 3.6 Docs / TODO

- One short subsection under structural mutations in `docs/DEEP_DIVE.md`
  (Hard Rule 11). Do not inflate README unless the flag becomes
  user-visible in the quick-start sense.
- If a TODO item is opened for this work, close it in the same commit
  that lands the implementation (standing rule in `docs/TODO.md`).

---

## 4. Out of scope for first landing

- MCTS neighbourhood expansion via SJT moves (can reference this
  handover later).
- Metropolis proposal distribution that only offers adjacent swaps.
- Automatic “interesting window” discovery beyond existing structural
  heuristics.
- Recursive non-Even implementation; Even’s form is sufficient and
  amortised O(1).
- Any change to the existing `_swap_pair` call sites or Fisher-Yates
  paths.

---

## 5. Suggested commit sequence

1. `core/sjt.py` + `tests/test_sjt.py` (pure algorithm, no registry).
2. Registry entries + operator handlers + regression tests.
3. DEEP_DIVE note.

Each step independently revertable. Do not collapse them.

---

## 6. Verification checklist for the implementer

- [ ] Read `AGENTS.md` Hard Rules 0–12, 16, 23, 38–41.
- [ ] Confirm no existing SJT / plain-changes implementation
      (`rg Steinhaus|Johnson.Trotter|plain.?changes|SJT`).
- [ ] Position the new operators in the structural category only;
      do not touch the bit/byte/regularity bands.
- [ ] `lizard --CCN=15` clean on `sjt.py`.
- [ ] Full `pytest` green; no new failures under `ExhaustivePool`.
- [ ] Short campaign on an existing target (`png_read` or `sqlite_read`)
      shows the new arms appearing in Elo / operator stats and does not
      collapse eps.
- [ ] Close any related TODO in the landing commit.

---

## 7. Relation to the combinatorics handover

`handover_combinatorics_permutations_2026-09-02.md` already catalogued
C(n,2) and n! generators and implemented `_swap_pair` / `_swap_tuple`.
This document adds the missing **adjacent-transposition Gray-code
generator**. It does not reopen the C(n,2) vs C(n,2+k) discussion; it
only supplies a different distance metric on the same symmetric group.

---

*End of handover.*

---

# Addendum, 2026-09-04: audited, measured, and not adopted

The plan above was audited against live source and then tested. The premise
holds and the algorithm is correct; the operator it proposes does not earn its
place. Recorded here rather than deleted, because "considered and rejected" and
"never considered" are different states.

## Corrections to the survey above

- **§1 says 15 `_swap_pair` call sites.** It is 16 (mp3.py joined). All 16 sit
  inside `*_chunk_mutate` operators, and all of those are in the `format` band.
- **§1's table omits `_swap_tuple`**, which exists (`core/mutations/generic.py`)
  with the even-m rotation / odd-m parity fallback, and has **zero production
  call sites** — only `tests/test_swap_tuple.py`. The tree already carries one
  unwired permutation primitive.
- **Base is stale.** The doc states `be05bd7`; it landed six commits later.
  Nothing in those six affects the analysis.
- **The band is wrong.** §1 surveys the `format` band and §3.2 lands SJT in
  `structural`. An adjacent swap of two bytes is not an adjacent swap of two
  ISOBMFF boxes, and §2's locality argument (length fields, checksums, relative
  offsets) only has teeth in the band the plan does not target. §6's
  "do not touch" list omits `format` — the one band where the case was strong.

## §3.3's state design cannot work

Three independent reasons, all measured:

1. `id(seed_buf)` is address reuse. Five sequential `bytearray(64)`
   allocations produce **1 distinct id**.
2. The "weak-key LRU" alternative does not exist: `weakref.ref` on `bytes` and
   `bytearray` raises TypeError.
3. Window selection follows the block operators, which redraw
   `idx = rng.randint(...)` every call.

Simulating 20,000 execs: the random-slice branch **hits** 51.3% of the time and
the whole-buffer-small branch 99.985% — every hit a recycled address, 5,269 of
10,269 carrying a window created for a differently-sized buffer. The generator
does not restart; it silently continues an unrelated input's walk. The fix, had
the operator been worth landing, is `ctx.seed_meta` (already a
`MutationContext` slot) and a mandatory randomised start, not the optional one
§3.1 describes.

## §0 item 3 is not delivered

SJT is deterministic and draws nothing, so `ExhaustivePool` has no tree to
enumerate. The house falsification seam is
`tests/support/operator_env.py::make_minimal_fuzzer(pool=...)` (port item P1-5;
`tests/test_swap_pair.py` is the worked example), and a no-draw core yields
exactly one run under it. All the randomness lives in window selection, which
§3.4 explicitly declines to cover.

## The measurement

Full result in `docs/sweeps/region_order_attribution_2026-09-04.md`. Against a
target with per-region ground truth, five arms, five seeds:

- SJT walking from the identity is **indistinguishable from a random adjacent
  swap** on every seed, and shares its false-DEAD error on a cold-but-live
  region.
- SJT with a randomised start loses to a free permutation confined to one
  region (wrong on 2 of 5 seeds vs 0 of 5).
- With its state broken — which §3.3 guarantees — `sjt_adjacent_swap` degrades
  to a single adjacent swap from the parent, which saturates at **7 distinct
  outputs at n=8** against `_swap_pair`'s 28. A 4x regression against the
  primitive it claims to complement.

Why, and it is the general lesson: deciding whether an ordering matters wants
**independent samples** of S_n, not a **connected walk**. The Gray-code
property trades independence for locality. At n=8 the value that fires the cold
region does not reach slot 0 until step 5376 of 40320 — SJT's exhaustiveness is
asymptotic, and inside a bounded budget it is a local sampler.

## What did land

The audit's by-product, not its subject: `_last_mutation_offset` was published
from `select_position()` for operators that never read it, so the region
liveness estimator was fed a fabricated attribution. `_DELOCALISED_OPS` stops
that and `region_shuffle` supplies a confined ordering mutation that can report
a true offset. Status: **closed, not adopted.**
