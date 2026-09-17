# Dominator trees for AFLGo distance: control-dependence gating

**Status:** implemented, opt-in, off by default. Not wired into the CLI or
`_activate_distance` — construct `TargetDistance(..., gate_bonus=X)` directly
until the open question below is resolved and this is A/B-measured against a
real target. HEAD at time of writing: `ff48941`. New commit on top:
see the attached patch (base `ff48941`, no push credentials in this
environment).

## Trigger

Requested graph-theory survey of the fuzzer identified five algorithms not
yet exploited anywhere in the codebase (BFS distance, Tarjan SCC + Katz
centrality, and a min-cut-style isoperimetric heuristic already exist —
see `core/distance.py`, `core/horizon.py`/`core/schedulers/seed_katz.py`,
`core/target_difficulty.py`). Ranked #1 by leverage/effort: **dominator
trees**, because AFLGo's harmonic-mean BFS distance conflates "close" with
"necessary" — a block can be BFS-near a target while sitting on a branch
that never reaches it, and BFS-far while sitting on the one edge every
execution reaching the target must cross. This doc covers dominator trees
only; min-cut (#2), betweenness centrality (#3), Steiner trees (#4), and
articulation points (#5) from that survey are still open.

## What was built

### `core/dominators.py` (new module)

Intra-procedural dominator-tree computation over `FunctionCFG`
(`core/cfg.py`), using Cooper, Harvey & Kennedy's *"A Simple, Fast Dominance
Algorithm"* (2001) — a reverse-postorder dataflow fixed point with a
finger-search intersect — rather than Lengauer-Tarjan's link-eval forest.
CHK is worst-case O(N²) vs. LT's O(N log N), but near-linear in practice on
CFGs (few predecessors per block, shallow depth), and is dramatically
simpler to get right. Revisit only if profiling shows this hot on very
large functions — nothing in this patch calls it on more than one function
at a time, and `_MAX_CFG_BLOCKS` (4096, already existing in `distance.py`)
already caps which functions get CFG-level treatment at all.

Public surface:

- `compute_idom(cfg, entry=None) -> dict[int, int]` — immediate-dominator
  map. `entry` defaults to the block flagged `is_entry`. Unreachable blocks
  (dead code, or only reachable via an indirect jump/call the decoder
  couldn't resolve — see `cfg.py`'s `indirect_call`/`indirect_jump` flags)
  are simply absent from the result; callers must not assume every block in
  `cfg.blocks` has an idom entry.
- `dominates(idom, a, b) -> bool`, `dominator_chain(idom, b) -> list[int]`
  (nearest-first, entry last).
- `gate_blocks(cfg, targets, entry=None) -> set[int]` — union of proper
  dominator-chain blocks (excluding the targets themselves) over a set of
  target blocks. This is the "mandatory gates" query `distance.py` uses.

22 tests in `tests/test_dominators.py`, hand-derived oracles (chain,
diamond, nested-diamond-with-a-gate, loop with a back edge, unreachable
block, multi-target union, default-entry fallback) — no code echoes
production logic back at itself, per repo convention.

### `core/distance.py` integration

New constructor param `gate_bonus: float = 0.0` (validated to `[0, 1]`,
raises `ValueError` outside it). `0.0` is the exact pre-existing behavior —
every existing BFS distance value is untouched, bit-for-bit, when the
parameter is omitted. When `> 0`, `_compute_bb_values` calls `gate_blocks`
per target function after its harmonic-BFS pass and multiplies the
existing `_bb_value` of every gate block by `1 - gate_bonus`, so gates
outrank BFS-equidistant non-gate blocks without touching target blocks
(which stay at exactly `0.0`). New `is_gate(bb_addr) -> bool` accessor
(always `False` when `gate_bonus == 0.0`, since the gate set is only
computed when the discount is enabled).

8 integration tests in `tests/test_distance_gate_bonus.py`, driving
`_compute_bb_values` directly against a hand-built loop CFG (see the "open
question" section below for why a loop, specifically, was necessary to
exercise this meaningfully). No regressions in
`test_distance.py`/`test_distance_unit.py`/`test_distance_unreachable_penalty.py`/
`test_cfg.py`/`test_icfg.py`/`test_horizon.py`/`test_katz.py`/
`test_target_difficulty.py` (120 passed, 6 skipped — the skips are
pre-existing, `test_icfg.py` needs capstone which isn't installed in this
sandbox). A full-suite run (`pytest -q`, no path filter) was started but
this sandbox does not keep background processes alive across tool calls,
so it never finished — the targeted slice above is what's actually
verified; run the full suite yourself before trusting this beyond it.

## Open question found while integrating (not fixed — read before wiring this up)

`_compute_bb_values`'s module-level docstring (top of `distance.py`) says
the CFG-level distance is "BFS on the reversed intra-procedural CFG," but
the implementation (the `for t in tbbs:` loop, unchanged by this patch)
walks `cfg.blocks[current].successors` — the **forward** graph — starting
from the target block, not a reversed one. Concretely, this means the set
of blocks that get a baseline harmonic-BFS value is whatever the target can
reach going *forward*, not whatever can reach the target. In an acyclic
region, a target's own ancestors (the blocks that lead to it — the ones
directed fuzzing most wants to prioritize) never get a baseline value at
all and silently fall back to the coarse per-function CG distance in
`bb_distance()`. Ancestors only pick up a value if a loop back-edge happens
to make them forward-reachable from the target too (which is why
`tests/test_distance_gate_bonus.py` deliberately uses a loop, not a plain
diamond — a plain diamond made every ancestor invisible to the discount and
the first draft of that test caught nothing).

This is pre-existing behavior, not something this patch changes, and it
sits outside dominator trees entirely — flagging it here because:

1. It's the reason `gate_bonus` frequently discounts nothing for gates that
   are true ancestors with no loop back-edge to them (the code correctly
   skips rather than invents a value in that case — see
   `test_unreachable_gate_has_no_invented_value` — but "correctly does
   nothing" still means the discount under-delivers on exactly the targets
   that would benefit most: straight-line, non-looping approach code).
2. If this direction is ever intentionally reversed (walking predecessors
   instead of successors, matching the module's own docstring), the
   dominator-gate integration in this patch needs no changes — `gate_blocks`
   is computed from the real function entry independent of `_bb_value`'s
   BFS direction, and the "skip rather than invent" guard means it will
   just start discounting more blocks, not fewer.
3. Worth its own investigation before anyone spends time on min-cut (#2)
   or betweenness centrality (#3) from the original survey, since both
   would inherit the same direction question if layered on top of
   `_bb_value` rather than the raw ICFG.

## Design decisions

- **Intra-procedural only**, matching `FunctionCFG`'s scope. No attempt at
  interprocedural dominance (e.g., "function G is unconditionally called
  before any target in F is reached") — that's a real follow-up but a
  materially harder problem (needs the call graph's own dominance
  structure composed with each function's), and the intra-procedural case
  alone already required the loop-CFG test case above to get right.
- **Opt-in, default 0.0, not CLI-wired.** Matches this repo's established
  pattern for new-and-unmeasured signals (`target_difficulty.py`'s own
  header: *"Status (P2-1): diagnostic only... do not assume a scheduling
  role until that wiring is designed and reviewed"*). Given the open
  question above, wiring this into `_activate_distance`/the CLI before an
  A/B run would risk shipping a signal whose actual coverage (which gates
  get discounted vs. silently skipped) isn't well understood yet.
- **Discount, not replacement.** `distance.py` already has a lot of
  machinery (`pc_distance_table`, the SHM channel, `max_distance` caching)
  built around `_bb_value` holding meaningful floats. Scaling existing
  values down preserves every invariant those callers rely on (ordering,
  nonzero-ness, the target-is-0.0 special case) instead of introducing a
  parallel signal that would need its own plumbing throughout.

## Suggested next steps (not done here)

1. Resolve the BFS-direction open question above — determine whether
   `_compute_bb_values` should walk predecessors from the target (matching
   its own docstring) and, if so, whether that's a bugfix or an
   intentional design tradeoff worth documenting instead.
2. A/B `gate_bonus` on a real target (the repo already has
   `tools/bench_paired.py` for exactly this) before considering CLI
   wiring.
3. Min-cut (#2) and articulation points (#5) from the original survey are
   the next-highest-leverage items and would benefit from #1 being settled
   first, since both are natural companions to (or supersets of) dominance
   on a directed reachability question.

## Files changed

- `src/fuzzer_tool/core/dominators.py` (new)
- `src/fuzzer_tool/core/distance.py` (`gate_bonus` param, `_bb_gate`,
  `is_gate()`, docstring addition — no change to default-path output)
- `tests/test_dominators.py` (new)
- `tests/test_distance_gate_bonus.py` (new)
