# Dominator-gate follow-up: two bugs found while wiring it in

Base: `b8a39af` (the dominator-gate commit itself). Three commits on top,
attached as a `git am` patch stack, verified to apply cleanly on a fresh
clone of `b8a39af` and to pass 225/225 relevant tests (15 pre-existing
skips — capstone unavailable in this sandbox).

## 1. Fixed: `_compute_bb_values` walked the CFG forward, not reversed

This was the open question the original dominator-gate handover flagged
but didn't fix. `_compute_bb_values`'s BFS from each target walked
`cfg.blocks[current].successors` — forward from the target — which
computes "distance from target to b", not "distance from b to target."
In an acyclic region a target's own ancestors (the blocks that lead to
it — what directed fuzzing most wants to prioritize) never got a
baseline value at all and silently fell back to the coarser per-function
call-graph distance. It only ever looked correct in loop-shaped CFGs,
where a back edge can coincidentally make an ancestor forward-reachable
from the target too.

Fix: `dominators.py`'s predecessor map is now a public `predecessors()`
function (was a private helper used only inside `compute_idom`); the BFS
in `distance.py` walks that instead of `successors`. Confirmed a
plain diamond CFG (0→{1,2}→3(target), no back edge) now gives 0, 1, 2
their correct distances instead of silently omitting them —
`tests/test_distance_bb_values_ancestors.py`. Rewrote
`test_distance_gate_bonus.py`'s hand-derived expectations for the
corrected semantics: with the old loop-shaped CFG in that test, block 0
(a true dominator) used to have no baseline value at all — the fix
means every dominator of a reachable target is now guaranteed to be
found, since being a dominator implies lying on some forward path to
the target, which the reversed BFS will always discover.

## 2. Wired `gate_bonus` into `Fuzzer`/CLI

`Fuzzer.__init__` gained `gate_bonus=0.0` → `self._gate_bonus`;
`_activate_distance` forwards it into `TargetDistance(...)`; CLI gained
`--gate-bonus X` (float, `[0, 1]`, directed-mode only). Verified via
`test_regression_cli_fuzzer_kwargs.py` that the flag actually reaches
`Fuzzer()`.

Deliberately **not** added to `_HAIL_MARY_FLAGS`: it is conditionally
inert without `--target-functions` (which `--hail-mary` does not set —
same dependency already documented for `--canary-scheduler`/`--elo`),
and it's still explicitly unvalidated (no A/B run against a real
target). Documented next to `_HAIL_MARY_FLAGS` and verified end-to-end
that `--hail-mary --target-functions main` leaves `gate_bonus` at `0.0`.

## 3. Found and fixed while answering "isn't ICFG usable without targets?"

This question surfaced a real, independent bug, unrelated to the
dominator gate itself. Investigating whether `core/icfg.py`'s
whole-program ICFG requires `--target-functions` (empirically: it does
not — confirmed by loading a `TargetDistance` with no targets at all and
building a full ICFG from it, 12/12 functions decoded) led to
`services/katz_channel.py::KatzChannel.build()`, which is the only
caller of `build_interprocedural_cfg` in the whole codebase.

`KatzChannel.build()` rejected construction whenever `td.target_addrs`
was empty. But `build()` never passes `targets=` to the `TargetDistance`
it constructs — and its only caller, `services/fuzzer.py`, only invokes
it when `if not targets:` (K-Scheduler is documented there as "mutually
exclusive with directed mode," i.e. the *undirected*-campaign channel).
So `target_addrs` was **guaranteed empty every single time this method
ran**, for any binary, in any campaign. The check was unconditionally
true. K-Scheduler could never activate at all since the perf commit
that introduced the check (`fe8fd42`) — the commit's own message is
"skip Katz ICFG when no targets," which is exactly backwards: Katz
*is* the no-targets path.

This went uncaught because every existing Katz test
(`test_katz_channel.py`, `test_katz_beta.py`) constructs `KatzChannel`
directly via `__init__` with a hand-built ICFG, bypassing `.build()`
entirely. Nothing exercised the classmethod end-to-end.

Confirmed empirically, not just by inspection: reverting just this hunk
against a real trace-pc-marked ELF (built with gcc, manual
`__sanitizer_cov_trace_pc()` calls at each block since no clang was
available) makes `KatzChannel.build()` return `None` unconditionally;
with the guard removed it returns a channel with real nodes/edges. Also
confirmed nothing about the channel actually depends on `target_addrs`:
`horizon.py`/`schedulers/katz.py`'s centrality math reads only
`hit_counts` and ICFG structure, never `td.target_addrs`.

Fix: removed the guard, documented why in `build()`'s docstring. New
`tests/test_katz_channel_build.py` drives the real classmethod
end-to-end (gcc-built binary, no clang dependency) instead of
constructing `KatzChannel` directly, so a regression here is caught
again — verified it fails against the pre-fix code and passes against
the fix.

## Verification summary

- `git am` stack applies cleanly on a fresh clone of `b8a39af`.
- 225 passed, 15 pre-existing skips (capstone) across
  dominators/distance/cfg/horizon/target_difficulty/katz/icfg/
  node_bitmap/shm_distance_channel/regression_cli_fuzzer_kwargs/
  regression_hail_mary_gates/regression_analyzer_registry.
- Each of the three fixes has a dedicated test proven to fail on the
  pre-fix code and pass on the fix (verified via `git stash` on the
  relevant file).

## Still open

- `gate_bonus` remains unvalidated against a real target/campaign (the
  original handover's point 2) — an A/B run is still the natural next
  step now that it's actually reachable from the CLI.
- Whether a target-*free* dominance signal is worth building (e.g.
  gating on a function's own exit block, or on a block once it's
  discovered interesting at runtime) is still just a sketch, not
  implemented.
