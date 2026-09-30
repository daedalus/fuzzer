# Handover — ExhaustivePool bulk-draw budget and the coin-flip rewrite

**Date:** 2026-09-26. Closes items 2 and 3 of
`docs/handover/handover_combinatorics_permutations_2026-09-02.md` (pruned
2026-09-26). **Verified against:** `bdf381df` (rebased from `b06afa5c` mid-pass
— see "Rebase note" below).

---

## 1. `ExhaustivePool` bulk gate is now per-call *and* per-run, not all-or-nothing

**Where:** `core/exhaustive_pool.py`, `ExhaustivePool._bulk_guard` /
`randbytes` / `*_list` methods.

**Before:** any `randbytes(n>0)`, `randrange_list`, `randint_list`,
`choice_list`, `weighted_choice_list`, or `categorical` call raised
`BulkDrawError` unless the whole pool was constructed with
`allow_bulk=True` — regardless of how small the call actually was.

**Now:** a new constructor argument `max_bulk_paths_per_call` (default
`DEFAULT_MAX_BULK_PATHS_PER_CALL = 65_536`, i.e. a lone `randbytes(2)`)
auto-permits any single bulk call whose own path count is within the cap,
with no `allow_bulk` needed. The same cap also bounds the *running
product* of every bulk call's paths within one run, so two calls that
each fit alone (say two `randbytes(1)`, 256 paths each) can still combine
past the cap (256×256 = 65,536, still fits; a third would not) — this was
the risk the combinatorics handover flagged for a naive per-call-only
cap ("several `randbytes` calls in one path multiply; the cap must be
per path, not per call, or `max_runs` truncates silently"). The
cumulative counter resets at the start of every run. `allow_bulk=True`
still bypasses both limits entirely, for callers who have done the
arithmetic themselves.

Both refusals raise `BulkDrawError` naming the path count and the cap, so
the message tells the caller which of the two limits it hit.

**Tests:** `tests/test_exhaustive_pool.py::TestRefusals` — 8 new tests:
auto-enumeration under the cap for all five bulk methods, calls that
combine past the per-run cap, calls that combine within it, the cap
resetting between runs, `max_bulk_paths_per_call` being configurable, and
`allow_bulk=True` still bypassing a lowered cap. The existing
`test_bulk_draws_refuse_by_default` was renamed
`test_bulk_draws_refuse_when_over_cap` and its parameter values bumped
(`randrange_list(10, 4)` → `(10, 5)`, etc.) since several of the original
examples now fit under the new default cap and were no longer testing a
refusal.

No other module constructs `ExhaustivePool` with `allow_bulk` (grepped
`ExhaustivePool(` across `src/`), so this is a self-contained change.

---

## 2. Coin-flip idiom rewritten for every fixed-probability site

**Census (93 sites total, `grep -rnE "\.random\(\)\s*<" src/fuzzer_tool`,
one exhaustive_pool.py hit excluded as it's inside that module's own
docstring):**

- **80 sites, fixed literal probability** (0.5, 0.3, 0.75, 0.35, 0.15,
  0.7, 0.8, 0.6, 0.34, plus three uses of a module-level rate constant —
  `VIOLATE_RATE = 0.3` in `wfc_chunks.py`, `SPARK_RATE = 0.10` in
  `schedulers/pos_burn_front.py`, `_REPAIR_PROB = 0.75` in
  `mutations/lz4.py`, each used at exactly one call site). **Rewritten**
  this pass to `rng.randint(0, N-1) < K` for the minimal exact `(N, K)`
  — `0.5 → randint(0,1)==0`, `0.3 → randint(0,9)<3`, `0.75 →
  randint(0,3)<3`, `0.25 → randint(0,3)<1`, `0.35 → randint(0,19)<7`,
  `0.15 → randint(0,19)<3`, `0.7 → randint(0,9)<7`, `0.8 →
  randint(0,4)<4`, `0.6 → randint(0,4)<3`, `0.34 → randint(0,49)<17`,
  `0.10 → randint(0,9)<1`. 22 files touched (full list in `git diff
  --stat` on the patch): `services/operators.py`, `services/stats.py`,
  `core/ga.py`, `core/cvm.py`, `core/format_seed_generator.py`,
  `core/wfc_chunks.py`, `core/schedulers/pos_burn_front.py`, and 16
  files under `core/mutations/` (`structured.py` alone had 22 sites).
  The named-constant sites keep their float constant declaration
  untouched (each is otherwise unused, so nothing else reads it) with an
  inline comment at the call site recording which constant and value the
  new `(N, K)` pair matches.

- **13 sites, runtime-computed probability** — left as `rng.random() <
  p` because `p` is not known until the call (a scheduler's `epsilon` or
  `passive_decay`, a GA's `crossover_rate`/`mutation_rate`, an
  acceptance ratio, `seed_picker.py`'s `gen_rate`,
  `fuzzer.py`'s `p_accept`, `cvm.py`'s `self.p`,
  `analyzer_elo.py`'s `total_ucb / (total_ucb + total_elo)`, one
  `<=`-based comparison in `seed_tang.py`). Converting these needs a
  design decision this pass didn't make: a fixed small `N` can't
  represent an arbitrary runtime float exactly, and a large enough `N`
  to approximate it well (as raised in the original handover's "Open")
  reintroduces the bulk-budget problem item 1 above just fixed, applied
  per-draw instead of per-call. Left as the next open item; sites listed
  in `tests/test_exhaustive_pool.py`'s
  `test_continuous_error_names_the_cheap_fix` docstring update.

- **2 genuinely continuous sites, correctly untouched** —
  `block_shuffle_variable` (`core/mutations/generic.py`) draws real
  `expovariate(1.0)` gap lengths, and `webm_chunk_mutate`
  (`core/mutations/webm.py`) packs `self._rng.random() * 10` into an
  IEEE double for a WEBM float field. Neither is a probability
  threshold; traced both via a one-off harness to confirm before writing
  this up.

**Effect on the operator table** (`services/operators.py` dispatch,
8-byte seed, `max_len=8`, `TestOperatorEnumeration.SEED`/`MAX_LEN`):
operators reporting `"continuous"` (silently unreachable by
`ExhaustivePool`) dropped from 34 to 2 (the two genuine ones above). 17
moved all the way to `"enumerated"` (`pool.exhausted` true): `arithmetic`,
`bitcast_float`, `bitcast_int32`, `corpus_literal_insert`,
`count_overflow`, `cycle_lock`, `float_squeeze`, `interesting_16`,
`interesting_32`, `interesting_8`, `perm_lock`, `radamsa_num`,
`size_field_overflow`, `spectral_peak`, `type_promote`, `varsize`,
`webp_chunk_mutate`. The other 15 moved to an honestly-checked status
instead of a masked refusal — 9 to `"over_budget"` (now sampled via the
spread fallback: `birthday_collide`, `degenerate_geometry`,
`elias_delta`, `elias_gamma`, `gcd_worst_case`, `length_miscalculate`,
`monotone_fill`, `protobuf_chunk_mutate`, `swap_bytes`), 5 to
`"too_deep"` (`der_len_mutate`, `der_tag_mutate`, `der_tlv_insert`,
`der_tlv_reorder`, `rle`), 1 to `"bulk"` (`kmer_starve`, which turned
out to hit item 1's gate further down its own path once the coin flip
in front of it stopped masking it).

**Tests:** new `TestCoinFlipRewrite` class in
`tests/test_exhaustive_pool.py` — parametrized `pool.exhausted`
assertions over the 17 newly-fully-enumerable operators, a parametrized
"no longer masked as continuous" check over the 15 that moved to an
honest non-continuous status, and a regression floor asserting the
`"continuous"` set is now exactly
`{block_shuffle_variable, webm_chunk_mutate}` (so a reintroduced coin
flip anywhere in the table fails this test by name). Updated the stale
docstrings/floors in `test_a_substantial_share_of_operators_is_enumerable`
(floor raised 55 → 150, counts appended for 2026-09-26) and
`test_continuous_error_names_the_cheap_fix` (noted the fix date and the
~13 remaining sites) so they don't read as still describing the
pre-fix state.

### Fallout: tests that monkeypatched `.random()` directly

Rewriting the *production* call from `.random()` to `.randint()` breaks
any test that forced a branch by monkeypatching `.random()` on the rng
object, or by scripting a `ScriptedRng(randoms=[...])` value for that
call site — the value is now read from a different method/queue
entirely, so the old script either silently does nothing (the branch
falls through to whatever the *real* stream returns) or raises
`StopIteration` (a `ScriptedRng` in strict mode still expects its
`randoms` queue to be drained if declared, and its `randints` queue runs
out early because the new call wasn't accounted for). Found and fixed
five affected tests, all in files already touched by this pass:

- `tests/test_structured_mutations.py::TestCycleLock` (2 tests) —
  `monkeypatch.setattr(type(rp), "random", lambda self: 0.9)` →
  `monkeypatch.setattr(type(rp), "randint", lambda self, a, b: 1)`
  (`cycle_lock`'s `big_endian` draw is the only `randint` call in its
  path, so patching the whole method is still safe and exact).
- `tests/test_cvm.py::test_down_sample_can_return_none_when_still_full`
  — its `KeepAll` stub only implemented `.random()`; added
  `randint(self, a, b): return a` (the down-sample `<0.5` site is now
  `randint(0,1)==0`, so returning the lower bound keeps "always keep").
- `tests/test_lz4_mutator.py` (7 tests) — `_mut_checksum`'s
  `repair = rng.random() < _REPAIR_PROB` is now the third `randint`
  call in that path (after the two `randint`s that already pick
  `Lz4Op.CHECKSUM` and the `ChecksumTarget`), so every
  `randints=[..., ...], randoms=[_REPAIR or _CORRUPT]` script needed
  the repair/corrupt sentinel moved into the `randints` list at that
  position and the `randoms=` kwarg dropped. Redefined the two
  module-level sentinels from floats (`_REPAIR = 0.0`, `_CORRUPT =
  0.99`) to the `randint(0,3)` values that select the same branch
  (`_REPAIR = 0`, `_CORRUPT = 3`).
- `tests/test_regression_bugreport_easy_fixes.py::TestRadamsaMutateNumInjectedRng`
  (1 test) — same shape: `radamsa_mutate_num`'s sign draw is now a
  third `randint(0,1)` call after the two that already pick the
  op/scale; moved the sentinel from `randoms=[0.9]` into
  `randints=[9, 9, 1]` (1, i.e. non-zero, selects the same "val - n"
  branch 0.9 used to).

Grepped the whole `tests/` tree for `setattr(type(rp), "random"` and
equivalents to make sure these five were the complete set for the sites
touched this pass — nothing else matched.

### Verification

Ran (not the full ~10k suite, per Hard Rule 50): `test_exhaustive_pool.py`
(103, all passing, including the new tests above), every dedicated test
file for each of the 22 touched `core`/`services` modules
(`test_cvm.py`, `test_format_seed_generator.py`, `test_ga.py`,
`test_bench_paired_arms.py`, `test_bench_paired_pos_arms.py`,
`test_avif_sqlite_mutators.py`, `test_mutations_der.py`,
`test_formatfuzzer_mutator.py`, `test_jpeg_mutations.py`,
`test_lz4_mutator.py`, `test_regression_sqlite_target.py`,
`test_structured_mutations.py`, `test_webm.py`,
`test_weizz_structural.py`, `test_wfc_chunks.py`, `test_new_operators.py`,
`test_stats_reporter.py`, `test_mutations.py`, plus the module-level
tests for `gzip`/`arm`/`gif`/`isobmff`/`protobuf`/`jpeg2000`/`webp`/
`flac`/`generic` reached via `test_operator_smoke.py`,
`test_rng_threading.py`, `test_new_format_mutators.py`,
`test_ffmpeg_port_mutators.py`, `test_format_mutators.py`,
`test_afl_det.py`, `test_sleb128.py`, `test_regression_dict_quotes.py`,
`test_bit_translation_ops.py`, `test_regression_operator_registry.py`,
`test_periodicity.py`, `test_regression_span_relocate_dest.py`,
`test_swap_tuple.py`, `test_swap_pair.py`, `test_import_corpus.py`,
`test_corpus_literals_incremental.py`,
`test_regression_corpus_literal_bytearray.py`,
`test_regression_block_shuffle_variable.py`,
`test_regression_bugreport_easy_fixes.py`), and
`test_operator_smoke.py::test_all_ops_fire` (every registered operator
fired once through the real dispatch table with a real `RandPool`, no
crashes). All pass after the five fixes above. `ruff check` on every
touched file shows the same 5 pre-existing findings as unmodified
`master` (verified by `git stash`/`ruff`/`git stash pop`), none new.
`test_regression_sqlite_target.py::TestBuildWiring` (2 failures) and
`test_regression_no_op_mutations.py::TestStateGatedOperatorsAreNotNoOps::test_every_selectable_operator_is_reachable`
(1 failure) reproduce identically on unmodified `master` — confirmed
pre-existing and unrelated, not investigated further here.

### Rebase note

A `git pull` partway through this pass brought `HEAD` from `b06afa5c` to
`bdf381df` (18 commits: `--wall-order`/minimax alpha-beta phases 3-5,
`fractal_voronoi` cell sub-ops, `format_fsm` uncapped-`max_len` fix, and a
round of handover pruning). `services/operators.py` was the only file
touched both upstream and by this pass — upstream added the
`--wall-order` branch immediately around this pass's rewrite of
`_op_condstmt_solve`'s `target_value = ... rng.random() < 0.5 ...` line;
the line itself was untouched upstream (pure context in their diff), so
`git stash` / `git reset --hard origin/master` / `git stash pop`
auto-merged cleanly with no conflicts. Re-ran the operator census after
the merge: identical split (157 enumerated / 2 continuous / 46
over_budget / 15 too_deep / 7 bulk) against a larger table (227 operators
now vs. 208 before the pull). One new upstream test file,
`tests/test_regression_wall_order.py`, scripted the same now-rewritten
line with `ScriptedRng(randoms=[0.1])`; fixed its 4 tests the same way as
the pre-existing fallout above (`randoms=[0.1]` → `randints=[0]`, since
`0.1 < 0.5` and `randint(0,1)==0` both select the same branch).

### Still open

- ~~The 13 runtime-probability `.random() < p` sites~~ — closed, see below.
- `kmer_starve` now reaches item 1's bulk gate rather than the coin-flip
  refusal; not investigated in this pass — its own bulk call may now be
  worth auto-enumerating if it is small (not measured here).

### Closed 2026-09-30: runtime-probability sites out of scope

`ExhaustivePool` is built only in tests and enumerates only
`OperatorEngine` dispatch paths. No open site is on one;
`TestCoinFlipRewrite`'s floor (`continuous` == the 2 genuine operators)
passes on `e02546b`. Rewriting would bias probabilities
(`round(p*N)/N`), add per-draw cost on hot paths (Whittle drift loop,
Metropolis step) and shift the RNG stream under scripted tests, for no
enumeration gain. **Rule:** coin-flip rewrites apply only to code reachable
from operator dispatch.

Census on `e02546b` (`grep -rnE "\.random\(\)\s*<"`, 20 sites):

- Runtime `p`, original set (11 found; 2 of the 13 no longer match):
  `seed_picker.py:953`, `fuzzer.py:7090`, `ga.py:410,415`,
  `seed_tang.py:165`, `op_epsilon_greedy.py:62`, `op_whittle.py:391,406`,
  `op_monte_carlo.py:566`, `analyzer_elo.py:403`, `cvm.py:71`.
- Runtime `p`, schedulers added 2026-09-28: `op_credit.py:159`,
  `seed_residual.py:135`, `pos_boundary.py:193`.
- Fixed literal, added 2026-09-28: `pos_boundary.py:188`,
  `pos_cmplog.py:182`, `pos_levy.py:96,102`, `pos_lineage.py:134,142`.
  Also off operator paths; left as is.
