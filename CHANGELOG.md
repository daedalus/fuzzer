# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`--consolidated-prior {fixed,blup}`** (`core/blup.py`): consolidated v1/v2 category-prior strength fitted per category from its operators' dispersion (empirical-Bayes BLUP, ceiling 8). Ties `fixed` on the synthetic tournament; opt-in.
- **Touched-slot bitmap for seed stability calibration** (`ShmCoverage(touched_bitmap=True)`, on when `--calibrate-stability` > 0; `afl_shim.c`, `Fuzzer._unstable_from_bitmap`): the shim sets one bit per edge-table slot when it goes live in a generation (first claim / stale reclaim, not on hits); calibration decides stability from `OR & ~AND` over the runs' bitmaps and one id gather rather than a full scan and Python set algebra per run. Region appended after the distance tail (no offset moves, no layout bump); shim announces it with a magic word, so an older shim or an undersized segment falls back to the edge-set path unchanged. Falls back to edge ids when a later run is under generation tag 0 (table wiped) and vetoes on dropped edges, as before. 3-run analysis cost 1.4 ms -> 26 us (8192 entries), 8.8 ms -> 141 us (1M); shim +0.9% / +3.9% per exec with the region present, none without. The bit set is a plain OR (a threaded target can lose a bit, i.e. a false unstable edge); `-D__AFL_TOUCHED_ATOMIC=1` makes it atomic at +16-24% shim time per exec. `_repeat_runs` is now the shared run loop (module-level, so tests that borrow `_repeat_edge_sets` unbound still work). Tests: `test_shm_touched_bitmap.py`, `test_stability_touched.py`.
- **`fuzzer-tool compact-seeds -d CORPUS`** (`seed_zip.compact`): moves pruned seeds from `seeds.zip` to `seeds/pruned/` and rewrites the archive without them, their tombstones and re-admission duplicates. Offline, nothing deleted, replay-checked swap. 10k seeds / 90% pruned: 5.2 MB -> 0.4 MB, load 150 -> 20 ms. Tests: `tests/test_seed_zip_compact.py`.
- **`--zip-compact-ratio R`** (`seed_zip.compact_over`, `pruned_ratio`): compact at startup when pruned seeds exceed R x live seeds in `seeds.zip`. Off by default; excluded from `--hail-mary` (a float, not a strategy).
- **`--pos-kadane`: maximum-excess-gain window position arm** (`core/kadane.py`, `core/schedulers/pos_kadane.py`). Each bin scores its gains above the seed's pooled rate (`s - p0*n` over the `BinRates` counts `changed` uses); Kadane's maximum-sum run over those scores is the contiguous window of bins that paid most, and offsets are drawn inside it (10% uniform). Untried bins score 0, so a gap of them is bridged for free; ties go to the tightest window. `max_subarray` is vectorized (prefix sums + `minimum.accumulate`: 43 us vs 450 us for the loop at 4096 bins). `PositionKadaneScheduler.window()` returns the run as `(offset, length)`, not yet wired into `structured._region`. Opt-in: **not** implied by `--position-arena` or `--hail-mary` (unmeasured); adds `BinRates.width()`. Synthetic only: 80% of proposals in a planted 60-byte hot region after 500 rounds (uniform 3%). Window hook: `_region(..., window=)` fits the run to the operator's `min_len`/`align`/`max_len` (a run narrower than `min_len` is widened around its centre) and consumes no draw; `cusum_bias_run` and `monotone_fill` accept `window=`, and with `--pos-kadane` on, `OperatorEngine._regularity_windowed` hands them the parent's run on `WINDOW_P` (0.5) of calls (the rest, and every call with the arm off or no run yet, stay random). The other region operators are untouched. Credit still goes to the offset the mutation loop drew, not to the window. Tests: `tests/test_kadane.py`, `tests/test_pos_kadane.py`, `tests/test_kadane_region.py`.
- **`div_trap` regularity operator**: plants `(MIN, -1)` in two adjacent fields (width 1/2/4/8, little/big endian, either order) to reach signed-division overflow (SIGFPE on x86 `idiv`, panic in Rust/Swift). No single-field operator can produce the pair. Handover `docs/handover/handover_div_trap_2026-10-05.md`; tests `tests/test_regression_div_trap.py`.

### Changed

- **Edge table slot math is divide-free** (`afl_shim.c`): `edge_id % __afl_map_size` and the probe wrap `(pos + i) % __afl_map_size` ran a runtime 32-bit divide per edge and per probe step. Home slot is now a mask for power-of-two sizes and Lemire fastmod otherwise; probe wraps by subtraction, written to avoid `pos + i` overflow. Placement identical to `%` (fastmod checked against `%` on 16M cases), so no `__AFL_SHM_LAYOUT` bump. Microbench (50M edges, 2000 distinct): 4.2 -> 3.1 ns/edge at 8192, 4.2 -> 3.4 ns at 8000. Distance-table probe untouched. Tests: `tests/test_shim_map_slot.py`.

- **`--pos-saliency` refits on `MatrixSubstrate`'s cadence, not its own** (handover `docs/handover/handover_neuzz_port_2026-10-03.md` item 4: reuse it rather than copy NEUZZ's retrain trigger). The policy `MatrixSubstrate.maybe_refit` learned the hard way is now `core/edge_matrix.RefitCadence` (executions not calls; too few seeds refused BEFORE the clock is stamped; skip when no seed/edge was added; stamp only on success), used by the substrate (behaviour unchanged, `substrate.cadence`) and by `PositionSaliencyScheduler`. The saliency net's clock is `Fuzzer.exec_count` (it counted its own proposal calls, 2000 of which could be a handful of real execs) and `maybe_refit(exec_count)` is driven from the same discovery hook as the substrate's, with a cheap interval pre-check that does not walk the corpus. It also inherits the substrate's `coverage_trust` gate (`trust_fn`): with unstable edge ids it declines to fit, to propose, and to arm `saliency_ladder`, exactly when the matrix arms abstain; the F1 stability probe feeds it with or without a substrate. Standalone use (no `exec_count_fn`) keeps lazy fitting from `propose`/`record`/`warm`, with refused retries spaced 200 calls apart. Removes `RETRY_INTERVAL`.

### Fixed

- **`hierarchical_pooling` was overconfident** (`BayesianSeedQuality`, `MonteCarloScheduler`): the convex blend gave each unit ~h x the population's total evidence as pseudocounts. Now an empirical-Bayes prior `Beta(h m mu, h m (1 - mu))` from `core/blup.py`; the pooled seed pick is vectorised (3.3x faster).
- **`RandPool.sample` and Floyd sampling reach populations above 2**32** (`rand_pool.sample` int branch k=1/2, `_floyd_indices`): they still took one uint32 word (`_draw() % n`) after `randint`/`randrange` learned to compose words, so `sample(1 << 40, 1)` and Floyd over a 40-bit population never returned a value at or above 2**32. They now call `randrange`, which is the same single draw for `n <= 2**32` (seeded runs identical) and composed draws above it. Tests: `tests/test_regression_rand_pool_wide_range.py` (also pins the two-draw 64-bit composition, the unchanged narrow path, the list variants and the Ogg granule position).
- **No global `random` seeding** (Hard Rule 16): `Fuzzer.__init__` and `_reseed_after_stall` called `random.seed`, reseeding the process-wide stream. They now reseed the default `RandPool`. Former global draws in `shapley`, `qea` tournament, `tree_mutator`, `mi`, `edge_tracker.update_correlation`, `gradient_descent` and `analyzer_frameshift` use `RandPool` (`rng`, else the default pool). `np.random.seed` for QEA/Monte-Carlo unchanged. Tests and tools still call `random.seed`.
- **`seeds/` inside `seeds.zip` is the same level as the archive root, not a foreign tree** (`seed_zip._parse`): `seeds.zip` sits beside `seeds/`, so members are `ab/id_<h>`; an archive built from the `seeds/` directory itself (`seeds/ab/id_<h>`, `seeds/irreplaceable/..`, `seeds/.pruned/..`) was classified foreign and, under `--zip-seed-corpus`, re-adopted as a second `ab/id_<h>` copy of every seed. Leading `seeds/` components (also repeated, `seeds/seeds/..`) are now stripped when parsing, so those members count as live/protected/tombstone exactly like root-level ones, are never duplicated, and new members are never written under `seeds/`. `seeds/pruned/..` members stay out (pruned stays pruned); `compact-seeds` reads them by their real name and collapses prefixed/unprefixed duplicates. Tests: `tests/test_seed_zip_corpus.py`, `tests/test_seed_zip_compact.py`.
- **Hacker's Delight operators raised `TypeError` on `RandPool`** (`rightmost_*`, `same_popcount_*`): `_pick_window` drew `rng.randrange(0, n)`, but `RandPool.randrange` takes one argument, so every handler failed in a real run while the `random.Random` tests passed. Now `rng.randint`. Registry test `REGULARITY_OPS` also lacked `same_popcount_next/prev`. Regression: `test_operators_run_on_randpool`.
- **`seeds/` and `seeds.zip` are one pool; a seed is never stored in both**. Under `--zip-seed-corpus`, `_put_seed` checked only the zip, so a seed already held as a loose `seeds/` file (canonical `seeds/ab/id_<h>` or the flat `seeds/id_<h>` that load normalises to) was written again into `seeds.zip` whenever something re-offered it (seen-hash eviction, trim, a second tracker). It now skips the write when either location holds a live copy, for every tree (main, irreplaceable, crashing, timeouts). With the flag off, an existing `seeds.zip` is consulted the same way (cached read-only `seed_zip.peek`, keyed by mtime+size) so a zip seed is not copied out to a file. A tombstoned (pruned) seed is not 'held' and is still re-admitted. Duplicates already present are not removed.
- **`seeds.zip` members with arbitrary names are loaded as seeds**. `SeedZip` only recognised members named `ab/id_<hash>`; a zip built elsewhere (flat or nested names such as `a.avi`, `sub/b.ogg`) was indexed as empty, so `--zip-seed-corpus` loaded nothing from it even with the flag set. Foreign file members are now treated like the loose files `load_corpus` already accepts: keyed by content, duplicates collapsed. Under `--zip-seed-corpus` they are re-added once as canonical `ab/id_<hash>` members (so prune/retire can tombstone them and a pruned seed is not re-adopted); read-only loads serve them without touching the archive. Directories, absolute and `..` names stay refused.
- **`corpus/seeds.zip` is loaded without `--zip-seed-corpus`**. Only `seeds/` was read unless the flag was set; an existing `seeds.zip` was skipped with a warning. `load_corpus` now reads it through an unregistered read-only `SeedZip` (`seed_zip.open_readonly`): tombstones and protected trees (`crashing/`, `irreplaceable/`, `timeouts/`) are honoured, writes still go to files, the archive is not appended to. Limit: without the flag a zip-resident seed cannot be tombstoned on prune.
- **cmplog/COMPCOV string interceptors SIGSEGV on unterminated operands** (`strcmp`, `strcasecmp`, `wcscmp`, `wcscasecmp`, `strpbrk`, `strspn`, `strcspn`). After the real libc call they sized the cmplog record with an unbounded strlen/wcslen, but the real call may legally return at the first mismatch/match, so an operand that is unterminated and ends at a page boundary is fine for libc while the shim ran off the mapping and faulted (a target-looking crash with the shim on top). `--hail-mary` forces cmplog + compcov on, so string-heavy targets (ffmpeg) were exposed. Lengths now come from `__afl_safe_len` / `__afl_safe_wcslen`, bounded to the page holding `s[0]` and to ASAN-poisoned bytes. Regression: `tests/test_regression_shim_unterminated_operands.py` (native-libc control per case; SIGSEGV before, clean after).

- **`__afl_get_caller_ctx()` SIGSEGV on a junk saved frame pointer** (seen fuzzing a frame-pointer-less ffmpeg build with `--inprocess-direct`: `si_code=SI_KERNEL`, rbp-chain value `0x800000010000` left in rbp by `s337m_probe`). The walk bounded only the hop length (4 MiB above the shim frame), so a value just above `cur` but past the stack end passed the check and the load of `caller_fp[1]` hit unmapped/non-canonical memory. The early `cfp >> 47` canonical-address reject (`1b2961fa`) covers only the non-canonical case; a junk value in the canonical range just past the stack end (or misaligned) still faulted, so the walk is now also bounded to the calling thread's real stack (`pthread_getattr_np`, cached per thread), 8-aligned, and rejected on a sigaltstack. Regression: `tests/test_regression_ctx_walk_stack_bound.py` (SIGSEGV before, clean after, ASLR on and off).

- **`saliency_ladder` operator crashed, ignored the gradient sign and never resized** (review of `410a481`). `rng.randint(start, end)` is inclusive, so the rank-bucket pick indexed one past `top_indices` (`IndexError` on ~14% of calls at 100 B, ~100% at 5 B); the step direction was a coin flip (measured agreement with the true gradient sign 0.51); the docstring promised NEUZZ's insert/delete but there was none; and ranking used the MEAN of signed gradients over several targets, which cancel. Now `core/saliency_ladder.py` (pure, NEUZZ tiers 2/2/4/8.., log-uniform 1..255 steps, follows the sign with P=0.75, 20% block delete/insert at hot offsets) driven by `PositionSaliencyScheduler.gradient_info` (signed gradient of ONE drawn target edge). The registry gate now requires a FITTED model (`warm()`), and `record()` gives a due refit its chance so the model exists even when the arena rarely draws the arm. The operator returns a new buffer instead of mutating its input.
- **Saliency `target_selector` was dead and fed wrong ids.** It was never passed by `Fuzzer`, and `refit()` handed it a seed index (`next(iter(rows))`) as the "edge id", so every weight came back 0 and was silently ignored. It now gets the smallest real edge id of each column class, its weights REPLACE `1/support` (they were multiplied, counting rarity twice), a bad selector falls back to `1/support` instead of aborting the fit, and the cumulative target weights are installed together with the model (a failed fit used to leave new weights with an old net).

### Changed

- `--saliency-targets {gt,support}` (default `gt`): `gt` weights target edges by the inverse Simple Good-Turing adjusted count (`gt_rarity_selector`, log-log smoothed, never inverts the rarity order), `support` keeps plain `1/support`.
- `--pos-saliency` is part of `--hail-mary` (it was excluded as unmeasured; upstream lifted that).

### Removed

- `_dominator_selector`: a stub returning all-ones while its commit message claimed ICFG/dominator weighting. A real one needs the shim's edge-id -> basic-block map, which the fuzzer does not hold at runtime (see `docs/TODO.md`).
- `PositionSaliencyScheduler.signed_saliency` (mean of signed gradients: not meaningful).

### Added

- **`--grimoire`: grammar-free structure inference** (`core/grimoire.py`, `services/grimoire.py`; backlog A1, survey gap 5): each admitted seed is re-run once with spans removed (chunks, delimiter spans, bracket pairs); removals that still reach the seed's novel edges become gaps. Operators `grimoire_extend`, `grimoire_recurse`, `grimoire_string` recombine the surviving tokens across seeds. `--grimoire-max-execs` (default 512) caps probes per seed; seeds > 4096 B are skipped. Off by default; in `--hail-mary`. Unmeasured.
- **`--ltl HOA`: LTL property monitor** (port of LTL-Fuzzer, ICSE '22; `core/ltl.py`, `docs/handover/handover_ltl_2026-10-04.md`). Targets call `__fuzz_event(id)` / `__fuzz_event_at(id, offset)` (shim: folds event order into the edge map, appends `{id, offset, state-var digest}` to `$__LTL_EVENTS_OUT`; no SHM layout change). The Buchi automaton of the negated property runs over each trace; acceptance is saved as crash `ltl:trap` or `ltl:lasso` (candidate). A new monitor transition admits the mutant; `ltl_prefix` (block band) keeps the shortest prefix reaching each state with unfired transitions and appends a fresh tail.
- **Growing Tree seed arm + lineage shape** (handover `docs/handover/handover_maze_algorithms_2026-09-24.md` items 1-3): `LineageTree.shape()` (leaf / corridor fraction, max depth, mean unary-chain length; printed in the run summary); `--seed-newest-scheduler` / `--seed-newest-p` (Elo arm `seed_newest`: newest seed with probability p, else uniform; off by default, not in `--hail-mary`); `bench_paired` arms `seed-round-robin`, `seed-newest-p{0,30,60,100}`. Unmeasured: no paired benchmark yet.
- **`--checksum-sites`: TaintScope-style checksum repair** (`core/checksum_sites.py`; survey gap 2): on each seed, finds fields equal to CRC-32 / Adler-32 / CRC-16 / Fletcher-16 / 16-bit sum of a region, then recomputes them in every mutant (end of `OperatorEngine.mutate`, `REPAIR_P=0.9`). Covers header and mid-file fields (PNG-like chunks), not only a trailer. Off by default; in `--hail-mary`.
- **`--pos-finch`: Finch hot-byte position arm** (`core/schedulers/pos_finch.py`; handover `docs/handover/handover_paper_collection_survey_2026-10-02.md` gap 1): weights a byte by how many edges its byteflip moved (`OperatorEngine.effector_heat`), not just live/inert like `effector`, plus a bounded per-seed gain bonus. Needs `--deterministic`; implied by `--position-arena`.

- **`--pos-saliency`: NEUZZ-style learned position arm** (`core/nn_saliency.py`, `core/schedulers/pos_saliency.py`; handover `docs/handover/handover_neuzz_port_2026-10-03.md`, candidate 1). A one-hidden-layer numpy MLP is fitted corpus bytes -> edges (columns = seed-set classes, rare ones weighted `1/support`, output bias at base rate); offsets are drawn by the closed-form input gradient `W1 @ (1[h>0] * W2[:, k])` of a few drawn target edges, mixed 20% uniform, bytes past the 512-byte input cap keep their uniform share. Reimplemented from the algorithm (Neuzz++ is AGPL-3.0; no code shared). Opt-in: **not** implied by `--position-arena` and **excluded from `--hail-mary`** (unmeasured). Bench arm `pos-arena-saliency` (baseline `pos-arena-uniform`). Measured on synthetic planted-byte data only: 3.5-7x uniform saliency mass on the deciding bytes; fit 0.17 s at the 256 x 512 cap, propose 0.07 ms. Not benchmarked on a real target.

### Documentation

- **paper_collection survey handover** (`docs/handover/handover_paper_collection_survey_2026-10-02.md`): maps ~814 papers to existing fuzzer features; ranks six gaps (hot bytes, checksum repair, binary rewriting, resource feedback, Grimoire, evaluation stats). Analysis only.

### Fixed

- **`fnv1a_p`/`fnv1a_r` slowed every Redqueen pair** (`core/rq_encodings.py`): replacement variants were built before the input search, 400 non-matching pairs 0.009 s → 0.51 s. Restored lazy build after a pattern hit (0.010 s). FNV-1a now inverts only the constant, not ±64 neighbours: match found 2.18 s → 0.05 s per pair. 2-byte table is `dict[int, int]` (injective), built once via `functools.cache`. Comments no longer call FNV-1a a bijection (measured: ~1 preimage on average, none for ~1/3). Encodings with an empty chunk are dropped on both sides.
- **Doppler drop memory churned on huge corpora** (`core/power_doppler.py`): the 2¹⁵-key LRU forgot every dropped key before a larger cycle returned, so the horizon never widened. Now the bottom-k keys by `crc32_ieee` are kept: a fixed, never-empty subset of any cycle within the same bound (a halving crc threshold could empty out on all-odd crcs).

- **Doppler horizon compounded per returning seed** (`core/power_doppler.py`): each return doubled one shared horizon (N returns → 2^N), disabling abandonment; and the dropped-key memory (`max_seeds` keys) forgot keys before large corpora cycled back. Horizon is now `max(horizon, 2 × measured gap)`; memory is 2¹⁵ keys.

- **Doppler horizon thrashed on slow corpus cycles** (`core/power_doppler.py`): a cycle longer than the fixed abandon horizon dropped every frame one tick before its seed returned. The horizon now doubles when a dropped seed comes back.
- **Stale corpus-membership memo** (`services/fuzzer.py`): keyed on corpus length, so an in-place trim of the parent kept a "member" verdict. Hits are now re-validated by slot identity; misses are not memoized.

- **`--mod-solving trace` clobbered `targets`** (`services/fuzzer.py`): the trace block reassigned the directed-targets parameter, disabling the Katz channel and directing at the fuzz target. Renamed the local.
- **Doppler never scored corpora > 64 seeds picked in turn** (`core/power_doppler.py`): LRU evicted every partial frame. New seeds now wait for a slot; abandoned frames are scored early and freed. Flow-edge ids are int64 arrays under a global cap (frozensets could reach hundreds of MiB).
- **Doppler mixed targets' edges** (`services/fuzzer.py`): multi-target frames are keyed per target. Without SHM, `--schedule doppler` now falls back to `base` with a warning instead of reporting enabled.
- **Seed-arm ledgers recorded standalone-QEA parents**: `seed_meta` is not corpus membership. Gated on the seed picker's cached corpus key map, memoized per parent.

- **EEVDF pick scanned ineligible flows** (`core/fair_queue.py`): one deadline heap popped every flow with an earlier deadline but `ve > V` (5000 pops at 5000 flows). Now a `ve` heap feeds a deadline heap; amortized O(log n). Test: `test_eevdf_pick_does_not_scan_ineligible_flows`.
- **Seed-arm ledgers grew without bound**: non-corpus (Markov) parents were recorded, and departed seeds never left `ArmCounts`. Only corpus parents are recorded now; ledgers trim to 2x the live corpus.
- **Round robin was O(n^2) per pick** (`seed_round_robin`, `op_round_robin`): `x in list` per registered arm. Set membership now: 103 ms -> 0.7 ms per pick at 5000 seeds.
- **`cuckoo_seed_filter` broke 12 corpus-minimization / lineage tests** (`services/corpus_manager.py` read it unguarded) and was missing, with `swap_walk`, from `_HAIL_MARY_FLAGS`.
- **Stack depth was always 0** (`adapters/afl_shim.c`): `__afl_max_stack_depth` was reset and copied to SHM offset 0 but never assigned, so `read_stack_depth()` returned 0 and the stack-depth boost in `core/schedules.py` never fired. The shim now samples the frame address in `__afl_map_loc` (base = first sample after reset, live write). Tests: `tests/test_shim_stack_depth.py`. Per-edge cost unmeasured; boost effect on discovery untested.

### Added (sancov)

- **`indirect-calls` coverage** (`adapters/afl_shim.c`, `tools/build_targets.sh`): `__sanitizer_cov_trace_pc_indir` hashes `(call site, callee)` into a synthetic id (bit 31, base-relative). Callees outside the module are dropped. Build with `--indir-cov` or `--sancov=...,indirect-calls`. Tests: `tests/test_shim_indir_cov.py`, `tests/test_sancov_modes.py`. Discovery effect and map pressure unmeasured.

### Tests

- **Field isolation validated on a real crashing target** (`tests/test_isolate_fields_proto_target.py`): builds `targets/proto_target.c` with ASAN and checks `isolate_fields_failure` / `root_cause --isolate-fields` return exactly the header fields each of its four crashes needs (null deref, heap overflow, stack overflow, abort), excluding irrelevant fields. Skips without gcc/ASAN.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- **paper_collection survey handover** (`docs/handover/handover_paper_collection_survey_2026-10-02.md`): maps ~814 papers to existing fuzzer features; ranks six gaps (hot bytes, checksum repair, binary rewriting, resource feedback, Grimoire, evaluation stats). Analysis only.

### Fixed

- **`fnv1a_p`/`fnv1a_r` slowed every Redqueen pair** (`core/rq_encodings.py`): replacement variants were built before the input search, 400 non-matching pairs 0.009 s → 0.51 s. Restored lazy build after a pattern hit (0.010 s). FNV-1a now inverts only the constant, not ±64 neighbours: match found 2.18 s → 0.05 s per pair. 2-byte table is `dict[int, int]` (injective), built once via `functools.cache`. Comments no longer call FNV-1a a bijection (measured: ~1 preimage on average, none for ~1/3). Encodings with an empty chunk are dropped on both sides.
- **Doppler drop memory churned on huge corpora** (`core/power_doppler.py`): the 2¹⁵-key LRU forgot every dropped key before a larger cycle returned, so the horizon never widened. Now the bottom-k keys by `crc32_ieee` are kept: a fixed, never-empty subset of any cycle within the same bound (a halving crc threshold could empty out on all-odd crcs).

- **Doppler horizon compounded per returning seed** (`core/power_doppler.py`): each return doubled one shared horizon (N returns → 2^N), disabling abandonment; and the dropped-key memory (`max_seeds` keys) forgot keys before large corpora cycled back. Horizon is now `max(horizon, 2 × measured gap)`; memory is 2¹⁵ keys.

- **Doppler horizon thrashed on slow corpus cycles** (`core/power_doppler.py`): a cycle longer than the fixed abandon horizon dropped every frame one tick before its seed returned. The horizon now doubles when a dropped seed comes back.
- **Stale corpus-membership memo** (`services/fuzzer.py`): keyed on corpus length, so an in-place trim of the parent kept a "member" verdict. Hits are now re-validated by slot identity; misses are not memoized.

- **`--mod-solving trace` clobbered `targets`** (`services/fuzzer.py`): the trace block reassigned the directed-targets parameter, disabling the Katz channel and directing at the fuzz target. Renamed the local.
- **Doppler never scored corpora > 64 seeds picked in turn** (`core/power_doppler.py`): LRU evicted every partial frame. New seeds now wait for a slot; abandoned frames are scored early and freed. Flow-edge ids are int64 arrays under a global cap (frozensets could reach hundreds of MiB).
- **Doppler mixed targets' edges** (`services/fuzzer.py`): multi-target frames are keyed per target. Without SHM, `--schedule doppler` now falls back to `base` with a warning instead of reporting enabled.
- **Seed-arm ledgers recorded standalone-QEA parents**: `seed_meta` is not corpus membership. Gated on the seed picker's cached corpus key map, memoized per parent.

- **EEVDF pick scanned ineligible flows** (`core/fair_queue.py`): one deadline heap popped every flow with an earlier deadline but `ve > V` (5000 pops at 5000 flows). Now a `ve` heap feeds a deadline heap; amortized O(log n). Test: `test_eevdf_pick_does_not_scan_ineligible_flows`.
- **Seed-arm ledgers grew without bound**: non-corpus (Markov) parents were recorded, and departed seeds never left `ArmCounts`. Only corpus parents are recorded now; ledgers trim to 2x the live corpus.
- **Round robin was O(n^2) per pick** (`seed_round_robin`, `op_round_robin`): `x in list` per registered arm. Set membership now: 103 ms -> 0.7 ms per pick at 5000 seeds.
- **`cuckoo_seed_filter` broke 12 corpus-minimization / lineage tests** (`services/corpus_manager.py` read it unguarded) and was missing, with `swap_walk`, from `_HAIL_MARY_FLAGS`.
- **Stack depth was always 0** (`adapters/afl_shim.c`): `__afl_max_stack_depth` was reset and copied to SHM offset 0 but never assigned, so `read_stack_depth()` returned 0 and the stack-depth boost in `core/schedules.py` never fired. The shim now samples the frame address in `__afl_map_loc` (base = first sample after reset, live write). Tests: `tests/test_shim_stack_depth.py`. Per-edge cost unmeasured; boost effect on discovery untested.

### Added (sancov)

- **`indirect-calls` coverage** (`adapters/afl_shim.c`, `tools/build_targets.sh`): `__sanitizer_cov_trace_pc_indir` hashes `(call site, callee)` into a synthetic id (bit 31, base-relative). Callees outside the module are dropped. Build with `--indir-cov` or `--sancov=...,indirect-calls`. Tests: `tests/test_shim_indir_cov.py`, `tests/test_sancov_modes.py`. Discovery effect and map pressure unmeasured.

### Tests

- **Field isolation validated on a real crashing target** (`tests/test_isolate_fields_proto_target.py`): builds `targets/proto_target.c` with ASAN and checks `isolate_fields_failure` / `root_cause --isolate-fields` return exactly the header fields each of its four crashes needs (null deref, heap overflow, stack overflow, abort), excluding irrelevant fields. Skips without gcc/ASAN.

### Added

- **Per-position Good-Turing arm** (`--pos-good-turing`, `core/schedulers/pos_good_turing.py`): position-arena proposer drawing a seed's offset bin by Good-Turing discovery probability over edge identity (entropy handover §7.1). Implied by `--position-arena`; bench arm `pos-arena-good-turing`. Unmeasured.
- **Format-aware Adler-32 patcher** (`core/mutations/recompress.py::patch_adler`, wired into `_op_crc_learn`): with a recovered Adler-32 model, a bare zlib stream or a PNG's joined IDAT data gets its trailer rewritten over the inflated plaintext (raw inflate, so a stale trailer is irrelevant); touched IDAT chunk CRCs are recomputed as CRC-32, every other byte is kept. Before, the generic path wrote Adler-32 of `buf[:-4]` over the IEND CRC. Other int models, non-container input, and non-inflatable streams keep the old behaviour; a PNG with no patchable stream is returned unchanged.
- **Consolidated seed and position arms** (`--seed-consolidated-scheduler`, `--pos-consolidated`): `seed_consolidated` merges the OS/network seed arms (p2c over sfq flows, eevdf/drr cost, stride favored weight, aimd decay, round-robin sweep); `pos_consolidated` merges the learning position arms (uniform/boundary/levy/bin candidates scored by context x per-seed bin rates).
- **`--consolidated-v2`** (`core/schedulers/op_consolidated_v2.py`): consolidated v1 scored by an optimistic, tempered Thompson draw. +1.3-1.8% over v1 on all four `bandit_env` environments (20 paired seeds). Leads no-Elo precedence.
- **Power Doppler schedule** (`--schedule doppler`, `core/power_doppler.py`): per-seed ensembles of mutant hit counts; mean + SVD wall filter, CFAR χ² flow detection; flow power scales seed energy to `[1, max_mult]`.
- **OS / network scheduler ports**: seed arms `mlfq`, `stride`, `eevdf`, `bfq`, `sfq`, `codel`, `aimd`, `p2c` (`--seed-<name>-scheduler`) and op arms `op_stride`, `op_p2c` (`--op-stride`, `--op-p2c`); `core/fair_queue.py` gains `Stride` and `EEVDF`. Elo arms, in `--hail-mary`; seed arms also run without `--elo`. Falsification and adversarial tests. No paired benchmark yet.
- **Ten op mutators** for in-tree targets and text decoders: `json_mutate` (fuzzgoat), `sql_mutate` (sqlite SQL path), `ecdsa_field_mutate` (secp256k1), `recompress_lz4`, `recompress_png_idat` (format band, sniffer-gated); `encoding_wrap`, `escape_mutate`, `ascii_float` (structural); `utf16_transcode`, `nest_bomb` (radamsa). Tests: `tests/test_{json_mutate,sql_mutate,ecdsa_field_mutate,recompress_roundtrip,text_codec,nest_bomb}.py`. Discovery effect on fuzzgoat unmeasured.
- **`covering_array_webp`, `covering_array_isobmff`, `covering_array_zip`** operators (`core/mutations/covering_array_container.py`): pairwise sweep of WebP `VP8X` (riff_size, chunk_size, flags, reserved, width, height; 49 rows), ISO-BMFF `ftyp` (size, major brand, minor; 30 rows) and the first ZIP local file header (version, flags, method, sizes, name/extra length; 50 rows). Each gated on its own magic. Selection share unmeasured; ISO-BMFF `tkhd` not covered (variable depth).
- **Walker covering-array operators** (`core/mutations/covering_array_walk.py`): eight more operators for fields at data-dependent offsets, found by a bounded box/chunk/record walk: `covering_array_isobmff_tkhd` (moov/trak/tkhd, v0 and v1 layouts; works without `ftyp`), `covering_array_wav` (fmt fields, fmt/data/riff sizes), `covering_array_avi` (LIST hdrl/avih), `covering_array_webp_vp8` (lossy frame tag, start code, width/height; also behind a `VP8X` chunk), `covering_array_webp_vp8l` (signature + packed size word), `covering_array_zip_eocd`, `covering_array_zip_cd` (random central-directory entry) and `covering_array_zip_entry` (local header of entry 1 or later). An input missing any field declines, so each costs ~0.2-0.9 us on other formats. Selection share on real targets unmeasured.
- **Fair-queue schedulers** (`core/fair_queue.py`): `SmoothWRR` (nginx smooth weighted round robin), `DeficitRR` (carries unspent credit, bounded per pick) and `WeightedFairQueue` (SCFQ virtual time; idle flows re-enter at the clock, no banked credit). Garbage weights/costs are excluded or neutral, never a hang or a raise.
  - `--target-schedule wrr|wfq`: deterministic weighted round robin, or wall-time fair queuing, on 1/edges. WFQ charges the time between selects to the target that ran.
  - `--seed-drr-scheduler` (`core/schedulers/seed_drr.py`): seed-arena arm `seed_drr`, equal share of target time per seed from the cost ledger, favored seeds 2x. Identical to `--seed-round-robin-scheduler` when exec cost is flat. Enabled by `--hail-mary`. No paired benchmark yet.
- **Failure-inducing combination isolation** (`core/failure_inducing.py`): given a failing parameter row and an oracle, finds the minimal set of parameter values that cause the failure (FIC-style, ~k probes, memoized, budget-capped, optional sufficiency check). PNG IHDR adapter `isolate_png_ihdr_failure` and `root_cause --isolate-png-ihdr`, which adds the responsible IHDR fields to the report. Tested against mocked oracles only.
- **Generic field adapter** (`core/field_spec.py`, `root_cause --isolate-fields SPEC`): declare fixed-offset integer fields (`NAME@OFFSET:SIZE[be|le][=V|V...]`) for any format; boundary-value domains by width, plus the baseline's value at each offset, feed `failure_inducing.isolate`. Result under `custom_field_schema`. Live-tested against `targets/test_target.c`.
- **`--isolate-crash-fields` (in `--hail-mary`)**: for each novel crash, FIC-replay the target (<= 64 execs, standalone subprocess; stdin or file mode) over the fields `core/field_map` recognises (PNG, gzip, ...) and record the fields the crash needs as `failure_schema` in the crash `.json`/`.txt` sidecar (`services/crash_isolate.py`). In-process targets are replayed through a private subprocess-loader runner (coverage-free, own process), and the reference failure is re-measured on that backend. Live-tested: the fuzzer found a synthetic PNG 16-bit/palette crash and isolated `bit_depth=16 & color_type=3`.

- **`core/combinadic.py`**: Lehmer/combinadic rank and unrank for m-permutations and m-combinations (lexicographic, big-int safe), plus uniform without-replacement sampling in O(count) memory. Not wired into a mutator: m>2 swap yield is still unmeasured.
- **Covering-array constraints and t=3**: `covering_array.generate/verify_coverage/missing_tuples/required_tuple_count` take `forbidden=[{param: value}, ...]` (Moser-Tardos repair; one-parameter bans shrink domains); t=3 covered by tests.
- **`covering_array_gzip`** operator (`core/mutations/covering_array_gzip.py`): pairwise sweep of the RFC 1952 header fields (CM, FLG, MTIME, XFL, OS); available only on gzip magic. Selection share on a real gzip target unmeasured.
- **`tools/lib/factorial_design.py`**: Plackett-Burman screening designs (`design_matrix`, `fold_over`, `main_effects`, `screen`) for ranking hyperparameters in 12-16 runs instead of a grid.
- **`core/group_testing.py`** + `tools/bench_group_testing.py`: non-adaptive pooled which-items-matter inference (COMP/DD, exact for any d) and a binary-splitting baseline. Not wired into `tmin`/colorizer (see handover section 6.5 result).
- **`--second-order-blend W`** (default 0 = off): second-order operator chain `P(next | prev2, prev)` in `MonteCarloScheduler` (`core/op_chain2.py`, sparse, capped at 4096 contexts), backing off to `--pairwise-blend` on unseen contexts. Synthetic A/B only; real-target A/B not run.

### Changed

- **Deterministic stage: arith ±1..35 and 16/32-bit passes** (`services/operators.py::_deterministic_mutation_stream`): arith deltas were powers of two (half repeated a bitflip, none hit small length offsets). Now AFL's schedule minus flip 2/4/16/32: arith 8/16/32 ±1..`ARITH_MAX` and interesting 8/16/32, LE and BE, with AFL dedup — the stream holds no duplicate or seed copy. Upper-bound cost 485 mutants/byte (was 33), so a capped stage covers fewer bytes per run.

- **`--consolidated` → `--consolidated-v1`** (`op_consolidated.py` → `op_consolidated_v1.py`, `ConsolidatedScheduler` → `ConsolidatedV1Scheduler`, strategy `consolidated` → `consolidated_v1`, stats keys `consolidated_v1_*`). `--consolidated` kept as an alias. Elo ratings saved under `consolidated` do not carry over.

### Documentation

- **Combinatorics gap analysis** (`docs/handover/handover_combinatorics_permutations_2026-09-02.md` section 6): ranked remaining candidates (FIC-style isolation over covering-array rows, covering-array extensions, orthogonal designs, rank/unrank, group testing) and what to skip. Docs only.

### Fixed

- **`BloomFilter` scaling and API consistency** (`core/bloom.py`):
  - Tight `error_rate` values are now honoured. When `k * log2(m)` exceeds the 256-bit digest,
    positions come from Kirsch-Mitzenmacher double hashing (`h1 + i*h2`, `h2` odd) instead of
    clamping *k* (which made 1M@1e-6 realise 1.29e-6 and 2M@1e-9 realise 7.7e-9). Configs that fit
    in one digest keep their exact historic slice positions. The exec-dedup filter (500k@1e-3) now
    uses the full k=12 (was clamped to 11). `digest_limited` now means "double hashing in use".
  - Constructor validates arguments: `capacity >= 1`, `0 < error_rate < 1` (was a bare
    `math domain error`, or silent nonsense for `error_rate >= 1` / `capacity <= 0`).
  - `add_bytes` now deduplicates without `init_fuzzy()` (it used to return False forever), hashes
    the raw bytes so it shares a keyspace with `update_bytes` (was the hex string), skips
    different-length keys before calling `hamming_distance` instead of catching `ValueError`
    (mixed-length fuzzy add 573 us -> 30 us), auto-creates the recent-keys buffer, and `clear()`
    drops it.
  - `load_factor` uses `int.bit_count()` on the whole array: 36 ms -> 1.4 ms at 8M bits.
  - New introspection: `expected_fpr`, `over_capacity`, `memory_bytes`; docstring documents the
    power-of-two rounding trade-off (up to 2x memory, realised rate never worse than requested)
    and that overfilling degrades to "always maybe" without false negatives. The corpus filter
    intentionally gets no reset: `adapters/filesystem.py` treats a bloom miss as "new", so
    forgetting keys would cause duplicate saves.
  - Tests: `tests/test_bloom_scaling.py`; `tests/test_bloom_exec_dedup.py` digest-budget tests
    updated for double hashing.

- **`CuckooFilter` load scaling** (`core/cuckoo.py`):
  - Bucket count is now sized against `MAX_LOAD = 0.90` (was `capacity // bucket_size` rounded up
    to a power of two, which put capacities like 500_000 and every power of two at 0.95-1.00 load,
    past the ~0.96 kick-failure cliff; 18.7k failed adds at capacity 1_048_576).
  - A failed `add()` is now transactional: the kick chain is journalled and rolled back, so a
    rejected insert no longer drops an already-stored fingerprint (previously ~17.7k false
    negatives in that run). New `n_failed` counter; `update_bytes(reset_on_full=True)` starts a new
    generation when an add fails instead of silently not tracking the key.
  - Kicking uses a private `random.Random` (`rng_seed`), no longer consuming the global stream
    that `--seed` makes reproducible.
  - Default `fingerprint_size` 8 -> 16: realised FPR ~3% -> ~1e-4 (the bloom exec backend is 1e-3),
    making the `--exec-dedup-backend cuckoo` help text true. `expected_fpr` property added.
    Trade-off: exec-dedup at capacity 500_000 uses ~45 MB (was ~13 MB) because of the larger table.
  - Fingerprint mapping no longer folds 0 onto 1 (P(fp=1) was 2x uniform).
  - One item digest per operation instead of two, plus a memoised alt-index hash: `contains`
    3.7 us -> 1.6 us, `add` 4-6 us -> ~3 us.
  - Tests: `tests/test_cuckoo_load_scaling.py`.

- **`entropy_kl` seed scores tracked seed length.** Plug-in KL is biased upward by ~(K-1)/(2n)
  nats, so short seeds out-scored long seeds that truly diverged (Spearman(score, length) = -0.99
  on single-distribution corpora; seeds <= 64 B drew 2.25x their share of selection weight).
  `scores()` is now the KL's excess over the exact length-dependent null in null standard
  deviations (capped at 8); `raw_scores()` keeps the old number. Cost: the pick after a corpus
  admission is ~13 ms instead of ~1 ms.

### Added

- **Position-arena `effector`, `token`, `chunk`, `changed`, `rare_mask` arms** (`--pos-effector`,
  `--pos-token`, `--pos-chunk`, `--pos-changed`, `--pos-rare-mask`; implied by `--position-arena` and
  `--hail-mary`). `effector`: byteflip-LIVE bytes, kept after the deterministic queue drains.
  `token`: dictionary-token occurrences. `chunk`: container chunk headers from the format parsers.
  `changed`: pooled group testing on trace movement. `rare_mask`: FairFuzz branch mask on the seed's
  rarest edge. New helpers: `OperatorEngine.effector_live`, `wfc_chunks.detect_chunks`,
  `ShmCoverage.has_edge`. Unmeasured.
- **Position-arena `boundary` arm** (`--pos-boundary`, implied by `--position-arena` and
  `--hail-mary`): a content-only prior that proposes field boundaries. Each gap between two bytes
  scores +1 for a byte-class change, +1 for the byte after a delimiter (not inside a delimiter
  run), up to +1 for the entropy step between the 16-byte windows either side, +0.5 at the edge of
  a run of >= 4 equal bytes and +0.25 at 4-byte alignment; the 256 best per seed are kept and one is
  drawn in proportion to its score with a jitter of -1/0/+1. Sites past a shrunken buffer are
  dropped, not clamped. Only the first 64 KiB is scored (the unscored tail of a longer seed gets its
  share of uniform draws); 15% of draws decline to uniform. Stateless (`record` is a no-op), the
  per-seed table is an LRU of 256 derived entries and is not persisted. Off-policy extra, so
  `--pos-arena-arms uniform,boundary` and the new `pos-arena-boundary` bench arm isolate it.
  Reuses `_bytecls.py`. `fractal_voronoi.py`'s `_is_boundary` is a hash-driven partition of the
  index space, not a content detector, so nothing was reusable from it. Unmeasured (arena-only).

- **`--pos-arena-arms ARM[,ARM...]`**: restrict `--position-arena` to a named subset of arms
  (uniform is always kept; hyphens accepted; unknown names are a usage error). A dropped arm is
  neither proposed from nor credited off-policy, plays no Elo matches, and is left out of the
  startup banner and the pos-canary inspection. Makes a single arm A/B-able inside the arena.
  New bench arms `pos-arena-uniform` (control), `pos-arena-<arm>` for burn-front, kl-ducb,
  fractal, context, levy, round-robin and fibonacci, and `pos-arena-all`
  (`tools/lib/bench_paired.py`). Unmeasured.

- **Position-arena `levy` arm** (`--pos-levy`, implied by `--position-arena` and `--hail-mary`):
  keeps one anchor per seed (the offset of its last gain) and proposes the anchor plus a
  heavy-tailed Lomax step (`floor(1/u - 1)`, so half the draws hit the anchor byte, one in six a
  byte away, `P(step >= k) = 1/(k+1)`), reflected at the buffer edges; 5% uniform sparks. A run of
  32 consecutive misses on the seed drops the anchor and the arm declines until the next gain.
  Off-policy credited every round, LRU-bounded (256 seeds), persisted for `--resume`. Unmeasured
  (arena-only).

- **Position-arena `context` arm** (`--pos-context`, implied by `--position-arena` and
  `--hail-mary`): learns which byte contexts (byte class, previous class, position decile; 560
  cells) precede coverage gains, pooled across all seeds so a new seed starts warm. Draws 16
  uniform offsets and picks one by clamped rate ratio (`[0.25, 4]`); declines below 200 credited
  offsets. Off-policy credited every round, persisted for `--resume`. Shares `_bytecls.py` with the
  future `boundary` arm. Unmeasured (arena-only).

- **Position-arena `lineage` arm** (`--pos-lineage`, implied by `--position-arena` and
  `--hail-mary`): proposes offsets near the mutation sites that produced a seed
  (`seed_meta[...]["parent_sites"]`, delocalised operators' sites dropped when `parent_ops` is
  aligned) with two-sided geometric jitter (mean 8 bytes), reflected at the buffer edges, as a
  cold-start prior for the seed's descendants. Passive, no persisted state; joins the pool only
  while `--lineage` is on, the only mode that records the sites. Unmeasured (arena-only).

- **Position-arena `cmplog` arm** (`--pos-cmplog`, implied by `--position-arena` and
  `--hail-mary`): proposes redqueen offsets and Weizz-flagged spans (length, magic, checksum,
  input-to-state) as landing sites for every operator. Passive, no persisted state; joins the
  pool only while cmplog is live. Unmeasured (arena-only, like `kl_ducb`/`fractal`).

- **Bach spectral KL estimator** (`core/spectral_kl.py`, `EntropyKLSeedStrategy.spectral_scores()`,
  `miller_madow_scores()`): closed-form lower bound on KL via one generalized eigendecomposition,
  measured as a baseline next to the calibrated `entropy_kl` score (AUC 0.83 vs 0.97; not wired
  into selection). `calibrated_spectral_scores()` applies the same null calibration to it
  (Spearman with seed length -0.99 -> -0.04; AUC 0.83 -> 0.94, still below the calibrated
  plug-in's 0.97).
- **Spectral score row cache** (`EntropyKLSeedStrategy.spectral_scores()`): rows are cached per pool
  version, so a repeat call costs ~0.5 ms instead of ~230 ms at 2000 seeds; a corpus admission or
  eviction still recomputes the batch (~250 ms, dominated by the pool-dependent `eigh`).
- **`entropy_zscore` opt-in length calibration** (`EntropyZScoreSeedStrategy(calibrate_length=True)`,
  `EntropyLengthNull`): adds back the exact small-sample entropy bias and standardises by the null
  spread; seeds <= 64 B go from 0.62x to 0.98x of their fair weight share on synthetic
  single-distribution corpora. Default off, no CLI flag.


- **SanitizerCoverage modes**: shim callbacks for `inline-8bit-counters`, `inline-bool-flag`,
  `pc-table`, `trace-loads`/`trace-stores` (previously failed to link). Counters/flags fold into
  the edge map; loads/stores of globals become data-flow features. ELF helpers recognise
  `__sancov_bools`. `tools/build_targets.sh --sancov=MODES` selects them (validated;
  `verify_sancov` accepts counter/bool sections).
- **Minimax Phases 2-5 wired** (were dead code): `fuzz --op-minimax`, `--wall-order`,
  `--minimax-select`; `minimize --minimax-robust`; `bench_paired.py analyse --risk-matrix`
  prints the minimax-robust arm. Bench arms `elo-op-minimax`, `wall-order`, `minimax-select`.
  `--hail-mary` enables the three fuzz flags.
  Fixed on the way: `select_op_minimax` root searched the first 4 ops in list order and
  scored by posterior mean (no exploration); `_minimax_pick` scored seed size, not unique
  loss; `minimax_robust_pruning` compared seed count to edge count; the risk matrix took
  the worst seed (≈1.0 for every arm), read missing data as zero regret, and dropped the
  baseline arm.

- **Docker `ffmpeg` stage**: `docker build --target ffmpeg` bakes vendored FFmpeg and
  `ffmpeg_read_nosan.so`; `docker run -v DIR:/out` runs a resumable campaign. Image now
  installs compiler-rt (`libclang-rt-dev`), without which every sanitizer link failed.

- **`CorpusFlux.z_score()` / `is_significant_drift()`** (`core/analyzers/analyzer_corpus_flux.py`):
  standardizes the existing net/gross flux counts as `Z_n = net / sqrt(gross)`
  (CLT normal approximation, treating each admission/eviction as an i.i.d.
  +-1 step under a null of undirected churn), so a nonzero `turnover` can be
  flagged as a real directional trend versus noise from a handful of events.
  Surfaced in the `flux:` stats line as `z=<value>[*]` (`*` marks
  `|z| >= 1.96`).

- **`--no-calibration`**: skips `_calibrate_seed_baselines` (saves its startup time and RSS).

- **Heap trim per status line** (`adapters/libc_mem.py`): `print_stats()` runs glibc
  `malloc_trim(0)` each status line and shows the RSS released (`| trim: 512KB`).

- **Position schedulers and their Elo arena** (`core/schedulers/pos_base.py`,
  `pos_burn_front.py`, `services/position_arena.py`): position selection is
  now a formal third scheduling axis with a `propose`/`record` contract.
  `--position-arena` (needs `--elo`) puts uniform, sensitivity, TE, phase, MI,
  crash-MI, region and burn-front in one Elo tournament under `pos_` keys;
  `--burn-front` adds a Gaussian-conduction/fuel-burn proposer over byte
  offsets. Both off by default; `--position-arena` implies `--burn-front`, and
  `--hail-mary` enables both.
  `Arena`/`strategy_arena()` now partition the Elo keyspace (stats, report,
  canary floors); `pos_` keys previously would have fallen into the operator
  arena.

- **Crash-cluster chaining diagnostic** (`core/crash_metadata.py::detect_chained_clusters`):
  read-only check over `cluster_crashes`'s output that flags clusters
  single-linkage likely chained together (A~B and B~C both clear the
  clustering threshold, but A~C does not clear a looser core threshold).
  Does not alter clustering. Wired into `--report`'s Crash Signatures
  section, which now annotates flagged clusters `[CHAINED -- min pairwise
  similarity X.XX, verify this is one bug]`. Tunable via
  `configure_crash_cluster(core_threshold=...)` /
  `FUZZER_CRASH_CLUSTER_CORE_THRESHOLD` (default 0.5) and
  `FUZZER_CRASH_CLUSTER_DIAG_MAX_SIZE` (default 500, caps the O(k^2)
  all-pairs scan per cluster).
- **`--continuum-reward`** (`core/analyzers/analyzer_navier_stokes.py::frontier_weight`,
  `services/fuzzer.py::Fuzzer._continuum_reward_shape`): scales every scheduler's
  shared operator reward by the mean continuum pressure of the edges co-hit
  alongside a discovery — a round that opens unvisited territory pays close to
  1, one that fills a gap between well-owned edges pays close to 0. `--continuum-
  reward-floor` clamps the factor from below, same contract as
  `--shaped-reward-floor`. Off by default, out of `--hail-mary` (rescales the
  reward every strategy reads, same structural exclusion as `--shaped-reward`);
  independent of `--continuum` (which re-ranks invasion operators, not the
  reward) and of `--shaped-reward`/`--op-credit` (which price duplication, not
  location) — the two factors compose multiplicatively. Registered as bench arm
  `elo-continuum-reward` against baseline `elo`; paired A/B on fuzzgoat (12
  seeds, 2k execs) came back 4W/8L, median Δ -3.5 edges, McNemar p=0.388 — not
  resolved at this cell count, point estimate a loss. See
  `docs/learnings/2026-09-22-continuum-reward-ab-result.md`.

- **Math-port plan P1–P4** (`docs/handover/handover_math_port_plan_2026-09-21.md`):
  - **`montgomery_mutate`** (`core/mutations/montgomery.py`) — structure-aware
    REDC/Barrett constant injector gated on the secp256k1 field prime or
    curve order appearing as a BE 32-byte literal; registered in the
    `format` band with sniffer + `_op_montgomery_mutate` handler.
  - **Shared KS p-value helpers** (`core/ks_pvalue.py`) — two-sample
    asymptotic, one-sample asymptotic (Stephens), and Marsaglia exact CDF;
    `edge_tracker` / `randomness` re-export without behaviour change.

- **`wfc_reorder_learned` rolled out to ogg, flv, nal, asf, mpegts and zip** (`core/wfc_chunks.py`), joining isobmff/riff/webp/gif. Fixed three defects in the shared reorder core found on the way: `violate` mode indexed one cell past the grid when nothing was pinned last (swallowed by `mutate`, so a fraction of violate calls silently declined) and overwrote the pinned last cell when something was; chunks a collapse under-placed were appended after the pinned-last chunk; and strict mode's "never emits an unobserved adjacency" was only ever tested on a re-parsed output, which re-segments positional formats such as GIF.

- **Crash field map and baseline in crash sidecars** (`core/field_map.py`, `services/crash_explain.py`). A novel crash's `.txt`/`.json` now name the fields of the crashing input (PNG, gzip, ZIP, RIFF) and mark which changed against the parent seed it was mutated from, falling back to a hash-rehydrated parent, then the nearest corpus seed. Static only: nothing executes the target, and `changed` does not claim causation.

- **Wired the RO/RD temporal-orientation analyzers** (`core/occupation.py`,
  `core/ro_rd.py`, `core/causal_sector.py`) into `analyzer_registry`, record
  hooks, CLI, and reporting, per `docs/handover/handover_RoRd.md` Phases
  A/B/C1/E. Both new analyzers default off and never construct outside
  `REGISTRY.wire_all`, matching the `transfer_entropy` pattern.

  - **`--occupation`** (`Fuzzer(occupation=...)`) — after each new-coverage
    exec, folds `hit_counts` into `OccupationMeasure` via
    `LongitudinalRarity.observe`, exposing support size / entropy / history
    count in the live stats supplementary line.
  - **`--causal-sector`** (`Fuzzer(causal_sector=...)`) — soft-requires
    `--transfer-entropy` (registered directly after it in
    `analyzer_registry.py` so `f._te` exists first). Every 100 samples,
    `StatsReporter.update_causal_sector()` feeds `CausalSectorGraph.observe_flow`
    via a new adapter, `services/te_position.py::edge_sets_to_flow`, which
    reimplements the top-k/binary-series pairwise-TE logic directly on the
    edge-id `set`s already stored in `Fuzzer._te_edge_history` (avoids
    materializing dense bitmaps for `TransferEntropy.edge_to_edge_flow`).
    Reports sector node/edge counts and stability.
  - **RO/RD classification (metadata only)** — `classify_operator_name`
    tags each attributed operator edge as `ro`/`rd`/`neutral` into
    `Fuzzer._ro_rd_edge_counts`, reported alongside occupation/causal-sector
    stats. Classification never influences scheduling or selection (Phase
    C2 lineage-reverse wiring and C3 soft RO gating remain deferred, as does
    the optional Phase D seed_quality scoring — see the handover doc's
    updated acceptance checklist and decision ledger).
  - Both flags added to `_HAIL_MARY_FLAGS` in `cli/commands.py` and to the
    `--features` "Analysis" group summary.

- **Ported `CuckooFilter`, `F0Estimator` (CVM), and `Feistel` from AIscripts** in
  `src/fuzzer_tool/core/{cuckoo,cvm,feistel}.py`. Three statistical primitives
  cleaned and dropped into the core layer:

  - **`CuckooFilter`** (`core/cuckoo.py`) — a classic cuckoo hash filter with
    `add`/`contains`/`query`/`remove`/`update`/`clear`/`load_factor`, plus a
    `update_bytes(key, reset_on_full=False)` hot path mirroring
    `BloomFilter.update_bytes` so it can stand in for the bloom as an
    exec-dedup backend.  Supports deletions, which the bloom does not.
  - **`F0Estimator`** (`core/cvm.py`) — a streaming distinct-elements
    estimator (arXiv:2301.10191) with `update()`/`estimate()`/`clear()`/
    `size`, an (eps, delta) guarantee, and a `None` (perp) return when the
    down-sample cannot recover.  Feeds the edge-ID stream in
    `EdgeTracker.record_edges()` when `enable_f0=True`.
  - **`feistel_scramble`** (`core/feistel.py`) — a keyed bijective block
    transform, registered as a regularity-band mutation operator.

  `tests/test_cuckoo.py`: 23 tests covering add/query/remove/update/clear,
  load factor, parameter validation, cuckoo kicking, alt-index consistency,
  `n_added`, and the `update_bytes` generational reset.
  `tests/test_cvm.py`: 17 tests covering construction validation, the
  update/estimate contract, `clear`, down-sampling (including the perp
  return), and an (eps, delta) property test over 200 seeded runs.

- **`--exec-dedup-backend {bloom,cuckoo}`** in `src/fuzzer_tool/cli/commands.py`
  and `exec_dedup_backend` on `Fuzzer` (`services/fuzzer.py`).  The exec-dedup
  gate (`_dedup_mutate`) now picks its structure at construction time rather
  than hard-coding `BloomFilter`; the default is unchanged (`"bloom"`), so
  every existing campaign is byte-for-byte identical.  The Cuckoo backend is
  opt-in and exposes the same `update_bytes(key, reset_on_full=True)` contract
  the method already drives.  `tests/test_bloom_exec_dedup.py` gains
  `TestCuckooDedupBackend` (6 tests) driving the same `_StubFuzzer` against
  both structures.

- **Streaming F0 cardinality signal in `EdgeTracker`** (`core/edge_tracker.py`).
  Opt-in via `enable_f0=True`; when on, `record_edges()` feeds the edge-ID
  stream into an `F0Estimator` and three new methods are exposed:
  `estimate_distinct_edges_f0()`, `f0_calibration()` (estimate vs exact
  count, for tuning eps/delta on a real corpus), and `f0_plateau()` — a
  saturation flag that is a distinct subcritical signal from the stall
  window.  The exact count (`get_cumulative_edge_count()`) is always
  authoritative; the F0 estimate is a memory-bounded approximate sibling.
  `tests/test_edge_tracker.py` gains `TestF0Cardinality` (6 tests).

- **F0 plateau threaded into `CoverageRegime._classify`**
  (`core/coverage_regime.py`).  `observe()` accepts an optional
  `f0_plateau` flag; when True and no earlier branch fired, the regime is
  `SUBCRITICAL` with reason `"cardinality plateau (F0 estimate saturated)"`.
  The branch sits after the discovery-rate collapse check and does not
  depend on the stall window.  `None` (no estimator wired) is inert, so
  callers without an F0 tracker are unaffected.  `services/fuzzer.py` feeds
  `self._edge_tracker.f0_plateau()` into `observe()`; `tests/test_coverage_regime.py`
  gains 3 plateau tests.

- **F0-derived weight fed into `BayesianSeedQuality.record_outcome`**
  (`services/fuzzer.py`).  The `weight` parameter is the documented
  injection point for discovery rarity; when the F0 estimator is wired the
  weight becomes `observed / f0`, bounded in (0, 1] — near 1 when the
  estimate is saturated (each remaining discovery is rare), near 0 when
  lots is undiscovered (discoveries are common).  It never inflates a
  posterior beyond the default; it only ever re-weights.  `tests/test_seed_quality.py`
  gains `TestF0DerivedWeight` (3 tests).

### Fixed

- **`fractal_voronoi` ran without sub-operators and ignored `rng`**: the registered
  instance had no `cell_ops`, so it always XOR-fell-back, and output was a pure
  function of the input, so a re-picked seed replayed one mutant. Registration now
  passes six single-byte bijections (`DEFAULT_CELL_OPS`); each call draws one salt
  from `rng` and mixes it into every choice. `rng=None` keeps the legacy output.

- **`--secretary` no longer reweights seeds or triggers minimization**: its stop rule
  fires on a `1/t` discovery-rate envelope (rank capped at 20 vs threshold `n/e`), so a
  seed was cut 100x after 20 observations and never recovered; `--elo all` enabled it.
  Now display-only, as `SecretaryStopping` already documented (P2-4).
- **`edge_diagnostic.py op-caches`**: reads the fractal-voronoi / perlin caches from the
  registered mutator instances; the module globals it read were moved in `6b3b7c3b`.

- **`--resume` refused on Katz-capable targets**: `load_state` ran before the Katz
  channel was built, so the `node_channel` contract always mismatched.
- **`vendor_ffmpeg.sh` masked configure/make failures** (`pipefail`).

- **Unbounded fuzz-loop growth** (`core/analyzers/analyzer_prng_state_learner.py`, `core/cmplog.py`):
  `_run_history` capped inputs but not their site records (~1.9 MB/input on ffmpeg); now a total budget,
  current input never evicted. `hash_candidates` outlived pair eviction; now pruned with its pair. ffmpeg
  non-ASAN, 3.5k execs: RSS 2066 -> 1836 MB and flat, peak 2562 -> 2096 MB, exec speed unchanged.

- **Position-canary misreports** (`services/fuzzer.py`): the uniform-floor check no longer flags
  `pos_canary` (built to lose) as needing inspection on every `--position-arena` run, and the banner lists
  "canary" only when the arena fields it (`--position-arena` with `--elo`).

- **Katz horizon recompute held Python lists for every ICFG node** (`core/horizon.py`, `core/schedulers/seed_katz.py`):
  adjacency, shortcut walk, SCC and DAG depth run on CSR arrays; Tarjan only on nodes that can be on a cycle;
  `u_nodes`/`u_icfg_index` packed, `node_index` built on first use. One recompute on ffmpeg's ICFG:
  2.67M nodes 34.8 s / 1923 MB -> 12.4 s / 300 MB; 833k nodes 13.5 s / 550 MB -> 6.8 s / 68 MB. Output identical.

- **Katz channel kept ~1.9 GB of build-only state** (`core/icfg.py`, `core/cfg.py`, `services/katz_channel.py`):
  CFGs and `TargetDistance` are dropped after use, `node_addrs` is `array('Q')` + bisect (no `node_index` dict),
  `BasicBlock` is slotted with a shared empty `callees`, and ICFG assembly uses packed arrays. ffmpeg, retained
  2112 -> 217 MB, build peak 2525 -> 1613 MB, output byte-identical. `ASAN_OPTIONS` default gains
  `allocator_release_to_os_interval_ms=1000` (ASAN never released freed memory; no exec/s cost measured).

- **Seed calibration duplicated every seed's edge set** (`services/fuzzer.py`): `_calibrate_seed_baselines`
  passed `target_name`, filling `seed_target_edges`, which only multi-target mode reads and calibration
  skips. ffmpeg_read_asan.so, 480 seeds: calibration RSS growth 465 -> 336 MB.

- **Exec-time anomaly threshold leaked and slowed every exec** (`core/analyzers/analyzer_exec_time_anomaly.py`):
  all exec times were kept (32 B/exec) and median-sorted each exec (1.1 ms at 10k execs, 235 ms at 1M).
  Now an exact median over the last 4096. fuzzgoat, 26k execs: 172 s -> 125 s.

- **SHM segments outlived their `Fuzzer`** (`adapters/shm.py`): `atexit.register(self.cleanup)`
  pinned `ShmCoverage`, `DistanceTableShm` and `NodeBitmapShm`; each dropped `Fuzzer` kept 3
  segments attached until exit. Exit hooks now hold weak refs; the latter two gain `__del__`.
- **Minimax estimator and algorithm for fuzzer enhancement**
  (`src/fuzzer_tool/core/schedulers/mcts.py`,
  `src/fuzzer_tool/core/schedulers/monte_carlo.py`,
  `src/fuzzer_tool/core/cond_stmt.py`,
  `src/fuzzer_tool/core/smt_solver.py`,
  `src/fuzzer_tool/core/elo.py`,
  `src/fuzzer_tool/core/rate_distortion.py`,
  `src/fuzzer_tool/services/corpus_manager.py`,
  `src/fuzzer_tool/services/seed_picker.py`,
  `tools/bench_paired.py`).
  Five-phase implementation of minimax, alpha-beta pruning, and adversarial
  search applied to fuzzer scheduling:
  - **Phase 1 (P0):** `AlphaBetaMCTSSeedScheduler` — alpha-beta minimax
    over the lineage tree, replacing UCT descent for seed selection.
  - **Phase 2 (P1):** Minimax-robust scheduler selection — `EloTracker`
    extended with a risk matrix and `select_minimax_scheduler()` for
    worst-case-optimal scheduler choice. `bench_paired.py --risk-matrix`
    outputs the risk matrix from benchmark sweeps.
  - **Phase 3 (P1):** Comparison-wall solving as a minimax game —
    `solve_comparison_wall_minimax()` in `cond_stmt.py` with alpha-beta
    pruning; `Z3Solver.solve_comparison_wall()` in `smt_solver.py`.
  - **Phase 4 (P2):** Adversarial operator sequencing —
    `MonteCarloScheduler.select_op_minimax()` models operator selection as
    a two-player game with multi-ply lookahead.
  - **Phase 5 (P2):** Minimax-robust corpus admission —
    `RateDistortionCorpus.minimax_robust_corpus_admission()` and
    `minimax_robust_pruning()` minimize maximum coverage loss;
    `CorpusManager.minimax_robust_admission()` integration.

  See `docs/handover/handover_minimax_implementation_2026-09-01.md` for
  full documentation.

- **`invasion_select()` in `src/fuzzer_tool/services/seed_picker.py`**
  (percolation handover Module 4). Greedy, no-backtracking operator
  selection: picks the operator with the lowest resistance (inverse
  observed success rate, from any scheduler's `bandit_stats()`). Untried
  operators get resistance 0 (optimistic exploration, matching the UCB
  schedulers' posture) so they're tried before being written off. Returns
  `None` — signalling "stuck, reseed or switch strategy" — when every
  operator's resistance is at or above `INVASION_STUCK_THRESHOLD`, or when
  an explicit (non-`None`) `frontier_edges` collection is empty.

  `tests/test_invasion_select.py`: 11 tests covering selection,
  exploration-over-low-success-rate, deterministic tie-break, the stuck
  threshold, and adversarial all-zero-success cases.

  Wired into the Elo meta-scheduler as of the entry below.
- **Invasion percolation wired into the Elo meta-scheduler
  (`services/fuzzer.py`, `services/operators.py`, `cli/commands.py`).**
  `--invasion` (also enabled by `--elo all` and `--hail-mary`) adds
  `"invasion"` as a candidate on the Elo ballot, alongside `bandit`, `mopt`,
  `exp3`, etc. Requires `--mc-bandit` (invasion reads `f.mc.bandit_stats()`
  as its resistance signal rather than tracking its own arm state) — a
  warning is logged if enabled without it, and `select_op()`'s dispatch
  branch filters `bandit_stats()` down to the current call's candidate
  operators before selecting, since the unfiltered dict covers every
  registered arm and could otherwise return an operator outside the
  candidate set. Falls back to the random terminal when `invasion_select`
  reports "stuck". Deliberately has no fallback-chain branch outside Elo —
  the same `f.mc`/`mc_bandit` condition is already covered earlier in that
  chain by the plain `bandit` branch, so a second branch there would be
  dead code (the exact failure mode `test_regression_cmaes_elo_ballot.py`
  documents for cmaes). Also added to the opponent ballot
  `record_strategy_match` iterates over (`all_strategies`) — omitting it
  there is the same cmaes bug in a different spot, where a strategy's
  Elo rating only moves on the exec it wins and never on the execs it
  loses.

  `tests/test_invasion_elo_integration.py`: 7 tests — ballot inclusion,
  ballot absence without `mc_bandit`, candidate-set filtering, stuck
  fallback, no-Elo unreachability, and opponent-ballot inclusion.
- **`src/fuzzer_tool/core/target_difficulty.py` (percolation handover Module
  3, revised per Diskin–Easo–Radhakrishnan–Sudakov–Tassion, "Supercritical
  sharpness of percolation," arXiv:2603.03257).** Static pre-fuzz difficulty
  estimation via the isoperimetric function Φ(n) = min{|∂S| : S ⊂ V,
  n ≤ |S| < ∞} over a target's CFG, approximated by a greedy boundary-growth
  heuristic with multiple restarts (`estimate_isoperimetric_profile`).
  Accepts either an `InterproceduralCFG` (`core/icfg.py`) or a plain
  adjacency dict for testing. `estimate_percolation_threshold` kept as a
  cheap degree-only fallback for targets where a full profile isn't
  affordable. `estimate_growth_curve` Euler-integrates the reachable-cluster
  growth curve from a Φ profile (Theorem 3 of the same paper).

  Each restart grows once to the largest requested size and keeps a
  suffix-min of the boundary sequence, since a witness set of size k is also
  a valid witness for any n ≤ k — this is what keeps the returned profile
  non-decreasing in n (an independent per-size probe can otherwise report a
  smaller boundary at a larger n than at a smaller one, since the two sizes
  can take different greedy paths from the same seed).

  `tests/test_target_difficulty.py`: 17 tests including chokepoint,
  monotonicity, restart-improvement, and adversarial (non-positive sizes,
  negative step count) cases.

### Fixed
- **`os.environ` writes were never restored, leaking state across runs in
  the same process.** Bug report 2026-08-21 HIGH #10. `run()` only ever
  undid the cmplog shim's own `LD_PRELOAD` edit (`self._cmplog.restore_env()`)
  — `__AFL_DIST_SHM_ID`, `__AFL_SHM_ID`, `AFL_MAP_SIZE`, the ASAN `LD_PRELOAD`
  injection, and `UBSAN_OPTIONS` were all written directly into the process
  environment and never put back. That leaked into whatever ran next in the
  same process: the next target in a multi-target session, a caller
  embedding `Fuzzer` as a library, or the next test in a pytest run. Added a
  process-wide snapshot taken once in `__init__` (before any of the above
  can mutate it) and a restore that runs both at the natural end of `run()`
  and via `atexit` as a safety net for crashes/SIGTERM/SIGINT. Regression
  tests in `tests/test_regression_environ_restore.py`.
- **Findings #12 and #13 (2026-08-21 bug report) fixed and pruned from
  `docs/bugreport_2026-08-21_merged.md`** — multi-target mode reading edges
  from the wrong SHM segment, and `--inprocess-direct` freezing permanently
  on the first wild-pointer crash. See `a927e62` and `ffc4437` /
  `ca9fdd0`.
- **`RandPool.sample()` raised `TypeError` on a `range` population.** The
  dispatch was `isinstance(population, list | tuple | bytes)`, so a `range`
  missed the sequence branch, fell through to the int branch, and died on
  `k > population` comparing an int to a range. `rng.sample(range(n), k)` is
  the idiomatic `random.sample` call and roughly a dozen structure-aware
  mutators use it (`asf.py`, `riff.py`, `mp3.py`, `adts.py`, …). It only
  fires when the pooled RNG is active rather than stdlib `random`, so it
  presented as a seed-dependent flake in `test_mutate_includes_splice`
  rather than as what it is: an exception raised mid-campaign from an
  ordinary call. `bytearray` was missing from the same tuple and is added
  with it.
- **`DEFAULT_CC` swallowed the gcc-fallback warning on any box without clang,
  breaking every vendored-library compile.** `_pick_cc()` emitted its warning
  through `warn()`, which writes to *stdout*, and the whole of its stdout is
  captured by `DEFAULT_CC="$(_pick_cc)"` — so `DEFAULT_CC` became the warning
  text followed by `gcc`, which is not a command. Every helper that compiles
  with `$cc` redirects stderr to `/dev/null`, so the only visible symptom was a
  run of "objects failed" warnings and targets silently missing from the
  output. Redirected to stderr, with a regression test in
  `tests/test_regression_build_flags.py`.
- **`run_target_fast` had no timeout, deadlocked on chatty targets, and leaked
  the child it failed on.** Bug report 2026-08-21 CRITICAL #3. This is the
  *default* spawn-fallback path — `run_target` picks it whenever the run is
  neither `file_mode` nor cmplog — and it was the only backend not honouring
  `f.timeout`. Three defects, each independently fatal:

  | defect | old behaviour | measured |
  |---|---|---|
  | `os.waitpid(pid, 0)`, no deadline | one looping input hangs the campaign | SIGKILLed at 15s, never returned |
  | stderr read after the reap | >64 KiB fills the pipe; child blocks in `write()`, parent in `waitpid()` | SIGKILLed at 15s, never returned |
  | `except: return -2, str(e), 0` | crash attribution lost, child never reaped | — |

  After: the looper returns `rc=-1` at the deadline, and the 400 KiB-stderr
  target returns `rc=3` with 65536 B captured in 0.77s.

  The bound is enforced by `poll()` on the stderr pipe, which doubles as the
  liveness wait — one syscall per wakeup and **no threads**. A watchdog thread
  (as `run_target_stdin`/`run_target_file` use) would have been the obvious
  match, but this path exists specifically to create no threads, and this
  process forks elsewhere; see E3 and the note in `tests/conftest.py`. The
  child is also spawned into its own process group (`setpgroup=0`, the
  posix_spawn equivalent of the siblings' `preexec_fn=os.setsid`) so a timeout
  kill reaches grandchildren.

  No throughput cost: 980.3 eps vs 989.2 eps median over 5 interleaved repeats
  of 400 execs, ranges fully overlapping. `timeout=None` keeps the old
  unbounded behaviour for callers that pass nothing; `runner.py` forwards
  `f.timeout`, pinned by a wiring test.

  `_TRACKED_PIDS` and friends moved out of the "Stdin mode" section, since all
  three modes now track their children.

- **cmplog left its shim on the process-global `LD_PRELOAD` forever.** Bug
  report 2026-08-21 E2. `setup_env_for_run()` is called before *every*
  execution and set `_CMPLOG_OUT` and `LD_PRELOAD` with no way to undo them.
  The shim conflicts with the ASAN runtime, so any later subprocess exec ran at
  full speed and found **zero crashes** — a quiet wrong answer, not an error.
  `restore_env()` reverts both from a snapshot taken before the *first*
  mutation; re-snapshotting per call would capture our own preload and make
  restore a no-op, which is the original defect wearing the fix's clothes.
  Wired into `stop()` (previously dead code) and the end of `Fuzzer.run()`.

- **`_clean_env({})` returned the full parent environment.** Bug report
  2026-08-21 E2. `dict(env or os.environ)` is falsy for an empty dict, so a
  caller asking for a scrubbed environment silently got the opposite. `None`
  now means "inherit", any dict means "use exactly this".

- **A bare `pytest` could hang forever.** Bug report 2026-08-21 E1. 300s
  default per test, applied in `pytest_configure` rather than `addopts` so a
  dev env without `pytest-timeout` still runs. The method is `signal`, not
  `thread`: the thread method arms a `threading.Timer` per test and makes the
  pytest process multi-threaded for the entire session, which makes every fork
  in the suite riskier (E3, and `docs/handover/test_shm_hang_2026-08-14.md`).
  Measured — under `thread`, CPython emits its multi-threaded-fork
  DeprecationWarning on a test that is otherwise silent. The Z3 modules, which
  block inside native code where SIGALRM is never delivered, opt into `thread`
  individually via `pytest_collection_modifyitems`.

  Also added: an autouse `_env_isolation` fixture restoring the five
  `os.environ` keys production code mutates, with `--env-leak-strict` to fail
  and name the leaking test rather than quietly repairing it. This is what
  fixes the two ASAN tests that passed in isolation and failed in a full run.

### Changed
- **`beta_quantile` Newton fast-path** (`op_bayes_ucb.py`) — Cornish-Fisher
  seed polished with ≤2 Newton steps against the CF CDF; pure bisection
  remains the fallback (`use_newton=False` or residual miss).
- **`poly_mul` nibble-table path** (`gf2_common.py`) — same signature, fewer
  Python loop iterations on wide limbs (CRC / Rabin / Berlekamp-Massey).
- **Coverage-guided mode is the default; `--no-coverage` opts out.** `fuzz`
  required `-c/--coverage`, and forgetting it failed silently: no SHM bitmap
  was created, so seed scheduling, MI/TE/sensitivity position weighting,
  Elo/bandit operator scheduling, stall detection and corpus admission all ran
  on a constant-zero signal while the run reported healthy throughput.
  Measured, 1500 execs x 2 repeats:

  | target | coverage on | coverage off | cost | corpus on -> off |
  |---|---|---|---|---|
  | `targets/test_target` | 158.4 eps | 160.7 eps | 1.4% | 3.0 -> 1.0 |
  | `targets/png_read` | 128.1 eps | 138.9 eps | 7.8% | 14.5 -> 1.0 |
  | uninstrumented gcc build | 587.0 eps | 592.4 eps | 0.9% | 1.0 -> 1.0 |

  The corpus column is the decision: without coverage it never grows past its
  seeds, on any target, so the default mode of a coverage-guided fuzzer was
  blind mutation. `-c`/`--coverage` are still accepted and are now no-ops, so
  every existing script, README line and doc example keeps working.
  `--no-coverage` is a real mode, not a compatibility shim — crash and timeout
  detection do not need the bitmap.

  Scoped to `fuzz`. `tmin`, `rc` and `minimize` keep `-c` opt-in because it
  means something different there: in `tmin`/`rc` it only sets `AFL_MAP_SIZE`
  in the child env, and in `minimize` it selects a different algorithm — so
  flipping those would silently change what the command *does*, not how fast
  it runs. The four `use_coverage: bool = False` service signatures are also
  unchanged; the CLI is the layer that expresses this policy.

  Rejected on measurement: auto-selecting ptrace for uninstrumented targets.
  73.9 eps against 587 for the SHM path on the same binary — an 8x silent
  slowdown to buy 2 function-entry edges. It stays behind explicit `--no-shm`.

### Added
- **Vendored SQLite fuzz target (`targets/sqlite_read.c`).** Wraps the SQLite
  amalgamation (`tools/vendor_sqlite.sh` → `vendor/sqlite/`) as a `.so` target
  built like lz4_read and secp256k1_read: `sqlite3.c` compiled as its own TU
  without the shim, linked into the wrapper, `$SQLITE_DEFINES` shared by both
  sides so header and library cannot disagree about `SQLITE_*` options.

  The input carries **no mode-selector byte**, unlike `lz4_read.c`. The
  `sqlite_chunk_mutate` sniffer is `len(d) >= 100 and d[:16] == b"SQLite
  format 3\x00"`, so a prefix byte would shift the magic to offset 1, stop the
  sniffer firing, and flat-byte-mutate every database in the corpus while the
  campaign kept reporting edges — a total loss of structure-awareness with no
  symptom. The dispatch uses those same two conditions: magic → database image
  via `sqlite3_deserialize()`, anything else → SQL text.
  `tests/test_regression_sqlite_target.py` pins the agreement.

  DB path: `PRAGMA integrity_check(4)`, `sqlite_master` read, then up to 24
  table scans reading every column value (without a column read the b-tree
  walk stops at the cell boundary and the record decoder never runs).
  In-process safety, since `direct_lite` shares the fuzzer's process:
  `:memory:` only, `DEFENSIVE` + `TRUSTED_SCHEMA=0`, extension loading off,
  authorizer denying ATTACH/DETACH/PRAGMA on the SQL path, progress-handler
  opcode budget, and `sqlite3_hard_heap_limit64`. Verified over 3,500 execs of
  corrupt, truncated and random inputs: no crashes, RSS flat.
- **Startup warns when the target has no edge instrumentation.** With coverage
  on by default the common failure is no longer "forgot `-c`" but "target was
  never built instrumented", and both produce the identical symptom. `run()`
  previously printed `AFL instrumentation: detected` with no negative branch,
  so a bare target ran with `[*] Coverage: AFL SHM bitmap` on screen and found
  nothing. `Fuzzer._warn_uninstrumented()` names the target, the consequence
  and both ways out (rebuild, or `--no-coverage` on purpose); it fires once per
  run, per instance, and never when coverage was explicitly disabled. Also
  wired into the multi-target banner. This generalizes
  `_warn_no_coverage()`, which covered only in-process `.so` targets.
- **`afl_instrumentation_status()` replaces the boolean `_detect_afl` for any
  decision that warns.** It returns `present` / `absent` / `unknown`, because
  `nm` reports no symbols at all for a stripped binary and a boolean cannot
  tell "not instrumented" from "symbol table removed" — a stripped,
  instrumented target is normal, and warning on it is the fastest way to train
  the warning out of people. `_detect_afl` remains as the boolean face for the
  three call sites that only decide whether to print `[AFL]`. Rule verified
  against every target shape in `targets/`: instrumented ELF and `.so` ->
  present, plain gcc build -> absent, stripped (either kind) and missing file
  -> unknown.

### Known issues
- **ptrace coverage reports no ASAN crashes.** Measured on
  `targets/asan_target.c` (gcc `-fsanitize=address`), same seed and settings:
  0 crashes at both n=100 and n=400 under `--no-shm`, against 32 for the SHM
  path and 25 for `--no-coverage`. Pre-existing and unrelated to this change in
  cause — but it was invisible until now, because `_setup_ptrace` is gated on
  `use_coverage`: the `ptrace` case of `test_asan_all_modes` passes no `-c`, so
  `--no-shm` was inert and that parametrization silently duplicated
  `default_subprocess`. Making coverage the default turned the case honest and
  it failed immediately. Marked `xfail(strict=True)` so it reports loudly when
  ptrace crash reporting is fixed; the fix belongs with the ptrace runner, not
  with a CLI default.

### Removed
- **`_add_common_args()` in `cli/commands.py`, which had zero production call
  sites.** AST-checked: only `tests/test_commands.py` called it. It is named
  "arguments shared by fuzz and subcommands" and declared its own
  `-c/--coverage`, making it the obvious place to edit when flipping this
  default — an edit that would have changed nothing reachable by a user while
  turning `test_commands.py` red, a wrong signal twice over. The live flags are
  declared per-subparser. `TestGetDirs` now builds its namespace directly,
  which is what it was actually testing.

- **XOR-checksum recovery solves by GF(2) elimination instead of SAT; the 32-bit
  rung is live.** `xor_map_solver` ran one Z3 `Solver` per output bit, and at a
  32-bit field the per-bit solve blew past `_SOLVER_TIMEOUT_MS` — recorded in
  `docs/edge-coverage-analysis.md` as a deliberate miss on cost, with a real
  32-bit XOR checksum deferred to an offline pass. It was never a cost problem.
  Every constraint the module builds is `XOR_{i in S} w_i == c`, one linear
  equation over `F2`, and the coefficient matrix is the same for every output
  bit — only the right-hand side changes with `j`. One incremental Gauss-Jordan
  elimination over the augmented matrix, rows as Python ints with all output
  bits packed into the RHS word, recovers the whole map in `O(pairs * rank)`.
  Measured on one box, 32x32 over 64 pairs: **151 s** for the SAT path with the
  timeout lifted (4.7 s/bit), **0.5 ms** for elimination; at the default budget
  the old path returned `(None, False)` and recovered nothing. Three
  consequences beyond speed: the full 8/16/32 ladder is now reached on every
  call; recovery no longer needs the optional `smt` extra, so it works on any
  install; and a rejection is now a consistency proof rather than an expired
  timeout — `solve()` has no inconclusive third outcome. `IncrementalXorMapSolver`
  keeps its public API. `timeout_ms` is still accepted and stored but no longer
  gates anything, and `_SOLVER_TIMEOUT_MS` survives only for callers reading it.

### Added
- **`recover_xor_model` requires a full-rank system before accepting a model**
  (`require_determined=True`, new `IncrementalXorMapSolver.rank` /
  `.is_determined`). Reaching the 32-bit rung exposed a defect that cost had been
  hiding: an underdetermined system reproduces **every pair it was fitted on**,
  for any assignment of the free variables, so `verify_xor_model` — which checks
  against those same pairs — cannot reject it. Measured over 100 trials per cell:
  at 24 pairs, a 32-bit fit to CRC-32 or Adler-32 data was accepted **100/100**
  times, with 0.4% accuracy on held-out inputs. That is worse than recovering
  nothing, because every input the fuzzer then "repairs" carries a wrong checksum
  while the operator looks healthy. Requiring full rank makes acceptance mean "the
  observations admit exactly one linear map". Over 2100 trials across 8/16/32
  bits: **0 wrong models**, 100% held-out accuracy on every accepted model, and no
  true positives lost — the gate converts insufficient-evidence cases from *wrong*
  to *abstain*. Pass `require_determined=False` for the old behaviour.

  Not verified: neither the 32-bit recovery nor the gate has been A/B'd for edge
  coverage on a real target. The mechanism is tested; the coverage claim is not.

- **`estimate_map_size()` now reports which tier produced its answer.** Every
  call logs the block count, the source (`sancov_guards`, `sancov_cntrs`,
  `profile`, `branch_density`, `default`), whether that source is exact or an
  estimate, and whether the cap bound. `estimate_map_size_detail()` returns the
  same as a `MapSizeEstimate` for callers and tests that need to assert the
  tier rather than the number. A silent fallback from "exact" to "estimated" is
  what hid the `__sancov_cntrs`/`__sancov_guards` mismatch below for the entire
  life of the function: tier 3 always returns a plausible number, so nothing
  downstream could tell a measurement from a guess.
- **Forkserver on the default execution path** (`--no-forkserver` to opt out).
  `afl_shim.c` now installs an AFL-style forkserver at the end of its
  constructor (`__afl_start_forkserver()`, AFL's protocol on fds 198/199), so
  the target is exec'd once and each input costs a `fork()` from a process
  already past its ELF load, dynamic linker, libc init and ASAN init.
  Measured: 5.27x on `test_target`, 1.38x on an ASAN target with heavy static
  init, **2.77x end to end through the CLI** (484 -> 1341 eps). The ASAN
  figure is the one to plan around — forking a process carrying ASAN's shadow
  mapping is itself expensive, and every target here is built with ASAN.
  Enabled only for the set `run_target_fast` already handled; in-process,
  persistent, network, ptrace, cmplog, perf-counter, `file_mode`,
  `target_args` and multi-target runs are untouched. Targets must be rebuilt
  to benefit — an older target silently takes the fork+exec fallback.
  Note coverage from pre-fork init is no longer re-recorded per execution, so
  edge sets are **not comparable across `--no-forkserver`**; verified to be a
  strict subset differing by a constant, input-independent init set.

### Fixed
- **An unguarded `execve`/`execv` in the fork+exec launch paths (`PersistentRunner.start`,
  `TargetRunner`'s ptrace launch) let a failed exec turn the child into an
  orphaned duplicate of the entire parent process.** `fork()` duplicates the
  whole process, including the interpreter's call stack; the child continued
  straight into `execve`/`execv` with no `try`/`except` around it. When exec
  failed (missing/non-executable target), the raised exception unwound back
  through that *inherited* stack instead of hitting the `os._exit(127)` meant
  to catch it — which sat unreachable one statement later, after the
  exec call, not around it. In a pytest run this meant the child kept
  executing as a second, orphaned copy of the whole test session: it
  re-entered pytest's own test loop and re-ran every remaining test
  concurrently with the real parent, producing duplicated test output and,
  at full-suite scale, hangs from the two copies contending for the same
  shm ids and temp files. Both launch paths now wrap the setup/exec sequence
  in the forked child in `try: ... except BaseException: os._exit(127)`,
  matching the existing convention in `ptrace_available()`. Regression test:
  `tests/test_regression_persistent_execve_failure_exits.py` runs the
  offending test in an isolated pytest subprocess and asserts it reports
  exactly once and completes promptly.

- **Every un-stopped `ForkserverRunner` leaked a process, a thread and a SHM
  segment.** The stderr-drain thread was started as `target=self._drain_stderr`
  — a bound method, so the thread held the runner. That reference is
  self-sustaining: the thread blocks reading the child's stderr until the
  child exits; the child exits when `stop()` sends QUIT; `stop()` runs from
  `__del__`; and `__del__` cannot run while the thread keeps the runner
  reachable. Nothing broke the loop. Measured: two test files left **98 live
  runners with 98 live children**, and `gc.collect()` freed none of them —
  they were reachable, not garbage. A full-suite run peaked at ~185 orphaned
  `fuzz_loader`/target processes, each pinning a SHM segment in `dest` state.
  The thread now targets a module-level `_drain_stream(stream, sink)` and
  holds no reference to the runner; after the fix the same suite run peaks at
  0 orphans and 1 segment. This is a production defect, not only a test
  artifact: any `Fuzzer` not explicitly stopped leaked the same three
  resources.
- **A failed SHM attach was silent, on both sides.** All three early returns
  in `__afl_map_shm()` (`__AFL_SHM_ID` absent, unparseable, or `shmat()`
  failing) left `__afl_area` NULL, after which the target ran its full input
  and exited 0 having recorded nothing, and the fuzzer read back an all-zero
  header. Success and total failure were indistinguishable. This is the
  "Loose thread" in `docs/edge-coverage-analysis.md`, unresolved across four
  sightings for precisely that reason; the instrumented assertion caught the
  fourth (`rc=0 edge_count=0 diag=0x00000000 dropped=0 occupied=0 stderr=b''`),
  and `diag == 0` identifies it as a child that never attached rather than a
  parent that raced the read. A target that was asked for coverage and could
  not attach now writes one line to stderr naming which return fired, with
  `errno`. Running standalone (no `__AFL_SHM_ID`) stays silent. The wording
  contains none of the tokens `ExecutionRunner.is_crash()` scans stderr for,
  so a diagnostic cannot be misread as a crashing input.
- **Map sizing read the wrong sancov section, so it never once used a real
  block count.** `estimate_map_size()` lists "sancov guard count (exact)" as
  its first priority, but the only parser it had — `parse_sancov_offsets()` —
  matches `__start/__stop___sancov_cntrs`, the *inline-8bit-counters* section.
  Every target in this tree is built with `-fsanitize-coverage=trace-pc-guard`,
  which emits `__sancov_guards` instead, so priority 1 matched nothing and every
  target fell through to branch-density estimation. That estimate ran 4–16x
  high: `test_target` (91 guards) asked for 131072 entries instead of 8192, a
  1 MiB table memset before every execution to hold 4 edges. Added
  `parse_sancov_guard_count()` and wired it ahead of the counters path. Reset
  cost per exec on the affected targets: 40.8 µs → 4.1 µs, which is noise under
  fork+exec (0.997x on `test_target`) and ~12% of a 305 µs forkserver exec. The
  same over-estimate also fed `recommended_map_size()`.
- `estimate_map_size()` divided the `__sancov_cntrs` length by 4 "because guards
  are uint32_t". That section is 8-bit counters, one byte per block, so the
  fallback path under-sized by 4x on any externally built target that did carry
  it. No target here has the section, so nothing in-tree was affected.
- **`fuzz_loader.c` was never a forkserver.** It did `fork()` + `execl()` per
  input, so every execution paid the full ELF load + linker + libc + ASAN init
  anyway. `docs/edge-coverage-analysis.md` §1 prescribed deleting its
  bitmap-file round-trip; that was necessary but measured **0.99x** on an ASAN
  target on its own. See `docs/learnings/2026-08-14-forkserver-that-execs.md`.
- Forkserver coverage never reached the fuzzer: the loader read a bitmap from a
  file while the target wrote to SHM. The child inherits `__AFL_SHM_ID` and the
  shim's constructor attaches on its own, so the bitmap, the `_COV_BITMAP_OUT`
  setenv and the caller-side `memmove` are all gone.
- The forkserver parent recorded its own control flow into the coverage map
  after the per-exec reset, and advanced `__afl_prev_loc` between forks, so the
  same input produced different edge ids on its first executions (5, then 6,
  then a stable 6). `__afl_area` is now detached in the parent, which suppresses
  recording through the null check already at the top of `__afl_map_edge` — no
  added hot-path cost, and `prev_loc` is left untouched.
- The forkserver dropped the child's stderr. ASAN exits 1, so
  `SanitizerReport.parse(stderr)` is the only crash signal `is_crash()` has
  there — the path would have been silently blind to every ASAN finding.
- A zero-length `RUN` produced no reply, blocking the caller until its join
  timeout and then tearing down and restarting the loader.
- An oversized `RUN` was skipped without consuming its body, leaving the payload
  in the pipe to be parsed as the next command.
- Forkserver timeouts reported the raw wait status (`-SIGKILL`). `-9` is in
  `SIGNAL_CRASH_CODES`, so every slow input would have been filed as a
  fatal-signal crash. Now `-1`.
- `ForkserverRunner.__del__` raised `ValueError: write to closed file` through
  the ignored-exception path on every clean exit (`stop()` runs twice).
- The loader piped stdin only, silently moving targets that read `argv[1]` when
  `argc == 2` (`png_read.c`, `grep_read.c`) onto their 64KB stdin path. It now
  stages each input into a file and execs `<target> <file>` with stdin
  redirected from it, mirroring `run_target_fast`.
- `runner.py`'s forkserver branch never reset the edge map, so
  `is_new_coverage_with_edges()` would have seen every execution since start
  accumulated together.
- `ShmCoverage.resize()` allocates a new segment and removes the old one, but
  the loader's environment is only read at exec time — its children kept
  attaching to the removed segment. Both resize sites now respawn the loader.
- The loader was hardcoded to `gcc`; it now prefers `clang` (hard rule 4).
- **cmplog/edge shim merge: a preloaded `cmplog_shim.so` could silently zero a
  target's coverage.** `cmplog_shim.c` carried a second copy of the edge
  machinery behind `weak` definitions of `__afl_map_shm` / `__afl_map_reset` /
  `__sanitizer_cov_trace_pc_guard{,_init}`. `weak` only loses to a strong
  definition at *static* link time; at dynamic link time the first definition
  in the global lookup scope wins regardless of binding, and `LD_PRELOAD`
  precedes dependency `.so`s. Measured on a `.so` target built without
  `-Wl,-Bsymbolic`: `__afl_area` = `0x7f4c757d6018` with no preload, `(nil)`
  with the shim preloaded — the run recorded zero edges. The four
  `_tracecmp.so` targets (`png_read`, `zlib_read`, `gzip_read`, `jpeg_read`)
  are built without `-Bsymbolic`, so they were exposed. The comparison layers
  now live in `afl_shim.c` behind `-D__AFL_CMPLOG=1`, and the `LD_PRELOAD`
  artifact is built from the same source with `-D__AFL_PRELOAD_ONLY`, which
  defines none of the `__afl_*` symbols.
- cmplog shim: the coverage segment was attached twice per exec (2 `shmat`
  against 1 for the edge shim alone) — the shim's own constructor re-entered
  `__afl_map_shm`, which in a combined link resolved to the strong definition.
- cmplog shim: the comparison buffer was flushed on the **first** crash only.
  Its crash handler restored the previous disposition permanently, so every
  later crash in a persistent/`direct_lite` loop lost up to 256KB of buffered
  records. The flush now lives in `__afl_crash_handler`, before the
  `siglongjmp`, and runs on every crash.
- cmplog shim: `AFL_MAP_SIZE` was read as *bytes* there and as *entries* in
  `afl_shim.c`. A live `__afl_map_reset` from the old shim would have zeroed
  the 24-byte SHM header and the first ~1021 entries. Only one definition
  survives the merge, so the unit is unambiguous.
- cmplog shim: `real_memcmp` and friends were resolved in the constructor and
  dereferenced unconditionally — a NULL call for any comparison reaching the
  interceptor before that constructor ran. Tolerable for an `LD_PRELOAD`
  object that loads early; not for one constructor among many in the target.
  Resolution is now lazy, with a re-entrancy guard (`dlsym` calls the very
  functions being interposed) and naive fallbacks.
- The comparison logger is no longer instrumented by the coverage it enables.
  With `-include` the layer lands in the *target's* translation unit, so
  `-fsanitize-coverage=trace-cmp` instruments the record writer, whose own
  comparisons call back into it — unbounded recursion arriving as a
  stack-overflow SIGSEGV at startup. Reproduced on `gcc -D__AFL_CMPLOG=1
  -fsanitize-coverage=trace-cmp`; fixed with `__AFL_NO_COV`
  (`no_sanitize_coverage` / `no_sanitize("coverage")`) on every function in
  the layer, plus a thread-local re-entrancy flag for toolchains without the
  attribute. The old build dodged this by compiling the shim as a separate
  uninstrumented object; that protection does not survive the merge.
- `CmplogCollector.collect_tokens`: the optional PC field was parsed with
  `int(s)` — base 10 — while the shim writes it in the `%p` convention
  (`0x55f65c387346`). Every parse raised `ValueError` into a `suppress()`, so
  `pc` was silently `None` for every record ever written and `_pair_pc` has
  always been empty. Now `int(s, 0)`, which still reads plain decimal, so
  existing logs keep parsing.
- `ShmCoverage`: hit counts are now part of the novelty decision. The primary
  coverage path decided interestingness by `ids - _seen_edge_ids` — set
  membership only — so an input driving a loop 2 times and one driving it 128
  times were the same coverage, and loop-count-guarded branches (`if (n > 16)`,
  buffer-growth paths, parser backtrack limits) were invisible. The `count`
  field was maintained faithfully by the shim and read back by
  `get_edge_counts()`; nothing consulted it.
- `core/count_class.py`: added `bucket_bit`/`bucket_bits`, AFL's
  `count_class_lookup8` as *bits*. `classify_single` returns representative
  values (0, 1, 2, 3, 4, 8, …) which are not disjoint — class 3 is `0b11`,
  class 1 OR'd with class 2 — so a virgin map built from them silently drops
  a hit count of exactly 3 from any edge already seen once and twice.
  `classify_*` semantics are unchanged; the new ladder is separate.
- cmplog shim: `memmem`/`strstr`/`strcasestr` passed a hardcoded `-1` as the
  comparison result, so a *successful* substring match bypassed `log_cmp`'s
  `result == 0` filter and was pooled as an unsolved comparison.
- `runner`: the bounded wait for a child's first ptrace stop charged the full
  per-exec `timeout`, which the run loop then charged again — worst case 2x the
  configured budget. Capped independently by `_INITIAL_STOP_TIMEOUT` (1.0s).
- `CmplogCollector.start`: superseded digest-keyed shim objects and the legacy
  fixed-name `fuzz_cmplog_shim.so` are now pruned once the current object is on
  disk, instead of accumulating in `~/.cache/fuzzer_cmplog/` forever.

### Changed
- `src/fuzzer_tool/adapters/cmplog_shim.c` is **removed**. Its libc
  interposition and trace-cmp layers are in `afl_shim.c`; enable with
  `-D__AFL_CMPLOG=1` (needs `-ldl`). Off by default, which keeps the
  interposers and the `-ldl` dependency out of targets that do not want them,
  and keeps `__cmplog_reset` out of their symbol tables — that symbol is what
  `services/fuzzer.py::_detect_cmplog` reads to decide whether `direct_lite`
  is safe, so a layer that always defined it would make the probe a constant.
- `tools/build_targets.sh`: `$CMPLOG_SHIM` and the per-target `cmplog_shim.o`
  compile/link/cleanup dance are replaced by `$CMPLOG_CFLAGS` /
  `$CMPLOG_LIBS`. The `tailslayer_read` C++ target keeps cmplog off (the
  interceptors use C signatures; C++ overloads their const-ness — the old
  build compiled the shim as a separate C object precisely to dodge that).
  MSAN/TSAN targets keep cmplog off as before: unmeasured rather than assumed
  safe.
- The trace-cmp callbacks are compiled `visibility("hidden")` in
  `-D__AFL_CMPLOG=1` builds, so nothing — libasan's weak stubs, an older
  preloaded shim — can interpose them. `nm` reports them as `t` (local)
  rather than `T`; the build script's post-link check accepts both.
- The comparison log is written through a raw fd and `write(2)` rather than
  `FILE*`/`fwrite`. The pre-crash flush runs inside a signal handler, where
  stdio is not async-signal-safe; this also drops the stdio lock from a path
  that runs on every intercepted comparison. Record format is unchanged.
- The SHM virgin bucket map is indexed by `edge_id` into a dense `uint8` array
  rather than a dict, because it runs on every coverage-changing execution.
  Measured on 200k active edges: 1.8ms direct-indexed against 28.7ms via a
  sorted-array `searchsorted` and 108ms via a per-entry dict loop — 6-12% on
  top of the `set(...) - _seen_edge_ids` diff already on that path, against
  ~500% for the dict loop. Affordable because guard values are small
  sequential integers, so `edge_id = prev_loc ^ cur_loc` stays in a range of
  roughly `2 * guard_count` and XOR with a context term cannot widen it past
  its wider operand. `__AFL_CTX_BITS` in the 24..32 range is the exception and
  falls back to a dict (`VIRGIN_DENSE_MAX`).
- Comparison constants are now visible on optimized targets. `-fno-builtin-*`
  (`$NOBUILTIN_CMP`) keeps `memcmp`/`strcmp` at the PLT so the libc layer sees
  their operands at `-O2`; `-fsanitize-coverage=trace-cmp` cannot recover them
  at any optimization level, because SanitizerCoverage instruments IR `icmp`
  and clang's `ExpandMemCmp` runs after it. Measured on
  `targets/cmplog_exercise.c`: 0/10 constants at `-O2`, 10/10 with the flags.
- trace-cmp targets compile the callbacks in instead of relying on `LD_PRELOAD`.
  `-fsanitize-coverage` links compiler-rt's sancov runtime, whose weak no-op
  `__sanitizer_cov_trace_*cmp*` stubs win the symbol lookup against a
  preloaded shim; the callbacks fired 20 times and logged nothing.
- `WITH_TRACECMP` defaults to on (`--no-tracecmp` opts out) and now covers
  `cmplog_exercise` as well as `tracecmp_target`, built as `*_tcg`.
- mypy is ratcheted rather than permanently red: `strict = true` remains the
  target, the 114 modules that cannot yet satisfy it are exempted by name in
  `[[tool.mypy.overrides]]`, and the other 17 are checked strictly. New modules
  are strict by default. `tests/test_regression_mypy_ratchet.py` enforces that
  the list only shrinks.
- CI installs clang and the `smt` extra, and fails if either is missing. Those
  105 tests previously skipped silently.
- The initial-ptrace-stop regression test runs on an injected virtual clock
  instead of a wall-clock threshold, making the exec budget directly
  observable and the assertion coverage-insensitive.

## [0.1.0] - 2025-01-01

### Added
- Core mutation operators (bit flip, byte flip, interesting values, block ops, havoc)
- Dictionary support with token injection
- Markov chain byte-level generation and mutation
- Thompson sampling bandit for operator selection
- Cross-entropy method for per-position byte distribution learning
- Sanitizer output parsing (ASAN, MSAN, TSAN, LSAN, UBSAN)
- Crash deduplication via signature generation
- Coverage-guided mode with ptrace breakpoints
- Deep coverage via x86-64 decoder disassembly
- File-mode execution for file-reading targets
- CLI with argparse
- pytest test suite
- CI pipeline with GitHub Actions
