# TODO — fuzzer-tool Roadmap

> **Status note**: This roadmap tracks open work only. Closed items are pruned
> on sight — the shipped record lives in `CHANGELOG.md` and in git history.
>
> **Standing rule, earned the hard way.** An unchecked `[ ]` in this file is not
> evidence of anything. Repeated audits found entries marked pending that had
> been implemented for over a week, including the top open item under *Coverage
> & Instrumentation* — picked up by a planning pass on the strength of this file
> alone. Verify against the code before starting work, and when closing an item,
> close it in the same commit as the implementation.

## Coverage & Instrumentation
- [ ] **Leak severity is conditional on edge count** (2026-08-23) — the fixed header clobber only carries stale coverage forward on targets with more than `(map_size - 24) / 8` live edges (8,189 at the default 65,536-entry map). Measured: entry index IS the edge id, assigned sequentially by `__sanitizer_cov_trace_pc_guard_init`, and `png_read.so` has ~4,600 guards, so nothing in `targets/` reaches the threshold. Verified deterministically by planting an entry at index 20,000: reported as live coverage pre-fix, filtered post-fix. A vendored ffmpeg or grep build would cross it naturally — worth re-running the probe (`docs/learnings/2026-08-23-shmat-sentinel-and-header-clobber.md`) once one is available.
- [ ] **GEP yield is optimization-level dependent** (2026-08-28) — the layer-3 `trace-gep` callback only sees indices clang still models as a GEP when the sancov pass runs; at `-O1`+ an array index folded into an addressing mode is already gone (measured: 17 call sites emitted, none firing for the indexed load the test exercises; at `-O0` it fires). Vendored `--vendor-tracecmp` targets build at `-O2`, so the divisor half of trace-div/trace-gep is the half that pays there. Worth measuring real GEP yield on a vendored libpng build before deciding whether the flag earns its stream volume.
- [ ] **`reset_bitmap()` is redundant on the runner path** (2026-08-23) — `services/runner.py` calls `shm.reset_edge_map()` immediately before `run_one()`, and `run_one` has exactly one call site, so the full-table memset in `InProcessRunner.reset_bitmap()` (reached from `_run_c_direct`) resets a table the generation bump already invalidated. Dropping the call would save a `shm_size * 8` byte memset per execution on the hot path of the fastest execution mode. Not done here because it wants measuring on a built target, and clang was unavailable in the authoring environment. The header-clobber half of this was a real bug and IS fixed — see `docs/learnings/2026-08-23-shmat-sentinel-and-header-clobber.md`.

## Scheduling
- [ ] **Validity channel needs a harness convention per target** (2026-08-28) — `--reject-code` is opt-in because nothing in the tree emits a rejection code today: `targets/*.c` return 0 or crash. The channel is inert until a harness is taught to report rejection (libFuzzer's `return -1` convention is the closest analogue). Worth adding to one structured target — `png_read.c` returning the code on a signature mismatch — to measure whether valid-only coverage admits anything the main map does not.
- [ ] **SGFuzz transitions are edges, not a state tree** (2026-08-28) — `__sfuzz_state` hashes each transition into the edge map, which buys every existing coverage consumer for free but materialises no State Transition Tree, so nothing can schedule by state the way upstream SGFuzz does (energy toward rarely-reached states). Building the tree needs the transitions read out separately, not folded into the map. Also open: the instrumenter skips declaration initialisers (a comma expression there is a second declarator, not a transition), so the initial state is unreported and the first real transition sees 0 as its predecessor. And no target in `targets/` is instrumented yet — `tools/build_targets.sh` has no `--sgfuzz` pass, so the runtime is inert in the tree as shipped.
- [ ] **Collaborative scheduling across parallel workers** — parallel workers currently sync corpus but don't coordinate scheduling decisions. Could share exploration/exploitation state.
- [ ] **A/B the cost-based Boltzmann energy** (2026-08-29) — `SeedPicker._pick_boltzmann_seed` now takes its energy from the per-seed cost ledger (`effective_fuzz_count`) instead of `fuzz_count`. The gating measurement justified writing it (`docs/learnings/2026-08-29-per-seed-cost-ledger.md`) but the paired benchmark through `tools/bench_paired.py` was **not** run. It changes seed selection, and the argument for why it could make things worse stands untested: down-weighting expensive seeds is down-weighting deep paths on targets where depth costs time. Measured on `png_read` the old and new orderings agree only weakly (Kendall tau 0.46), so a difference should be visible if there is one. Note the change is arithmetically a no-op on a target with uniform exec cost, so `gzip_read` is the wrong target to benchmark it on — use `png_read` at `-m 65536` or a vendored `ffmpeg_read`.
- [ ] **Bound the edge tracker by memory, not by seed count** (2026-08-30) — `max_tracked_seeds` is a seed count, but what it exists to bound is the memory held by `seed_edges` and its eight companion maps, and the cost per seed spans 6x across our own targets: measured 95.3 KiB on a png_read-shaped target (~500 live edges) against 592.2 KiB on an ffmpeg_read-shaped one (8,189). The ceiling of 1,000 set in the prune fix is therefore 93 MiB on one target and 578 MiB on another for the same nominal setting. A budget in bytes, with the ceiling derived from observed average edges per seed, would hold a real bound on both. Not done with the prune fix because changing the unit is a larger change than unblocking the mechanism, and the count at least binds now. Measurements in `docs/learnings/2026-08-30-prune-ceiling-and-eviction.md`.
- [ ] **Should cumulative execution cost enter the eviction ordering?** (2026-08-30) — backlog D5's third consumer, now unblocked. `EdgeTracker._maybe_prune` orders evictions by how much unique coverage goes with the seed, which answers the original complaint that insertion order was a tiebreak with no defence behind it. Whether cost per retained edge should join that ordering is a separate and now testable question. Note the premise it was filed under was wrong in a way that raises its value: the fallback path was assumed to fire rarely, and measured on a corpus-shaped workload it evicted all 420 seeds while subsumption evicted 0 — every seed owns a unique edge because owning one is what got it admitted. This inherits the A/B requirement from the Boltzmann item above, since it changes which seeds exist.

## Integer-Modulus Checksum Recovery (follow-ons)
- [ ] **Weighted-sum multiplier sweep is a fixed candidate list** (`_MULTIPLIER_CANDIDATES`) — a target using an unlisted multiplier is missed entirely. Recovering `k` properly means root-finding mod `N`; Coppersmith's bound (`N^(1/deg)`) is useless at realistic data lengths, so the list is the pragmatic answer for now. Consider deriving candidates from cmplog constants instead of hardcoding.
- [ ] **`_extract_zlib_adler_pairs` only fires on valid streams** — `decompressobj` raises on an Adler mismatch, so mutated PNGs yield no pair. Pairs therefore come only from corpus seeds and successful recompressions. Reading the trailer without validating would widen the source but needs a raw-deflate path.
- [ ] **Fletcher-32 word endianness is swept, not detected** — both LE and BE are tried and verification arbitrates. Fine, but it doubles the general-path work for that family.
- [ ] **No format-aware patcher for integer checksums** — `_op_crc_learn` patches only the generic trailing field when an integer model is active. A real zlib/IDAT Adler patcher belongs in the `recompress_zlib` mutator, not here.
- [ ] **`field_constraints.py` bounded-integer pre-pass** (handover §1, deprioritized) — z3 is already fast on these small bitwidth systems, so the win is thin. Revisit only if the integer-checksum pattern proves out.

## Testing
- [ ] **One retry-until-random-hit test is left** (2026-08-29) — Hard Rules 39/40 landed and `tests/support/scripted_rng.py` is the shared helper, but `tests/test_new_operators.py::test_fuse_old` (~line 305) still loops 30 times waiting for `_op_fuse_old` to change the buffer and `break`s on the first hit. Last survivor of the 2026-08-24 determinism pass, and it tests luck: it can pass while the operator is broken and fail unreproducibly when it is not. `_op_fuse_old` draws through `self.f._rand_pool` (`rng.choice` over the fuse-memory ring), so the `ScriptedRng` seam used elsewhere applies directly — drive the exact draw, assert the exact output.

## Standing notes

These are not work items. They are the lessons the closed work left behind, and
they keep costing time when forgotten.

### Test quality

A sweep found **573 of ~4,956 tests (11.6%) whose every assertion is
value-free** — `is not None`, `isinstance`, `len(...) > 0`. That is what let
the dictionary and grammar bugs through: output was well-typed and
semantically wrong. An invariant audit over all 134 operators (18,090
invocations across nine input formats, checking type / max_len / exceptions)
found **zero** violations, so the operators themselves are sound; the risk
concentrates in PARSERS, where a wrong byte is still a valid byte. The
`test_hex_escape` case is the worse variant: an assertion strong enough to
pass review that pins the defect. Prefer asserting values, and derive the
expected value from the spec, not from a run.

### Caveat for any future per-operator rate measurement
`_format_available` offers a never-yet-seen format on non-matching input 2% of
the time (`_FORMAT_BOOTSTRAP_RATE`), so a sniffer-gated operator's *denominator*
includes selections where it was never going to fire. On a corpus containing
none of its format, an operator's reported success rate is essentially the
trickle rate, and says nothing about correctness. Compare against a corpus that
actually contains the format before concluding an operator is broken.

This also made the no-op regression sweep seed-dependent: a trickle-offered
operator that correctly declines to mutate looks like a pure no-op. Fixed by
giving every sniffer-gated operator a matching sample in the battery; keep it
that way when adding operators.


### Verified not-a-bug
- `--cmplog` under `--inprocess-direct` (reported as "Cmplog: disabled" in the
  FFmpeg session). The wiring is correct: `_detect_cmplog` finds the exported
  `__cmplog_reset`, and the direct_lite path prints "compiled into target .so
  (direct_lite compatible)" and collects pairs. Reproduced on
  `targets/cmplog_exercise.c` built with `-D__AFL_CMPLOG=1`: 420 tokens /
  450 pairs.
  The session recipe simply never enabled it — `-c` is `--coverage`, and
  `--cmplog` has no short form. Add `--cmplog` explicitly to the FFmpeg recipe.
