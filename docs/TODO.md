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
- [ ] **Per-seed cost is accumulated and only ever read as an average** (2026-08-29) — `meta["total_time"]` is written in `Fuzzer.fuzz_one` and all three readers (`Fuzzer._cull_queue`, `Fuzzer.run`, `CorpusManager.auto_minimize_corpus`) divide it by `fuzz_count`. Stale-seed detection in `StatsReporter._print_summary_seeds` therefore uses a count (`fuzz_count >= 50`) where cost is the question, and `SeedPicker._pick_boltzmann_seed` uses `E = log(fuzz_count + 1)` for the same reason. **The measurement gates the work**: if `total_time / fuzz_count` is tightly clustered across a real corpus then the count and the cost agree and none of this is worth writing — that result should be recorded in `docs/learnings/` and this item closed. Start with the stats one; it is display-only. Analysis in `docs/handover/handover_persistence_mechanics_2026-08-29.md`, backlog entry D5.

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
