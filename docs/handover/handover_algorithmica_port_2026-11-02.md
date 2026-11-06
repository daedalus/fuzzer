# Handover: what ports from en.algorithmica.org (Algorithms for Modern Hardware)

**Date:** 2026-11-02. **Verified against:** `7c1de57d`. **Status:** analysis only, no code changed.
**Source:** https://en.algorithmica.org/hpc/ (Sergey Slotin, 2021-22 draft; the English site hosts nothing else useful).

## 1. Verdict

Little ports directly. Two reasons, both verified in the repo:

- The C side already applies most of the book (see section 2).
- Time is spent in Python. `handover_shm_hugepages_ab_2026-10-28.md` section 4: target time is ~0.15 ms of a 1.5-5 ms cycle; `handover_unapplied_optimizations_2026-10-24.md` budgets ~6.8 ms/exec fuzzer-side vs ~1.8 ms in the target. Cache-line and SIMD tricks in the shim cannot move that.

## 2. Already covered (do not port)

| Book topic | Where it is in the repo |
|---|---|
| SIMD, masking, movemask+ctz | `afl_shim.c` `__afl_probe_vec` (AVX2 / SSE2 paths), `__afl_probe_insert` |
| Cold paths, no-instrument attributes | `afl_shim.c`: `cold`, `noinline`, `__AFL_NO_COV`, 7 uses of `__builtin_expect`/`__AFL_LIKELY` |
| Direct-mapped caches | `afl_shim.c` `__afl_ctx_base_cache` (keyed by return address) |
| AoS vs SoA / scan layout | **Already implemented**: `--touched-scan`, see section 3 |
| Hugepages / paging | Measured, no gain, not shipped: `handover_shm_hugepages_ab_2026-10-28.md` |
| Blocked Bloom filters | Evaluated and rejected: `handover_sbbf_vs_flat_bloom_folding_2026-10-21.md` |
| Pointer alternatives | `handover_tiny_pointers_shim_probe_2026-10-13.md` |
| Fenwick / prefix-sum sampler | `core/fenwick_sampler.py` |
| Eviction policies | `core/lru.py`, adaptive LRU in `smt_solver` |
| NTT, Montgomery (secp256k1 only), number theory | `core/ntt.py`, `core/mutations/montgomery.py`, `core/numtheory.py` |
| Binary search | `bisect` is C in CPython; `core/edge_set.py` bisects a sorted `array('I')` |

Not found in the shim: `__builtin_prefetch` (0 matches). No port is proposed: the table is probed within a bounded window and the shim is not the bottleneck.

## 3. Correction: item "#2 AoS/SoA scan layout" was already done

An earlier version of this analysis proposed wiring the touched-slot bitmap into `ShmCoverage._scan`, relying on `handover_touched_bitmap_2026-10-07.md` ("not wired into `_scan`"). That handover is stale. In `7c1de57d`:

- `adapters/shm.py` `_scan`: when `touched_scan` and `touched_supported`, live slots come from `_touched_slots()` (bitmap), then the same generation and non-zero-id filter.
- `services/fuzzer.py:1023` `resolve_touched_scan`: `None` means auto, on from `TOUCHED_SCAN_MIN_ENTRIES = 65536`; `--touched-scan` forces on, `--no-touched-scan` forces off. Falls back to the table walk if the shim has no bitmap.
- Measurements are in `docs/DEEP_DIVE.md` ("Touched-slot bitmap"): 2.5x/1.3x at 65536, 7.2x/3.3x at 262144, 21x/9x at 1M, no gain at 8192. Folding the bitmap was measured and rejected there.
- The non-bitmap path (`_live_entries`) tests the generation byte with a stride-8 `uint8` compare, which touches every cache line of the table. For maps under 65536 entries that is the faster option (measured), so no layout change is proposed.

Action: the stale statement in `handover_touched_bitmap_2026-10-07.md` can be updated to point at `DEEP_DIVE.md`. Not done here.

## 4. Open candidates, highest value first

### 4.1 Low-noise benchmark metric (book: Benchmarking, Getting Accurate Results)

**Status (2026-11-05): implemented** as `cpu_user` / `cpu_sys` per cell (`tools/lib/child_rusage.py`, `os.wait4` rusage; `--metric cpu_user`). `perf stat` not used: not installed here. No A/A noise comparison against wall time yet (TODO).

Evidence: `tools/lib/bench_paired.py`, `bench_replicated.py` and `noise_probe.py` contain no instruction-count, CPU-time or `perf` metric (grep for `perf_counter|process_time|instructions|perf stat|cycles|rusage|ru_utime|taskset|governor` over `bench_paired.py`, `benchmark.py`, `profile_hotpath.py` returned nothing; `bench_replicated.py` times with `time.time()`). Paired cells compare edge counts, and `noise_probe.py` documents that a cell is not reproducible under the same arm because of wall-clock state. The hugepages A/B had an A/A control drifting -1.0% to -3.7%, the size of every effect measured.

Proposal for throughput A/Bs (T1-T11 of the optimizations handover): add an optional metric of user-mode instructions retired (`perf stat -e instructions:u`) or `resource.getrusage(RUSAGE_CHILDREN).ru_utime`, reported next to wall time, with the A/A control (Hard Rule 46) on the same metric. Instruction counts are insensitive to frequency scaling and neighbours; they do not capture cache or branch effects, so state that in any verdict.

Not verified: `perf` and `valgrind` are not installed in the sandbox used for this analysis, and whether `perf` works in the target VM (hardware counters are often blocked in VMs) is unknown. `ru_utime` needs nothing extra and is the safe first step.

### 4.2 Generic Montgomery constants (book: Montgomery Multiplication)

**Status (2026-11-02): implemented** as `montgomery_word_mutate` (see CHANGELOG). Recognition is by exact literal from `WORD_MODULI`, not by shape; the helpers above are fixed at 256/64 bits, so the new code has its own width-parameterised `word_constants`. Unmeasured on a target.

`core/mutations/montgomery.py` only sniffs the secp256k1 field prime and curve order (`sniff_secp_modulus`, `_modulus_from_seed`). Proposal: a second sniffer/mutator for odd 32- and 64-bit moduli found in the seed, injecting `n' = -n^-1 mod 2^w` (Newton iteration `x *= 2 - n*x`, as in the book), `R mod n`, `R^2 mod n`. Helpers `_redc_n0_prime`, `_montgomery_r`, `_montgomery_r2`, `_barrett_mu` already exist; check they are width-parameterised before reuse (not verified, they are written for the 256-bit/64-limb case). Gate like the existing operator (registry `_AVAILABLE` predicate) so it is silent on seeds without a plausible modulus. Value is speculative: it only helps targets with word-size Montgomery code.

### 4.3 Fast inverse square root constants (book: Fast Inverse Square Root)

**Status (2026-11-02): implemented on request, in `prime_inject`'s table** (`core/mutations/prime_inject.py`, `_RSQRT_MAGIC`: `0x5F3759DF` in the 4- and 8-byte tables, `0x5FE6EB50C7B537A9` in the 8-byte one; appended last so `values_for(4)[0]`, the empty-input constant, is unchanged). The expected benefit is low: (a) the constant lives in the target's code, not its input, so injecting it feeds the target nothing it reads unless the target parses such a magic number from input; the input-side edge cases of `i = MAGIC - (i >> 1)` (negative floats, NaN, denormals) are already in `structured.py`'s float patterns, and fast rsqrt is straight-line, so there is no branch to reach. (b) It was deliberately not put in `INTERESTING_UNSIGNED_32` (feeds the hot `_op_interesting_32`; one more entry shifts every seeded `rng.choice` index) nor in `INTERESTING_32` (the AFL table the deterministic stage sweeps). (c) No 64-bit interesting table exists, which is why the f64 constant went into `prime_inject`. Cost: 2 of the 33 width-8 entries (about 6% of width-8 `prime_inject` writes) and 1 of the 29 width-4 entries (about 3%); widths 1 and 2 are unchanged. Unmeasured on a target.

`core/mutations/structured.py` ~line 998 has float bit-pattern pairs (denormals, ulp neighbours, NaN payloads). Grep for `0x5f3759df`, `0x5fe6eb50` and `rsqrt` over `src/` and `dictionaries/` found no such constants (only the word `rsqrt` in `ruby.dict` and `x86.dict`). Adding `0x5f3759df` (f32) and `0x5fe6eb50c7b537a9` (f64) to the interesting-values set is a few lines. These are integers, not float patterns, so they probably belong with the integer interesting values rather than the `(f64_bits, f32_bits)` table; check which table the call sites read. Benefit is speculative.

### 4.4 Sampling profiler for the Python hot path (book: Statistical Profiling)

**Status (2026-11-05): implemented** as `--profile-mode sample` (SIGPROF sampler, `adapters/stack_sampler.py`) and `--profile-mode perf` (`sys.activate_stack_trampoline`, 3.12+; exits on 3.11). `tools/profile_hotpath.py` still uses cProfile. See DEEP_DIVE §Benchmarking, Profiling modes.

T4-T11 savings are 1-10 us per exec, below the overhead of a deterministic profiler like cProfile. Python 3.12 (sandbox: 3.12.3) supports `perf` with `PYTHONPERFSUPPORT=1` / `-X perf`; `pyproject.toml` has `requires-python >=3.11`, and 3.11 lacks it. `tools/profile_hotpath.py` exists; what profiler it uses was not checked.

## 5. Rejected

- Eytzinger layout, S-trees, SIMD binary search, wide segment trees: in CPython the lookups are C `bisect` calls or numpy; no interpreter-level gain.
- Binary GCD, integer factorization: `math.gcd` is C; `numtheory.factorint` is documented instant at 64 bits.
- Prefetching, cache associativity, alignment/packing in the shim: not measurable from Python.
- Branchless programming in the shim: edges are counted, not branched on, in the hot path.

## 6. Not done

Nothing implemented. No benchmark run. Sandbox lacks `perf`/`valgrind`, so none of section 4 was measured. Suggested order: 4.1 first, since without a stable metric the throughput items cannot be shown to help.

**Stale (2026-11-05):** 4.1-4.4 are now implemented; see their status lines.

## 7. Re-check, 2026-11-05 (verified against `44f3e74`)

An independent pass over the book (source read from `algorithmica-org/algorithmica` on GitHub; the site is proxy-blocked here) agreed with sections 2 and 5 and found the items below.

### 7.1 Open

| Item | Evidence | Port | Caveat |
|---|---|---|---|
| A/A noise for `cpu_user` | 4.1 TODO | A/A run, `--metric cpu_user` vs wall | Gates every throughput item |
| O(n) weight rebuild per pick (book: prefix sums) | `services/seed_picker.py` `_pick_katz_seed:764`, `_pick_branch_seed:789`, `_pick_tang_seed:819`, `_pick_aflgo_seed:1151`, `_pick_boltzmann_seed:1371`, `_pick_ecofuzz_seed:1447`: Python loop builds weights, then a linear cumulative scan | numpy weights + `searchsorted`, or cached weights in `core/fenwick_sampler.py` updated per seed and rebuilt when a shared input changes (`mean_exec_time()`, Boltzmann `_temperature`, AFLGo `max_distance`) | Elo arms: run a fraction of picks. Swapping only the scan for `bisect` leaves the O(n) rebuild |
| `% n` range reduction (book: Division, Lemire) | `core/rand_pool.py:353` and `randint`/`choice`/`shuffle` paths, plus bulk `randint_list` (`choice_list`) and every other `% width` site in `rand_pool.py` | Lemire multiply-shift with rejection | Removes bias, no CPython speed gain. Changes every seeded stream |

### 7.2 Checked and withdrawn

- AVX2 probe as default: measured, rejected (`docs/DEEP_DIVE.md` "SIMD probe runs, opt-in"). Default stays scalar.
- Cache-line bucket homes (`pos & ~7`): measured, slower everywhere (`docs/DEEP_DIVE.md:669`).
- Huge pages, SoA scan layout: already settled (sections 2, 3).
- `__afl_probe_distance` hang: `docs/afl_shim_hotpath_analysis.md` item 5 says it "never terminates" on a dense table. False: the loop is bounded by `i < size` (`afl_shim.c:3139`). Dense table = O(size) per edge, not a hang. That doc is stale.
- Two-digit LUT in `__afl_put_i64` (`afl_shim.c:3972`): callers are cmp/site dumps and crash records, not per edge. Not proposed.
- Stale comment: `afl_shim.c:1754` still says `% map_size`; the code multiplies and shifts.
