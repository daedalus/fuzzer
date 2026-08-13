# fuzzer-tool — Deep Dive

*This is the comprehensive reference documentation, moved from the original README. For a quick overview, see [README.md](../README.md).*

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/daedalus/fuzzer)

Coverage-guided binary fuzzer with static target analysis, statistical novelty scoring, Markov chain generation, Monte Carlo mutations, and format-aware grammar mutations.

Honest Caveats:
- This fuzzer is developed entirely with AI-assistance.
- Probably the most complex and dense fuzzer from the information theory standpoint but also the slowest. The tradeoff is speed for edge discovery novelty.
For production and sensitive binaries using AFL family fuzzers is the best course of action.

## Features

### Mutation & Generation
- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit, signed + unsigned boundary), arithmetic (1/2/4/8-byte, LE/BE), block insert/delete/duplicate, bit-offset flip/span (arbitrary bit positions for DEFLATE/JPEG), havoc mode (with stall-recovery escalation), TLV-aware mutation, token shuffle, security-sensitive string injection (44 curated SQL/XSS/traversal/format strings), magic values table (229 boundary values across all widths/endians), in-place ASCII number arithmetic, chunk shuffle (boundary-preserving), block shuffle variable (variable-width blocks via order-statistics spacings), compound dictionary insert, punctuation insertion, gradient-descent search (Angora-style byte-level optimization toward cmplog operands)
- **Operator performance**: `type_replace` uses a precomputed 256-byte translate table (184x faster), PNG/BMP random generation uses `random.randbytes()` instead of Python loops (16x faster), `colorization` uses a module-level lookup table, RedQueen fallback hoists `bytes(buf)` outside the while loop (saves 4/5 full-buffer copies), `_op_line_mutate` uses `bytearray.split()` instead of `bytes().split()` (avoids redundant buffer copy)
- **Import-time memory reduction**: `LOOKUP_U16` in `core/count_class.py` is now lazily built (PEP 562 module `__getattr__`) instead of at import — numpy's vectorized path handles all non-empty trace buffers, so the 65,536-entry table (~2.6 MB) is dead weight and is only constructed on first access (which never happens in normal operation). `maybe_refit` in `core/schedulers/monte_carlo.py` snapshots the previous CEM `byte_freq` by reference swap instead of a deep copy (`_prev_byte_freq = self.byte_freq` — safe because the dict is reassigned, not mutated, during refit), saving ~1.3 MB per refit. `mutate()` in `services/operators.py` skips the redundant `bytearray(bytearray_slice)` copy when an operator already returned a bytearray — the slice is reused directly (all non-havoc operators return `bytearray`; the havoc path returns early with `bytes`).
- **array.array for homogeneous numeric data**: per-seed MinHash signatures (`MinHashLSH.signatures` in `core/edge_tracker.py`) are stored as `array('Q')` instead of `list[int]` — each 64-entry signature drops from ~36 B/element (PyLong objects) to 8 B/element, saving ~1.8 KB per seed (MBs on large corpora); `compute_signature`/`corpus_minhash` return arrays and `add()` normalizes lists (the JSON load path) via `array('Q', sig)`, with `save()` serializing via `.tolist()`. The MI tracker's `edge_marginal` (`core/mi.py`) is `array('Q')` instead of a dense `list[int]` over edge-index space (up to 65,536 entries) — bounded at 2^64 per edge count, JSON round-tripped via `.tolist()`/`array('Q', em)`. Transient numeric accumulators use the same treatment: the Levenshtein DP rows (`prev`/`curr` in `core/similarity.py`, both the bytes and token variants) are `array('i')` — two rows of `len(a)+1` ints drop from ~48 B/element to 4 B/element, cutting the DP peak ~5.9x (48 MB → 8 MB at 1 MB inputs) — and the JS-divergence accumulators `js_values` in `core/markov.py` (`_js_between_snapshots`, up to `MAX_TRANSITIONS` entries) and `core/schedulers/monte_carlo.py` (`_compute_js`, per byte position) are `array('d')`. **Cold bounded histories** follow the same rule (verified runtime-neutral — flat self-times in the 10k profile, wall clock within noise): `_corpus_size_history` (`corpus_manager.py`, `array('I')`, state.json round-trips via `list(arr[-500:])`), the fuzzer's three tuple histories (`_discovery_execs`/`_discovery_edges`, `_crash_rate_execs`/`_crash_rate_counts` — `array('Q')` pairs, `_entropy_execs`/`_entropy_vals` — `array('Q')`+`array('d')`, consumed via `zip(..., strict=True)` windows), the edge_tracker coverage timeline (`_coverage_execs`/`_coverage_edges`, `array('Q')` pairs, `edge_tracker.json` format unchanged), elo's `_prediction_errors`/`_best_win_rate` (`array('d')`), and the redqueen pair-length index (`array('I')`, bisect consumer unchanged). Deliberately NOT converted: the EdgeTracker edge-count maps stay sparse dicts — `array.array` loses runtime on scalar read-modify-write (`record_edges`) and Python-level iteration, and the heavy readers (`shannon_entropy_global`, `simpson_diversity_global`) are already numpy-vectorized via `np.fromiter(hits.values())`; a dense `array('Q')` rewrite was designed (commit-ready plan) then dropped on benchmark evidence (array scalar ops ~2x slower than list; iteration slower; memory win only above ~50% map saturation) plus the collision-merge semantic change a modulo-remap would introduce.
- **Adaptive havoc sub-mutation weighting** (on by default; `--no-adaptive-havoc` to disable): `_apply_single_mutation` picks one of 11 inline branches (bit flip, byte set, byte swap, insert byte, delete block, CRC32 repair, swap regions, endianness swap, byte insert, random byte, shuffle range). That pick was `r[0] % 11` — flat odds forever — while every top-level operator above it is scheduled by a bandit fed real success rates. Branches are now drawn from a precomputed 256-slot inverse-CDF table rebuilt from `hits / trials` per branch, credited on the same signal the outer schedulers use: trials increment when a branch is applied, hits when the resulting execution finds new coverage (deferred to `Fuzzer._record_outcome`, since havoc mutates long before the verdict exists). Guard failures count as trials without hits, so a branch that no-ops on this target's input sizes decays — the same treatment `_last_ops_effective` gives no-op operators. A 15% uniform mixing floor (EXP3-style) guarantees every branch keeps ≥3 of the 256 slots, so a branch that only pays off on larger inputs can climb back rather than being starved permanently; counts halve past 100k trials to stay responsive to a target whose reachable behaviour shifts mid-run. Counts persist across `--resume` keyed by branch *name*, not index. Implementation is the cheapest form that consumes the signal: the table lookup measured 154 ns/draw against 250 ns for a bisect over an 11-float CDF and 237 ns for `array('d')` counters (`tools/bench_havoc_subop.py`), and the table refresh is triggered once per mutation round rather than per sub-mutation (havoc applies 2-8 per call, 8-16 under stall recovery).
- **Length boundary operator**: systematically tries input lengths at boundary values (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024, 4096) — discovers length-sensitive unsigned integer underflows
- **Unsigned boundary values**: interesting values include small values (0-5) and unsigned max values (0xFF, 0xFFFF, 0xFFFFFFFF) for triggering unsigned arithmetic underflows
- **Crash-MI-guided mutation**: CrashMITracker identifies byte positions and values correlated with crashes, biasing mutation position selection and interesting value selection toward crash-relevant bytes
- **CrashMITracker memory pruning**: automatically caps per-position byte-value tracking to top 32 entries every 500 execs — prevents tracker JSON from growing unbounded (was 7.5MB per 15k execs, now ~500KB)
- **Weighted length distribution**: length_boundary operator weights small lengths (0-16) 10:1 over large ones (512+) — 4096-byte inputs dropped from 4.7% to 0.5% of picks, stabilizing EPS
- **Corrupted state recovery**: seed_meta entries with keys > 256 chars (tracker JSON loaded as corpus) are skipped on load and save — self-heals bloated state on first run via `cleanup_legacy()` (15MB → 6KB)
- **Grammar-aware mutations**: format-specific structure-aware mutations for PNG (IHDR, IDAT, CRC, filter types, interlace), JPEG (SOF, DHT, DQT, DRI, SOS, scan data), BMP (header fields, pixel data), gzip (header flags, deflate stream, trailer, extra fields), zlib (CMF/FLG header, deflate stream, Adler-32 trailer), **PGS** (segment-type/header/payload mutations on PCS/WDS/PDS/ODS/END segments with correct reserialization), **ISO-BMFF** (box-type/size/ftyp/handler-type/codec mutations on recursively-nested MP4/MOV containers), **NAL** (H.264/H.265 start-code-delimited NAL unit type/ref_idc/slice/SPS/PPS mutations), **Protobuf** (tag renumbering, wire-type change, varint/length re-encode, field delete/duplicate/swap/splice on wire-2/3 fields with byte-preserving `raw_between` gaps), **GIF** (LSD dimensions/flags, GCT colors, image descriptor/min-code-size, sub-block length rewrite/insert/delete, extension label swap, block duplicate/truncate — byte-identical reserialization), **WebP** (RIFF chunk type/size, VP8 frame/VP8L header/VP8X flags, ANMF fields, chunk swap/duplicate/delete/truncate; chunk size written verbatim), **WebM** (EBML element ID swap, size-vint rewrite incl. all-ones unknown-size, codec swap, container nest/unnest, leaf data, element duplicate/delete/swap, TimecodeScale/Duration, truncate — vint re-encode only when payload length changed), **ZIP** (LFH/CD/EOCD field patching: method swap, CRC incl. recompute, csize/usize rewrite, name, flags kept bit-3-consistent with data descriptors, modtime/date, EOCD field corruption, entry swap/duplicate/delete, data truncate; local offsets recomputed on serialize; zip64 sentinels rejected), **BER/DER** (`core/mutations/der.py`; sniffer-gated on a leading `30`/`31` SEQUENCE/SET tag with a plausible length byte — `der_len_mutate` short<->long form flips / shrink / grow / BER-indefinite lengths, `der_tag_mutate` class/constructed-bit/number/2-byte-tag corruption, `der_tlv_reorder` sibling shuffle/duplicate/remove inside constructed values, `der_tlv_insert` fresh or truncated TLV splicing; byte-minimal re-serialization keeps untouched subtrees verbatim, ancestor lengths cascade-recompute; depth-capped parsing keeps opaque leaves instead of failing — targets X.509 / EC-key / ECDSA-signature material like `secp256k1_read.c`), **x86/x86-64** (opcode-class swap via primary/0F tables, modrm field flips, imm/disp mutate with field-width clamping, prefix toggle, NOP replace, insn delete/duplicate/swap/splice via length-only decoder with 1-byte resync on unknown opcodes), **ARM** (A32/T32 word bitflip/arith/interesting, condition-nibble, branch imm24, register-field flips, word swap/duplicate/delete/truncate, T32 pair mutation with thumb-stream heuristic), and **format lock** (magic-prefix detection with protected-byte-tail-havoc for autoprobe targets)
- **Regularity operators** (`core/mutations/structured.py`, category `regularity`): fourteen constructive inverses of the diehard/dieharder statistical battery. Each dieharder test defines a statistic `S` over a byte stream with a known distribution under the uniform null and asks whether `S` is improbably far from the mean; these operators build buffers whose `S` sits in the far tail instead — the region random havoc never reaches, and where table-driven parser and algorithm fast paths live. All are length-preserving so they compose with the length-changing operators rather than competing with them, and all stay on the `RandPool`/stdlib-`random` intersection like the rest of `core/mutations/`. Most overwrite a bounded region (`MAX_REGION` = 4096); `kmer_saturate` tiles the whole buffer, because the missing-k-mer count it drives is a property of the buffer a reader samples, not of one region inside it. Round trips are asserted against the matching detectors in `core/randomness.py` wherever one exists:
  - `gcd_worst_case` (← `marsaglia_tsang_gcd`): consecutive Fibonacci pairs, which maximise Euclid's iteration count below a bound (Lamé's theorem) — ~92 steps for a 64-bit pair against ~38 for uniform operands. Drives `gcd`/`av_reduce`-style loops to their worst case: rational normalisers, aspect-ratio and timebase code, bignum fast paths. A multiplier is applied half the time so the gcd is non-trivial and the "common factor found" branch is exercised too.
  - `monotone_fill` (← `dab_filltree`): strictly monotone runs of fixed-width words, degenerating a BST or interval map into a linked list — symbol tables, ZIP central directories, font cmaps.
  - `kmer_saturate` (← `diehard_opso`/`oqso`/`dna`/`bitstream`): de Bruijn sequence B(k,n) via iterative FKM, so every k-mer of order `n` appears exactly once. Maximum branch diversity per byte through a lexer or DFA; zero missing words against the ~141,909 a uniform stream leaves.
  - `kmer_saturate_bits` (← `diehard_bitstream`, bit-exact): same FKM generator over a binary alphabet, but packed one symbol per *bit* instead of per byte. `kmer_saturate`'s `k=2` shape only ever varies the top bit of each output byte (the `step` scaling in `de_bruijn_bytes` holds the low 7 bits at zero), so it exhausts every n-bit window only at byte-aligned phase; a bit accumulator that starts reading mid-byte — Exp-Golomb fields in an H.264 RBSP, protobuf varint continuation bits, any packed bitfield struct — never sees most of its reachable states. Bit-tight packing makes the exhaustive-window guarantee hold at all 8 phases, and costs 8x less buffer to do it.
  - `kmer_starve`: the opposite tail — a 2–4 symbol alphabet that locks a state machine into one region of its transition table and holds it there.
  - `rank_deficient` (← `diehard_rank_32x32`/`rank_6x8`): GF(2) matrices built as XOR combinations of a small basis, forcing rank ≤ rows/2. Snapped to a block boundary, because an unaligned run of dependent rows reads as full rank again at the reader's own alignment. Reaches the "not invertible" branch of erasure/Reed-Solomon and LDPC decoders.
  - `perm_lock` (← `diehard_operm5`, `rgb_permutations`): sorted, reverse-sorted, all-equal, organ-pipe and half-interleaved word sequences — the classic O(n²) inputs for comparison sorts over parsed records.
  - `lag_correlate` (← `rgb_lagged_sums`): exact periodicity at a chosen lag, which is also the maximum possible LZ77 match length — decompression-bomb shape, RLE and delta-filter pathologies.
  - `spectral_peak` (← `dab_dct`): spectrally degenerate blocks (pure cosine at an 8-point bin, DC-only, Nyquist alternation, impulse, full-scale alternating AC) that pin the position of the largest DCT coefficient and reach IDCT saturation/clamping arithmetic.
  - `birthday_collide` (← `diehard_birthdays`, `dab_birthdays1`): words in arithmetic progression, so every birthday spacing is identical — the hash-flooding shape for tables, dedup logic and bloom filters.
  - `invariant_break` (← `rgb_persist`): the inverse of the usual protection — freezes the bytes the corpus varies and scribbles only on the ones it never does (magic numbers, version fields, structural constants), whose validation code is therefore least explored. Uses both fully-locked offsets and partial masks, writing only the locked bits so the varying bits stay self-consistent. Gated on a corpus of ≥16 samples; the `CorpusInvariants` scan is cached and rebuilt only on corpus growth.
  - `degenerate_geometry` (← `diehard_parking_lot`, `rgb_minimum_distance`): coincident, collinear and origin coordinate tuples — the div-by-zero/NaN inputs for hull, triangulation, collision and area code. Snapped to the point stride for the same reason as `rank_deficient`.
  - `float_squeeze` (← `diehard_squeeze`): IEEE-754 patterns that break convergence loops — one ulp either side of 1.0, denormals, infinities, quiet and signalling NaN payloads, negative zero, with f64/f32 encodings paired by value class rather than truncated.
  - `popcount_lock` (← `diehard_count_1s_byte`/`_stream`): bytes pinned to a single Hamming weight, collapsing the five-letter popcount alphabet to one — bit-packed formats, UTF-8/Base64 validity classes, ECC and constant-weight codes, SIMD popcount scalar tails.
  Note on provenance: dieharder is GPL-2 and this tool is MIT; the module contains none of its code. The constructions derive from the public test descriptions (Marsaglia's `tests.txt`, the dieharder manual) and the underlying combinatorics.
- **Tree mutator** (`lightweight_tree_mutate`): Radamsa-style delimiter-based tree mutations (delete, duplicate, swap, stutter) with correct round-trip invariant — unmatched delimiters are preserved, never healed
- **FrameShift**: automatic length-field tracking — discovers and adjusts length/count fields during insertions/deletions, applied as universal post-processing after every mutation
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus, generate statistically similar inputs, persist across runs
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Surprisal-weighted rewards**: all scheduling mechanisms (bandit, MOpt, Replicator, Elo) weight discovery credit by `1 - bitmap_density` — rare discoveries in sparse coverage regions get more credit than discoveries near already-saturated areas
- **Perplexity-gated generation**: model quality dynamically scales generation rate (more generation when model is lost, less when well-calibrated); rejects extreme-perplexity outputs as pure noise

### Thermodynamic Scheduling & Corpus Admission

- **Simulated annealing temperature**: `self._temperature` decays linearly from 1.0 to 0.1 over `--anneal-budget` iterations. Originally used only for Monte Carlo CEM elite retention (Metropolis criterion in `add_elite`), now also governs Boltzmann seed selection and Metropolis corpus admission.
- **Boltzmann seed selection** (`--boltzmann`, requires `--anneal-budget > 0`): replaces the 15-signal weight-soup (`_compute_weights()`) with a single formula — `P(seed) ∝ exp(-E/T)` where `E = log(fuzz_count + 1)`. Rare seeds (low hit counts) get exponentially more weight as T cools. At T=1.0 (hot), all seeds have roughly equal probability. At T=0.1 (cold), rare seeds dominate by factors of ~10^(fuzz_count_ratio). Implemented as a separate Elo-eligible strategy (`_pick_boltzmann_seed()` in `seed_picker.py`) alongside `"weighted"`, `"pareto"`, `"markov"`, etc.
- **Metropolis corpus admission** (`--metropolis`, requires `--anneal-budget > 0`): swaps the hard `is_new_coverage()` boolean gate for probabilistic acceptance — non-improving inputs are admitted with `P = exp(-ΔE/T)` where `ΔE = 1.0` (unit cost for any non-covering input). At T=1.0: 37% of exploratory junk passes through (may unlock paths later). At T=0.1: effectively 0%, converging to today's strict coverage-only rule. Implemented in `fuzz_one()` (fuzzer.py) after the existing coverage/crash gate.
- **Fluctuation-theorem diagnostics** (`--fluctuation-theorems`): tracks mutation trajectories and computes Jarzynski/Crooks estimators over operator-selection probabilities. The work functional is `w_i = -log(max(p_i, ε))` per operator step, with trajectory work `W(τ) = Σ w_i`. `WorkFunctional` accumulates online means of `exp(-βW)` per state (keyed by hit-edge-set hash) and reports `ΔF̂` estimates on the stats line. Phase 1 is diagnostics-only: estimates are not fed back into scheduling. State persists under `StateStore["fluctuation"]` and survives `--resume`. See `core/fluctuation.py` (`TrajectoryRecord`, `WorkFunctional`, `snapshot`/`restore`, `jarzynski_estimator`, `crooks_forward_reverse`) and `services/stats.py` for the periodic display.

### Directed-Distance & Power Schedules

- **AFLGo directed distance** (`--target-functions`, accepts function names, hex addresses, or `file.c:line`): `core/distance.py` implements AFLGo's exact two-level distance. `d_cg(f) = |T_f| / Σ_t 1/(1 + d_bfs(f,t))` — the harmonic mean of (1 + shortest call-graph path) over reachable target functions via reverse BFS. **CFG distance** `d_cfg(b) = |T_b| / Σ_t 1/(1 + d_cfg_path(b,t))` — the harmonic mean over target basic blocks within a function's intra-procedural CFG (built by `core/cfg.py` from the pure-Python x86-64 decoder: block splitting at control-flow, successors, callsite map; decoder gaps like `ret imm16`/`loop` compensated). Target blocks get distance 0; blocks in functions without target blocks fall back to the 0-based CG distance. `file.c:line` targets are resolved through pure-Python DWARF parsing (`core/dwarf.py` — `.debug_line`/`.debug_info`/`.debug_abbrev`/`.debug_str*` for DWARF 4 and 5, incl. compressed sections, strx indexed strings, and LLVM's format-before-count v5 file tables). The abbrev table is scanned for the DIE's abbrev code rather than assuming the first entry is the CU's (gcc orders helper DIEs like formal_parameter first), and `DW_FORM_implicit_const` embedded SLEB values in the abbrev table are skipped — this fixes DWARF 5 binaries from gcc 13+/14+ defaults, which previously failed to load silently (`load()` returned False and `file.c:line` targets resolved to nothing). The old implementation read `st_size` at the wrong Elf64_Sym offset (24 instead of 16) and used file offsets as virtual addresses — both fixed; the ELF loader now translates vaddr→file offset via PT_LOAD segments (PIE/.so-correct).
- **AFLGo power schedule** (`--schedule aflgo`, `--aflgo-cooling exp|log|lin|quad`, `--t-x MINUTES`): the exact `calculate_score()` distance section from AFLGo's afl-fuzz.c. Temperature `T` follows the chosen cooling over t_x minutes to exploitation (`exp`: 1/20^progress; `log`: 1/(1+2·ln(1+progress·13358.7268297)); `lin`: 1/(1+19·progress); `quad`: 1/(1+19·progress²)); `p = (1−nd)(1−T) + 0.5T` with `nd = (d−min)/(max−min)` normalized over the observed queue (min/max tracked per run, excluding the no-data sentinel); `factor = 2^(2·log2(32)·(p−0.5))` — symmetric around 1.0: early campaigns treat every seed equally, late campaigns give near-target seeds up to 32× energy and far seeds as little as 1/32×.
- **AFLGo SHM-tail distance channel**: distance builds (`build_targets.sh --distance`, target compiled with `-fsanitize-coverage=trace-pc -D__AFL_DISTANCE_MODE`) have the shim accumulate per-block distances in `__sanitizer_cov_trace_pc()` — the PC (relative to the dladdr-derived object base) probes an open-addressing table of `{key, dist}` entries (packed 12-byte layout; the 4-byte header holds the slot *capacity*, a power of two ≥ 2×entries so empty slots exist, and the builder hash-inserts at `key % capacity` with linear probing to mirror the shim's probe — uploaded by the fuzzer at startup via `DistanceTableShm`/`__AFL_DIST_SHM_ID`), accumulating sum/count into the 16-byte SHM **tail** (after the edge table: `u64 dist_sum`, `u64 dist_count`), written at reset, at process exit (subprocess runs never call reset), and per-iteration in in-process modes via `__afl_dist_flush` (direct_lite has no process boundary, so the runner flushes the tail after each `run_one`). Per-execution `avg_distance = sum/count/100` is read straight from the tail and preferred over Python-side computation; blocks without a table entry don't count (AFLGo semantics). The table's PC keys are recovered by scanning text for `call __sanitizer_cov_trace_pc` sites (modern clang emits no `__sancov_pcs` for trace-pc) mapped to valued blocks via the CFGs — `TargetDistance.pc_distance_table()`. The shim's sanitizer-coverage callbacks are hidden-visibility so a libasan LD_PRELOAD cannot interpose over them in PIE builds. `tools/gen_distance_table.py` emits the table as C or text for inspection. Without the table (count==0) everything degrades to the Python-side path. Works in subprocess, direct_lite, and persistent modes (ASAN direct_lite requires libasan preloaded at fuzzer-process start — the `use_direct_lite` gate). The periodic stats line shows live distance when directed mode is active: `dist: avg:<tail avg> min:<observed min> max:<observed max>` (or `no-data`). `build_targets.sh --distance` builds both `*_dist.so` (no-ASAN) and `*_dist_asan.so` (ASAN) variants with the cmplog shim linked in, so `--cmplog` keeps them in direct_lite mode. Startup reports `[*] Distance instrumentation: detected` when the target carries the channel (the shim's `__afl_dist_flush` or a defined `__sanitizer_cov_trace_pc`), mirroring the AFL-instrumentation check. With `--elo` in directed mode, `aflgo` joins the Elo-arbitrated seed-strategy pool — a distance-pure arm picking `P(seed) ∝ exp(-2·norm_dist)` (distinct from the generic `weighted` arm, which blends distance with speed/size/entropy).
- **AFLGo distance-annealed schedule** (`--schedule go`, requires `--target-functions`): wires the precomputed `avg_distance` (per-seed distance to directed targets) and `_anneal_progress` (exploration/exploitation annealing variable) into `SeedScorer.score()` for mutation budget scaling. During exploration phase (`anneal_progress` ≈ 0): uniform energy. During exploitation phase (`anneal_progress` → 1): `energy *= exp(β · (1 - norm_dist))` where `β = anneal_progress * 5`, capping at 100x. Seeds near the target get exponentially more mutations as the campaign matures. Previously these metrics only influenced seed selection but not mutation intensity.
- **Power schedules** (`--schedule base|fast|coe|rare|mopt|lin|quad|go|aflgo`): AFL++ power schedules ported to control mutation budget per seed via `SeedScorer`. Each schedule modifies a base score (100) by frequency-based factors. Honggfuzz-style novelty decay, density, fertility, freshness, and entropy factors are applied multiplicatively on top.
- **Bayesian seed quality** (`--bayesian`): `BayesianSeedQuality` (`core/seed_quality.py`) maintains a Beta-Bernoulli posterior per seed over `P(outcome = new_coverage)`. Thompson sampling naturally balances explore/exploit without a manual temperature knob — unexplored seeds have high posterior variance and get sampled. The `record_outcome()` feedback loop is now wired in `fuzz_one()` (was previously a dead code path with all posteriors stuck at Beta(1,1)). State is persisted to `seed_quality.json` and restored on resume.

### Static Target Analysis
- **TargetProfiler**: ELF static analysis at startup — extracts string constants, function boundaries, magic bytes, and input format hints
- **Auto-populated dictionary**: interesting strings (format specifiers, error messages, keywords) and magic bytes extracted from `.rodata`
- **Format-aware seed generation**: produces structurally meaningful initial seeds (PNG headers, text protocols, JSON, XML, HTML) based on inferred format
- **Informative Bayesian priors**: `format_operator_priors()` seeds the Thompson-sampling bandit's Beta prior toward structure-aware operators (e.g. `png_chunk_mutate`) and dictionary operators when static analysis detects a matching format or extractable tokens, instead of always starting from the uninformative Beta(1, 1)
- **Hot-function weighting**: seeds exercising high-branch-density functions get a proportional boost in selection
- **Crash ETA estimation**: blends static risky density (keyword heuristic) with dynamic I(byte_position; crash) mutual information from actual executions, plus Good-Turing edge estimates and calibrated discovery rate — the MI signal strengthens as fuzzing accumulates near-miss data

### Coverage & Scoring
- **AFL count-class bucketization**: `classify_counts()` collapses raw hit counts into 9 logarithmic buckets (0, 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+) before coverage comparison — eliminates noise from count-magnitude jitter and provides cleaner signal for JS-divergence/Wasserstein diversity scoring; `new_bits()` provides AFL-style overlap/new-edge detection on classified bitmaps
- **Morris probabilistic counting (a=30)**: log-scale edge hit counters prevent overflow and provide frequency information for scheduler decisions; estimate formula `a * ((1+1/a)^v - 1)` converts back to approximate counts
- **AFL SHM bitmap** coverage for instrumented targets (~65-200 eps). **Sparse 8-byte entry hash table** replaces the traditional fixed-size byte bitmap — each SHM entry stores `{edge_id: uint32_t, count: uint32_t}` with open-addressing linear probing. Edge IDs are full 32-bit `(prev_loc ^ cur_loc)` values, eliminating silent bucket collisions. The hash table load factor replaces birthday-collision as the resize signal. No Morris counting needed (32-bit saturating counters). **Auto-resize on stall** is enabled by default (`--resize-map-on-stall` / `--no-resize-map-on-stall`) — when load factor exceeds 0.7, SHM grows to reduce collision risk and expose new edges.
- **SHM front-header metadata**: the SHM segment begins with a fixed 24-byte header (offset 0–23), followed by the edge table at offset 24, followed by a 16-byte **distance tail** (offset 24 + table_bytes: `dist_sum` u64, `dist_count` u64 — written by distance builds, always allocated by Python, zeroed on reset). Layout: `stack_depth` (u32 @0), `_pad0` (u32 @4), `path_hash` (u64 @8), `edge_count` (u64 @16). The front header is never moved, even when the edge table is resized — only the table grows, keeping metadata access O(1) and resize-safe. `edge_count` is a **cumulative** insert-only counter that survives across executions: the C shim maintains a static `__afl_total_edge_count` that is never reset, and writes it to the SHM header on each new-slot insertion. This provides a correct O(1) fast-path for coverage-change detection — `is_new_coverage()` reads 8 bytes and compares against the last known value; if the count differs, the slow path determines whether any edge_ids are genuinely new. The previous per-execution counter caused false-negatives when consecutive executions touched the same number of distinct edges (fixed).
- **Ptrace edge coverage** with deep x86-64 decoder disassembly for closed-source binaries (~18-20 eps)
- **In-process execution**: persistent subprocess mode (~65-120 eps) with auto-restart on crash
- **Stack depth tracking**: SHM front-header (offset 0) tracks max stack depth per iteration via `__sancov_lowest_stack` hook (C shim) or approximation from edge count (Python fallback)
- **Path hash**: rolling 64-bit hash (`hash = hash * 31 ^ edge_id`) maintained in SHM front-header (offset 8) for collision-resistant path identification and seed diversity scoring
- **Hardware perf counters** (`--hw-perf`): `perf_event_open(2)` for instruction count, branch count, branch misses via `CAP_PERFMON` — provides execution-depth signals beyond edge coverage
- **Length-edge tracking**: correlates input length with coverage edge discovery — biases seed selection and length-changing mutations toward productive lengths
- **Per-target SHM coverage**: multi-target mode tracks coverage independently per target binary
- **Cross-target seed scoring**: seeds that found edges in the least-covered target get boosted proportionally to the coverage gap
- **Branch density**: per-target static analysis metric (conditional branches/KB) with average across targets
- **Auto-sized edge bitmap**: `estimate_map_size()` from branch density × .text size replaces hardcoded 65536
- **Good-Turing estimation**: prospective edge discovery count with saturation confidence
- **KS significance testing**: replaces fixed JS thresholds with sample-size-aware p-values
- **CRPS scoring**: proper scoring rule for execution time calibration (fixed indicator direction bug); computed per-execution as a numpy-vectorized prefix recurrence (3.9x, equivalence-tested vs the legacy walk)

### Distribution Diagnostics
- **Running statistics** (`core/running_stats.py`): Welford/Pébay online algorithm for O(1) mean, variance, stddev, skewness, and excess kurtosis — unbounded or sliding-window variants
- **Execution time tail-risk detection**: skewness > 2.0 flags algorithmic-complexity inputs (regex backtracking, hash-flood) that occasionally trigger big execution-time excursions
- **Critical slowing down skewness tier**: three-signal detector (variance + autocorrelation + skewness) — rising right skew upgrades the verdict to "approaching transition, and it looks productive"
- **Chi-squared test suite** (`core/chi_squared.py`): four test families — goodness-of-fit, homogeneity, independence (contingency-table), and p-value via regularized incomplete gamma (series + Lentz's continued fraction). Supports Cramér's V effect-size measurement on any contingency table. Powers the operator heterogeneity test and coverage-column homogeneity detector.
- **Operator success heterogeneity test** (`--chi2-operator-interval N`): periodic diagnostic that builds a 2×K contingency table from per-operator `op_counts`/`op_success`, runs chi-squared independence, and logs at `info` when p<0.05 — detecting when some operators consistently outperform others. Effect size reported via Cramér's V.
- **Coverage-column homogeneity detector** (`CoverageHomogeneityDetector` in `critical_slowing.py`): tracks per-column edge discovery totals over a sliding window and tests spatial uniformity of coverage via chi-squared goodness-of-fit. When homogeneity is rejected, the fuzzer's coverage is spatially clustered — edges concentrate in a subset of columns, which may signal biased exploration.
- **Per-operator reward moments**: UCB-style exploration bonus (`mean + k * stddev`) with kurtosis-scaled stability guard — high-kurtosis operators require more observations before trusting their stddev-based bonus
- **Format learner z-score gate**: replaces fixed `delta != 0` threshold with z-score-based outlier detection; MAD fallback under high kurtosis for robustness against zero-inflated coverage deltas
- **Corpus bloat early-warning**: rising right skew in seed file sizes is a leading indicator of bloat that precedes RSS threshold tripping
- **Bounded memory structures**: all accumulative data structures (correlation matrix, coverage timeline, cmplog tokens/pairs, Shapley attribution edges, stderr buffer, seed secretary, seen hashes) are capped via module-level constants — RSS plateaus instead of growing linearly with exec count
- **Report distribution diagnostics**: stddev, skewness, and kurtosis for exec time, discovery rate, per-operator rewards, and seed sizes

### Statistical Region Profiling (`--region-profile`)
- **Per-seed window labelling** (`core/randomness.py::profile_buffer`): slides the dieharder-derived battery (monobit, runs, byte chi-square, serial, 32×32 binary rank, k-mer occupancy, birthday spacings, lagged autocorrelation) over 4 KiB windows and labels each `incompressible` / `tabular` / `textual` / `repetitive` / `mixed`, with a Shannon and printable-fraction cross-check.
- **Position weighting**: `RegionProfile.mutation_weight()` turns each label into a multiplier on that region's byte-selection probability — `incompressible` 0.15 (bit flips inside a deflate payload die at the CRC), `tabular` 1.6 (offset and length fields are where arithmetic operators pay off), `textual` 1.3, `repetitive` 0.6. `OperatorEngine._region_weighted_position()` samples a region proportionally to weight × length, then a uniform offset inside it, and joins the existing MI/TE/sensitivity/crash-MI candidate set in `select_position()`.
- **Caching**: the battery costs ~1 ms per 4 KiB window, so profiles are cached per seed by content hash in `OperatorEngine._region_cache` (bounded at 64 entries, cleared wholesale on overflow — the profiles are cheap to rebuild and an LRU would cost more bookkeeping than it saves). Off by default: worth paying only on structured targets.
- **Buffer clamp**: region bounds come from the seed, but the position must land in the buffer, which earlier operators in the same mutation round may already have resized.

### FFT Periodicity Detection
- **Record-size inference** (`core/periodicity.py::estimate_record_size`): Wiener-Khinchin autocorrelation via FFT (`irfft(|rfft(x - mean)|²)`, O(N log N)) with Hanning windowing and lag-0 normalization. Returns the smallest lag with a locally-dominant autocorrelation peak — the inferred record stride in bytes, when one exists. Significance gated by `max(min_rel_peak, 4.0/√n)` (multiple-comparisons-aware ~4σ bound over ~n/3 scanned lags; white-noise ac[k] ~ N(0, 1/n)). Returns `None` for genuinely aperiodic input (e.g. iid-random record contents) rather than a spurious stride. **Analysis window cap** (`DEFAULT_MAX_WINDOW` = 16 × `DEFAULT_MAX_LAG` = 4096 bytes): the lag scan never exceeds lag 256, so only the first 4096 bytes are FFT'd — a 2.8 MB seed cost ~2.4 s/call uncapped, ~0.4 ms windowed, with byte-identical results for buffers ≤ 4096 bytes (the sigma bound uses the window length, which calibrates that window's own noise floor)
- **Consumers of the inferred stride**: `chunk_shuffle(data, stride=...)` uses it as the chunk size when `stride >= 2` and `len(data)//stride >= 2` (50% probability gate in the operator handler, else legacy `randint(1,4)`); `format_learner` receives it via `set_record_stride()` and adds a 0.05 confidence boost to stride-aligned field hypotheses; `TreeMutator.parse(chunk_size=...)` overrides its inferred chunk-size fallback; BMP WFC uses it as pixel-tile width when it divides the row stride evenly (else per-pixel fallback). PNG WFC is a documented no-op (chunk-type semantics, no byte alignment)
- **Seed metadata persistence**: `record_stride` computed eagerly per seed (a pure function of seed bytes — idempotent, no double-counting) and round-tripped through `state.pkl.gz` (via `StateStore`) save/load allowlists and the corpus save path
- **Spectral time-series diagnostics** (`services/report.py::_spectral_diagnostics`): rfft power spectra (DC bin excluded, `dominant_period = n / peak_bin`) on two series — exec-time samples (`ExecutionTimeTracker._times`) and discovery-rate (first-differences of cumulative edges per sync interval, min 50 samples). A discovery-rate peak at a short period flags possible corpus-sync artifacts imitating genuine discovery waves
- **Hidden-periodicity significance** (`core/periodicity.py::detect_periodicity`): the dominant non-DC bin is scored with Fisher's g-test — largest periodogram ordinate / total power against the exact closed-form null (`P(G > g) = Σ_k (-1)^(k-1) C(m,k) (1-kg)^(m-1)`, `m = floor((n-1)/2)` full bins, DC and Nyquist excluded from the peak search, terms in log space for large `n`). Rejects white noise at the nominal alpha rate (default 0.05) instead of a fixed peak/median ratio, and the sum denominator keeps `peak_strength` bounded in [0, 1] on clean signals
- **No-scipy discipline**: FFT work stays within numpy (`np.fft.rfft`/`irfft`, `np.hanning`) — real inputs only, Hermitian symmetry exploited (~2× faster, half the memory)

### Checksum Learning (Berlekamp-Massey)
- **Core algorithm** (`core/berlekamp_massey.py`): the Berlekamp-Massey algorithm over GF(2) finds the shortest LFSR generating a binary sequence (`berlekamp_massey(bits) -> (L, C)`); *L* (linear complexity) is a structuredness metric that complements Shannon entropy and FFT periodicity — a sequence can look uniform under both while still being generated by a short linear recurrence. `recover_lfsr(values, width)` recovers a reflected-form connection polynomial from sequential LFSR state values (the lane-0 bit stream, one step apart). `compute_checksum(data, poly, width, init, final_xor, reflect_in, reflect_out)` computes a CRC-like checksum with a recovered polynomial in either shift convention (non-reflected MSB-first with the normal-form poly, or reflected LSB-first with the reversed-form poly, e.g. `0xEDB88320`); with the full zlib configuration (reflected, init/final_xor `0xFFFFFFFF`) it reproduces `zlib.crc32` exactly. `recover_polynomial_gcd(pairs, width)` is the independent-pair recovery path: the generator divides every syndrome `M(x)·x^W + C`, so the GCD of syndromes over GF(2) recovers it (the result may carry extra factors — round-trip verification against observed pairs is the authoritative check, and masking to *width* bits makes multiples equivalent for computation).
- **ChecksumLearner** (`core/checksum_learner.py`, attached as `f.checksum_learner`): collects `(data, checksum)` pairs from three sources — (1) **format-aware extraction** (`_extract_png_pairs` parses PNG chunks and yields `(chunk_type + chunk_data, crc)`; `_extract_zip_pairs` uses the local-file-header crc32 field; `_extract_gzip_pairs` reads the trailer crc — payload extraction for gzip members is a known limitation, yielding placeholder pairs), (2) **cmplog heuristics** (`extract_cmplog_pairs` scans `f._cmplog.pairs` for 4-byte operands found in the input buffer, pairing data-before-checksum), (3) **state restore**. Recovery order is **GCD first** (works for independent pairs — the normal case for format extraction), **BM fallback** (sequential LFSR outputs, rarely satisfied by realistic corpora). A candidate polynomial is only accepted when it reproduces ≥ 2 distinct observed checksums (`_verify`), which rejects the GCD residue that mismatched `(init, final_xor)` configurations produce (e.g. standard PNG CRCs use `final_xor=0xFFFFFFFF` and are deliberately not recovered — the standard poly needs no recovery). Recovery is re-attempted only when the pair set has grown since the last attempt (`ensure_poly` guards on `_pairs_attempted_at`): the `crc_learn` availability gate calls `ensure_poly()` once per fuzz iteration, and re-running the full GCD/BM recovery on every call for pairs that cannot verify collapsed eps to single digits (~56 s of a 103 s fuzz profile). On success the model is activated module-wide via `set_active_model` and persisted to `state.pkl.gz` (via `StateStore`) as the `checksum_learner` section.
- **`crc_learn` operator** (`core/operator_registry.py` adaptive band; availability gated on `f.checksum_learner.ensure_model()` — either a recovered GF(2) polynomial or a recovered integer-modulus model): `_op_crc_learn` (`services/operators.py`) patches checksum fields using the recovered polynomial — format-aware for PNG (re-serialize every chunk CRC) and ZIP (patch the local-file-header crc32 field), falling back to last-4-bytes patching for unknown formats. The format-aware patchers stay on the GF(2) polynomial by design (PNG/ZIP CRCs are CRC-32 by specification); a recovered integer model drives only the generic trailing-field patch, sized from the model's own field width so a 2-byte Fletcher-16 field is not zero-padded into 4 and clobbering real data.
- **Integer-modulus checksum recovery** (`core/int_checksum.py`, `core/int_checksum_solver.py`): the GF(2) machinery above cannot represent Adler-32, Fletcher-16/32, or bespoke `sum(data[i] * k^i) mod N` schemes — these are integer-linear over Z/NZ, not GF(2)-affine — so a target gating on one had that field permanently rejected and every path downstream of validation unreachable by mutation. `int_checksum.py` is the dispatch module (parallel to `crc32.py`, same lock pattern): an `IntModel` is `(kind, modulus, multiplier, init_a, init_b, word_bytes, out_bits, big_endian)` where `kind` is `weighted_sum` or `fletcher` (the latter covers Adler-32, which is the same two-running-sums shape with a prime modulus). Both families evaluate in closed form — `a_n = init_a + S1` and `b_n = init_b + n*init_a + S2 (mod N)` — since reduction mod N is a ring homomorphism, so the running loop is never needed; `S2` is a vectorized index-weighted dot product with an int64-overflow guard above `2**21` words. `int_checksum_solver.py` recovers unknown models: a common-model fast path (Adler-32/Fletcher-16/32/plain sums) first, then general modulus recovery.
- **How modulus recovery works**: with `c_i = R_i + b (mod N)` and `R_i` the exact unreduced raw sum, each pair gives `R_i - c_i = N*q_i - b`. The congruences are exact and the checksum field is fully observed, so differencing kills `b` outright: every pairwise difference is an exact integer multiple of `N`, and `gcd` over them recovers `N` (or `m*N` for a small `m`, from which `N` falls out by cofactor enumeration). Exact, microseconds, no C extension.
- **Preconditions and failure modes**: the raw sum must actually *wrap* the modulus for it to be visible (for `k=1`, pairs of ≳ `2N/255` bytes, i.e. ~512 for Adler-32) — below that the modulus is genuinely under-determined and any `N` above the largest observed sum fits the evidence equally well. This is why the integer path must **not** reuse the GF(2) path's `_GCD_MAX_PAIR_DATA_BYTES = 256` cap, which would filter out precisely the pairs carrying the signal; it instead prefers the *longest* pairs, and its cost is bounded differently (with `multiplier=1` the raw value is one small int regardless of length). Recovery is fail-closed: across 2400 trials the failures were only `gcd == 0` (no wrap) or `gcd == m*N` for small `m` (recovered by cofactor enumeration), never a wrong modulus. `min_pairs` was characterized empirically (~12 for reliable exact recovery), not inherited from the GF(2) path.
- **Outlier tolerance** (`_consensus_candidates`): a single-chain GCD is all-or-nothing — one corrupt pair drags it to 1 and recovery fails outright (measured 0/200), which matters because `extract_cmplog_pairs` is explicitly heuristic and *will* emit bad pairs. The consensus path takes GCDs over deterministic index triples (most of which avoid any given bad pair) and scores each candidate by congruence-class size: every honest pair satisfies `r_i = -b (mod N)`, so the true modulus collapses all good residuals into one class while a corrupt pair joins only with probability `1/N`. Restores 199-200/200 with 1-3 corrupt pairs out of 12-24, with no regression on clean sets. `init` offsets are likewise taken by majority vote rather than anchored on a possibly-corrupt pair 0, and modulus lower bounds count how many pairs fit rather than taking `max()` (an outlier could otherwise inflate the bound past the true modulus).
- **Integration**: `ChecksumLearner` gains `_int_model` alongside `_poly`, attempted only after every GF(2) path fails to verify (the families are disjoint, so a verified GF(2) model is definitive). The same >= 2-distinct-pair gate guards activation, tightened to 4 where pairs allow, plus a requirement that matched pairs carry >= 2 distinct checksum values so a degenerate constant-output model cannot pass. `_extract_zlib_adler_pairs` reads the Adler-32 trailer from an embedded zlib stream (PNG IDAT payloads, or a bare stream) — one layer below the chunk CRC-32 already extracted, with bounded inflate output so a zip bomb cannot stall `fuzz_one()`. `compute_int_checksum()` is deliberately **separate** from `compute_checksum()`: the latter is called by `_patch_png_crc`/`_patch_zip_crc` meaning "the CRC-32 this format specifies", and routing an integer model through it would write an Adler-32 into a CRC-32 field and silently corrupt every mutated PNG. The model is persisted to the `checksum_learner` state section as `int_model` and re-activated on restore, with type validation so a corrupt state file yields `None` rather than a model that explodes later inside a mutation.
- **Configurable CRC-32 dispatch** (`core/crc32.py`): the single point of dispatch for all CRC-32 operations. `crc32(data)` uses the module-level active model — `None` or the standard poly shortcuts to hardware `zlib.crc32` (hot path unchanged), any non-standard recovered model uses the software LFSR (`compute_checksum`). `set_active_model`/`set_active_poly` (with `threading.Lock` for worker-thread safety) are called by the fuzzer on recovery and on `state.pkl.gz` restore. Format mutators (`png.py`, `zip.py`, `gzip.py`), the SMT PNG helper (`smt_solver.py`), `seed_picker.py`, tool scripts, and `edge_tracker.py` (which uses the explicit `crc32_ieee` alias so MinHash bucket keys never shift with a learned model) all import from here instead of calling `zlib.crc32` directly.
- **Known boundary**: MT19937 (used by `np.random.seed()` in RandPool) is a twisted GFSR with tempering, not a plain LFSR — standard BM does not apply; recovery there would require untempering + twist inversion, which is out of scope.

### Scheduling Intelligence
- **Operator registry** (`core/operator_registry.py`): single source of truth for mutation operators. Every operator (name, category band, availability predicate, `_op_<name>` handler) is registered exactly once in `REGISTRY`; `OperatorEngine.build_dispatch()`/`build_ops()` (`services/operators.py`), `_register_arms()` (`services/fuzzer.py`), and `OPERATOR_CATEGORIES` (`core/operator_categories.py`) all derive from it. Schedulers never hardcode operator lists — they learn available ops through the services layer, which queries the registry. Adding an operator means one registration plus its handler; the legacy `MUTATIONS`/`FORMAT_MUTATIONS`/`DICT_MUTATIONS` lists in `core/mutations/generic.py` are backward-compat views only. Registration order is deterministic (category-band order, names sorted within a band) so op selection is reproducible for a fixed seed; gated ops (dictionary, markov, cem, grammar, cmplog, flag, per-input redqueen) carry availability predicates mirroring the historic `build_ops()` conditions, and `colorization` is dispatch-only (registered, never selectable).
- **MB / CBH constraint search** (`core/mb_cbh.py`): Angora's `search/mb.rs` (magic-bytes) and `search/cbh.rs` (climb-hill) ported as two operators, `magic_byte_search` and `climb_hill`, both in the adaptive band gated on `_has_cmplog_pairs`. Both optimize the same objective as the gradient-descent port — make some window of the input equal a cmplog operand — but with deliberately different character, which is why all three earn a slot. `magic_byte_search` does no descent at all: it plants the operand verbatim at a candidate site and randomizes a bounded number of bytes around it (never clobbering the planted window), solving one-step the common case where a magic number simply has to appear. `climb_hill` is stochastic: random position in the scored window, single random **bit flip**, accept on improvement. Bit flips rather than whole-byte randomization matter — the objective is Hamming distance, so a bit flip improves it with probability (differing bits)/8, whereas a uniformly random byte value improves with probability ~1/256 and the stuck counter always fires before convergence (measured: 0/200 seeds solved with byte randomization, 172/200 with bit flips). Bit flips also keep CBH distinct from `gradient_descent`, whose ladder takes *arithmetic* steps (±1..±8). **Shared limitation worth knowing:** CBH and `gradient_descent` both locate their window via `_candidate_positions`, which derives sites from byte-value overlap between input and operand — when an input shares no bytes with the operand there is no signal, site selection degrades to random offsets, and the search optimizes an arbitrary window (measured: 163/200 seeds solved with two bytes differing, but 0/200 at the target offset when nothing overlaps, though the climb itself still succeeded at *some* site in 135/200). MB has no such dependency and bootstraps exactly the overlap the other two need, so the pairing is complementary rather than redundant.
- **Class-based mutator interface** (`core/mutator_interface.py`): port of wtf's `Mutator_t` pattern. The function-based operator model (`_op_<name>` on `OperatorEngine` + a registry entry) is unchanged and every built-in still uses it; this adds a parallel path for *self-contained* mutators that carry their own state and register as one object. `MutatorBase` is an ABC with `mutate(data, rng, max_len, **ctx) -> bytes | None` (returning `None` declines, matching the `_op_*` convention), plus optional `on_new_coverage(seed, new_edges)` and `is_available(fuzzer, data)` hooks. `REGISTRY.register_mutator()` wraps an instance as an `OperatorSpec` carrying a `mutator` reference instead of a `handler_name`; `dispatch()` resolves it through `_mutator_adapter()`, which bridges the bytes/rng signature to the `(buf, byte_idx, data)` handler shape and **applies the `max_len` clamp centrally** — third-party mutators are not trusted to do it, since an operator silently growing the buffer past `max_len` is precisely the class of bug that produced the `_op_fuse_this` OOM. Exceptions from both `mutate()` and `on_new_coverage()` are contained and logged so a misbehaving external mutator cannot take down the fuzz loop. `on_new_coverage` is the feedback hook function-based operators have no equivalent for, motivating mutators that adapt to coverage. Scaffolding is opt-in: nothing is registered by default, so the shipped operator table is byte-identical until something calls `register_mutator()`.
- **CondStmt first-class model** (`core/cond_stmt.py`, `adapters/track_parser.py`): Angora-style comparison tracking ported as a first-class data model. `CondStmtBase`/`CondStmt` represent a single comparison (`cmpid`, `op_a`, `op_b`, `width`, `result`, `pc`, `state`) with `CondState` lifecycle (`ONE_BYTE/UNSOLVED/SOLVED/UNSOLVABLE/TIMEOUT`). `offsets` is computed lazily from the current input via `update_from_input()`. Constructed backward-compatibly from existing cmplog pairs (`from_cmplog_pair`) and structured track records (`from_track_record`). Dedup key is `(op_a, op_b, width)` — `cmpid` is excluded because it is a monotonically increasing identity, not a branch identity. The SMT bridge (`smt_solver.py::ConcolicTrace.to_cond_stmts()`) converts raw cmplog text into `CondStmt` objects; the operator handler `_op_condstmt_solve` (`services/operators.py`) solves unsolved comparisons via the SMT engine, gated on cmplog availability. Track parser (`adapters/track_parser.py`) reads Angora text/JSON track files (`cmpid context order arg1_hex arg2_hex size condition [pc_hex]`) and raw cmplog text. Registry entry `condstmt_solve` is in the adaptive band with `_has_cmplog_pairs` gate; `build_ops()`/`build_dispatch()`/`_register_arms()` and `OPERATOR_CATEGORIES` derive automatically from the registry.
- **Seed-level energy multiplier** (`SeedScorer`): AFL++ power schedules (FAST/COE/RARE/MMOPT/LIN/QUAD) scale `mutations_per_input` per seed — fast seeds with high coverage get more mutation attempts, heavily-fuzzed seeds get fewer. Honggfuzz power factors (novelty decay, density, fertility, freshness, CMP progress, entropy penalty, timeout penalty) applied multiplicatively on top of schedule scoring
- **Elo arbitration** (`--elo`): combined operator + seed scheduling via Bayesian Elo rating system. All available strategies (bandit, mopt, replicator, cem, exp3, eps_greedy, hierarchical, gp_ucb for operators; weighted, pareto, format, ga, qea, bayesian, markov for seeds) run in shadow; Elo Thompson-samples which to trust each iteration. Ratings decayed periodically to model non-stationarity. Competition is **group-scoped**: operator schedulers only compete against other operator schedulers, seed strategies only against other seed strategies (namespaced `seed_<name>`), and the random fallback (stall-recovery `random_stall`, random seed pick) is the only non-competitor in either group. `--elo all` enables every scheduler and mutation-stack feature (metropolis, shapley, mi-guided, secretary, wfc, lineage, `--schedule fast`) rather than merely listing them as available; the per-run convergence report prints only schedulers actually selected. **Schedulers are independent**: each scheduler in `core/schedulers/` never imports or calls another scheduler (they compete; only Elo arbitrates between them), sharing only neutral dependencies like the operator→category taxonomy in `core/operator_categories.py` (derived from the operator registry; imported by the hierarchical and gp-ucb schedulers)
- **EXP3 adversarial bandit** (`--exp3`): adversarial bandit algorithm for operator selection in non-stationary reward environments. Uses importance-weighted rewards with exponential weight updates — automatically tracks the best operator even when reward distributions shift over time. Exploration rate `gamma` controls the trade-off (default 0.1, range [0,1]). Window decay discounts old observations to adapt to changing conditions.
- **Epsilon-greedy with annealing** (`--eps-greedy`): classic exploration/exploitation strategy with exponential epsilon decay. Starts fully exploratory (epsilon=1.0) and anneals toward exploitation at rate `decay` (default 0.9995, min epsilon 0.01). Q-values track per-operator average reward with incremental updates. Serves as the simplest baseline for comparing against more complex bandit algorithms.
- **Hierarchical bandit** (`--hierarchical-bandit`): two-level Thompson sampling — first picks an operator category (bit/byte/block/dict/structural/radamsa/format/adaptive) via Beta posterior sampling, then selects an operator within the chosen category. Success/failure feedback updates both the category-level and operator-level posteriors, so operators sharing a category compete less destructively. Arm decay gradually discounts old observations.
- **GP-UCB bandit** (`--gp-ucb`): Gaussian Process Upper Confidence Bound for operator selection using a simplified RBF kernel over one-hot-by-category features. Running-moments track per-operator mean and variance; UCB score = mu + beta * sigma selects operators with high potential. The kernel captures covariance between operators in the same category, allowing information sharing. Unobserved operators get a fixed exploration bonus until `min_samples` (default 3) observations accumulate.
- **CEM in Elo** (`--mc-cem`): when CEM byte distribution is fitted, "cem" competes alongside bandit/mopt/replicator as an operator scheduling strategy — Elo learns when CEM-generated byte values outperform other selection methods
- **Markov-gen in Elo** (`--markov-gen`): Markov chain seed generation is now arbitrated alongside weighted/pareto/ga/qea/bayesian — Elo picks the seed strategy that finds the most new coverage
- **Jaccard index**: average pairwise edge-set overlap (xxhash-fast) for corpus redundancy monitoring
- **FMM-clustered overlap density** (`--overlap-density`): per-seed pairwise edge-set overlap via Fast Multipole Method decomposition — MinHash LSH clusters seeds by coverage similarity, computes exact overlap within clusters and centroid-approximated overlap across clusters. Used as a weight modifier (penalises redundant seeds) or 4th Pareto dimension (`--overlap-mode pareto4d`). O(N·C) instead of O(N²), 4-7× speedup vs naive at N=500-3000 with MAE ~0.04
- **Subsumption weighting**: MinHash-approximated Jaccard for continuous seed deprioritization
- **Hitcount diversity (JS divergence)**: seeds with unusual frequency profiles get boosted
- **Wasserstein spatial diversity**: seeds exploring different code regions get boosted
- **Weight caching**: recomputes only when corpus/edge-count changes (733x speedup on 200+ seeds)
- **Perplexity (MDL codelength)**: structurally novel seeds get 1.0-2.0x weight
- **NCD similarity**: Normalized Compression Distance between corpus entries
- **Simulated annealing**: temperature-scaled exploration/exploitation balance
- **Hamming bitmap distance**: fast byte-level seed-to-seed similarity on edge bitmaps (numpy `count_nonzero` over zero-copy uint8 views above 64 bytes — ~10-100x for large equal-length inputs; genexpr below, matching `levenshtein_align`'s dispatch)
- **Near-duplicate detection**: finds seed pairs with near-identical coverage via Hamming + LSH

### Information Theory
- **Mutual information** (`--mi-guided`): I(byte_position; coverage) guides mutation toward positions that actually control code paths. The joint distribution is **memory-bounded** (`MAX_JOINT_CELLS`, `MAX_EDGES_PER_CELL`, `MI_MAX_POSITIONS` in `core/mi.py`): record() rejects new cells once the joint-cell budget is exhausted (existing cells keep incrementing) instead of evicting — eviction of least-observed positions thrashed (re-observed -> re-victimized). `mi.json` is loaded only with `--resume`, and `max_positions` is capped independent of `max_len`, so a stale oversized file (`--elo all` on a real corpus previously OOM-killed at 7.5GB / 796MB mi.json; RSS now flat ~229MB) cannot re-trigger the blowup. The `sum(edge_marginal)` used by `mi()` is computed via a cached zero-copy `np.frombuffer` view over the `array("Q")` (49-76x faster than Python-level sum; the view is released before the array grows — `array.array` refuses to resize while a frombuffer view exports its buffer). Raw shim edge IDs are full 32-bit hashes (`caller_ctx ^ prev_loc ^ cur_loc`, unmasked by design for collision-free SHM identity), so `record()` folds them into `[0, map_size)` (AND-mask on power-of-two maps, modulo otherwise) before indexing the dense marginal — an unmasked high hash used to force a multi-GB allocation and MemoryError on first sighting
- **Rényi entropy** (`--renyi-weight`): generalized entropy spectrum for seed weighting — boosts seeds exercising rare (cold) edges
- **Rate-distortion corpus minimization** (`--rate-distortion`): optimal compression of corpus preserving coverage diversity
- **Transfer entropy** (`--transfer-entropy`): directional causal flow between byte positions and coverage edges
- **Shannon entropy rate tracking**: global edge-hit distribution entropy sampled periodically; confirms genuine stall (no new edges + flat entropy rate) vs. transient redistribution before activating random-mode recovery
- **Index of Dispersion** (Fano factor, D = σ²/μ): sliding-window variance-to-mean ratio on the incremental edge-discovery rate — resolves Allan variance's blind spot: a buffer full of zeros (genuine stall, D « 0.3) vs. rare bursts (bursty exploration, D › 1.5). D › 1.5 overrides stall recovery; D « 0.3 confirms it with higher aggression. Also available as a standalone `DispersionIndex` class for any per-operator or per-signal dispersion analysis.

### Game Theory
- **Shapley value** (`--shapley`): per-edge frequency-weighted operator attribution — credit distributed proportional to co-occurrence frequency, not naive full credit to all stacked operators
- **Replicator dynamics** (`--replicator`): evolutionary game theory scheduling — operators grow proportionally to fitness, converging to evolutionarily stable strategies
- **MOpt PSO** (`--mopt`): particle swarm optimization over operator distributions (alternative to Thompson sampling)

### Genetic Algorithm Lifecycle (`--ga`)
- **Finite population**: replaces monotonically growing corpus with bounded, evolving population (`--ga-pop-size`)
- **Unified fitness function**: single score combining novelty (edge coverage), diversity (Wasserstein distance), freshness (recency), and mutation potential
- **Fitness-proportional parent selection**: tournament selection for crossover parents instead of random corpus picks. Uses rank-based order-statistics — `rank = int(N * (1 - (1-U)^(1/k)))` — reducing `k` random draws to 1 call + 1 exponentiation. Pre-sorted pool fast path avoids resorting when the pool is already descending.
- **Speciation**: MinHash LSH-based species partitioning prevents dominant lineages from monopolizing selection
- **Generational replacement**: periodic evolution cycles with elitism — top fraction always survives, low-fitness individuals culled
- **Crash preservation**: crash-triggering seeds get infinite fitness bonus, never culled
- **State persistence**: population and generation state saved to `ga.json`, survives `--resume`

### Quantum-Inspired Evolutionary Algorithm (`--qea`)
- **Amplitude encoding**: each bit represented as a qubit-like probability amplitude pair (α, β) with α² + β² = 1 — "this bit is P(0)=α² likely to be 0" rather than a committed value
- **Rotation gate feedback**: amplitudes incrementally updated after each evaluation — nudging toward or away from collapsed values depending on coverage outcome
- **Collapse-only evaluation**: concrete bytes sampled from amplitudes at evaluation time, preserving uncertainty between generations
- **Built on GA infrastructure**: reuses the existing FitnessFunction, Speciation (MinHash LSH), and generation lifecycle
- **Breeding by collapse + crossover**: parents' amplitudes collapsed to bytes, two-point crossover applied, child amplitudes biased toward result
- **Diversity preservation**: continuous per-bit uncertainty maintains diversity longer than committed-value GA or batched CEM refits
- **State persistence**: population and amplitude state saved to `qea.json`, survives `--resume`
- **Note**: `--qea` and `--ga` are mutually exclusive; `--qea` takes precedence if both are set

### Wave Function Collapse (`--wfc`)
- **Constraint-satisfaction generation**: WFC solves local adjacency constraints via min-entropy collapse and AC-3 propagation, producing novel-but-valid chunk orderings for structured formats
- **1D chunk reordering**: replaces random chunk swaps in PNG/JPEG/gzip structural mutations with WFC-valid orderings that respect format-specific adjacency rules (IHDR first, IEND last, ancillary interposition, IDAT contiguity)
- **2D pixel generation**: per-row WFC for locally-coherent pixel data in BMP/PNG raw payloads, using adjacency learned from existing corpus pixels
- **No topology assumptions**: unlike `grammar.py` (recursive CFG) and `markov.py` (causal left-to-right), WFC handles arbitrary flat adjacency constraints without causal ordering or recursion depth limits
- **Defensive posture**: bounded backtrack (max 3 restarts), capped AC-3 iterations (5000) with greedy fallback, seeded RNG for tmin reproducibility
- **Guarded integration**: WFC operators run at warm tier (per-input format mutation, not per-execution bit flip), and are disabled when `--wfc` is not set
- **Independent mode**: `--wfc` is orthogonal to `--ga`/`--qea` — it controls structural generation inside format mutators, not seed selection

### Multi-Target Fuzzing
- **Shared corpus**: fuzz multiple binaries with the same corpus — inputs that find coverage in one target can discover paths in others
- **Glob expansion**: `targets/fuzz_*` expands to all matching executables, automatically skips non-binaries (`.c`, `.py`, `.sh`, etc.)
- **Per-target SHM**: each target gets its own shared memory region for independent edge tracking
- **Weighted round-robin**: targets with fewer discovered edges get proportionally more execution time
- **Cross-target seed scoring**: seeds productive for the least-covered target get boosted in selection
- **Per-target stats**: startup shows `[AFL]`/`[no-AFL]` detection, branch density per target; live stats show edge counts per target
- **AFL detection**: binary checked for `__afl_area`/`__afl_map_shm` symbols via `nm` at startup

### Corpus Management
- **Delta-encoded corpus**: parent-child diffs for small mutations (< 25% change), periodic full snapshots every 20 generations
- **xxhash dedup**: ~13x faster than SHA-256 for corpus deduplication
- **Delta snapshotting**: caps chain depth at 20 hops, prevents unbounded reconstruction cost
- **Auto-minimize**: corpus pruning guided by Wasserstein spatial diversity
- **Hamming fuzzy dedup**: near-duplicate detection via Hamming distance on equal-length seeds (`--fuzzy-dedup N`)

### Mutation Lineage Tree (`--lineage`)
- **Weighted parent-pointer forest** keyed by seed hash: every corpus seed records its parent seed key, the mutation operators + byte sites that produced it (edge weight = operator attribution), and its node weight = new coverage edges contributed at insertion
- **Branch-level pruning**: `auto-minimize` drops an entire unproductive subtree when `recent_credit == 0` (no coverage gained since the last minimize) and `subtree_weight < 1.0` (structural γ-discounted edge weight), instead of pruning just the low-scoring seed; mandatory/fresh/irreplaceable seeds are protected from branch drops
- **Causal crash-path replay** (`tmin --lineage --corpus-dir DIR`): walks the parent-key chain from the crash sidecar's `parent_seed` through `state.pkl.gz` (via `StateStore.get("corpus")`) seed_meta, rehydrates each ancestor from disk (full seeds, pruned seeds, and delta records via `rehydrate_by_hash`), and uses the smallest rehydrated ancestor that still triggers the pinned crash signature as the delta-debugging start
- **LCA-based diversity scoring**: `_compute_weights` boosts seeds whose lineage subtree is far from a sampled set of peers (mult = `1.0 + 0.5 * diversity`, diversity = mean LCA distance over a 64-seed sample normalized by tree depth)
- **Persistence**: lineage fields (`parent_key`, `parent_ops`, `parent_sites`, `new_edge_count`, `coverage_edges_baseline`) round-trip through `state.pkl.gz` (via `StateStore`); the tree is rebuilt idempotently from `seed_meta` on resume — never re-derived from runs, so no double-counting
- **Trim carries lineage**: coverage-guided trimming replaces a seed with the half-length cut inheriting the original's parent edge plus a synthetic `("trim", cut_point)` op, keeping crash chains intact across the trim point

### Crash Analysis
- **Sanitizer detection**: automatic ASAN/MSAN/TSAN/LSAN/UBSAN crash classification
- **Crash minimization**: delta-debugging with signature-matching to prevent drift to unrelated bugs
- **Corpus minimization**: greedy set-cover over SHM edge bitmaps (`minimize` subcommand)
- **Crash exploitability tiers**: ASAN_EXPLOITABILITY classification in reports
- **Levenshtein crash clustering**: groups crashes with similar stack traces (same root cause, different offsets)
- **Fuzzy corpus similarity**: Hamming + Levenshtein + 4-gram Jaccard for crash-to-corpus nearest-neighbor search
- **Crash stack hash**: hashes last 3 nibbles of each PC in top 7 frames (14 with sanitizers) for deduplication; single-frame crashes masked to prevent false uniqueness
- **Blocklist/allowlist** (`--crash-blocklist`/`--crash-allowlist`): skip known crash stack hashes or override blocklist for specific crashes
- **Smaller crash replacement** (`--save-smaller`): replace crash triggers with smaller inputs for the same stack hash
- **Fault-address extraction**: the ptrace runner captures the kernel-reported faulting address (`si_addr` via `PTRACE_GETSIGINFO`) plus register state (`PTRACE_GETREGS`) at the fatal-signal stop (`runner.py::_capture_crash_state`), before the tracee is reaped. Non-sanitizer crash signatures become `signal:N@0xaddr`, so NULL-deref, wild-pointer, and stack-overflow crashes that share a signal number dedup separately (address-bearing fallback signatures use exact matching only — the Levenshtein fuzzy matcher stays gated to sanitizer signatures because `normalize_frame()` strips `0x`-addresses, which would silently re-group them). The address is written to the `.txt` sidecar (`fault_addr:`), and the sidecar register section (previously always zeroed) is now populated. With `--trace-crashes`, trace reports show both `RIP:` (`crash_rip` — where the crashing instruction lives) and `Fault:` (the memory address it touched, from GDB `$_siginfo`). si_addr semantics: SIGSEGV/SIGBUS → faulting memory address (valid for NULL-deref/wild-pointer bucketing); SIGILL/SIGFPE → faulting instruction address. User-raised signals (si_code ≤ 0, e.g. `kill(SIGSEGV)`) are ignored since their si_addr is meaningless.
- **Fault-address capture is per-mode** (the ptrace runner above is only one of three `.so` execution paths). (1) **Persistent loader** (ASAN/cmplog-fallback `.so` mode): the fork-per-call grandchild self-traces with `PTRACE_TRACEME` after `os.setsid()` (`persistent_loader.py`), its direct parent reaps it with a `WUNTRACED` stop loop (`os.waitpid` returns a `WIFSTOPPED` stop for a traced child even without `WUNTRACED`), captures si_addr + rip/rsp/rbp at the fatal stop, then `PTRACE_CONT`s the signal so the guarded call (or default disposition) still runs. Fault state relays P2→P1→P0 over the trailing RC-line tokens (`RC <rc> <bmp_len> <fault_addr> <rip> <rsp> <rbp>`, `-` when absent). (2) **direct_lite** (default `.so` mode — crashes reported only as a negative rc from `__afl_guarded_call`, no address): on a crash with no captured address, `fuzz_one()` re-runs the input once through the subprocess loader script self-traced with `PTRACE_TRACEME` (`TargetRunner._run_triage_ptrace`, gated on a lazy `ptrace_available()` probe). Latency is paid only on the rare crashing input. Subprocess/forkserver/network modes stay diagnostic-poor (address only via the ASAN report text when present). Registers surface in the sidecar via `corpus_manager.save_crash` gating (regs now flow when captured OR `ptrace_cov` is set, not only ptrace mode).
- **GDB crash replay in the report sidecar**: every saved crash (`.txt`) now embeds a `=== GDB crash replay ===` section automatically when gdb is installed — best-effort, ~1s on the rare crashing input (`corpus_manager.save_crash` → `CrashTracer.gdb_replay`). `.so` targets, which can't be exec'd, are driven through a ctypes harness that loads the library and calls the probed fuzz entry point (`fuzz_shm_run`, else first exported `fuzz_*` symbol) with the crash input on stdin; gdb's own stdin is redirected since gdb has no `run <` redirection. The section carries the signal, `RIP`/`Fault` (from `$_siginfo`'s `si_addr`), registers, backtrace, and — when the target carries DWARF (all `tools/build_targets.sh` builds use `-O2 -g`) — `file:line` with argument values plus a source listing of the crashing line (`frame 1` + `list`). `--trace-crashes` additionally writes the full standalone `.trace` report (with strace). The pre-existing tracer flaw (input passed as ARGV instead of stdin) is fixed for the `.so` path (stdin); the standalone-executable path still passes the input as argv[1].
- **Ptrace runner fixes** (found while wiring the above, both pre-existing): (1) the breakpoint handler read `rsp` from register offset 176 (`128+48`) instead of 152 — offset 176 is `gs_base` (0 for the main thread), so `rsp > 0x1000` was always false and every instrumented function's first instruction was skipped, corrupting the tracee into spurious stack-address SIGSEGVs with zero edge coverage; (2) the wait loop tested `status == 0` from `os.waitpid()` instead of the returned PID, conflating "no event" `(0, 0)` with a clean exit `(pid, 0)` — every rc=0 exit was misreported as `-2` ("exec failed") and its input saved to the corpus as "interesting", polluting the corpus in ptrace mode.

### Observability
- **Branch density**: per-target static analysis at startup (`cond branches/KB`) with average across targets
- **Per-target coverage stats**: live display shows `targets: name1:N name2:N name3:N` (edge counts per target)
- **AFL detection**: binary checked for `__afl_area`/`__afl_map_shm` symbols via `nm` — shows `[AFL]`/`[no-AFL]` per target
- **GA/QEA lifecycle stats** (`--ga`/`--qea`): stat line shows `ga: gen=3 pop=200 spc=5 fit=0.42` (or `qea:`) — generation number, population size, species count, best fitness
- **Mutual information stats** (`--mi-guided`): stat line shows `mi: obs=5500 pos=4096` — total observations, tracked byte positions
- **Elo arbitration stats** (`--elo`): stat line shows `elo: meta=bandit seed=weighted top=seed_bayesian(1561)` — active meta-strategy, seed strategy, top-rated strategy
- **Sensitivity stats** (`--sensitivity`): stat line shows `sens: 34 seeds` — seeds analyzed for byte-level sensitivity
- **Transfer entropy stats** (`--transfer-entropy`): stat line shows `te: 42 edges` — causal byte→edge mappings discovered
- **Secretary stats** (`--secretary`): stat line shows `sec: 5 tracking` — secretary-problem instances active
- **Shapley stats** (`--shapley`): stat line shows `shap: 75 ops` — operators with Shapley attribution data
- **FrameShift stats**: stat line shows `fs: 3 rel` — active length-field relations auto-discovered
- **Misc auto-stats**: stat line shows `pruned:2 dup:5` — corpus auto-prunes, duplicate rejections
- **Bayesian stats** (`--bayesian`): stat line shows `bayes: 695 seeds 5500 obs` — seed quality tracking state
- **Markov context count**: stat line shows `ctx: 562347` — contexts seen by the Markov chain/ensemble
- **Replicator dynamics** (`--replicator`): stat line shows `rep: dom=bit_flip ops=78` — dominant operator and active operator count
- **MOpt PSO** (`--mopt`): stat line shows `mopt: 5p` — active particles in PSO swarm
- **Jaccard index**: corpus redundancy metric (`| jac: 0.XX`)
- **Diversity score**: Wasserstein spatial diversity (`| div: N`)
- **`--report` flag**: full explainability report with coverage, mutations, perplexity, corpus health, edge map. The Run Summary opens with the **exec line** (target invocation reconstructed from `target` + `target_args` + `file_mode`, rendering `{file}` as `@@` AFL-style) plus the full **invocation** (`sys.argv` captured in `cmd_fuzz`) and input mode. A **Configuration** section (seed, schedule, mutations/input, resume, max len, timeout, map size, multi-target count, in-process/direct-lite/persistent, extra crash codes, ASAN/UBSAN targets, cmplog, dictionary, grammar, markov, operator count) appears after the Run Summary, and a **Crash Signatures** histogram (from `crash_sigs`, keyed by `SanitizerReport.signature` or `signal:N@0xaddr` fallback, top 2 frames appended when tracked) follows Crash Analysis.
- **`--plot FILE` flag**: self-contained HTML report (`core/plotting.py::generate_html_report`) with inline SVG charts (edges, exec rate, corpus size, crashes, operator success rate, operator usage counts). The summary block shows exec count/crashes/corpus, target, exec line, full fuzzer-tool invocation, input mode, coverage mode, max len, timeout, and cmplog state; all exec-line strings are HTML-escaped. Title carries the target basename (`Fuzzer Report — <target>`).
- **`--replay-N` flag**: background crash reproducibility scoring
- **Per-seed cost tracking**: wall-clock time per seed for cost-aware scheduling
- **Discovery rate**: edges per 1k execs over sliding window
- **Bitmap density**: map occupancy percentage (saturation detection)
- **Dup rejection rate**: duplicate-rejection as saturation signal

### Performance
- **Weight caching**: 733x speedup on `_pick_seed` with 200+ seeds
- **Lazy watchdog**: `Event.wait(timeout)` eliminates busy-poll overhead on fast processes
- **xxhash dedup**: 13x faster than SHA-256 for corpus operations
- **Redqueen xform sort cache**: cmplog pairs sorted once per pair-list change (not per invocation) — eliminates ~2,558 redundant O(N log N) sorts per run, saving ~4-5s. Hot path uses filtered scan with early break on the pre-sorted list
- **Elo K-factor cache**: `_effective_k()` computed once per iteration instead of ~78K times across record_match and record_strategy_match calls — saves ~1s per run by eliminating redundant sum-of-squares
- **CMPLOG hex decode**: `binascii.unhexlify` replaces `bytes.fromhex` for ~2x faster hex decoding in collect_tokens
- **SHM hotpath optimization**: edge_count O(1) fast-path (8-byte header read) skips full-table scan when no new edges exist at all — saves the numpy vectorized scan entirely, not just filtering cost. Combined `is_new_coverage_with_edges()` reduces two parallel methods to one. Per-iteration edge cache in `fuzz_one` eliminates redundant Python loops over 8192 SHM entries (was 2.5s/500 iters, now 0.024s); ~2.4x total speedup, ~3.3x more EPS (132→429)
- **Tree mutator optimization**: `__slots__` on `_Node`, pre-computed delimiter lookup tables, inlined `_find_delim` in parse loop, iterative `_collect_nodes`, RandPool passthrough — `partial_parse` 2.2x faster (0.155s→0.070s)
- **RandPool vectorized batches**: `randint_list`/`randrange_list`/`random_list` use numpy vectorized modulo + `tolist()` instead of Python list comprehensions
- **RandPool in format mutations**: all format-specific mutation classes (zlib, gzip, jpeg, png, bmp) and grammar mutations now route through RandPool via `rng=None` parameter passthrough — reduces stdlib `random.randint` calls ~17%, total function calls reduced 2.8M per 1k iterations
- **Seed picker batch entropy**: `_compute_weights` pre-computes all seed entropies and mean entropy in a single pass before the per-seed loop, eliminating O(n²) redundant `shannon_entropy_seed` calls — total function calls reduced 26% (5.5M -> 4.1M per 1k iterations)
- **Distance max_distance caching**: `max_distance` property cached (only computed once since `_distances` doesn't change during fuzzing), pre-computed in `_compute_weights` and passed to `_weight_entropy_and_distance` — `max_distance` 14x faster, `_weight_entropy_and_distance` 7x faster
- **SHM data minimization**: three-tier reduction of SHM bitmap data movement (~170MB/s saved):
  - **Tier 1 — numpy flatnonzero**: replaces Python `for` loop over 1MB bitmap in `record_edge_lifetimes` with `np.flatnonzero()` — saves ~2GB total data movement from Python iteration
  - **Tier 2 — zero-copy numpy views**: replaces `bytes()` allocations with `np.frombuffer()` zero-copy views at 5+ call sites (distance, Shapley, length tracker, edge lifetimes) — saves ~2.2GB total bytes allocation
  - **Tier 3 — inline tobytes/memmove chain**: replaces `classified.tobytes() + ctypes.memmove` in `_is_new_coverage_numpy` with direct numpy array slice assignment — saves ~1MB allocation + 1MB copy per numpy-scan (~2.6GB total)
- **Operator-selection hot path**: ~60% of `fuzz_one` time was operator *selection* (not application). Three caches cut it in half on the `--elo all` config:
  - **Thompson draw cache** (`MonteCarloScheduler`): draws are cached per arm keyed on the effective `(alpha, beta)`; `record()`/decay change the key, so stale entries miss naturally. Draws are additionally force-refreshed every 16 selects (`_draw_refresh_interval`) so an arm whose posterior never moves cannot keep a frozen lucky draw and starve the other arms — the unbounded version collapsed exploration (one op dominated, eps 427→47, RSS 1.5GB on dimension-mutating PNG targets). Cuts `gammavariate` 950K→194K and `betavariate` 519K→98K calls per 1k execs
  - **`RunningMoments.stddev` cache**: GP-UCB read stddev per arm per select (299K reads/run), each recomputing `math.sqrt(variance)`. Now cached, invalidated on `update()`/`load()` — cumtime 0.139s→0.020s
  - **Elo meta-strategy resolved once per exec**: `select_op` called `elo.select_strategy` on every mutation (~21x/exec), each doing a gauss sample per rated strategy. Now cached in `f._meta_strategy_cached`, reset at the top of `mutate()` and re-validated against the available set — `select_strategy` 23K→2K calls, `gauss` 175K→7.8K calls per 1k execs
  - Net: `operators.select_op` cumtime 4.30s→1.96s per 1k execs; EPS improved ~1.4-1.9x on the png_read_dist.so workload (259-369 → 486-541) with stable RSS
- **Branch-pruning hot-path micro-opts (2026-08-06)**: cProfile on `png_read_dist.so` (2k execs) shows per-exec Python cost concentrated in `mutate`/`available`/`_check` (~44µs), seed picking (~50-97µs, weight-cached), `_check_new_coverage` (~23µs), `_compute_crps` (~15µs). Two branch/work reductions landed: (1) `_compute_crps` hoisted the per-call `import numpy` to module scope and replaced `np.sum(cd²·diff)` with `np.dot(cd², diff)` — one fewer temp array allocation per exec (this runs once per exec); (2) `select_position` gates the per-mutation `get_weighted_position` call on `_use_sensitivity` like its MI/TE siblings (the tracker is never populated when disabled, so the call always returned None after paying lookup+branches). Measured: consistent ~2-3% faster wall-clock (66.6s vs 68.8s per 100k execs, alternating identical-corpus runs), within measurement noise. CPython microbenchmark caveat: naive "branchless" patterns mostly lose — `min()`/`max()` builtins are slower than `if`, and bool-multiply clamps are slower than the branch; the wins are ternaries, `x = x or cond` short-circuit accumulation (2.2x), and removing redundant per-exec calls/allocations. The bigger structural lever (per-exec re-evaluation of data-independent availability predicates in `REGISTRY.available`) remains open
- **JPEG length-field clamp**: `JpegMarker.serialize()` truncated segment data to 65533 bytes before packing the 16-bit length field. A corrupt length field that makes the parser absorb a >64KB input as one segment previously crashed the whole fuzzer with `struct.error: 'H' format requires 0 <= number <= 65535`

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Basic fuzzing
fuzzer-tool fuzz ./target

# Coverage-guided with dictionary
fuzzer-tool fuzz -c -D dictionary.txt ./target

# In-process mode (fastest for .so targets)
fuzzer-tool fuzz libfoo.so --inprocess --inprocess-func target_func -c

# File-based target
fuzzer-tool fuzz -F -A "{file}" ./target

# With Markov and Monte Carlo
fuzzer-tool fuzz --markov --markov-gen --mc-bandit --mc-cem ./target

# Grammar-aware PNG fuzzing
fuzzer-tool fuzz targets/png_read -c -D dictionaries/png.dict -g dictionaries/png.gram

# ASAN fuzzing (auto-detected, catches heap-buffer-overflow, use-after-free, etc.)
fuzzer-tool fuzz targets/asan_target

# Cmplog comparison tracing test (memcmp/strcmp/strncmp/memchr/strcasecmp/strncasecmp/memmem/strstr/strcasestr)
fuzzer-tool fuzz targets/cmplog_exercise --cmplog

# Compiler-IR comparison tracing test (requires clang -fsanitize-coverage=trace-cmp)
fuzzer-tool fuzz targets/tracecmp_target --cmplog

# fgrep SIMD/regex/BMH search fuzzing
fuzzer-tool fuzz targets/fgrep_read

# fgrep-specific fuzz targets (ASAN-instrumented)
# Regex compilation — adversarial patterns against regcomp()
fuzzer-tool fuzz targets/fuzz_regex_compile

# Pattern matching — fixed patterns, fuzzed data against regexec/SIMD search
fuzzer-tool fuzz targets/fuzz_pattern_match

# Full search pipeline — end-to-end search_data() with SIMD, regex, output
fuzzer-tool fuzz targets/fuzz_search_pipeline

# Multi-target: fuzz multiple binaries with shared corpus (glob supported)
fuzzer-tool fuzz targets/fuzz_regex_compile targets/fuzz_pattern_match targets/fuzz_search_pipeline -c -d corpus/fgrep

# GNU Grep exec-based fuzzing — exercises /usr/bin/grep across regex, fixed, PCRE modes
# Input format: mode byte | pattern length | pattern | text
# Modes: 0=basic_regex, 1=extended_regex, 2=fixed, 3=PCRE, 4=icase, 5=word, 6=line, 7=invert, 8=combined_flags
fuzzer-tool fuzz targets/grep_read -c -F --no-shm

# GNU Grep vendoring setup (downloads source + runs configure)
tools/vendor_grep.sh

# GNU Grep in-process .so mode (with ptrace coverage for the grep child)
fuzzer-tool fuzz targets/grep_read.so -c -F --no-shm

# Tailslayer hedged reader fuzzing (in-process .so mode, ~66 eps)
fuzzer-tool fuzz targets/tailslayer_read.so -c --inprocess

# FFmpeg demux+decode pipeline (vendored FFmpeg 7.1, exercises 300+ format/codec paths)
fuzzer-tool fuzz targets/ffmpeg_read -c

# Multi-target with glob — skips .c/.h/.py automatically
fuzzer-tool fuzz 'targets/fuzz_*' -c -d corpus/fgrep

# Two-pass workflow: fast fuzz without ASAN, then verify crashes with ASAN
fuzzer-tool fuzz targets/fuzz_*_nosan -c -d corpus/fast/
fuzzer-tool verify targets/fuzz_search_pipeline corpus/fast/crashes/

# Auto sanitizer replay: fuzz no-ASAN target, auto-replay crashes on ASAN/UBSAN variants
fuzzer-tool fuzz targets/png_read.so -d corpus/ \
  --inprocess-direct --cmplog \
  --asan-target targets/png_read_asan.so \
  --ubsan-target targets/png_read_ubsan.so

When `--asan-target` or `--ubsan-target` is set, the fuzzer schedules every
no-ASAN crash for replay on the corresponding sanitizer-instrumented target.
Replays run in a subprocess (fork+exec with LD_PRELOAD) because ASAN detection
does not work in-process via ctypes/direct_lite (shadow offset mismatch on
48-bit systems). Reports are saved as JSON alongside the crash file:
`crashes/<sig>_sanitizer_report.json`. Replays run periodically during the
fuzz loop (every 500 iterations alongside existing crash reproducibility checks).

# Resume a previous fuzzing session
fuzzer-tool fuzz ./target -c --resume

# Full report after run
fuzzer-tool fuzz ./target -c -n 5000 --report report.txt

# Rank corpus seeds by interestingness
fuzzer-tool rank ./target -d corpus -n 20

# Dump top 10 most interesting seeds to files
fuzzer-tool rank ./target -d corpus -n 10 --dump top_seeds
```

## Fuzzing Options

| Flag | Description |
|------|-------------|
| `-c` | Enable coverage-guided mode |
| `--no-shm` | Skip AFL SHM, force ptrace |
| `--deep-coverage` | Capstone-based basic block discovery |
| `-F` | File mode (write input to temp file) |
| `-D FILE` | Load dictionary tokens |
| `--no-adaptive-havoc` | Draw havoc's 11 inline sub-mutations uniformly instead of weighting them by measured new-coverage rate (weighting is on by default; use this as the A/B baseline) |
| `-g GRAMMAR` | Grammar-aware mutations (built-in: png, json, http_request, elf) |
| `--cmplog` | Comparison tracing via LD_PRELOAD (or compile `cmplog_shim.c` into target .so for direct_lite compatibility) |
| `--cmplog-workdir` | Directory for cmplog runtime log files (default `/tmp/<target>.cmplog`). Use a disk-backed path when `/tmp` is a small tmpfs to avoid filling it. |
| `--markov-gen` | Markov-generated seeds (rate adapts to model quality via perplexity) |
| `--mc-bandit` | Thompson sampling operator selection (Brier score calibration) |
| `--mc-cem` | Cross-Entropy Method byte distribution |
| `--mopt` | MOpt PSO operator scheduling (alternative to bandit) |
| `--replicator` | Replicator dynamics operator scheduling (evolutionary game theory) |
| `--exp3` | EXP3 adversarial bandit operator scheduling (non-stationary rewards) |
| `--exp3-gamma FLOAT` | EXP3 exploration rate in [0,1] (default: 0.1) |
| `--eps-greedy` | Epsilon-greedy operator scheduling with exponential annealing |
| `--eps-greedy-epsilon0 FLOAT` | Initial epsilon for epsilon-greedy (default: 1.0) |
| `--eps-greedy-decay FLOAT` | Epsilon decay rate per pull (default: 0.9995) |
| `--hierarchical-bandit` | Hierarchical bandit operator scheduling (category → operator) |
| `--gp-ucb` | GP-UCB operator scheduling with RBF kernel covariance |
| `--gp-length-scale FLOAT` | GP kernel RBF length scale (default: 1.0) |
| `--gp-beta FLOAT` | GP-UCB exploration parameter (default: 2.0) |
| `--shapley` | Shapley value operator attribution (fair credit distribution) |
| `--mi-guided` | Mutual information guided mutation (target high-MI byte positions) |
| `--renyi-weight` | Rényi entropy weighting in seed selection (boost cold-edge seeds) |
| `--transfer-entropy` | Transfer entropy causal tracking (byte→edge influence detection) |
| `--inprocess` | Persistent subprocess mode (auto-restart on crash) |
| `--resume` | Resume from saved state |
| `--profile-hotpath` | Profile the fuzz run with cProfile; prints tottime/cumtime/ncalls tables and dumps stats (ignored with `--jobs > 1`; suppresses the periodic `[*] execs:` status line for clean output) |
| `--profile-out PATH` | cProfile dump path for `--profile-hotpath` (default `/tmp/fuzzer_hotpath.prof`) |
| `--crash-codes N` | Additional exit codes to treat as crashes |
| `-j N` | Parallel fuzzing with N workers |
| `--elo` | Elo arbitration between operator strategies (bandit/mopt/replicator/cem/exp3/eps_greedy/hierarchical/gp_ucb) and seed strategies (ga/qea/weighted/pareto/format/bayesian/markov); `--elo all` also enables every scheduler plus the mutation-stack features (metropolis/shapley/mi-guided/secretary/wfc/lineage/`--schedule fast`), and the convergence report lists only schedulers actually used |
| `--sensitivity` | Per-byte sensitivity analysis (Lyapunov exponent) for mutation targeting |
| `--region-profile` | Statistical region profiling for mutation targeting (labels seed windows incompressible/tabular/textual/repetitive) |
| `--secretary` | Secretary-problem optimal stopping for seed/operator/corpus scheduling |
| `--bayesian` | Bayesian methods: Thompson-sampled seed selection, hierarchical operator priors, Bayesian coverage growth model |
| `--ga` | Genetic algorithm lifecycle mode (bounded population, speciation, crossover) |
| `--qea` | Quantum-inspired evolutionary algorithm (amplitude encoding, rotation gate feedback) |
| `--wfc` | Wave Function Collapse structural generation (chunk reordering, pixel generation) |
| `--enable-smt-z3` | Z3-based SMT solving for arithmetic constraint solving on cmplog pairs |
| `--hw-perf` | Hardware performance counters via perf_event_open (instructions, branches, misses) |
| `--schedule base\|fast\|coe\|rare\|mopt\|lin\|quad\|go\|aflgo` | AFL++ power schedule (`aflgo` = exact AFLGo distance annealing) |
| `--aflgo-cooling exp\|log\|lin\|quad` | Cooling schedule for the `aflgo` power factor (default exp) |
| `--t-x MINUTES` | AFLGo time-to-exploitation in minutes; temperature cools to 1/20 over this window |
| `--markov-order N` | Markov chain order(s), comma-separated (e.g. '0,1,2' for ensemble) |
| `--save-smaller` | Replace crash triggers with smaller inputs for the same stack hash |
| `--crash-blocklist FILE` | Skip crashes matching these stack hashes |
| `--crash-allowlist FILE` | Override blocklist for specific crash hashes |
| `--coverage-log FILE` | Append (timestamp, edge_count) lines for coverage-over-time plots |
| `--coverage-report FILE` | Dump edge coverage map to JSON on exit |
| `--max-corpus N` | Auto-minimize corpus at N entries |
| `--corpus-boost MAX_LEN` | Resize corpus seed lengths to truncated normal distribution N(mean, std), capped at MAX_LEN |
| `--boost-mean FLOAT` | Target mean for normal distribution (default: corpus_boost/2) |
| `--boost-std FLOAT` | Target std for normal distribution (default: corpus_boost/6) |
| `--boost-pad {repeat,zero,random}` | Padding mode for undersized seeds; repeat cycles existing bytes (AFL-style), zero pads with \\0, random appends uniform random bytes |
| `--replay-n N` | Replay each crash N times for reproducibility scoring |
| `--asan-target PATH` | Path to ASAN-instrumented .so/executable for auto sanitizer crash replay |
| `--ubsan-target PATH` | Path to UBSAN-instrumented .so/executable for auto sanitizer crash replay |
| `--report [FILE]` | Generate explainability report (stdout or file) |
| `--stats-interval N` | Print live stats and dump stats file every N iterations (default: 1000) |

## Subcommands

| Command | Description |
|---------|-------------|
| `fuzz` | Run coverage-guided fuzzing (default) |
| `rank` | Rank corpus seeds by interestingness (edge coverage, rarity, subsumption) |
| `minimize` | Minimize corpus by removing redundant inputs |
| `sweep` | Linear corpus scan — replay every seed without mutations or scheduling (find missed crashes) |
| `tmin` | Minimize a crash to smallest reproducer |
| `replay` | Replay a crash input against the target |
| `verify` | Re-run crashes with ASAN target to confirm memory bugs |
| `estimate` | Estimate execs to first crash via static analysis + calibration |
| `import` | Import corpus from AFL/libFuzzer/honggfuzz |

### Rank Seeds

Rank corpus seeds by a composite interestingness score based on edge coverage, singleton edge rarity, subsumption (irreplaceability), and coverage proximity.

```bash
fuzzer-tool rank <target> -d <corpus> [-n TOP] [--dump PREFIX]
```

| Flag | Description |
|------|-------------|
| `-d DIR` | Corpus directory |
| `-n N` | Number of top seeds to show (default 10) |
| `--dump PREFIX` | Dump top seeds to files `PREFIX.0`, `PREFIX.1`, ... |

### Sweep

Linearly replay every seed in the corpus without mutations, scheduling, or minimization — finds crashes the target would have triggered on corpus seeds but that were missed during normal fuzzing.

```bash
fuzzer-tool sweep <target> -d <corpus> [-c]
```

| Flag | Description |
|------|-------------|
| `-d DIR` | Corpus directory (reads from `seeds/` and `irreplaceable/`) |
| `-c` | Enable coverage tracking (not used for scheduling, just reporting) |
| `--timeout MS` | Per-seed timeout in milliseconds (default: same as fuzz mode) |

No mutations, no scheduler, no coverage-guided feedback loop. Each seed is executed once as-is.

### Estimate Crash ETA

Estimate executions to first crash using static risky density, Good-Turing edge estimates, and optional calibration runs.

```bash
fuzzer-tool estimate <target> --corpus <dir> [--calibrate N]
```

| Flag | Description |
|------|-------------|
| `--corpus DIR` | Corpus directory for Good-Turing edge estimation |
| `--calibrate N` | Number of calibration executions (default: 1000) |

## Coverage Modes

| Mode | Flag | Throughput | Notes |
|------|------|-----------|-------|
| SHM bitmap | `-c` (default) | 65–200 eps | For AFL-instrumented targets |
| In-process | `--inprocess` | 65–120 eps | Persistent loader with crash recovery |
| In-process direct | `--inprocess-direct` | 2k–34k eps | No crash isolation; afl_shim crash handler gives _exit(128+sig) exit codes |
| Ptrace basic | `-c --no-shm` | ~20 eps | Function-entry breakpoints |
| Ptrace deep | `-c --no-shm --deep-coverage` | ~18 eps | Capstone BB discovery |

## State Persistence

```bash
fuzzer-tool fuzz ./target -c -n 10000
fuzzer-tool fuzz ./target -c --resume -n 10000
```

State files:
- `state.pkl.gz` — **single compressed pickle** via `core/state_store.py:StateStore`; all sections below are stored as named sections inside this one file. Replaces the eleven per-component JSON files. Legacy JSON files are auto-migrated on first `--resume` and cleaned up via `cleanup_legacy()`.

  Sections:
  - `corpus` — exec counts, crash sigs, op stats, seed metadata, lineage depths
  - `edge_tracker` — per-seed edge coverage, cumulative edges, global hit counts
  - `markov` — persisted Markov chain transitions
  - `mi` — mutual information tracker (byte-to-coverage correlations); loaded only with `--resume`
  - `elo` — Elo ratings for operator and seed strategies
  - `qea` — QEA population amplitudes and generation state
  - `ga` — GA population, generation, and fitness state
  - `crash_mi` — crash-path MI tracker
  - `sensitivity` — sensitivity scores
  - `length_tracker` — length-vs-edge correlation
  - `seed_quality` — Bayesian seed quality posteriors

  Use `--no-save-state` to skip writing the file entirely.

## Kalman Filter Online Estimation

The fuzzer includes a self-contained Kalman filter implementation (`src/fuzzer_tool/core/kalman.py`) for online denoising and uncertainty quantification of noisy scalar signals.  Available in 1D (value-only) and 2D (constant-velocity: value + derivative) variants, plus a `RobustKF` subclass with Huber innovation gating and adaptive measurement-noise estimation.

### Applications

1. **Denoised execs/sec for stats and budget allocation** — the interval EPS rate (execs since last stats-tick / elapsed since last stats-tick) is a noisy per-interval signal, unlike the monotonic campaign-average. `stats.py:print_stats()` feeds this interval rate into a 2D `RobustKF` (value + derivative) via `predict(dt=interval_elapsed); update(interval_rate)`. The 2D model properly scales process noise by dt/dt²/dt³, important because the stats interval varies across the campaign. The filtered state-tuple and uncertainty are exposed as `fuzzer._eps_filtered` (value) and `fuzzer._eps_uncertainty`, and used in:
   - Dict-entry pruning (`fuzzer.py`, replaces the raw 10-sample sliding window)
   - Stats-interval calculation (`fuzzer.py`): the first tick prints at ~1 second of work (1x EPS) so the first `[*] execs` line appears promptly; subsequent ticks are spaced at ~10 seconds of work using 10x the mean of the last 10 avg-eps samples (`fuzzer._eps_history`, one per tick, via `_last_avg_eps()`), falling back to the fixed `stats_interval` while the window fills so a single inflated warm-up reading can't stretch the gap between stats lines

2. **Critical-slowing-down denoising** — `CriticalSlowingDown` accepts an optional `denoiser` (any `KalmanFilter` instance). When provided, `observe(value)` runs `predict(dt=1.0); update(value)` internally and stores `kf.estimate` instead of the raw discovery rate. This reduces false "stalled" calls from single-execution noise spikes.

3. **Adaptive network settle time** — `NetworkRunner` accepts an optional `settle_kf` parameter. When a `KalmanFilter` is attached, the `_settle()` sleep uses the KF's filtered estimate (clamped to `[0.5×initial, 10×initial]`) instead of the fixed `settle` time. An external measurement loop can feed observed edge-plateau latencies into the KF, making settle self-tuning.

### Design

- **Separate `predict(dt)` and `update(z)`** — the caller is responsible for calling both in sequence each cycle. `update()` does NOT call `predict()` internally, preserving correctness for irregular-time-step callers. Constant-time-interval callers always use `kf.predict(1.0); kf.update(obs)`.
- **Plain Python, no numpy** — 1D and 2D matrix operations are explicit list-of-list arithmetic. For dim ≤ 2 the operations are trivial and the numpy dependency is avoided.
- **Huber gating is one-shot** — when the Mahalanobis distance of the innovation exceeds the threshold, the measurement noise (R) is inflated for that single `update()` step only. The inflated R is NOT persisted, preventing a single outlier from making the filter untrusting of subsequent normal measurements for many steps.
- **Adaptive R** — a slow-timescale (gain `~0.02`) exponential window on innovation RMS nudges the effective measurement noise to match observed innovation statistics. This handles non-stationary observation noise without manual retuning.

### CSD autocorrelation protection

The autocorrelation leg of `CriticalSlowingDown` is decoupled from the Kalman denoiser: raw (unfiltered) values are stored separately in `_raw_history` and used exclusively for the lag-1 autocorrelation computation. The denoised values in `_history` continue to feed the variance and skewness legs (which benefit from noise reduction without the autocorrelation inflation problem). This prevents the KF's IIR-smoothing from producing false-positive transition warnings on flat traces.

### Persistent state

- `kalman.json` in the corpus directory saves/loads the filter state (x, P, R_eff, innovation RMS) on shutdown and resume.

## ELF Binary Static Analysis

The fuzzer includes a built-in ELF analysis engine for extracting DIV/IDIV constants from compiled binaries using the pure-Python x86-64 instruction decoder:

### DIV/IDIV Constant Extraction

```bash
# Extract constant divisors from a compiled binary
python -c "
from fuzzer_tool.core.elf import extract_div_constants
d, w = extract_div_constants('targets/png_read')
print('Constant divisors:', d)
print('Weak modulus PCs:', w)
"
```

- **Backward scan**: traces register writes backward from DIV/IDIV to find `mov $K, %reg` (constant-assignment) that feeds the divisor register
- **Forward modulus extraction**: detects `cmp $K, %edx` patterns after DIV/IDIV, mapping CMP addresses to the same divisor — even when the remainder is copied through an intermediate register (`mov %edx,%eax; cmp $K,%eax`)
- **Pure-Python decoder**: x86-64 instruction decoder with no external dependencies
- **CET/IBT aware**: correctly consumes `endbr64` (F3 0F 1E FA) and multi-byte NOP (0F 1F) alignment instructions
- **Register-to-register MOV**: handles `89 /r` and `8B /r` encodings for remainder propagation tracking
- **Dynamic remainder tracking**: `_rem_regs` set expands through MOV copies and shrinks on register overwrites, enabling `weak_mod_pcs` detection through the common GCC `mov %edx,%eax; cmp $K,%eax` idiom

## In-Process Execution

### Direct ctypes (`--inprocess-direct`)
Calls target function directly via `ctypes.CDLL`. Catches SIGSEGV/SIGABRT via signal handler. ~2k–34k eps.

### Persistent subprocess (`--inprocess`)
Keeps one Python subprocess alive. Fork-per-call with `os.setsid()` for process group isolation. Timeout enforced via outer threaded readline. Auto-restarts on subprocess death. Throughput monitoring detects sustained slowdowns (below 10% of calibrated baseline) and auto-restarts the loader. ~65–120 eps.

### ASAN support
Automatically detects ASAN-instrumented targets by checking for `__asan_init` symbols. For `.so` targets, the fuzzer preloads libasan via `ctypes.CDLL(mode=RTLD_GLOBAL)` with a `verify_asan_link_order=0` shim, enabling ASAN-instrumented shared libraries to load in `--inprocess-direct` (direct_lite) mode. If ctypes preloading fails, auto-detected `.so` targets fall back to the persistent subprocess loader.

**Known limitation (Layer 1)**: ASAN-detected memory bugs (heap/stack buffer overflow, use-after-free) do NOT trigger in direct_lite mode due to interactions between mid-process ctypes loading, PLT resolution ordering, and ASAN's initialization sequence. See [`docs/ASAN-LIMITATION.md`](docs/ASAN-LIMITATION.md) for full investigation details, ruled-out hypotheses, and workarounds.

**Layer 2 resolved**: ASAN-generated crash reports are captured via stderr pipe redirection in direct_lite mode, with `halt_on_error=0` preventing ASAN from aborting the process. The existing `SanitizerReport.parse()` pipeline detects crashes from the captured ASAN report text. This is enabled automatically for all ASAN-instrumented `.so` targets.

### AFL shim crash handlers (`afl_shim.c`)
The coverage shim compiled into every fuzz target now installs C-level signal handlers for `SIGSEGV` and `SIGABRT` that call `_exit(128 + sig)`. This ensures the process exits with a meaningful signal-indicating exit code when the target crashes, even when Python-level signal handlers cannot fire (e.g. during a `ctypes` call). The persistent subprocess loader detects these exit codes and converts them to negative signal codes (`-6` for SIGABRT, `-11` for SIGSEGV), enabling proper crash reporting in all execution modes.

### Hybrid abort interception
In non-ASAN builds, `afl_shim.c` intercepts `abort()` calls via a preprocessor macro, redirecting them to a static helper that writes `[shim] abort() intercepted` to stderr and returns (instead of killing the process). This prevents false crash detections from library assertion failures (e.g. FFmpeg's ~1600 `av_assert0` call sites). In ASAN builds (`__SANITIZE_ADDRESS__`), the override is excluded so `abort()` raises `SIGABRT`, which the signal handler chains to ASAN's own handler — letting ASAN produce diagnostic output before termination. The macro approach avoids the GCC "noreturn function does return" warning by never re-declaring `abort()` directly, and works correctly in both standalone binary and `.so` (ctypes/in-process) contexts.

### SHM resize in inprocess mode
When collision risk exceeds the threshold, the bitmap SHM is resized. In inprocess mode, this patches the target's `__afl_area` pointer to the new SHM segment and invalidates the cached SHM attachment, so coverage writes don't go to freed memory. The target's compiled-in `__afl_map_mask` is not updated (static variable), so the target underutilizes the new bitmap — but writes remain in-bounds.

### Timeout in direct mode
`--inprocess-direct` and direct_lite mode enforce timeout via `SIGALRM` + `setitimer`. Previously these modes had no timeout protection — a hanging target would freeze the fuzzer.

## Corpus Minimization

```bash
# Basic minimization (greedy set-cover)
fuzzer-tool minimize ./target -d corpus -c

# Rate-distortion optimal pruning (preserves coverage diversity)
fuzzer-tool minimize ./target -d corpus -c --rate-distortion --target-frac 0.95
```

### Irreplaceable Seeds

Seeds placed in `corpus/irreplaceable/` are never pruned by minimization. When the minimizer's set-cover identifies mandatory (keystone) seeds that uniquely cover edges, those seeds are promoted to irreplaceable — copied to `irreplaceable/` and the original in `seeds/` is removed. On subsequent runs they are loaded alongside regular seeds and bypass all pruning logic.

Use `corpus/irreplaceable/` for seeds that must always be in the active set (e.g., known reproducers, structural format seeds).

## Test Suite

2532+ tests covering all modules, including 67 regression tests for historical bugfixes (`tests/test_regressions.py`). Run with:

```bash
pip install -e ".[dev]"
pytest
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
ruff format src/ tests/
```

## Benchmarking

Compare fuzzer configurations on a target:

```bash
# 4-way comparison: baseline vs enhanced vs enhanced+ vs optimal
tools/bench.sh targets/png_read 10000
```

Configurations:
- **baseline**: no features
- **enhanced**: elo + bandit + mopt
- **enhanced+**: all enhanced + markov + replicator + shapley + renyi + transfer-entropy + grammar
- **optimal**: elo + mopt + replicator + markov ensemble (orders 0,1,2,3) + markov-gen
  - Best edge coverage at -n 1k (sweep-validated: 74 edges vs 61 baseline, 70 enhanced+)

For a broader sweep across individual features and many combinations (instead
of these four named configurations), use `tools/bench_sweep.sh`. Both scripts
share common helpers (SHM cleanup, log metric extraction, coverage
verification) from `tools/lib/bench_common.sh`.

### SMT Solver Evaluation (`--enable-smt-z3`)

The SMT solver (Z3) attempts to solve arithmetic constraints discovered by cmplog, generating inputs that satisfy specific branch conditions rather than relying solely on random mutations. **Concolic mode is now the default** (`--mod-solving concolic`), providing full constraint modeling with z3 across whole execution traces. Override with `--mod-solving heuristic` or `--mod-solving trace` if needed. The concolic whole-input solve is capped at `_CONCOLIC_MAX_BYTES` (32 KiB) in `core/smt_solver.py`: it models one z3 `BitVec` per input byte, and over multi-MB seeds that model alone transiently exceeds a GB of memory (measured ~1.3 GB spikes), enough to OOM the fuzzer. Oversized inputs skip the whole-input solve (the per-pair cmplog solving is unaffected).

**30k-iteration comparison, zero corpus, `targets/png_read_tracecmp_asan.so`:**

| Metric | SMT (z3) | No SMT |
|--------|----------|--------|
| Avg EPS | 485.4 | 444.3 |
| Corpus growth | 1→139 entries | 1→126 entries |
| SHM max edge IDs | 273 | 245 |
| Stalls | 1 (0.6% recovery) | 4 (16.2% recovery) |
| SMT solve rate | 10/39 (26%) | N/A |

**Verdict**: From a cold start, SMT provides a modest but real advantage — ~9% higher throughput, ~75% fewer stalls, and slightly higher edge coverage (273 vs 245 max edge IDs). The solver fires on ~26% of cmplog arithmetic constraints when starting from a fresh corpus (higher solve rate on simpler constraints). The effect is amplified over the pre-warmed case, where stale constraints reduce the solve rate to ~6%. At this scale the advantage is incremental, not transformative — the SMT overhead is negligible, so there is no reason to leave it off when cmplog is already enabled.

## Building Targets

```bash
# Build all targets (ASAN + no-ASAN executables and .so shared libraries)
tools/build_targets.sh

# ASAN only
tools/build_targets.sh --asan

# No-ASAN only (faster)
tools/build_targets.sh --fast

# Build .so targets with cmplog compiled in (for direct_lite compatibility)
tools/build_targets.sh --cmplog
tools/build_targets.sh --asan --cmplog        # ASAN + cmplog

# Build with compiler-inserted edge coverage and compiler-IR comparison tracing (requires clang)
tools/build_targets.sh --clang-scov
tools/build_targets.sh --asan --clang-scov    # ASAN + compiler-inserted coverage

# Build vendored (libpng+zlib) targets with compiler-IR comparison tracing
tools/build_targets.sh --vendor-tracecmp
tools/build_targets.sh --vendor-tracecmp --asan   # With ASAN (two-step build)

# Build a single target by name (saves time when iterating on one target)
tools/build_targets.sh --target ffmpeg_read
tools/build_targets.sh --target ffmpeg_read,test_target          # comma-separated
tools/build_targets.sh --target ffmpeg_read --target test_target # repeatable
tools/build_targets.sh --fast --target test_target               # combined with existing flags

# List all available target names
tools/build_targets.sh --list-targets
```

The build script compiles every target as both an executable and a `.so` shared library, in ASAN and no-ASAN variants:

- `*.so` (base, no suffix) — No-ASAN, directly loadable via `ctypes.CDLL()` for high-throughput in-process fuzzing
- `*_asan.so` — ASAN-instrumented, requires libasan (falls back to subprocess mode automatically)
- `*_nosan.so` — Explicit no-ASAN variant (backward-compatible, same as base)

Every invocation prints a **build feature matrix** before compiling: an always-on text table
listing each feature flag (`cmplog`, `tracecmp`, `clang-scov`, `vendor-tracecmp`, `distance`,
`ffmpeg-sancov`), the sanitizer variant set being built (ASAN/UBSAN/No-ASAN), and whether
optional target groups (fgrep, tailslayer, lz4, secp256k1) will build or be skipped based on what the
script found on disk — so the exact effect of the passed flags is visible at a glance.

**Dual vendored FFmpeg builds**: FFmpeg fuzz targets require coverage-instrumented
vendored static libraries. Since the vendored FFmpeg `.a` files are compiled with
`-fsanitize-coverage=trace-pc-guard,trace-cmp`, they embed undefined references to
ASAN runtime symbols when built with `-fsanitize=address`. To prevent false ASAN
detection in nosan targets, the build script maintains **two separate FFmpeg build
directories**:

- `vendor/ffmpeg/` — Coverage-only (no ASAN), for `*.so` (nosan) and `*_ubsan.so` targets
- `vendor/ffmpeg_asan/` — Coverage + ASAN, for `*_asan.so` targets

The build script (`build_vendored_ffmpeg_sancov` in `build_targets.sh`) manages
both variants. The ASAN variant is a source copy of `vendor/ffmpeg/` reconfigured
with `-fsanitize=address`. `build_simple_so_targets` selects the correct path based
on the `$suffix` parameter (`_asan` → `vendor/ffmpeg_asan/`, otherwise
`vendor/ffmpeg/`).

### Vendored libsecp256k1 target (secp256k1_read)

`targets/secp256k1_read.so` wraps the vendored libsecp256k1 v0.8.0
(`tools/vendor_secp256k1.sh` → `vendor/secp256k1/`), mirroring the lz4_read wiring:
a `.so`-only target built in `build_standalone_so_targets()` via
`compile_secp256k1_objects()` (library TUs compiled separately, without the shim,
then linked with `-Wl,--export-dynamic`). The build enables every module present
in `src/modules/` (ecdh, recovery, extrakeys, schnorrsig, …) with
`-DENABLE_MODULE_<UPPER>` per TU — no config.h or `-DSECP256K1_BUILD` needed
(`src/secp256k1.c` self-defines it).

The input layout is mode-bit flagged (byte 0 arms one or more of the DER/compact
signature, pubkey+ECDH, BIP-340 Schnorr, and recovery surfaces; byte 1 carries the
recovery recid; bytes 2.. are the payload, capped at 256). Length-sensitive parsers
probe truncation points so boundary coverage emerges, and
`secp256k1_context_static` + fixed well-known keys keep the target allocation-free.

**API discipline (found by fuzzing this target)**: libsecp256k1 *zeroes* its
opaque objects (`secp256k1_pubkey`, `secp256k1_ecdsa_signature`, …) when a parse
fails (`memset(pubkey, 0, …)` in `src/secp256k1.c`), and any later use of a zeroed
object hits the illegal-argument callback, which `aborts` on
`secp256k1_context_static`. A truncation loop that re-parses into the same object
clobbers a previously-parsed value with the zeroed result of a failed iteration —
the wrapper must keep the last *successful* parse in a separate object and never
touch the loop-scratch object afterwards. Verified: 52 edges / ~79% map saturation
on a 5k-exec campaign with zero crashes.

### Build-time Cmplog for .so Targets

By default, `--cmplog` uses `LD_PRELOAD` to intercept comparison functions, which requires a process boundary (fork+exec). For `.so` targets in `direct_lite` mode, this doesn't work — no exec occurs.

**Solution: compile cmplog into your .so at build time.** Link `cmplog_shim.c` (needs `-ldl`) alongside your target:

```bash
gcc -shared -fPIC -O2 \
    -include src/fuzzer_tool/adapters/afl_shim.c \
    src/fuzzer_tool/adapters/cmplog_shim.c \
    targets/lz4_read.c \
    -o targets/lz4_read.so \
    -llz4 -ldl
```

The fuzzer auto-detects the built-in cmplog by scanning for the `__cmplog_reset` symbol and keeps using `direct_lite` mode (no fork overhead). The log file is truncated between executions via `__cmplog_reset()`, which the fuzzer calls via ctypes after reading tokens.

Cmplog runtime log files are written to `--cmplog-workdir` (default `/tmp/<target>.cmplog`). Use a disk-backed path when `/tmp` is a small tmpfs to avoid filling it.

To verify cmplog is active from the .so itself, check the startup output:
```
[*] Cmplog: compiled into target .so (direct_lite compatible)
```

### Compiler-IR Comparison Tracing (trace-cmp)

Symbol-based cmplog intercepts libc functions, but at -O2 both clang and GCC fold small constant-length `memcmp`/`strcmp` into inline integer compares — no libc call exists to intercept. This is exactly the pattern for format-signature detection (PNG magic, protocol headers, etc.).

**Performance note**: cmplog can produce thousands of comparison pairs per execution
when the library code is heavily instrumented (e.g., with trace-pc-guard coverage).
Each execution writes CMP lines to a log file, and the Python `collect_tokens()`
parses them all (14-23ms for 5000 pairs). To prevent an EPS cliff, the fuzzer
uses **adaptive periodic collection**: once the pair pool exceeds 2000 entries,
cmplog data is collected only 1 in 20 iterations, amortizing the parsing cost
to ~1ms per iteration while still discovering new tokens.

**trace-cmp** uses Clang's `-fsanitize-coverage=trace-cmp` instrumentation to
insert callbacks at the IR level. It catches every `icmp` in the IR — inline
integer comparisons, byte checks, switch dispatch — none of which the libc
layer can see.

It does **not** recover folded `memcmp`/`strcmp` constants, at any
optimization level. SanitizerCoverage instruments IR `icmp`, and clang's
`ExpandMemCmp` is a CodeGen pass that runs *after* it, so the comparison
trace-cmp actually sees is `memcmp_result == 0`: it logs the literal pair
`(0, 1)`, and only later does the memcmp become `cmpl $0x6C504D43,(%rbx)`.
On an -O2 trace-cmp build of `cmplog_exercise.c`, 11 of 20 logged pairs were
that degenerate `(0, 1)` — pool noise, not input-to-state evidence.

**`-fno-builtin-<fn>` is what recovers the constants.** It keeps the call at
the PLT so the libc interceptors see the real operands, while leaving every
other -O2 optimization in place. `$NOBUILTIN_CMP` in `tools/build_targets.sh`
carries the flag set. The two layers are complementary, not alternatives.

Constants from `cmplog_exercise.c` reaching the pair pool, seed
`AAAAAAAAAAAAAAAA` (10 magic constants in the target):

| build | constants | unique operands |
|---|---|---|
| `-O2` | 0/10 | 5 |
| `-O2` + trace-cmp, preloaded shim | 0/10 | 5 |
| `-O2` + trace-cmp, linked shim | 0/10 | 12 |
| `-O0` | 9/10 | 21 |
| `-O2 $NOBUILTIN_CMP` | 10/10 | 24 |
| `-O2 $NOBUILTIN_CMP` + trace-cmp, linked | **10/10** | **36** |

**The shim must be linked in, not LD_PRELOADed.** `-fsanitize-coverage` links
compiler-rt's sancov runtime, which ships *weak no-op definitions* of
`__sanitizer_cov_trace_{,const_}cmp{1,2,4,8}`. The executable is searched
before LD_PRELOAD libraries in the global symbol lookup order, so those stubs
win and the preloaded shim is never reached — 20 call sites in the binary, 0
lines in the log. A strong definition in the same link beats the weak stub;
`build_tracecmp_targets` compiles `cmplog_shim.c` to an object and links it.

Both layers live in the same shim (`cmplog_shim.c`) and write to the same
`_CMPLOG_OUT` file; the collector parses both transparently.

```bash
# trace-cmp is ON by default when clang is available
tools/build_targets.sh --clang
tools/build_targets.sh --no-tracecmp        # opt out

# Explicit
tools/build_targets.sh --asan --cmplog --tracecmp --clang
```

Targets built by this path are suffixed `_tcg`
(`targets/cmplog_exercise_tcg`, `targets/tracecmp_target_tcg`, plus `.so`
variants). The build verifies after linking that the callbacks are `T`
(strong) rather than `W` (weak no-op), and that `memcmp` survived as a call.

The trace-cmp shim intercepts:
- `__sanitizer_cov_trace_cmp{1,2,4,8}` — typed comparison callbacks
- `__sanitizer_cov_trace_const_cmp{1,2,4,8}` — constant-operand variants
- `__sanitizer_cov_trace_switch` — switch statement tracing

#### Vendored trace-cmp targets (libpng + zlib)

The `--vendor-tracecmp` flag rebuilds zlib and libpng from `vendor/` with
`-fsanitize-coverage=trace-cmp,trace-pc-guard`, then links targets against
the instrumented static libraries:

```bash
tools/build_targets.sh --vendor-tracecmp            # Non-ASAN .so targets
tools/build_targets.sh --vendor-tracecmp --asan     # ASAN + tracecmp two-step build
```

Output goes to `targets/png_read_tracecmp.so`, `targets/zlib_read_tracecmp.so`,
`targets/gzip_read_tracecmp.so`, and (with `--asan`)
`targets/png_read_asan_tracecmp.so`. The separate `_tracecmp` suffix avoids
clobbering regular builds.

#### ASAN + tracecmp two-step build

When a target is compiled with both `-fsanitize=address` and
`-fsanitize-coverage=trace-cmp`, ASAN's LD_PRELOAD provides its own
`__sanitizer_cov_trace_cmp*` no-op stubs that would override the tracecmp
shim's logging implementations. The fix: compile `tracecmp_shim.c` with
`-fvisibility=hidden` and link it INTO the target `.so` so the callbacks
resolve locally rather than through the PLT/GOT:

```bash
# Step 1: compile tracecmp shim with hidden visibility
clang -O2 -g -fsanitize=address -fvisibility=hidden -fPIC -c \
    src/fuzzer_tool/adapters/tracecmp_shim.c -o /tmp/tracecmp_shim.o

# Step 2: compile target + link shim together
clang -O2 -g -fsanitize=address -fsanitize-coverage=trace-cmp,trace-pc-guard \
    -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
    -o targets/my_target_asan.so targets/my_target.c /tmp/tracecmp_shim.o \
    vendor/libpng/.libs/libpng16.a vendor/zlib/libz.a -lm
```

The fuzzer auto-detects the compiled-in tracecmp symbols and uses
`direct_lite` mode with ASAN LD_PRELOAD:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 \
python -m fuzzer_tool fuzz targets/my_target_asan.so -c --cmplog -d corpus
```

#### Auto-detection

The fuzzer auto-detects tracecmp targets by scanning for
`__sanitizer_cov_trace_cmp1` in the binary. When found:
- Direct_lite mode is used (no subprocess overhead)
- The tracecmp shim is preloaded into the process before the target .so
- `_CMPLOG_OUT` is set before loading so the constructor opens the log file
- After each execution, the 256KB internal buffer is flushed to disk and
  comparison operands are extracted as dictionary tokens

### Compiler-Inserted Edge Coverage (`--clang-scov`)

The default coverage scheme uses hand-placed `__afl_map_edge()` calls in
wrapper targets. This only covers the wrapper code — the fuzzer cannot see
which internal code path executed inside the library being fuzzed.

**Clang `-fsanitize-coverage=trace-pc-guard`** solves this by having the compiler
insert `__sanitizer_cov_trace_pc_guard()` at every edge, which the runtime shim
(`afl_shim.c`) delegates to `__afl_map_edge()` → the AFL SHM bitmap.

**Important**: Clang zero-initializes guard variables by default. The shim's
`__sanitizer_cov_trace_pc_guard_init` **must** assign each guard a unique non-zero
value. Without this, `__sanitizer_cov_trace_pc_guard` returns immediately on
`*guard == 0` and every edge is silently skipped — the most common reason for
"0 edges discovered" with trace-pc-guard.

The `--clang-scov` flag or the default `.so` build (when clang is available) passes
`-fsanitize-coverage=trace-pc-guard` to fgrep library compilation. Fgrep `.so`
targets now get full compiler-inserted edge coverage automatically.

```bash
# Build with compiler-inserted edge coverage (requires clang)
tools/build_targets.sh --clang-scov

# Combined with ASAN
tools/build_targets.sh --asan --clang-scov
```

This builds two variants of library-wrapping targets (png_read, zlib_read,
gzip_read, jpeg_read) using vendored library sources compiled with sancov
instrumentation. The vendored sources live in `vendor/` and are compiled
as `.o` files with the same flags, then linked statically.

The existing manual `__afl_map_edge()` calls in wrappers remain — they become
named semantic checkpoints on top of full automatic coverage.

For targets with source already compiled by the build script (fgrep, tailslayer),
the sancov flag is applied directly to their compilation. fgrep is vendored from
[daedalus/fgrep](https://github.com/daedalus/fgrep) into `vendor/fgrep/`.

## Troubleshooting

### Zero edges discovered (ASan + LD_PRELOAD conflict)

If the fuzzer runs but reports `edges: 0` and `map: 0.0%`, the target is likely crashing before AFL instrumentation initializes. The most common cause is `LD_PRELOAD` entries (e.g. `ksm_preload.so`) that load before the ASan runtime, triggering the error:

```
ASan runtime does not come first in initial library list
```

The fuzzer strips conflicting `LD_PRELOAD` entries automatically, but if you set `LD_PRELOAD` manually, ensure it does not contain sanitizer-incompatible libraries. Verify by running:

```bash
python3 -c "
import os, sys; sys.path.insert(0, 'src')
from fuzzer_tool.adapters.process import _clean_env
print(_clean_env(os.environ).get('LD_PRELOAD', '(stripped)'))
"
```

If this prints `(stripped)`, the environment is clean.

## Findings

Bugs discovered by fuzzing with this tool are documented in `docs/FINDINGS/`:

- **[FINDINGS/fgrep.md](FINDINGS/fgrep.md)** — Unsigned integer underflow in fgrep AVX2 search (3 bugs, severity MEDIUM)
- **[FINDINGS/ffmpeg.md](FINDINGS/ffmpeg.md)** — Reachable `av_assert0(0)` in FFmpeg 7.1.3 when `avcodec_send_packet` is called on a subtitle decoder (severity HIGH — denial-of-service via 46-byte crafted input)

## License

MIT

</description>
<｜｜DSML｜｜parameter name="file_path" string="true">/home/dclavijo/my_code/fuzzer-new/docs/DEEP_DIVE.md
