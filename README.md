# fuzzer-tool

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/daedalus/fuzzer)

Coverage-guided binary fuzzer with static target analysis, statistical novelty scoring, Markov chain generation, Monte Carlo mutations, kernel crash verification, and format-aware grammar mutations.

Honest Caveats:
- This fuzzer is developed entirely with AI-assistance.
- Probably the most complex and dense fuzzer from the information theory standpoint but also the slowest. The tradeoff is speed for edge discovery novelty.
For production and sensitive binaries using AFL family fuzzers is the best course of action. 

## Features

### Mutation & Generation
- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit, signed + unsigned boundary), arithmetic (1/2/4/8-byte, LE/BE), block insert/delete/duplicate, bit-offset flip/span (arbitrary bit positions for DEFLATE/JPEG), havoc mode (with stall-recovery escalation), TLV-aware mutation, token shuffle, security-sensitive string injection (44 curated SQL/XSS/traversal/format strings), magic values table (229 boundary values across all widths/endians), in-place ASCII number arithmetic, chunk shuffle (boundary-preserving), compound dictionary insert, punctuation insertion
- **Operator performance**: `type_replace` uses a precomputed 256-byte translate table (184x faster), PNG/BMP random generation uses `random.randbytes()` instead of Python loops (16x faster), `colorization` uses a module-level lookup table
- **Length boundary operator**: systematically tries input lengths at boundary values (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024, 4096) — discovers length-sensitive unsigned integer underflows
- **Unsigned boundary values**: interesting values include small values (0-5) and unsigned max values (0xFF, 0xFFFF, 0xFFFFFFFF) for triggering unsigned arithmetic underflows
- **Crash-MI-guided mutation**: CrashMITracker identifies byte positions and values correlated with crashes, biasing mutation position selection and interesting value selection toward crash-relevant bytes
- **CrashMITracker memory pruning**: automatically caps per-position byte-value tracking to top 32 entries every 500 execs — prevents tracker JSON from growing unbounded (was 7.5MB per 15k execs, now ~500KB)
- **Weighted length distribution**: length_boundary operator weights small lengths (0-16) 10:1 over large ones (512+) — 4096-byte inputs dropped from 4.7% to 0.5% of picks, stabilizing EPS
- **Corrupted state recovery**: seed_meta entries with keys > 256 chars (tracker JSON loaded as corpus) are skipped on load and save — self-heals bloated state.json on first run (15MB → 6KB)
- **Grammar-aware mutations**: format-specific structure-aware mutations for PNG (IHDR, IDAT, CRC, filter types, interlace), JPEG (SOF, DHT, DQT, DRI, SOS, scan data), BMP (header fields, pixel data), gzip (header flags, deflate stream, trailer, extra fields), and zlib (CMF/FLG header, deflate stream, Adler-32 trailer)
- **Tree mutator** (`lightweight_tree_mutate`): Radamsa-style delimiter-based tree mutations (delete, duplicate, swap, stutter) with correct round-trip invariant — unmatched delimiters are preserved, never healed
- **FrameShift**: automatic length-field tracking — discovers and adjusts length/count fields during insertions/deletions, applied as universal post-processing after every mutation
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus, generate statistically similar inputs, persist across runs
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Surprisal-weighted rewards**: all scheduling mechanisms (bandit, MOpt, Replicator, Elo) weight discovery credit by `1 - bitmap_density` — rare discoveries in sparse coverage regions get more credit than discoveries near already-saturated areas
- **Perplexity-gated generation**: model quality dynamically scales generation rate (more generation when model is lost, less when well-calibrated); rejects extreme-perplexity outputs as pure noise

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
- **SHM front-header metadata**: the SHM segment begins with a fixed 24-byte header (offset 0–23), followed by the edge table at offset 24. Layout: `stack_depth` (u32 @0), `_pad0` (u32 @4), `path_hash` (u64 @8), `edge_count` (u64 @16). The front header is never moved, even when the edge table is resized — only the table grows, keeping metadata access O(1) and resize-safe. `edge_count` is a **cumulative** insert-only counter that survives across executions: the C shim maintains a static `__afl_total_edge_count` that is never reset, and writes it to the SHM header on each new-slot insertion. This provides a correct O(1) fast-path for coverage-change detection — `is_new_coverage()` reads 8 bytes and compares against the last known value; if the count differs, the slow path determines whether any edge_ids are genuinely new. The previous per-execution counter caused false-negatives when consecutive executions touched the same number of distinct edges (fixed).
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
- **CRPS scoring**: proper scoring rule for execution time calibration (fixed indicator direction bug)

### Distribution Diagnostics
- **Running statistics** (`core/running_stats.py`): Welford/Pébay online algorithm for O(1) mean, variance, stddev, skewness, and excess kurtosis — unbounded or sliding-window variants
- **Execution time tail-risk detection**: skewness > 2.0 flags algorithmic-complexity inputs (regex backtracking, hash-flood) that occasionally trigger big execution-time excursions
- **Critical slowing down skewness tier**: three-signal detector (variance + autocorrelation + skewness) — rising right skew upgrades the verdict to "approaching transition, and it looks productive"
- **Per-operator reward moments**: UCB-style exploration bonus (`mean + k * stddev`) with kurtosis-scaled stability guard — high-kurtosis operators require more observations before trusting their stddev-based bonus
- **Format learner z-score gate**: replaces fixed `delta != 0` threshold with z-score-based outlier detection; MAD fallback under high kurtosis for robustness against zero-inflated coverage deltas
- **Corpus bloat early-warning**: rising right skew in seed file sizes is a leading indicator of bloat that precedes RSS threshold tripping
- **Bounded memory structures**: all accumulative data structures (correlation matrix, coverage timeline, cmplog tokens/pairs, kernel crashes, Shapley attribution edges, stderr buffer, seed secretary, seen hashes) are capped via module-level constants — RSS plateaus instead of growing linearly with exec count
- **Report distribution diagnostics**: stddev, skewness, and kurtosis for exec time, discovery rate, per-operator rewards, and seed sizes

### Scheduling Intelligence
- **Seed-level energy multiplier** (`SeedScorer`): AFL++ power schedules (FAST/COE/RARE/MMOPT/LIN/QUAD) scale `mutations_per_input` per seed — fast seeds with high coverage get more mutation attempts, heavily-fuzzed seeds get fewer. Honggfuzz power factors (novelty decay, density, fertility, freshness, CMP progress, entropy penalty, timeout penalty) applied multiplicatively on top of schedule scoring
- **Elo arbitration** (`--elo`): combined operator + seed scheduling via Bayesian Elo rating system. All available strategies (bandit, mopt, replicator, cem for operators; weighted, pareto, format, ga, qea, bayesian, markov for seeds) run in shadow; Elo Thompson-samples which to trust each iteration. Ratings decayed periodically to model non-stationarity
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
- **Hamming bitmap distance**: fast byte-level seed-to-seed similarity on edge bitmaps
- **Near-duplicate detection**: finds seed pairs with near-identical coverage via Hamming + LSH

### Information Theory
- **Mutual information** (`--mi-guided`): I(byte_position; coverage) guides mutation toward positions that actually control code paths
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

### Crash Analysis
- **Sanitizer detection**: automatic ASAN/MSAN/TSAN/LSAN/UBSAN crash classification
- **Kernel crash verification**: async dmesg streaming for kernel-level crash detection
- **Crash minimization**: delta-debugging with signature-matching to prevent drift to unrelated bugs
- **Corpus minimization**: greedy set-cover over SHM edge bitmaps (`minimize` subcommand)
- **Crash exploitability tiers**: ASAN_EXPLOITABILITY classification in reports
- **Levenshtein crash clustering**: groups crashes with similar stack traces (same root cause, different offsets)
- **Fuzzy corpus similarity**: Hamming + Levenshtein + 4-gram Jaccard for crash-to-corpus nearest-neighbor search
- **Crash stack hash**: hashes last 3 nibbles of each PC in top 7 frames (14 with sanitizers) for deduplication; single-frame crashes masked to prevent false uniqueness
- **Blocklist/allowlist** (`--crash-blocklist`/`--crash-allowlist`): skip known crash stack hashes or override blocklist for specific crashes
- **Smaller crash replacement** (`--save-smaller`): replace crash triggers with smaller inputs for the same stack hash

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
- **Misc auto-stats**: stat line shows `pruned:2 kcrash:1 dup:5` — corpus auto-prunes, kernel crashes, duplicate rejections
- **Bayesian stats** (`--bayesian`): stat line shows `bayes: 695 seeds 5500 obs` — seed quality tracking state
- **Markov context count**: stat line shows `ctx: 562347` — contexts seen by the Markov chain/ensemble
- **Replicator dynamics** (`--replicator`): stat line shows `rep: dom=bit_flip ops=78` — dominant operator and active operator count
- **MOpt PSO** (`--mopt`): stat line shows `mopt: 5p` — active particles in PSO swarm
- **Jaccard index**: corpus redundancy metric (`| jac: 0.XX`)
- **Diversity score**: Wasserstein spatial diversity (`| div: N`)
- **`--report` flag**: full explainability report with coverage, mutations, perplexity, corpus health, edge map
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

# Multi-target with glob — skips .c/.h/.py automatically
fuzzer-tool fuzz 'targets/fuzz_*' -c -d corpus/fgrep

# Two-pass workflow: fast fuzz without ASAN, then verify crashes with ASAN
fuzzer-tool fuzz targets/fuzz_*_nosan -c -d corpus/fast/
fuzzer-tool verify targets/fuzz_search_pipeline corpus/fast/crashes/

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
| `-g GRAMMAR` | Grammar-aware mutations (built-in: png, json, http_request, elf) |
| `--cmplog` | Comparison tracing via LD_PRELOAD (or compile `cmplog_shim.c` into target .so for direct_lite compatibility) |
| `--markov-gen` | Markov-generated seeds (rate adapts to model quality via perplexity) |
| `--mc-bandit` | Thompson sampling operator selection (Brier score calibration) |
| `--mc-cem` | Cross-Entropy Method byte distribution |
| `--mopt` | MOpt PSO operator scheduling (alternative to bandit) |
| `--replicator` | Replicator dynamics operator scheduling (evolutionary game theory) |
| `--shapley` | Shapley value operator attribution (fair credit distribution) |
| `--mi-guided` | Mutual information guided mutation (target high-MI byte positions) |
| `--renyi-weight` | Rényi entropy weighting in seed selection (boost cold-edge seeds) |
| `--transfer-entropy` | Transfer entropy causal tracking (byte→edge influence detection) |
| `--inprocess` | Persistent subprocess mode (auto-restart on crash) |
| `--resume` | Resume from saved state |
| `--crash-codes N` | Additional exit codes to treat as crashes |
| `-j N` | Parallel fuzzing with N workers |
| `--elo` | Elo arbitration between operator strategies (bandit/mopt/replicator/cem) and seed strategies (ga/qea/weighted/pareto/format/bayesian/markov) |
| `--sensitivity` | Per-byte sensitivity analysis (Lyapunov exponent) for mutation targeting |
| `--secretary` | Secretary-problem optimal stopping for seed/operator/corpus scheduling |
| `--bayesian` | Bayesian methods: Thompson-sampled seed selection, hierarchical operator priors, Bayesian coverage growth model |
| `--ga` | Genetic algorithm lifecycle mode (bounded population, speciation, crossover) |
| `--qea` | Quantum-inspired evolutionary algorithm (amplitude encoding, rotation gate feedback) |
| `--wfc` | Wave Function Collapse structural generation (chunk reordering, pixel generation) |
| `--enable-smt-z3` | Z3-based SMT solving for arithmetic constraint solving on cmplog pairs |
| `--hw-perf` | Hardware performance counters via perf_event_open (instructions, branches, misses) |
| `--schedule base\|fast\|coe\|rare\|mopt\|lin\|quad` | AFL++ power schedule |
| `--markov-order N` | Markov chain order(s), comma-separated (e.g. '0,1,2' for ensemble) |
| `--save-smaller` | Replace crash triggers with smaller inputs for the same stack hash |
| `--crash-blocklist FILE` | Skip crashes matching these stack hashes |
| `--crash-allowlist FILE` | Override blocklist for specific crash hashes |
| `--coverage-log FILE` | Append (timestamp, edge_count) lines for coverage-over-time plots |
| `--coverage-report FILE` | Dump edge coverage map to JSON on exit |
| `--max-corpus N` | Auto-minimize corpus at N entries |
| `--corpus-bust` | Resize corpus seed lengths to truncated normal distribution N(mean, std), capped at `--max-len` |
| `--bust-mean FLOAT` | Target mean for normal distribution (default: max_len/2) |
| `--bust-std FLOAT` | Target std for normal distribution (default: max_len/6) |
| `--bust-pad {repeat,zero,random}` | Padding mode for undersized seeds; repeat cycles existing bytes (AFL-style), zero pads with \\0, random appends uniform random bytes |
| `--replay-n N` | Replay each crash N times for reproducibility scoring |
| `--report [FILE]` | Generate explainability report (stdout or file) |
| `--stats-interval N` | Print live stats and dump stats file every N iterations (default: 1000) |

## Subcommands

| Command | Description |
|---------|-------------|
| `fuzz` | Run coverage-guided fuzzing (default) |
| `rank` | Rank corpus seeds by interestingness (edge coverage, rarity, subsumption) |
| `minimize` | Minimize corpus by removing redundant inputs |
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
- `state.json` — exec counts, crash sigs, op stats, seed metadata, lineage depths
- `edge_tracker.json` — per-seed edge coverage, cumulative edges, global hit counts, hit counts
- `markov.json` — persisted Markov chain transitions
- `mi.json` — mutual information tracker (byte-to-coverage correlations)
- `elo.json` — Elo ratings for operator and seed strategies
- `qea.json` — QEA population amplitudes and generation state
- `ga.json` — GA population, generation, and fitness state

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

## Test Suite

2398+ tests covering all modules, including 67 regression tests for historical bugfixes (`tests/test_regressions.py`). Run with:

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

The SMT solver (Z3) attempts to solve arithmetic constraints discovered by cmplog, generating inputs that satisfy specific branch conditions rather than relying solely on random mutations. **Concolic mode is now the default** (`--mod-solving concolic`), providing full constraint modeling with z3 across whole execution traces. Override with `--mod-solving heuristic` or `--mod-solving trace` if needed.

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
```

The build script compiles every target as both an executable and a `.so` shared library, in ASAN and no-ASAN variants:

- `*.so` (base, no suffix) — No-ASAN, directly loadable via `ctypes.CDLL()` for high-throughput in-process fuzzing
- `*_asan.so` — ASAN-instrumented, requires libasan (falls back to subprocess mode automatically)
- `*_nosan.so` — Explicit no-ASAN variant (backward-compatible, same as base)

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

To verify cmplog is active from the .so itself, check the startup output:
```
[*] Cmplog: compiled into target .so (direct_lite compatible)
```

### Compiler-IR Comparison Tracing (trace-cmp)

Symbol-based cmplog intercepts libc functions, but GCC -O2 inlines small constant-length `memcmp` into integer compares — no libc call exists to intercept. This is exactly the pattern for format-signature detection (PNG magic, protocol headers, etc.).

**Performance note**: cmplog can produce thousands of comparison pairs per execution
when the library code is heavily instrumented (e.g., with trace-pc-guard coverage).
Each execution writes CMP lines to a log file, and the Python `collect_tokens()`
parses them all (14-23ms for 5000 pairs). To prevent an EPS cliff, the fuzzer
uses **adaptive periodic collection**: once the pair pool exceeds 2000 entries,
cmplog data is collected only 1 in 20 iterations, amortizing the parsing cost
to ~1ms per iteration while still discovering new tokens.

**trace-cmp** solves this by using Clang's `-fsanitize-coverage=trace-cmp` instrumentation, which inserts callbacks at the IR level — after the compiler has already inlined/folded comparisons. This catches every `icmp` that survives optimization.

Both shims coexist: symbol-based (cmplog_shim.c) for explicit libc calls + compiler-based (tracecmp_shim.c) for inlined comparisons. They export different symbols, write to the same `_CMPLOG_OUT` file, and the collector parses both transparently.

```bash
# Build targets with trace-cmp (requires clang)
tools/build_targets.sh --tracecmp --clang

# Build with both cmplog and trace-cmp
tools/build_targets.sh --asan --cmplog --tracecmp --clang
```

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

Bugs discovered by fuzzing with this tool are documented in `docs/`:

- **[FINDINGS.md](docs/FINDINGS.md)** — Unsigned integer underflow in fgrep AVX2 search (3 bugs, severity MEDIUM)
- **[FINDINGS-ffmpeg.md](docs/FINDINGS-ffmpeg.md)** — Reachable `av_assert0(0)` in FFmpeg 7.1.3 when `avcodec_send_packet` is called on a subtitle decoder (severity HIGH — denial-of-service via 46-byte crafted input)

## License

MIT
