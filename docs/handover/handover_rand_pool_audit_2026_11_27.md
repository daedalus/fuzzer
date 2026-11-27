# Security & Correctness Audit — `src/fuzzer_tool/core/rand_pool.py`

| | |
|---|---|
| **Project** | [`daedalus/fuzzer`](https://github.com/daedalus/fuzzer) — information-dense, coverage-guided binary fuzzer |
| **Commit audited** | `c994adc188f181d9b0ca03971bfeb73d40962c4e` (`feat: CVE-derived mutators…`, 2026-11-27, branch `master`) |
| **File** | `src/fuzzer_tool/core/rand_pool.py` — 1,140 lines, class `RandPool` + 4 singleton-management functions |
| **Audit date** | 2026-11-28 |
| **Verification environment** | Python 3.12.16, numpy 2.1.3 (repo requires `numpy>=2.0`) |
| **Method** | Full manual read; call-site analysis (40+ modules); CLI/wiring analysis; git-history review; 44-case empirical verification script; project test suite run |

---

## 1. Executive summary

`rand_pool.py` is the fuzzer's single source of randomness — repo rule (`AGENTS.md`, Hard Rule 16): *"For random always use the prng in `src/fuzzer_tool/core/rand_pool.py`."* It implements a numpy/PCG64-backed batched RNG pool (4,096 pre-drawn uint32 words per refill, with a pre-computed mod-256 mirror), deliberately deterministic, never reading OS entropy.

The module is in materially good shape. Its genuinely tricky boundary logic — pool spanning, refill invalidation on reseed, `randbytes` chunking at exact pool multiples, 64-bit rejection sampling, Floyd sampling, stream domain separation via `SeedSequence` tags, and the aux Poisson stream — all **verified correct empirically** (44/44 checks), and the project's own 172-test suite passes. The code shows four prior audit-fix rounds in git history, and the comments document past defects with unusual honesty.

That said, the audit found **one high-severity design/default-configuration issue** and **three medium issues**:

1. **Default campaigns have zero entropy.** `seed=None` maps to the published constant `0x5EED_5EED`, and the CLI's `--seed` **defaults to 42** — so every default run of the fuzzer, on every machine, replays the identical mutation stream. For a fuzzer this defeats the instrument's purpose (R‑01).
2. **The modulo-bias safety comment is wrong by ~3.5 orders of magnitude**, and the actual threshold (`_MODULO_SAFE_MAX = 2^30`) admits up to **25 % relative bias** at the top of its accepted range — demonstrated empirically at 704σ (R‑02). Latent today (no current caller sits in the bad zone), but the documentation actively invites future callers into it.
3. **`categorical()` silently accepts negative and NaN weights** its sibling methods reject, producing degenerate sampling output (one arm permanently unreachable) (R‑03). Latent: its only current caller feeds it Dirichlet draws.
4. **`inject_entropy`'s determinism guarantee is conditional** on campaign topology (wall-clock mode, TCP-ingested seeds), and its mixed-in "history" is quantized to refill boundaries (R‑04).

No memory-safety, secret-handling, or injection issues exist in this file — it neither reads secrets nor processes untrusted input beyond numeric arguments; its failure modes are statistical and behavioral.

**Bottom line:** safe to keep using; fix R‑01 (default seeding policy) and the R‑02 documentation/threshold mismatch before they compound; harden `categorical` (R‑03) opportunistically.

---

## 2. Scope and methodology

- **In scope:** `rand_pool.py` at the commit above; its immediate integration surface — `services/fuzzer.py` (pool construction/wiring), `cli/commands.py` (`--seed`), `services/corpus_manager.py` (`inject_entropy` call site), and callers of the wide-range/sampling APIs (`core/mutations/*`, `core/dirichlet.py`).
- **Out of scope:** the other ~280 core modules (including `rng_health.py`, `randomness.py`, `exhaustive_pool.py`), the Rust components, and build tooling.
- **Procedure:**
  1. Line-by-line read of all 1,140 lines.
  2. Static call-site analysis of 40+ modules that touch `RandPool`, plus the CLI argument definitions and the singleton wiring in `Fuzzer.__init__`.
  3. Review of the file's git history (41 commits since 2026‑07‑24, four of which are prior audit-fix rounds: `3712812e`, `de86a607`, `ca1fa4ff`, `f4b9beec`).
  4. An empirical verification harness (44 assertions, all passing — summary in Appendix A) covering: modulo-bias magnitude, seeding/determinism, `inject_entropy` semantics, sampler uniformity (1M-trial χ² tests), `randbytes`/`_take` boundary behavior, API consistency, weighted/categorical validation, aux-stream independence, and entropy gates.
  5. Run of the project's own tests for this module (`test_rand_pool.py`, `test_rand_pool_modulo_hinf.py`, `test_rand_pool_samplers.py`, `test_randpool_cycle_audit.py`): **172 passed** in 3.19 s.

---

## 3. Architecture overview (as verified)

The pool dispenses 32-bit words from a 4,096-entry batch (`_pool`, numpy uint32) with a pre-computed mod-256 mirror (`_m256`) and Python-list mirrors (`_pool_l`, `_m256_l`) for fast scalar access. Three streams coexist:

| Stream | Backing | Used by |
|---|---|---|
| Pool-backed | `_pool` / `_m256` words | `random`, `randrange`/`randint` (+list forms), `randbytes`, `choice`, `_randbelow`, `random_between`, `sample` fast paths, Floyd sampling, small shuffles |
| Generator-backed | `self._rng` directly | `shuffle` (n ≥ 8), `sample` (k ≥ 3, non-Floyd), `weighted_choice_list`, `categorical`, continuous/distribution helpers |
| Auxiliary | `_aux_rng` (seed-derived, spawn-key tagged) | `poisson_array` only |

Seed handling: `None` → fixed constant `_NO_SEED_FALLBACK` (`0x5EED_5EED`); negative ints → tagged `SeedSequence` (`numpy` rejects raw negatives); the process-wide default singleton gets an extra tag so it never mirrors a `RandPool(seed)` built from the same campaign seed; the aux stream gets a spawn-key tag. `Fuzzer.__init__` (fuzzer.py:2728–2741) sets `self.seed = seed`, reseeds the default singleton, and builds the campaign pool `RandPool(seed=seed)`. The startup banner prints the seed (`[*] Seed: …`).

Entropy gates (opt-in `min_entropy` / `min_hinf` constructor args) measure the mod-256 batch after each refill, retry up to 3 times with a stream-derived reseed, then raise `RuntimeError` on starvation.

---

## 4. Findings summary

| ID | Severity | Title | Location |
|---|---|---|---|
| R‑01 | **High** (fuzzer effectiveness) | Default campaign entropy is fixed and published: CLI `--seed` defaults to 42; `seed=None` → constant `0x5EED_5EED` | `cli/commands.py:8097-8103`; `rand_pool.py:83-126` |
| R‑02 | **Medium** | Modulo-bias bound misstated ~3.5 orders of magnitude; `_MODULO_SAFE_MAX = 2^30` admits 25 % relative bias | `rand_pool.py:77-81`, fast paths at 541-542, 559-567, 419-423, 519, 663-665 |
| R‑03 | **Medium** (latent) | `categorical()` accepts negative/NaN weights; siblings validate | `rand_pool.py:986-1001` |
| R‑04 | **Medium** | `inject_entropy`: determinism guarantee conditional; mixed history quantized to refills; restarts aux stream | `rand_pool.py:348-389`; `corpus_manager.py:1450-1460` |
| R‑05 | **Low‑Med** (latent) | "Not thread-safe" singleton with no runtime guard | `rand_pool.py:43`, `1073-1140`, `_draw` 319-324 |
| R‑06 | **Low** | Scalar vs vectorized API error semantics inconsistent | `rand_pool.py:409-423`, `496-525`, `800-850`, `391-407` |
| R‑07 | **Info** | `random()` has 2⁻³² resolution, not 2⁻⁵³ | `rand_pool.py:425-427` vs `_random53` 460-468 |
| R‑08 | **Info** | Minor notes: gate-retry reseed drops stream tagging; `configure_rand_floyd()` default-arg disables; entropy-getter contracts differ | `rand_pool.py:280`, `102-108`, `391-399` vs `401-407` |

---

## 5. Detailed findings

### R‑01 [High — fuzzer effectiveness] Default campaigns have zero entropy

**Evidence.** Three independent facts compose into one behavior:

1. The module deliberately never reads OS entropy, and maps `seed=None` to the fixed, source-published constant `_NO_SEED_FALLBACK = 0x5EED_5EED` (rand_pool.py:83-86, 116-126). Verified empirically: two `RandPool()` instances produce **byte-identical streams**, equal to `RandPool(0x5EED5EED)`.
2. The CLI defines `-s/--seed` with `type=int, default=42` ("RNG seed for reproducibility (default: 42)", commands.py:8097-8103).
3. `Fuzzer.__init__` threads that seed into both the campaign pool and the default singleton (fuzzer.py:2728-2741), so both streams are pure functions of `42` by default.

**Impact.** Every default `fuzzer-tool` campaign — every user, every machine, every target, every rerun — replays the *identical* mutation stream from the first exec. Concretely, for a fuzzer:

- Re-running a campaign without manually choosing a new seed explores the same input space again; CPU hours are spent re-fuzzing an already-fuzzed stream. (The repo's own `--clock virtual` help text confirms the design intent that `--seed` makes runs replay *exactly*.)
- The stream is not merely deterministic, it is **published** — anyone with the source can precompute every mutation the default campaign will make, forever.
- The stochastic-search machinery in this repo (QEA, Thompson sampling, GA, MCTS schedulers, ~40 scheduler modules) is fed by this same stream — all of its diversity collapses to a constant when the seed is defaulted.

This is the opposite of the established AFL/libFuzzer posture: random-by-default with the chosen seed *printed* for reproducibility, deterministic only when explicitly asked.

**Mitigating context.** The behavior is *documented*, deliberate (module docstring: "every pool, seeded or not, produces a byte-for-byte reproducible stream"), test-pinned (`test_regression_unseeded_reproducibility.py` asserts two unseeded pools match and equal the explicit fallback), and the seed is printed in the banner. Any user passing `-s <random>` gets full diversity. It is also a genuine determinism architecture: nothing in `src/` touches the global `random` or `np.random` (their `test_regression_no_global_random.py` enforces this). The fix should therefore land in the CLI layer, not in the module.

**Recommendation.** Keep `RandPool` exactly as deterministic as it is — the defect is in the default *policy*, not the class. In the CLI: default `--seed` to a fresh OS-random value (`secrets.randbits(63)` or time/pid mix), print it in the banner (the banner already prints it), and let `-s` remain the explicit-replay switch. This converts "reproducible but pointless" into "diverse and reproducible-on-demand" with a one-line change. (Framing reference: CWE‑340, generation of predictable values — noting this is a fuzzing-effectiveness defect, not a cryptographic exposure: the pool generates test inputs, not keys.)

---

### R‑02 [Medium] Modulo-bias documentation contradicts its own math; threshold admits 25 % relative bias

**Evidence.** The safety comment (rand_pool.py:77-81) reads:

> `draw % n` over-weights the lowest `2**32 % n` values by one count in `2**32 // n`: <= 1e-7 relative bias for n <= 2**20, 2:1 near n = 2/3 * 2**32. Above this bound the range APIs rejection-sample (`_randbelow`) instead.
> `_MODULO_SAFE_MAX = 1 << 30`

The first clause defines **relative** bias: one extra count per `q = 2**32 // n`, i.e. relative bias = 1/q. Under that definition:

- For **n ≤ 2²⁰**: worst realized case is n = 1,048,575 (q = 4,096, r = 4,096 > 0) → relative bias **1/4096 = 2.44 × 10⁻⁴**, i.e. **2,441× larger than the documented "1e-7"**. (The 1e-7 figure only holds as *absolute* per-value probability excess, which is always ≤ 2⁻³² for any n — making the threshold meaningless under that reading.)
- For **n ≤ 2³⁰** (the actual gate): the worst zone is q = 4, e.g. n = 954,437,177 (r ≈ n/2) → relative bias **25 %** on the low half of the range. Empirically verified over 40,000,000 draws: P(x < r) = 0.55564 vs the uniform 0.50000 (biased model predicts 0.55556) — a **704σ** deviation from uniform.
- For the small ranges that dominate the hot path the bias genuinely is negligible: n = 256 → 6.0 × 10⁻⁸; n = 25 → 5.8 × 10⁻⁹ (verified).

The comment thus contains three mutually inconsistent bounds — "1e-7 at 2²⁰", "2:1 at ⅔·2³²", and the implemented "rejection above 2³⁰" — none of which describes what the threshold actually permits (25 %).

**Impact.** Latent. Audited callers all sit in safe territory: byte/word-scale ranges (bias ≤ 10⁻⁷-class) and wide spans such as `randint(0, 2**31)` in the png/jpeg/bmp mutators (width 2³¹+1 > 2³⁰ → rejection path, verified unbiased). But the comment actively invites the next contributor to call `randrange(n)` with any n ≤ 2³⁰ under the belief that bias is ≤ 10⁻⁷, and a distribution-silent 25 % skew in a scheduler or fitness-proportional selector would be very hard to notice downstream.

**Recommendation.**
1. Fix the comment to state the real formula and numbers: relative bias = 1/(2³² // n); ≤ 2.4 × 10⁻⁴ at n ≈ 2²⁰, up to 25 % near n ≈ 9.5 × 10⁸, 2:1 near ⅔·2³².
2. Pick a policy and make code and comment agree, e.g.: lower `_MODULO_SAFE_MAX` to 2¹⁶ (relative bias ≤ 1.5 × 10⁻⁵, rejection is cheap and vectorizable via a masked redraw), or keep 2³⁰ and state plainly that up to 25 % relative bias is accepted on the fast path. Option 1 is safer given how this class is the mandated RNG for everything.

---

### R‑03 [Medium, latent] `categorical()` accepts negative and NaN weights that its siblings reject

**Evidence.** `weighted_choice` (line 704) and `weighted_choice_list` (line 734) both raise `ValueError` on negative weights. `categorical` (986-1001) only checks `len(probs) == 0` and `cum[-1] <= 0`:

- Negative weights make the cumsum non-monotone; `np.searchsorted` on a non-monotone array is undefined behavior territory. Empirically, `categorical([3.0, -1.0, 3.0], 20000)` returned counts `[8014, 0, 11986]` — the middle arm is **permanently unreachable** and the two others are 60/40 skewed (uniform would be ~6,667 each).
- NaN weights pass the guard because `NaN <= 0` is `False`. `categorical([1.0, nan], 50)` degenerates to *all* draws = index 1, no exception.

**Impact.** Latent: the only current caller (`core/dirichlet.py:207`) feeds it Dirichlet draws, which are always finite and non-negative. But Hard Rule 16 funnels *all* future randomness needs through this class, and every other weighted-sampling method here validates. The failure mode is silent degenerate sampling — the kind of bug that quietly biases a scheduler for weeks.

**Recommendation.** One guard, O(n) against the existing O(n) cumsum:

```python
probs = np.asarray(probs, dtype=np.float64)
if probs.size and (not np.isfinite(probs).all() or (probs < 0).any()):
    raise ValueError("categorical: weights must be finite and non-negative")
```

---

### R‑04 [Medium] `inject_entropy`: conditional determinism, refill-quantized history, aux-stream restart

**Evidence.** `inject_entropy` (348-389) re-derives the whole stream as SHA-256(bit-generator state ‖ raw bytes) → `SeedSequence`. Wired at exactly one call site: `CorpusManager` mixes each newly admitted corpus seed's bytes into the campaign pool (corpus_manager.py:1450-1460), whose comment claims "for a fixed --seed, the sequence of admitted seeds is itself a deterministic function of the run."

Verified properties (all deterministic, none buggy in isolation):

- Same generator-state + same raw bytes → same new stream (reproducible).
- Different generator-states + same bytes → different streams.
- Empty `raw` is a no-op.

Three nuances matter for the audit:

1. **The "prior state" is the PCG64 bit-generator state, quantized to refill boundaries.** Two pools that consumed 50 vs 90 draws from the *same* 4,096-word batch (no refill in between) hold identical bit-generator states, so injecting the same bytes into both yields the **identical** new stream — the exact number of draws consumed does not enter the mix. The mixed history advances only every 4,096 draws. (The docstring says "a snapshot of the pool's current … state" — accurate but easy to over-read.)
2. **The end-to-end determinism claim is conditional on campaign topology.** With `--clock virtual` and no external seeds, the admission sequence is a pure function of the seed, and the comment holds. With `--tcp-listen-port` (externally supplied, timing-dependent seed ingestion) or `--clock wall` (host-time-dependent scheduling/admission), *which* inputs get admitted — and therefore the injection sequence — varies run to run, and every downstream mutation diverges. The module docstring's "byte-for-byte reproducible stream" is true of the pool object in isolation, but the wired system's reproducibility depends on the campaign mode.
3. **Side effect on the aux stream.** `inject_entropy` re-derives `_aux_rng` (line 387), restarting the `poisson_array` stream at every corpus admission. This is documented in `poisson_array`'s docstring, but it means post-injection Poisson draws depend on the *number of injections*, another coupling between corpus I/O events and scheduler randomness.

**Recommendation.** Document the conditions in the module docstring (reproducible *given a deterministic admission sequence*); consider including `self._idx` (and the buffered-word flags) in the hashed state if exact draw-position sensitivity is ever wanted; consider whether corpus events should feed a *derived* stream rather than reseed the shared campaign stream, which would decouple scheduler randomness from corpus-save timing entirely.

---

### R‑05 [Low-Medium, latent] Thread-safety is documented but unguarded

**Evidence.** The module states "The pool is not thread-safe" (line 43). `_draw` (319-324) does `self._idx += 1` on shared state; `_refill` swaps `_pool_l`/`_m256_l`. The class is exposed process-wide via `get_default_rand_pool()` (1081-1101).

Current call sites were audited for concurrency: the mutation loop and all scheduler/mutator draws are single-threaded; background threads (TCP seed inbox, frontier oracle, stderr readers, malloc-trim) never touch a `RandPool`; the `ThreadPoolExecutor` sites (`mutant_build.py`, `mutant_kill.py`, `frontier_cov.py`) compile/execute only — `mutant_build.py:355` constructs and consumes its own pool in the calling thread.

**Impact.** No live bug. But a future parallelized mutation loop (an obvious optimization) sharing one pool or the singleton would silently duplicate/skip draws — no exception, just correlated mutants and a non-reproducible interleaving. `get_default_rand_pool`'s "everyone must fetch the shared instance" design makes the singleton the natural thing a refactor would reach for.

**Recommendation.** Add a cheap ownership assertion (record creator thread id; assert in `_refill` under a `--debug` flag or `PYTHONDEV`), or provide a documented `threading.local` default pool for parallel contexts, so the contract fails loudly instead of silently.

---

### R‑06 [Low] Inconsistent error semantics between scalar and vectorized forms

**Evidence (all verified at runtime).**

- `randrange(0)` and `randint(5, 3)` raise `ValueError` — added deliberately, per their docstrings, to expose callers' empty-range bugs (commit `40463a5c`).
- The vectorized forms of the same operations — `randrange_list(0, 10)`, `randrange_list(-5, 10)`, `randint_list(5, 3, 10)` — silently return `[]`, re-hiding exactly the bug class the scalar forms were changed to surface.
- `sample(population, k)` clamps `k > n` down to `n`; `random.sample` raises `ValueError("Sample larger than population")`.
- "Not refilled yet" is reported three different ways: `pool_entropy()` returns `None`, `measure_entropy()` raises `RuntimeError`, `pool_min_entropy_now()` raises `RuntimeError`.

**Impact.** Minor. Empty-range bugs in list-form call paths pass silently; ported stdlib code may be surprised by clamping instead of raising. In a codebase this disciplined about exposing caller bugs, the asymmetry looks unintentional.

**Recommendation.** Make the list forms raise the same `ValueError` (or, if clamping is a deliberate fuzzer ergonomics choice, say so in all four docstrings); pick one not-refilled contract.

---

### R‑07 [Info] `random()` has 32-bit resolution

`random()` returns `_draw() / 2³²` — 2³² distinct values, granularity 2⁻³² (verified: every draw is an exact multiple of 2⁻³²). `_random53` (CPython's 27+26-bit construction) exists but only `geometric` uses it. Consequence: an event coded as `rng.random() < p` is *impossible* for p < 2⁻³² ≈ 2.33 × 10⁻¹⁰. Current epsilon-style callers use p ≥ 0.05, so there is no practical impact; this is a property to know when adding very-low-probability behaviors (use `_random53` there). The `geometric` docstring's tail-cap arithmetic (22× mean at p = 10⁻⁶ for the old 32-bit construction) checks out.

### R‑08 [Info] Minor notes

- **Gate-retry reseed drops stream tagging** (line 280): after an entropy-gate retry, the main generator is re-created from a plain stream-derived int seed rather than a tagged `SeedSequence`. Still a pure function of the original seed (deterministic), and the aux stream is deliberately not re-derived there; the tags' cross-stream-collision purpose only applies at construction. No action needed beyond awareness.
- **`configure_rand_floyd(enabled=False)`** (line 105): calling it without arguments silently *disables* Floyd sampling; a default of `True` would be less surprising.
- **`import hashlib` inside `inject_entropy`** (line 381): fine for a rarely-called path; noted only for style consistency.
- **Entropy-gate calibration is sound**: the docstring's batch statistics (H_inf ≈ 7.2 typical; 2.6 % of healthy batches < 7.0; 25 % < 7.1) are consistent with extreme-value computation for the max of 256 Poisson(16) counts (exact tail: P(max ≥ 33) ≈ 3.3 % maps the 7.0 threshold, P(max ≥ 30) ≈ 29 % the 7.1 threshold — the slightly lower simulated figures reflect bin dependence); the recommended ≤ 6.5 threshold corresponds to P(max ≥ 46) ≈ 1.9 × 10⁻⁷ per batch. Verified: healthy streams pass 6.5; an unsatisfiable 7.9 raises `RuntimeError` after 3 retries.

---

## 6. Verified-correct behaviors (positive results)

All of the following were verified empirically against the audited commit (44/44 checks passed):

| Area | Verification |
|---|---|
| `randbytes` boundaries | Exact pool-multiple requests never replay a batch (the `else: _idx = _POOL_ENTRIES` rewind guard works); `randbytes(8192)` equals two fresh batches concatenated; partial consumption stays aligned across the 4,096 boundary. |
| `_take` oversized batches | Spans refills correctly and returns copies (the historical truncation bug — commit `215c26e6` — stays fixed). |
| `reseed` | Discards up to 4,096 buffered old-stream draws; post-reseed stream equals a freshly constructed pool's stream. |
| Floyd sampling (`_floyd_indices`) | Uniform over *ordered* samples: n=5, k=3 over 60 outcomes, 1M trials × 3 seeds, χ²/df = 0.93–1.01 (p ≈ 0.46–0.62). k = n produces uniform permutations (χ²/df = 0.93). Linked-list insertion is correct. |
| `_pair` | Uniform over the 20 ordered distinct pairs at n = 5 (χ²/df = 1.14); always distinct. |
| `_randbelow` | 64-bit spans unbiased; top bit reachable (50.2 % set rate); power-of-two mask path bias-free; rejection loop mirrors CPython. |
| `random_between` | Rejection-sampled, uniform (χ² = 11.0, df = 9 over 100k draws). |
| Stream domain separation | Default-pool stream ≠ `RandPool(42)` stream (tag); negative seeds accepted (raw negatives would crash numpy) and ≠ +42; aux stream independent of main draws and advanced only by aux use. |
| `inject_entropy` | Deterministic given (state, raw); empty injection is a true no-op. |
| `weighted_choice` | Rejects negative weights; zero-weight entries never selected; the `bisect … hi = n-1` clamp prevents the float-rounding `IndexError` (CPython-mirroring fix, commit `ca478fed`) — survives weights of 1e300. |
| `randint_list` fast path | The `_m256` bulk path consumes the same words and returns the same values as the scalar `randint` path (stream-compatible). |
| `random_sequential` | Bit-identical to repeated `random()` calls. |
| Entropy gates | Healthy stream passes `min_hinf = 6.5`; unsatisfiable threshold raises after exactly 3 retries; retry reseed is stream-derived (deterministic) and correctly leaves the aux stream untouched. |
| No OS-entropy leakage | No `os.urandom`/`secrets`/`/dev/urandom`/RDRAND anywhere in the module (`os` is imported solely for `environ`); determinism claims hold. |
| Project tests | 172/172 pass for the module's four test files. |
| `sample(range(10M), 5)` | 0.02 ms — the numpy `choice(replace=False)` delegation is O(k) (Floyd internally), not O(n); no hidden scaling trap. |

Additionally noteworthy as design strengths: the `SeedSequence` tagging scheme for domain separation is a clean solution to cross-stream mirroring; the pool/list-mirror split with per-refill `tolist()` is a sound hot-path optimization; and the comments' regression archaeology (each historical bug is documented at its fix site) materially lowers the risk of reintroduction.

---

## 7. Test coverage assessment

The module is among the best-tested in the repo — 396 test files reference `RandPool`, and the four dedicated files alone carry 162 test functions (172 test cases, all passing). Observed gaps relative to the findings above:

- **No test pins the documented bias bound.** A test computing `max(1/(2**32//n))` for the accepted range — or asserting the comment's figures — would have caught R‑02's 2,441× misstatement.
- **`categorical` is exercised with zero-weight vectors, but never with negative or NaN weights** — precisely the unvalidated input class (R‑03); adversarial-weight tests exist only for `weighted_choice`/`weighted_choice_list`.
- **No concurrency tests** — consistent with the documented "not thread-safe" scope, but see R‑05. (The unseeded `seed=None` byte-identity property, by contrast, *is* pinned, by `test_regression_unseeded_reproducibility.py`.)

---

## 8. Conclusion

`rand_pool.py` is a carefully engineered, unusually well-documented hot-path RNG whose hard parts are demonstrably correct. Its problems are not in what it does but in what it promises: a default seeding policy that turns the fuzzer's default mode into a global replay (R‑01 — the one finding I would fix before the next release), a bias-safety comment that understates real bias by orders of magnitude while the threshold silently permits 25 % skew (R‑02), one unvalidated public API on the repo's mandatory RNG surface (R‑03), and a determinism narrative that is conditional on campaign modes the module itself doesn't control (R‑04). All fixes are small, local, and testable; none require architectural change.

---

## Appendix A — Empirical verification log (summary)

44 assertions, 44 passed, 0 failed. Environment: Python 3.12.16, numpy 2.1.3, commit `c994adc`.

| Group | Checks | Result |
|---|---|---|
| A. Modulo-bias math (worst-case bounds; 40M-draw empirical demonstration of the 25 % zone, z = 704σ; small-n sanity) | 5 | all confirmed the analysis |
| B. Seeding/determinism (unseeded identity, published-constant stream, tag disjointness, negative seeds, reseed invalidation) | 6 | pass |
| C. `inject_entropy` (refill-quantized history, determinism, no-op) | 5 | pass |
| D. Samplers (Floyd uniformity, `_pair`, 64-bit `_randbelow`, `sample` clamp) | 5 | pass |
| E. `randbytes` boundaries | 3 | pass |
| F. Scalar vs list error semantics | 5 | pass (documents the inconsistency) |
| G. Weighted/categorical validation | 6 | pass (documents the gap; [8014, 0, 11986] skew reproduced) |
| H. Stream model / aux stream / fast-path equivalence | 5 | pass |
| I. Entropy gates and uniformity | 4 | pass |
| J. Stress evidence (not assertions): Floyd at 1M trials × 3 seeds; `sample` scaling | — | χ²/df = 0.93–1.01, p ≈ 0.46–0.62; `sample(range(10M), 5)` = 0.02 ms |

## Appendix B — Reproducing the two headline findings

```python
import sys

sys.path.insert(0, "src")  # repo root on sys.path
import numpy as np
from fuzzer_tool.core.rand_pool import RandPool

# R-02: 25% relative bias inside the documented "safe" range (q = 4 zone).
n = 954_437_177
q, r = divmod(2**32, n)  # q=4, r=477_218_588
p = RandPool(seed=1234)
draws = p.randint_list(0, n - 1, 10_000_000)
print(float((np.array(draws) < r).mean()))  # ~0.5556  (uniform would be ~0.5000)

# R-03: categorical accepts invalid weights silently.
p2 = RandPool(seed=2)
out = p2.categorical(np.array([3.0, -1.0, 3.0]), 20_000)
print(np.bincount(out).tolist())  # [8014, 0, 11986] -- arm 1 unreachable

# R-01: default campaigns share one stream.
a = [RandPool().randrange(1 << 30) for _ in range(10)]
b = [RandPool().randrange(1 << 30) for _ in range(10)]
print(a == b)  # True -- everywhere, forever
```

*Audit performed on 2026-11-28 against commit `c994adc`. Findings R‑01/R‑02/R‑03/R‑04 are new relative to the four prior audit-fix rounds visible in the file's history (`3712812e`, `de86a607`, `ca1fa4ff`, `f4b9beec`).*

---

## Resolution (2026-11-27)

| ID | Status |
|---|---|
| R-01 | Fixed in CLI: `--seed` default `None` -> `resolve_seed` (random 63-bit, printed). `RandPool` unchanged. |
| R-02 | Fixed: `_MODULO_SAFE_MAX = 2**20`, comment rewritten with the real formula; wider ranges rejection-sample (seeded streams change for n > 2**20 only; buffers > 1 MiB take the scalar path in `randint_list`/`randrange_list`). |
| R-03 | Fixed: `ValueError` on negative/NaN/inf weights, after the existing `IndexError` sum check. |
| R-04 | Draw position added to the mixed state; conditional determinism documented in the module and `inject_entropy` docstrings. Aux-stream restart kept (documented). |
| R-05 | Opt-in owner check in `_refill` (`FUZZER_RAND_THREAD_CHECK=1` / `-X dev`); owner is the first refilling thread. |
| R-06 | Documented, not changed: list forms return `[]` on empty ranges (bulk callers size possibly-empty buffers), `sample` clamps `k > n`, not-refilled contract stated per method. |
| R-07 | Documented: `random()` resolution is 2**-32; use `_random53` for rarer events. |
| R-08 | `configure_rand_floyd()` default is now `True`. Retry-reseed tagging left as is (deterministic). |

Tests: `tests/test_regression_rand_pool_audit_2026_11_27.py` (22). Pre-existing, unrelated failure: `test_scheduler_convergence.py::TestSchedulerSeedability` (UniformScheduler).
