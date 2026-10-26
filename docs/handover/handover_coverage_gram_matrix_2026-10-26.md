# Coverage Gram matrices: B = A·Aᵀ and C = B·Bᵀ

**Date:** 2026-10-26
**Scope:** what the seed×edge incidence matrix A (S×E) gives us through its Gram matrix B = AAᵀ and through C = BBᵀ = B² = A(AᵀA)Aᵀ, and how that maps onto what the repo already has.
**Status:** analysis only. Nothing implemented, nothing measured on a real corpus.

## 1. Definitions

A[i,e] = 1 if seed i reaches edge e (binary; see §6 for hit counts). B is S×S, symmetric, PSD. G = AᵀA is the E×E edge co-occurrence matrix and shares B's nonzero eigenvalues.

## 2. What B gives

| Quantity | Formula | Meaning |
|---|---|---|
| Coverage size | B[i,i] | edges seed i covers |
| Overlap | B[i,j] | edges shared by i and j |
| Complementary | B[i,j] = 0 | no overlap |
| Subsumption | B[i,j] = B[i,i] | seed i's edges ⊆ seed j's (i is redundant) |
| Duplicate | B[i,i] = B[j,j] = B[i,j] | identical coverage |
| Jaccard | B[i,j] / (B[i,i] + B[j,j] − B[i,j]) | size-aware similarity |
| Cosine | B[i,j] / √(B[i,i]·B[j,j]) | size-independent similarity |
| Row sum | Σⱼ B[i,j] = Σₑ A[i,e]·dₑ | dₑ = seeds hitting e; low = covers rare edges |
| Rarity-weighted | A·D⁻¹·Aᵀ, D = diag(dₑ) | IDF-style; rare edges dominate |
| Rank | rank(B) = rank(A) | linearly independent coverage vectors |
| Spectrum | nonzero eig(B) = eig(AᵀA) | fast decay = low-dimensional coverage |
| Graph | B as weighted adjacency | components = coverage islands; Fiedler vector = natural corpus split |

## 3. What C adds

C = B² is a 2-step path count: C[i,j] = Σₖ B[i,k]·B[k,j]. Equivalently C[i,j] = Σₑ,f A[i,e]·G[e,f]·A[j,f].

- **Indirect relatedness.** B[i,j] = 0 but C[i,j] > 0: no shared edges, but both overlap a bridge seed k (contribution B[i,k]·B[k,j]).
- **Structural equivalence.** cos of B's rows, C[i,j]/√(C[i,i]·C[j,j]), says two seeds relate to the corpus the same way even when disjoint (alternative paths through one code region).
- **Hub weighting.** Row sum of C = Σₖ B[i,k]·rₖ, with rₖ the row sum of B. High = overlaps with hub seeds.
- **Spectral smoothing.** Eigenvalues become λ², same eigenvectors as B. Wider gaps make spectral clustering cleaner. Cᵏv converges to the top eigenvector (eigenvector centrality of seeds).
- **Effective dimension.** tr(C) = ‖B‖_F² = Σλ². Participation ratio (tr B)² / tr C ≈ number of independent coverage directions.
- **No new information.** rank(C) = rank(B) = rank(A) and the eigenvectors are identical to B's. C is a nonlinear reweighting; the new content is the path-based view.

## 4. Limits

- B holds pairwise intersections only. Union coverage and A itself are not recoverable (A only up to an orthogonal transform).
- S×S is quadratic. Use sparse A, MinHash/LSH, or a truncated SVD of A.
- C amplifies hubs quadratically and fills in even when B is sparse. Never materialize it: Cv = A(Aᵀ(A(Aᵀv))), O(nnz(A)) per product.
- Raw hit counts skew B toward hot loops (diagonal becomes Σ count²). Binarize, log-scale, or rarity-weight first.

## 5. Where this already lives in the repo

Checked on a fresh clone at HEAD `177eacf7`. None of it materializes B or C; each uses a cheaper proxy for one of the readouts above.

| Readout | Existing code | Gap |
|---|---|---|
| Edge-side Jaccard, leader clustering | `core/edge_clusters.py` (`CLUSTER_JACCARD`, `MAX_CLUSTER_CLASSES`, `LEAD_BLOCK`) | works on edges (the G side), not seeds |
| Seed cosine on hashed edge sets, rarity quality | `core/schedulers/seed_dpp.py` (`HASH_BITS = 8`, `q = (Σ 1/freq)^θ`, Cholesky d² update) | uses B's cosine only against the last K picks; no global spectrum |
| MinHash near-duplicate lookup | `services/corpus_manager.py` (`find_similar`, `min_jaccard`) | pairwise lookup, no subsumption test |
| Seed overlap threshold in picker | `services/seed_picker.py:2321` (`_overlap_min_jaccard`) | threshold only |

Not found: subsumption test from B, spectral/participation-ratio saturation signal, bridge-seed score, structural-equivalence clustering.

## 6. Candidate work, in order of expected value

1. **Participation-ratio / spectrum analyzer.** New analyzer under `core/analyzers/` (follow `analyzer_registry.py`). Truncated SVD of sparse A every N refits; report top-k singular values, PR = (Σσ²)² / Σσ⁴, and its trend as a saturation signal next to `analyzer_corpus_flux.py`. Cost O(k·nnz(A)). Falsifiable on synthetic corpora: duplicated seeds must lower PR, orthogonal seeds must raise it.
2. **Subsumption in corpus minimization.** Exact test B[i,j] = B[i,i] on sparse A (or per-edge-owner counts) as a cheap pre-pass before the set-cover work in `services/minimize.py`. Needs a check that the existing minimizer does not already catch it.
3. **Bridge-seed score.** Betweenness-like score from the 2-step term: Σ over pairs (i,j) with B[i,j] = 0 of B[i,k]·B[k,j], computed matrix-free. Feed as a seed-scheduler weight; needs a paired A/B before wiring (no arm has measurements yet).
4. **Matrix-free power iteration** for Cv and eigenvector centrality, shared with 1.

Prerequisites for all four:
- Decide binary vs log-scaled counts once, in one function, and reuse it. Rarity weighting (A·D⁻¹) is the default for 3.
- Cap S the way `edge_clusters.py` caps classes (`MAX_INCIDENCE_CELLS`, `REPORT_NNZ_BUDGET`) so no S×S array is ever built.
- Per Hard Rules in `AGENTS.md`: any random draw goes through the existing RNG conventions; persisted state goes through the state store.

## 7. Open questions

- Is the edge-class canonicalization (`EdgeCanonicalizer`) the right A to use? It merges edges that every seed hits together, which removes redundant columns but changes B's diagonal.
- Does the PR signal add anything over CorpusFlux's z-score? Needs a side-by-side on the fuzzgoat and FFmpeg corpora before it earns report space.
- Bridge seeds may already be what `pos_centroid` gaps and DPP picks surface from the edge side. Measure overlap of the three selections before adding a fourth.

## 8. Real-data results (fuzzgoat, 2026-10-26)

**Setup.** `tools/build_targets.sh --fast --clang-scov` on a fresh `vendor/fuzzgoat` clone (clang 18.1.3), `tools/corpus_fuzzgoat.py` (250 generated inputs: 125 valid JSON, 125 byte-level mutations), `tools/edge_diagnostic.py matrix --repeats 0 --save` for the per-seed (id, count) columns. A = 250 seeds × 797 edge ids, nnz 34,477, density 0.173. Ids carry context bits (`ctx_bits` 8; 560 context-free families), so A is context-sensitive. One target, one run, a generated rather than fuzzer-evolved corpus, no ground-truth tracer. Scripts: `gram.py`, `gram2.py` (not committed).

**B = AAᵀ**
- No seed pair is disjoint (0 of 31,125): 48 edges are hit by every seed and 63 by ≥90%. Mean Jaccard 0.526, median 0.467. After dropping edges owned by ≥90% of seeds, 425 pairs (1.4%) are disjoint.
- Subsumption works: 713 exact-duplicate pairs (133 redundant seeds) plus 22 strictly subsumed seeds; the B test alone removes 150 of 250 with no loss of union coverage. The tool's own greedy set cover keeps 49, so the B test is a safe pre-pass, not a minimizer.
- rank(A) = rank(B) = rank(C) = 114. GF(2) rank is also 114 and 117 distinct rows exist, matching the existing `matrix` output.
- Raw participation ratio is uninformative: PR(B) 2.17 vs 2.23 ± 0.00 under degree-preserving swap randomization (5 draws), with the top eigenvalue carrying 67% of the trace. That is the shared parser core, not corpus structure.
- It becomes informative once the core is down-weighted: core-stripped PR 4.36 vs null 5.32 ± 0.02; rarity-weighted (A·D⁻¹ᐟ²) PR 13.80 vs null 23.52 ± 0.15. The corpus is more concentrated than its degree margins predict.
- PR over corpus prefixes is flat (2.2, 2.1, 1.8, 2.0, 2.2, 2.2 at S = 25..250); rank grows 25, 41, 65, 87, 106, 114 while edges grow 550 to 797. Rank growth looks like the better saturation signal; neither was checked against future discovery.

**C = B²**
- Identity C = A(AᵀA)Aᵀ holds exactly; the matrix-free product matches the dense one to 2.6e-8.
- C adds almost nothing here: top eigenvector overlap with B is 1.000000, λ_C/λ_B² = 1.000000, hub score (row sum of C) vs row sum of B Spearman 0.997, and B-cosine vs C-cosine pair ranking Spearman 0.935. Average-linkage clusterings differ by ARI 0.89 (k=4) and 0.86 (k=8).
- Structural-equivalence cosine saturates: mean 0.969 vs 0.680 for direct cosine. With 17% density and a dense core, every seed looks alike to its B-row.
- Bridging is vacuous on raw A (no disjoint pairs) and trivial after core stripping: all 425 disjoint pairs are 2-step bridged. The bridge score correlates 0.86 with the B row sum and −0.34 with rare-edge ownership, so it is mostly a hub score.

**Negative result: validity.** AUC for valid vs mutated is 0.748 for coverage size B[i,i], 0.729 for the B row sum, 0.746 for the rarity row sum, 0.741 for the C row sum, 0.739 for the top rarity-B eigenvector. Everything tracks coverage size. Cluster purity against validity is 0.52 to 0.58 vs a 0.50 baseline for both B-cosine and C-cosine, at k = 2, 4, 8. The Fiedler split I first reported as 125/125 was a median-split artifact and is withdrawn.

**Effect on §6**
1. Spectrum analyzer: keep, but rarity-weighted or core-stripped and always against a swap-randomized null; raw PR is not usable. Rank growth is a second candidate. Neither is validated as a saturation signal.
2. Subsumption pre-pass: confirmed (150/250 on this corpus). Check it against `services/minimize.py` before adding.
3. Bridge-seed score: demote. On this data it duplicates the hub score.
4. C-based clustering: not worth it here. C is B reweighted, and the pair rankings agree at 0.935.
5. Before any of this ships, repeat on an evolved corpus and on a second target (png_read or gzip_read), since a generated 250-input corpus with a 48-edge shared core may not generalize.

## 9. Second target: png_read (2026-10-26)

**Setup.** libpng 1.6.48 and zlib 1.3.1 cloned from GitHub (`glennrp/libpng`, `madler/zlib`), compiled by hand with `-fsanitize-coverage=trace-pc-guard -fno-omit-frame-pointer` and linked with `targets/png_read.c` and `afl_shim.c` (`-D__AFL_CMPLOG=1`) into an executable. This was not `tools/build_targets.sh`: the script expects `vendor/libpng/.libs/libpng16.a` from an autotools build and `vendor_libpng.sh` downloads from hosts unreachable here. The system libpng was not used, because it is uninstrumented. Corpus: `tools/corpus_png.py --count 250`, which produced 156 files (104 start with the PNG signature, 52 are raw zlib `.bin` streams). Same collection as §8. A = 156 × 1829, nnz 48,834, density 0.171; 4 edges are owned by every seed, 251 are singletons, 632 have ≤3 owners. Fuzzgoat was re-run through the same generalized script and reproduces §8 exactly (PR 2.17, 4.36 vs 5.32, 13.80 vs 23.47; 150/250 removable). Script: `gram3.py`.

**Results**
- Subsumption/duplicates: 1,399 duplicate pairs (84 redundant seeds), 3 strictly subsumed, 87 of 156 (56%) removable by the B test, union coverage preserved (checked). Greedy set cover keeps 38. Most of the redundancy is the raw `.bin` seeds: they reach 5 edges in total (4 shared with every seed, 1 exclusive), because the wrapper rejects them at the signature check. 1,824 edges are reached only by signature-passing seeds.
- Raw PR is again uninformative: 1.95 vs null 2.04 ± 0.00. Core stripping is a no-op here (only 4 core edges): 1.95 vs 2.03. Rarity-weighted PR is 6.32 vs null 7.81 ± 0.02, the same direction as fuzzgoat (13.80 vs 23.47): the corpus is more concentrated than its degree margins predict.
- C vs B again: top-eigenvector overlap 1.000000, hub score rho 1.000, B-cos vs C-cos pair ranking rho 0.997. Mean C-cosine is 0.861 vs 0.449 for B, i.e. saturated.
- Components: after dropping the 4 shared edges, B has exactly 2 connected components, 104 signature-passing seeds and 52 raw ones. That is the "coverage islands" readout from §2 working, but trivially: one island is 52 copies of a rejected input. The 5,408 cross-island pairs are disjoint and C = 0 for all of them, so there is nothing for a 2-step bridge to bridge. The bridge score is constant. On fuzzgoat it was a hub score (§8); here it is undefined. Neither target gives a usable signal.
- Label recovery is perfect and uninformative: AUC for signature vs raw is 1.000 for coverage size, B row sum, rarity row sum and C row sum. Size-residualized AUC is 0.42 (rarity-B eigenvector 1) and 0.60 (row sum): no structure beyond size. Rarity-B-cosine clustering gets purity 1.0 at k = 2, 4, 8; rarity C-cosine gets 0.667 at k = 2 (the baseline, ARI vs B −0.018, an outlier gets split off instead of the islands) and 1.0 at k = 4 and 8 (ARI 0.92, 0.98).
- Prefix saturation (random order, mean of 5), seeds / edges / rank / PR: 15 / 1004 / 11.2 / 1.70; 39 / 1424 / 23.8 / 1.84; 78 / 1684 / 42.0 / 1.93; 117 / 1770 / 57.2 / 1.94; 156 / 1829 / 72.0 / 1.95. Rank saturates at 72, which is also the number of distinct rows in A, and PR is nearly flat.

**Limits.** Generated corpus, no mutations, no fuzzer-evolved seeds; the hand-built instrumentation was not cross-checked against `build_targets.sh`; ids are context-sensitive; one run per target; the signature-vs-raw split is a gift from the wrapper's own check, so it says little about general corpora.

**Effect on §6 and §8**
1. Subsumption pre-pass: now confirmed on both targets (60% and 56% of seeds removable, union preserved). Highest-confidence item. Check against `services/minimize.py` before adding.
2. Spectrum analyzer: raw PR is uninformative on both targets. Only the rarity-weighted PR separates from the null, and the effect is not tied to any outcome yet. Not ready; needs a link to future edge discovery before it can be called a saturation signal.
3. Bridge-seed score and C-based clustering: drop. On two targets C is B reweighted, the bridge score is a hub score or undefined, and C-cosine is saturated.
4. Connected components of B after stripping near-universal edges is cheap and found the real islands on png_read. Worth a diagnostic line, not a scheduler.
5. Still open: an evolved corpus, and a target where inputs aren't rejected wholesale at an early check (e.g. ffmpeg or sqlite).

## 10. Evolved corpora, 10k execs per target, incl. ffmpeg (2026-10-26)

**Setup.** Each target was fuzzed with `fuzzer-tool fuzz <target> -d <dir> --max-execs 10000` (default flags, one run each), starting from the generated seeds of §8/§9. The corpus analysed is every unique file under `<dir>/seeds/` outside `pruned/`, `irreplaceable/` and `timeouts/` (650, 477 and 765 files). It is not the fuzzer's live set: the fuzzgoat run's own report says 509 seeds. Collection, script and null model as in §8/§9 (`gram3.py`, now also with a `jsonc` mode that labels fuzzgoat seeds by `json.loads`, and a `none` mode).
- fuzzgoat and png_read: same binaries as before.
- ffmpeg: `tools/vendor_ffmpeg.sh --nosan --minimal` (clang, trace-pc-guard + trace-cmp; component set limited to mov/matroska/wav/flac/mp3/ogg and a few audio codecs), then `targets/ffmpeg_read.c` linked by hand against the four static libraries with `-lz -lbz2`. Not the versioned `ffmpeg_read_<ver>_*` targets of `build_targets.sh`. Seeds: 14 from `tools/gen_ffmpeg_803_seeds.py` plus 5 written by hand (3 WAV, 1 MP3 frame run, 1 FLAC header). Many of the 14 target formats are not compiled into the minimal build.

| | fuzzgoat | png_read | ffmpeg (minimal) |
|---|---|---|---|
| seeds × edges (density) | 650 × 2003 (0.081) | 477 × 2401 (0.197) | 765 × 8594 (0.170) |
| edges owned by all / ≥90% / singletons | 46 / 52 / 852 | 1 / 4 / 365 | 1 / 333 / 850 |
| removable by B test (union preserved) | 306 (47%) | 111 (23%) | 209 (27%) |
| seeds kept by B test / greedy set cover | 344 / 205 | 366 / 149 | 556 / 228 |
| rank(A) | 397 | 398 | 583 |
| PR(B) raw vs null | 3.06 vs 3.30 | 1.92 vs 1.97 | 2.04 vs 2.14 |
| PR(B) core-stripped vs null | 5.39 vs 7.38 | 1.93 vs 1.97 | 2.01 vs 2.27 |
| PR rarity (A·D⁻¹ᐟ²) vs null | 10.01 vs 25.81 | 9.90 vs 10.96 | 12.68 vs 14.66 |
| C vs B: hub rho / pair-cos rho | 0.996 / 0.957 | 1.000 / 0.973 | 0.999 / 0.961 |
| disjoint pairs after core strip | 5,261 (2.5%), all 2-step bridged | 22,895 (20.2%), none bridged | 764 (0.3%), none bridged |
| ARI(B-cos, C-cos), k = 4 / 8 | 1.000 / 0.829 | 0.132 / 0.189 | 1.000 / 0.996 |

**Reading**
- Subsumption survives evolution, but the share drops from 56–60% (generated corpora) to 23–47%, and the B test keeps 1.5–2.4× as many seeds as greedy set cover. It finds the lossless exact-subset cases only. Union coverage was preserved in all three.
- Raw PR stays near its null everywhere (within 3–8%). Rarity-weighted PR is below the null in all three, by a wide margin only on fuzzgoat (10.0 vs 25.8); on png_read and ffmpeg it is 10–14% below. Same caveat as §9: nothing ties it to future discovery.
- Rank grows nearly linearly in the corpus prefix instead of saturating (fuzzgoat 55 at 65 seeds to 397 at 650; ffmpeg 73 at 76 to 583 at 765), while PR is flat. On a 10k-exec corpus rank is tracking the number of distinct profiles, not a plateau.
- C is again B reweighted (top eigenvector overlap 1.000000 on all three). rank(C) comes out slightly below rank(B) (527 vs 583 on ffmpeg, 394 vs 397 and 397 vs 398 on the others): this is `matrix_rank`'s tolerance on a matrix whose spectrum has been squared, not a real rank drop (C = B² has B's rank).
- png_read is the one place C changes the answer, for the worse: rarity C-cosine clustering stays at the baseline purity (0.889) at k = 4 and 8 while B-cosine reaches 1.0 (ARI between them 0.13 and 0.19). Mean C-cosine is 0.951 vs 0.548 for B: saturated.
- Bridge score: vacuous or undefined in all three (fuzzgoat: all 5,261 disjoint pairs bridged but the score only tracks hubs; png_read and ffmpeg: no bridges). The 764 disjoint ffmpeg pairs equal S−1, consistent with one seed whose non-core edges nobody else shares; I did not look at which seed it is.
- Validity labels (fuzzgoat: 185 parse / 465 don't; png_read: 423 signature / 54 raw): AUCs for coverage size, B row sum, rarity row sum and C row sum are 0.78 / 0.82 / 0.78 / 0.83 on fuzzgoat and 1.000 on png_read. Size-residualized AUC is 0.37–0.65 in both: no structure beyond coverage size. Purity of the rarity-B and rarity-C clusterings on fuzzgoat is 0.72 vs a 0.715 baseline; raw B-cosine reaches 0.883 at k = 8, but purity rises mechanically with the number of small clusters and I did not control for that.

**Limits.** One 10k-exec run per target with default flags and no repeats; corpus is the retained unique files, not the live set; ffmpeg is a minimal component build linked by hand, with many seed formats not compiled in; no timing and no check against later discovery; labels exist only for fuzzgoat (JSON parse) and png_read (signature).

**Effect on §6, §8, §9.** No change to the ranking: (1) subsumption pre-pass stays the only item with a clear result, now on six corpora; (2) the spectrum analyzer stays unvalidated, and rank growth is not a plateau signal at this corpus size; (3) bridge-seed score and C-based clustering stay dropped. The "evolved corpus" open item of §9 is closed for these three targets; a target without an early wholesale reject, a longer run and a link to future edge discovery are still open.
