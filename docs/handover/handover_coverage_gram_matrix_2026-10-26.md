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
