# Handover: Portable items from AIscripts → fuzzer-tool

**Date:** 2026-09-07
**Source:** https://github.com/daedalus/AIscripts
**Target:** https://github.com/daedalus/fuzzer
**Author context:** Same author; AIscripts is the experimental playground, fuzzer is the production-grade information-dense binary fuzzer.

## Summary

AIscripts contains many experimental algorithms. Only a small subset has a clean, high-value fit for the fuzzer’s core loop (seed → mutate → execute → measure → schedule). The fuzzer already ships a sophisticated Bloom filter (`core/bloom.py`), extensive bandits/schedulers under Elo arbitration, Markov, GA, constraint helpers, checksum learning, etc. Ports should be surgical, follow existing conventions in `AGENTS.md` / `operator_registry.py` / scheduler interfaces, and avoid new heavy dependencies on the hot path.

## Prioritized portable items

### 1. Cuckoo Filter (High priority) — `cuckoofilter.py`

**Why:**
Fuzzer uses Bloom for exec-dedup and near-duplicate detection. Cuckoo filters support **deletions**, have lower false-positive rates for the same space in many regimes, and store fingerprints. Useful for:
- Corpus / path / edge tracking that needs eviction
- Generational or sliding-window membership
- Alternative to Bloom when exact removal is required

**Port notes:**
- Original depends on `mmh3`. Prefer pure-Python / `hashlib` (matching `bloom.py` style) or make `mmh3` optional.
- Expose the same style of API as `BloomFilter` where possible (`add`, `query`/`contains`, `remove`, `update` check-then-add).
- Place in `src/fuzzer_tool/core/cuckoo.py`.
- Wire optionally behind the existing exec-dedup path or as a parallel structure.

**Status in this handover:** Cleaned pure-Python version provided in the accompanying patch series.

### 2. CVM F₀ Estimator (High priority) — `CVM.py`

**Why:**
Streaming approximate distinct-elements (F₀) count with tunable ε/δ. Complements existing Chao2 rarity, Renyi spectrum, rate-distortion, and edge-tracker modules without materializing the full set. Ideal for:
- Estimating unique edges / paths / comparison sites seen so far
- Novelty / coverage cardinality signals for seed quality or scheduling

**Port notes:**
- Tiny pure-Python implementation of the textbook CVM algorithm (arXiv:2301.10191).
- Place in `src/fuzzer_tool/core/cvm.py` (or under a `stats/` / `coverage/` subpackage if preferred).
- Integrate into `edge_tracker`, coverage regime, or seed-quality scoring.

**Status in this handover:** Cleaned version provided.

### 3. Minimal Feistel Network (Medium) — `minimal_feistel_network.py`

**Why:**
Compact, invertible 64-bit Feistel. Useful as:
- Building block for new adaptive / regularity operators that need bijective byte permutations
- Keyed deterministic transforms
- Structured mutation helper

**Port notes:**
- Pure Python, no extra deps.
- Place in `src/fuzzer_tool/core/feistel.py` or under `mutations/`.
- Can become a new operator or helper used by existing adaptive/havoc paths.

**Status in this handover:** Cleaned version provided.

### 4. Cassowary linear constraint solver (Medium) — `cassowary.py`

**Why:**
Classic incremental linear arithmetic solver. Fuzzer already has `field_constraints.py`, `path_constraints.py`, `cond_stmt.py`, frameshift, and cmplog-style solving. Cassowary can strengthen length-field repair, structural constraints, or path-condition handling.

**Port notes:**
- Needs careful integration; do not replace existing solvers wholesale.
- Review for numerical stability and performance on the hot path.
- Not included as a full drop-in in the first patch series (larger surface).

### 5. Meta Gradient Descent / MGD (Research / Medium-High) — `MGD.py`

**Why:**
Meta-optimizer that differentiates through a training (or campaign) process to tune meta-parameters. Could sit above the Elo + bandit layer to adapt operator weights, temperatures, elite fractions, etc.

**Port notes:**
- Requires a differentiable surrogate of campaign reward (edge discovery rate, crash rate, …).
- Higher effort; keep experimental / optional.
- Not included in the first patch series.

### 6. Lower / optional

| Item | Notes |
|------|-------|
| `semantic_cache.py` | Embedding + FAISS cache. Heavy (torch/FAISS). Better as offline corpus analysis or slow adaptive mode than hot-path. |
| `evolve.py` | LLM self-improvement loop. Fits “skills / AGENTS” meta-tooling, not runtime. |
| `binary_transformer.py` / `BNNN.py` | Binary / 1-bit nets. Research direction for learned mutators. |
| `findaes.py` + `crypto_key_scanner.py` | Useful post-crash / crypto-target skills under `skills/` or `tools/`. |
| `bloomfilter.py` / `mbf.py` | Superseded by existing `core/bloom.py`. |
| LLM compression (ZeroMerge, DLFloat, SeedLM, …), matrix kernels, PhyloLM, etc. | Orthogonal to binary fuzzing. |

## Integration conventions (must follow)

From `AGENTS.md` / existing code:

- New core modules live under `src/fuzzer_tool/core/`.
- New mutators register in `operator_registry.py` (category band + availability predicate).
- Schedulers implement the `select_op` / `record` / `bandit_stats` interface and register in the appropriate strategy list.
- Prefer pure Python or already-vendored deps; avoid new hard dependencies on the hot path.
- Surgical changes only; match naming, error handling, and comment style.
- Add tests under `tests/` following existing patterns.
- Document in CHANGELOG and, if user-visible, in README / SPEC.

## Recommended next steps

1. Apply the accompanying git-am patch series (Cuckoo + CVM + Feistel).
2. Add unit tests and a minimal integration point for Cuckoo (e.g. optional exec-dedup backend) and CVM (coverage cardinality signal).
3. Evaluate Cassowary and MGD in a later research branch.
4. Consider packaging crypto scanners as optional skills.

## Files in this handover package

- `docs/HANDOVER_AISCRIPTS_PORT.md` (this file)
- Cleaned modules under `src/fuzzer_tool/core/`:
  - `cuckoo.py`
  - `cvm.py`
  - `feistel.py`
- Git am-compatible patch series (zipped) for easy application.

---

*Generated as a handover artifact from analysis of both repositories.*
