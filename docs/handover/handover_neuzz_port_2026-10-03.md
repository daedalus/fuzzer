# Handover: porting NEUZZ / NEUZZ++ ideas to the fuzzer (2026-10-03)

Status: **analysis only, nothing implemented.** Based on reading `boschresearch/neuzzplusplus`
(`a104ef5`, ESEC/FSE'23 artifact) against fuzzer HEAD `ae5335c`. Read: `README.md`,
`neuzzpp/models.py`, `neuzzpp/mutations.py`, `scripts/train_cov_oracle.py`. NOT read:
`data_loaders.py`, `preprocess.py`, `utils.py`, `aflpp-plugins/ml-mutator.c`. Nothing was executed.

## 1. What NEUZZ++ does

- **Model** (`models.py::MLP`): input = seed bytes /255, post-padded to `max_file_size`
  (default: 80th percentile of seed length); Dense(4096, ReLU) -> Dense(n_edges, `logits`) ->
  sigmoid. Loss = binary cross-entropy, Adam, CosineDecayRestarts, output bias initialised from
  class frequencies. Output = edge bitmap.
- **Retraining** (`train_cov_oracle.py`): driven by seed-count / time (`model_needs_retraining`),
  early stopping on PR-AUC. Talks to the AFL++ custom mutator over named pipes.
- **Gradient** (`mutations.py::compute_gradient`): pick target edge `e` (random, or random
  never-covered edge via `choose_rand_unseen_edges`), take d logit_e / d input, sort bytes by
  |grad| descending, keep top `NEUZZPP_MAX_GRADS` (default 32), keep sign.
- **Mutation** (`generate_one_mutation`): for round k in 0..n_iter, bytes with rank in
  [2^(k-1), 2^k) are stepped +1..+up_steps and -1..-down_steps along the gradient sign
  (clipped to 0..255); then `INS_DEL_RATIO` (0.2) x len(signs) alternating block delete / block
  insert at the high-gradient locations, block lengths from AFL havoc buckets (32/128/1500/32768).

## 2. What the fuzzer already has (overlap)

| NEUZZ piece | Existing fuzzer piece |
|---|---|
| training matrix (seed x edge) | `core/edge_matrix.py` (`MatrixSubstrate`, `MatrixFold`, canonical edge classes, refit cadence, `NNZ_BUDGET`) |
| per-byte importance | `pos_changed`, `pos_effector`, `pos_finch`, `pos_rare_mask`, `pos_cmplog`, `pos_good_turing` (bin-pooled, content-blind) |
| "gradient" mutation | `core/gradient_descent.py` (Angora GdSearch port), `core/gradient_cmp.py` (honggfuzz): cmplog/Hamming based, **no learned model** |
| numpy | already a hard dependency (`numpy>=2.0`) |
| rare/frontier targeting | `op_good_turing`, `pos_good_turing`, `dominators.py`, `icfg.py` |

No NEUZZ / neural / surrogate code exists in the repo.

## 3. Port candidates (ranked)

1. **`PositionSaliencyScheduler` (genuinely new).** Existing position schedulers pool by offset
   bin; a learned model conditions on seed *content*. Implement a small numpy MLP:
   - Inputs: seed bytes (/255), capped length (cap + hidden width are the cost knobs).
   - Outputs: **canonical edge classes** from `EdgeCanonicalizer`, not raw bitmap columns, to
     keep the output layer small and stable.
   - Gradient is closed-form for one hidden ReLU layer:
     `g = W1 @ (1[h>0] * W2[:, e])` (h = pre-activation, e = target class). No autograd.
   - Plug in via the `pos_base` `propose/record` interface; add as an arm in the position arena
     (`--pos-arena-arms`) and A/B against `pos_changed` / `pos_effector`.
   - Refit on `MatrixSubstrate`'s cadence (`DEFAULT_REFIT_INTERVAL`), gated by `coverage_trust`.
2. **Target-edge selection.** NEUZZ's "unseen edge" choice is weak: an edge with only negative
   labels has an uninformative gradient. Prefer rare / frontier classes (Good-Turing
   discovery probability, dominator-frontier edges) so the target has some positive support.
3. **Sign-directed ladder operator.** (offsets, signs) -> 2^k top-byte walk with +-1..255 steps
   plus insert/delete at hot spots. Overlaps the `gradient_descent` ladder (`_STEPS`), so register
   it as a single operator and let the op arena judge it rather than special-casing.
4. **Reuse refit cadence/trust gating** from `MatrixSubstrate` instead of NEUZZ's retrain logic.

## 4. Not worth porting

- AFL++ custom-mutator C glue and named-pipe protocol (fuzzer is in-process).
- TensorFlow / Keras (breaks the stdlib+numpy constraint).
- CosineDecayRestarts, PR-AUC/F1 evaluation plumbing, TensorBoard callbacks.
- `AFL_DISABLE_TRIM` requirement (AFL++-specific).

## 5. Risks and caveats

- **License:** NEUZZ++ is **AGPL-3.0**; the fuzzer is **MIT**. Reimplement from the algorithm /
  paper only. Do not copy code, structure, or comments.
- **Weak prior evidence:** from memory (UNVERIFIED, check the paper "Revisiting Neural Program
  Smoothing for Fuzzing", arXiv 2309.16618): the FSE'23 evaluation found NEUZZ-family gains
  largely do not reproduce against a properly configured AFL++ baseline. The repo's own record is
  also negative on learned/physics schedulers (Q-learning/SARSA, `op_tang`, Navier-Stokes
  `ContinuumField`, 4W/8L, McNemar p=0.388).
- **Cost unmeasured:** a 4096-wide dense layer over long inputs may dominate exec time. Cap input
  length and hidden width; benchmark refit + gradient time per call before wiring.
- **Label quality:** training labels come from the edge matrix; if `coverage_trust` is low
  (unstable edge ids) the model learns noise. Gate on it.

## 6. Suggested plan if pursued

1. `core/nn_saliency.py`: pure-numpy MLP (fit, `edge_grad(seed, e)`), unit tests incl. numeric
   gradient check against finite differences.
2. `core/schedulers/pos_saliency.py`: `PositionSaliencyScheduler` (`name = "saliency"`), tests.
3. Wire as `--pos-saliency` (default off, **excluded from `--hail-mary`**) and add to the
   position arena arm set.
4. Benchmark: `bench_paired.py` + `pos-arena-*` on fuzzgoat and a format target; ship only on a
   paired win. Record negative results either way.
5. Optional follow-up: ladder operator (item 3) and frontier-edge target picker (item 2).

## 7. Open questions

- Hidden width / input cap that keeps refit under the campaign-loop stall budget.
- Whether per-position saliency adds anything over `pos_changed` when `moved()` is available
  (the cheaper signal may already capture most of it).
- Whether the model should predict canonical classes or per-seed residual (`seed_residual`).
