# entropy_kl: length bias found and fixed (2026-09-27)

**Origin:** analysing https://francisbach.com/spectral_log_density_estimation/ against the fuzzer.
The article's estimator does not help here (see "Not done"); reading the KL code for where it
might turned up the bias below.

## Finding

`seed_entropy_kl.py` scored `KL(P_s || Q)` from the seed's own <= 4096-byte histogram. Plug-in KL
is biased upward by ~(K-1)/(2n) nats (K = 256 bins), so:

- A 16-byte seed drawn from the pool's own distribution (true KL 0) scored ~1.7 nats; a 4 KiB seed
  that truly diverged (0.32 nats) scored 0.36. Short seeds win regardless of content.
- Selection is proportional to score, so on corpora whose seeds are all one distribution (6 runs,
  400 seeds, lengths 16-4096): Spearman(score, length) = -0.99, seeds <= 64 B drew 2.25x their
  uniform share of the weight.

Reproduce: `tests/test_regression_entropy_kl_length_bias.py::test_control_raw_scores_still_show_the_bias`.

## Fix

`scores()` = `clip((KL - E0(n)) / sd0(n), 0, Z_CAP)`; `raw_scores()` is the old number.

- `E0(n)`: exact `E[KL(P_hat_n || Q)]`, `null_kl_bits`. With c_b ~ Bin(n, q_b) and
  `E[c f(c)] = n q E[f(Y+1)]`, Y ~ Bin(n-1, q): `E0 = sum_b q_b (E ln(1+Y_b) - ln q_b) - ln n`.
  Checked against enumeration at n = 1 (equals H(q)) and n = 2, and against (K-1)/(2n ln 2) at
  n = 4096 for uniform q. The chi-square expansion is not used: it overshoots ~5x for n < K.
- `sd0(n)`: 200-draw Monte-Carlo, fixed-seed private generator (a calibration table, not a
  scheduling decision, so not on the campaign RandPool).
- Both on a power-of-two grid up to the cap, log-log interpolated.

Measured (6 runs), raw -> calibrated: Spearman(score, len) -0.99 -> +0.01; <= 64 B weight share
2.25x -> 0.99x; AUC diverging-vs-null (10 % shifted seeds) 0.73 -> 0.98.

**Mean subtraction alone was tried first and is not enough.** It fixes the rank correlation
(AUC 0.95) but short seeds are far noisier, so weight proportional to the clipped excess left
them at 2.3-3.0x their share, no better than raw. Dividing by the null spread fixed that.

## Miller-Madow baseline (added same day)

`miller_madow_scores()` subtracts `(K_hat - 1) / (2 n ln 2)` bits, K_hat = distinct byte values in
the sample. Mean AUC over 6 mixed-length corpora (diverging vs matching seeds): raw 0.72,
Miller-Madow 0.75, calibrated z 0.97. It only removes a sliver because for n < K most bins are
empty, so K_hat is far below the K the bias scales with. Kept as a measured baseline, not wired
into selection. `tests/test_entropy_kl_miller_madow.py` holds the AUC helper (rank-based, with a
control on uninformative labels) and asserts raw < Miller-Madow < calibrated.

## Bach spectral estimator (added same day)

`core/spectral_kl.py` implements the closed form from the article (Eq. 7 collapsed by one
generalized eigendecomposition): `F = sum_i ((mu_p - mu_q)^T v_i)^2 f(lambda_i)/(lambda_i - 1)^2`,
`f(t) = t ln t - t + 1`, with ridge `1e-3` on both covariances and 32 nibble features (one-hot high
nibble + one-hot low nibble). A batch of seeds against one pool is one stacked `eigh`.
`EntropyKLSeedStrategy.spectral_scores()` exposes it in bits.

Checked against: quadrature of Eq. (7) with a linear solve per node (rel 1e-6, with a
self-comparison control); exact reduction to plug-in KL for one-hot features; `F <= KL` over 20
random pairs; `F = 0` at `p = q`; degenerate pools (unseen bins, point mass).

Measured over 6 mixed-length corpora: AUC raw 0.72, Miller-Madow 0.75, **spectral 0.83**,
calibrated z 0.97. Spearman(score, length) on single-distribution corpora: raw -0.994, spectral
-0.988. So the spectral bound separates diverging seeds better than plug-in KL but does **not**
remove the length bias with these features and ridge; it stays a baseline, not the scheduling
score. `spectral_scores()` on 2000 seeds takes ~240 ms, fine for offline comparison, too slow to
run on every pick. Not tried: a larger ridge, richer features, or applying the null calibration to
the spectral score itself.

## Null calibration applied to the spectral score (added same day)

`calibrated_spectral_scores()` = `clip((F - mean0(n)) / sd0(n), 0, Z_CAP)`, the same construction as
`scores()`. F has no closed-form null, so `mean0` and `sd0` are Monte-Carlo (200 draws per grid
point, fixed-seed private generator, `null_spectral_curve`) on the same power-of-two grid,
log-log interpolated, rebuilt when the pool drifts 2 % in L1.

Over 6 runs: Spearman(score, length) on single-distribution corpora -0.988 -> **-0.043**
(range -0.09..-0.01); <= 64 B weight share 0.92-1.18x uniform (mean 1.09). Mean AUC diverging vs
matching, mixed lengths: raw plug-in 0.72, Miller-Madow 0.75, raw spectral 0.83, **calibrated
spectral 0.94**, **calibrated plug-in 0.97** (range 0.95-0.99). So calibration removes the length
bias from the spectral score and lifts it 0.83 -> 0.94, but it still trails the calibrated
plug-in on this benchmark; the shift here is a high-byte mass move that the nibble features only
partly express, so a benchmark whose divergence lives in shared structure may rank them
differently. Not measured on real corpora.

Cost: null-curve rebuild ~310 ms, and `spectral_scores()` is recomputed on every call (~240 ms for
2000 seeds), so this is an offline comparison, not a per-pick score. Not tried: caching the
spectral rows across calls, larger ridge, richer features.

## Costs and limits

- Pick after an admission into a 2000-seed corpus: ~13 ms (was 1.0 ms). `E0` is rebuilt on each
  pool change because a 2 % drift moves a z-score ~0.3; the Monte-Carlo spread is reused until
  the pool drifts 2 % in L1. A pick that changes nothing is unchanged (~0.3 ms).
- `sd0` carries ~5 % Monte-Carlo error; a reused build and a fresh one differ ~7 %.
- `Z_CAP = 8` is a heuristic: it stops a 100-sigma outlier from starving every other seed.
- `mean_kl` in status/report is now the mean z-score, not bits.
- The pool contains the seed being scored, biasing large seeds' KL slightly down
  (`seed_entropy_loo` addresses that). Not corrected here.
- All measurements use synthetic i.i.d. seeds; the repo has no real corpus fixtures. Real seeds
  are structured, so magnitudes will differ. Worth re-measuring on a real campaign corpus.

## Not done

- Bach's spectral estimator (feature-based, closed form): with one-hot features it reduces exactly
  to plug-in KL; with nibble features it cut the bias 2-8x but left ranking length-dependent.
  Untested idea: shared operator features in `op_tpe` if its per-operator counts are too sparse.
- The sibling seed arms (`seed_entropy_zscore`, `seed_entropy_loo`) were not audited for the same
  bias.
