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
