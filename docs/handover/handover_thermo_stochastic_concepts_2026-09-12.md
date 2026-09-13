# Handover: thermal/stochastic-process concepts vs. current tree (2026-09-12)

Base commit: f17a3ef49f47277cb911ef9d5a996e240db48694

Requested analysis: how do thermal equilibrium, Brownian motion, the
equipartition theorem, dynamic equilibrium, the binomial distribution,
quadratic variation, power spectral density, and least squares map onto
this codebase. Verdict per concept below, checked against what already
exists rather than assumed absent.

## Already implemented — no new work justified

- **Power spectral density.** `core/periodicity.py` already computes the
  PSD via `numpy.fft.rfft` and inverts it (Wiener-Khinchin) for
  autocorrelation-based record-size and cadence detection, gated by
  Fisher's g-test.
- **Least squares.** Two independent instances: `allan_variance.py`'s
  `noise_slope()` fits a log-log line to Allan deviation vs. averaging
  time by closed-form OLS; `edge_tracker.py` fits the coverage-saturation
  curve `E(t) = A*(1-exp(-k*t))` via Levenberg-Marquardt nonlinear least
  squares.
- **Binomial distribution.** `services/report.py` computes Wald and
  Wilson binomial-proportion confidence intervals; `randomness.py`'s
  `repeat_test` scores adjacent-repeat counts against
  Binomial(n-1, 1/alphabet). `discovery_uniformity.py`'s negative-binomial
  severity fit is the adjacent generalization (Poisson rate itself
  Gamma-distributed).
- **Quadratic variation.** This *is* what Allan variance computes:
  `σ²(τ) = ⟨(x[i+τ]-x[i])²⟩/2` is a normalized sum of squared increments
  at lag τ, and `noise_slope()` already extracts the scaling exponent
  (the same information a Hurst-exponent estimate would give) via the
  least-squares fit above. A separate "quadratic variation module" would
  duplicate this.
- **Brownian motion as an implicit null.** The Allan-variance slope
  classification (`slope ≈ 0` vs `slope > 0.1`) is already a diffusive-
  vs-non-diffusive test on the edge-discovery series, which is the
  practical content Brownian motion would contribute to that signal.

## Genuine gap — concrete, falsifiable, proposed

- **Brownian motion / quadratic variation, applied to `distance.py`
  instead of the discovery-rate series.** The AFLGo directed-distance
  signal (mean CFG/CG distance of a seed's trace to target blocks) is
  currently consumed by the `aflgo` power schedule but nothing checks
  whether directed scheduling is actually producing directed *progress*.
  Reusing the existing log-log least-squares slope machinery from
  `allan_variance.py` against the seed-distance trajectory instead of
  the edge-discovery trajectory gives a three-way, falsifiable
  classification: slope ≈ 1 (ballistic/directed — distance falls faster
  than diffusion predicts), slope ≈ 0.5 (Brownian — no net directional
  progress beyond a random walk over reachable blocks), slope < 0.5
  (sub-diffusive/trapped — distance-guided scheduling stuck against a
  structural barrier). This is a real question the tree cannot currently
  answer: whether `aflgo` scheduling is earning its annealing weight or
  running no better than undirected search on this target. Falsifier:
  simulate a pure random walk over a synthetic reachability graph and
  confirm the estimator recovers slope ≈ 0.5 before trusting it on real
  distance traces.
- **Dynamic equilibrium, applied to corpus size.** `corpus_manager.py`
  tracks additions and prunes but nothing distinguishes a corpus that has
  plateaued because both processes have gone to ~0 (stalled campaign)
  from one that has plateaued because additions and prunes are both
  active and balanced (a healthy campaign still discovering redundant-
  but-covering seeds and pruning them at the same rate net size is
  flat). Corpus size alone cannot tell these apart; gross flux
  (net = additions − prunes, tracked separately from |additions| +
  |prunes|) can. This is a one-line addition to the existing tick
  bookkeeping in `corpus_manager.py`, reusing `DispersionIndex` /
  `RunningMoments` already in the tree rather than adding new statistical
  machinery — the gap is which signal is tracked, not a missing
  algorithm. Falsifier: on a synthetic run with additions and prunes
  both forced to zero, the two cases must be distinguishable from the
  flux log even though final corpus size is identical in both.

## Weak or rejected fit — do not force

- **Thermal equilibrium / equipartition theorem.** `temperature_control.py`'s
  "temperature" is already a simulated-annealing schedule metaphor, not a
  thermodynamic state; there is no literal ensemble of independent
  quadratic degrees of freedom here. Equipartition's substantive claim —
  that at equilibrium every quadratic degree of freedom carries equal
  mean energy — has no natural target: the whole point of Thompson-
  sampling/UCB operator scheduling is to allocate budget *unequally*,
  toward operators with higher measured reward. Forcing an "equal energy
  per operator" framing would fight the actual optimization goal instead
  of describing it. `discovery_uniformity.py`'s spatial-homogeneity
  chi-squared test already covers the one adjacent question worth asking
  (is coverage discovery uniform across regions), and it earns that
  through a Poisson-process argument, not a thermodynamic one. No action
  recommended beyond what already exists.

## Recommendation

Implement the two "genuine gap" items above if this line of work is
wanted; skip the equipartition/thermal-equilibrium framing as not
adding falsifiable content beyond `discovery_uniformity.py` and
`temperature_control.py` as they stand.
