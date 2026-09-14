# Handover: thermal/stochastic-process concepts vs. current tree

**Date:** 2026-09-12, **audited and re-prioritised 2026-09-13**,
**implementation audited and closed out 2026-09-13 (later)**
**Base of original analysis:** `f17a3ef`
**Base of the audit revision:** `94d1741`
**Base of this revision:** `388df9f`

**Status: P0-T1, P0-T3, P2-T4 and P4-T6 are closed; P3-T5 is withdrawn
pending its three design questions; E-T7 was raised by the P0-T1 fix.**
§4 and §5 are the implementation audits of P0-T1/P2-T4 and P0-T3. §6 is a
parallel session's closeout of P4-T6 and P3-T5. §7 records a state-file
defect found while wiring P4-T6 that is unrelated to this document and
more serious than anything in it.

**Note on §4/§5.** `eddb9a0` rewrote this document and dropped both
(777 → 474 lines, 357 lines removed against 48 added) while adding what
is now §6. They are restored here rather than left out: §5.1 in
particular records four measured approaches to the same problem of which
three do not work, and "absent from the doc" and "considered and
rejected" are different states — only one of which should be
re-proposable. §6 is kept verbatim; it is new information and its
correction about `scaling_exponent.py` stands.

**Companion:** `handover_pending_2026-09-06.md` — the tier scheme below (P0/P1/P2/P3/P4)
is that document's, so these items can fold into it without re-ranking.

Requested analysis: how do thermal equilibrium, Brownian motion, the
equipartition theorem, dynamic equilibrium, the binomial distribution,
quadratic variation, power spectral density, and least squares map onto
this codebase.

**The 2026-09-12 revision of this document answered the question by
inventory — "does a module for this exist" — and four of its five
"already implemented, no new work justified" entries are wrong in the
way that matters: the module exists, and it does not compute the thing
it is named after, or its null hypothesis is contradicted by another
module in the same tree.** Everything below was re-measured in a
container at `94d1741`. Numbers are from that run unless stated.

A `[ ]` is not evidence and neither is a module name. The lesson of this
audit is narrower and worth keeping: **a concept is not "covered"
because a file is named after it.** Three of the four corrections below
were found by deriving what the code computes and comparing it to what
the file claims, not by looking for missing files.

---

## 0. Verdict table

| Concept | 09-12 verdict | 09-13 verdict |
|---|---|---|
| Power spectral density | covered, no work | **covered, but its null is wrong for the series it is applied to** → P0-T3 |
| Least squares | covered, no work | covered; two secondary defects fold into P0-T1 |
| Binomial distribution | covered, no work | **confirmed covered, and better than claimed** — no action |
| Quadratic variation | "is what Allan variance computes" | **right about this code, wrong about why** → P0-T1 |
| Brownian motion (implicit null) | covered via Allan slope | **the estimator cannot separate the two hypotheses** → P0-T1 |
| Brownian motion on `distance.py` | genuine gap, implement | genuine, but **gated on two facts the document got wrong** → P3-T5 |
| Dynamic equilibrium (corpus flux) | genuine gap, implement | confirmed sound → P4-T6, **CLOSED `8bdf52e` + wiring**, see §6 |
| Thermal equilibrium / equipartition | rejected, no action | **rejected for a provable reason instead of a metaphorical one; and the tree already has the module the rejection overlooked** → P2-T4 |

---

## 1. What the 09-12 revision got right

Kept, re-verified, no action:

- **PSD machinery exists.** `core/periodicity.py` computes the power
  spectrum via `numpy.fft.rfft` and inverts it (Wiener–Khinchin) for
  autocorrelation-based record-size detection, gated by Fisher's
  g-test. `power = power * power.conj()` at `:137-139` is a real PSD,
  not the "forgot to square it" error this shape of code usually has.
  The Hanning window and the `SIGMA_CUTOFF / sqrt(w)` noise floor are
  both correct and both documented.
- **Nonlinear least squares exists.** `edge_tracker.py`
  `bayesian_coverage_growth_model` fits `E(t) = A*(1-exp(-k*t))` by
  Levenberg–Marquardt with a Laplace posterior covariance and a
  delta-method `p_stalled`. Also a closed-form log-log OLS in
  `allan_variance.py::noise_slope`.
- **Binomial distribution is covered, and more completely than the
  09-12 text says.** It credits `report.py` with "Wald and Wilson"
  intervals. The tree is past that: `_wilson_interval` (`report.py:88`)
  exists *specifically because* the Wald interval degenerates at small
  `p` and can emit negative lower bounds, and its docstring says so.
  Discovery probabilities are exactly the small-`p` regime, so this is
  a closed item, not a live one. Do not re-propose a binomial-CI fix.
- **Dynamic equilibrium on corpus size** is a real gap and the 09-12
  framing of it survives audit — see P4-T6.

---

## P0 — defects in shipped code

### P0-T1. `core/allan_variance.py` does not compute the Allan variance, and the fatigue threshold misfires on stationary long-memory noise

`adev(tau)` (`allan_variance.py:112-126`) computes

    sqrt( 0.5 * mean_i (x[i+tau] - x[i])^2 )

That is the **second-order structure function** — the variogram, i.e.
normalised quadratic variation at lag `tau`. It is a *first* difference
of the raw samples. The Allan variance is a first difference of
*block averages*, equivalently a *second* difference of the cumulative
series. They are different estimators with different noise-type
signatures, and separating white from flicker noise is the entire
historical reason the Allan variance exists.

Measured, 200 replicates, N=256, `noise_slope()` vs. a textbook
overlapping Allan variance on the same series:

| series | tree slope | true Allan slope | tree `noise_type()` |
|---|---|---|---|
| white i.i.d. | 0.001 ± 0.021 | −0.532 ± 0.115 | `active` 200/200 |
| flicker 1/f | **0.106 ± 0.034** | −0.041 ± 0.082 | **`fatiguing` 117/200** |
| random walk | 0.441 ± 0.124 | 0.408 ± 0.167 | `fatiguing` 200/200 |
| linear downtrend | 0.632 ± 0.017 | 0.837 ± 0.016 | `fatiguing` 200/200 |

The white-noise column differs by exactly the 0.5 the theory predicts,
which confirms the identification rather than merely suggesting it.

**Why this is P0 and not a naming nit.** `_FATIGUE_SLOPE_THRESHOLD =
0.1` sits precisely where a **stationary** long-memory process lands.
Sweeping the spectral exponent of a stationary 1/f^beta rate series
(400 replicates each):

| beta | mean slope | fraction called `fatiguing` |
|---|---|---|
| 0.00 | −0.001 | 0.000 |
| 0.50 | 0.024 | 0.000 |
| 0.75 | 0.057 | 0.025 |
| **1.00** | **0.105** | **0.593** |
| 1.25 | 0.177 | 0.927 |
| 1.50 | 0.256 | 0.988 |

A campaign whose discovery rate has 1/f structure and **no decline at
all** is classified as approaching stall on a coin flip. That
classification is not a display: `fuzzer.py:5895-5901` halves the stall
threshold on `noise == "fatiguing"` (`max(self._stall_threshold // 2,
100)`), which pulls `_stall_recovery_enter` forward, which is what
drives `--reseed-on-stall` and the bitmap resize. And the detector is
**default-on**: its `AnalyzerSpec` (`analyzer_registry.py:324-330`) has
no `available` gate, so it is in the "always constructed" class.

The class docstring already says "Classification is tailored to
edge-discovery-rate signals, not generic noise theory", which is honest
about the thresholds being empirical. It does not rescue the module
header, which says "Overlapping Allan variance for noise-type
identification" — noise-type identification is the one thing this
estimator cannot do.

**Two secondary defects in the same file, which must move with the
above and not separately, because both change `adev` values and
therefore the classification:**

- `count = n - 2*tau` (`:122`), but a lag-`tau` first difference has
  `n - tau` valid pairs. Discarded pairs: 1.6% at tau=4, 14.3% at
  tau=32, **33.3% at tau=64** — precisely the noisiest point, which the
  unweighted fit then over-weights. `adev` also requires `2*tau+1`
  samples where `tau+1` would do.
- `noise_slope` is an **unweighted** OLS over at most five points
  (tau=4..64). Allan/variogram estimates at large tau have far fewer
  equivalent degrees of freedom, so equal weighting is the textbook
  weighted-least-squares case. Measured on ground truth, 600
  replicates, weights ∝ `count/tau`: Brownian RMSE **0.134 → 0.095**
  and bias −0.044 → −0.023; white noise is a wash (0.0212 → 0.0239).
  So weighting helps exactly in the non-stationary case the classifier
  acts on, and is neutral elsewhere.

**The decision to make on paper first.** Two routes, and they are not
interchangeable:

1. **Rename and recalibrate.** Keep the variogram, fix the name and the
   header formula, and re-derive `_FATIGUE_SLOPE_THRESHOLD` with a
   stated margin against the 1/f null instead of against nothing. Cheap,
   no new estimator, but permanently gives up white-vs-flicker.
2. **Implement the real Allan variance** (second difference of the
   cumulative series) and re-derive all three thresholds
   (`_ADEV_ACTIVE_THRESHOLD`, `_ADEV_STALL_THRESHOLD`,
   `_FATIGUE_SLOPE_THRESHOLD`) against it. Gains the noise-type
   separation the module header claims; costs a full recalibration of a
   default-on stall path.

**Acceptance test, either route:** the beta sweep above, as a table, with
a stated maximum false-`fatiguing` rate at beta=1.0. The current value
is 0.593. A fix that does not report this number has not been tested.
**Falsifier before trusting any replacement:** feed synthetic white,
flicker and random-walk series and confirm the estimator recovers the
slope its own theory predicts — the table above is the harness.

**Open question this audit does not answer:** whether real discovery
series actually have beta near 1. I could not measure it —
`_discovery_edges` reached only 13 snapshots in a 20k-exec run (see §3).
This does not soften the item: a threshold with no margin against the
most plausible stationary alternative is a defect whether or not that
alternative is currently realised.

### P0-T3. Fisher's g-test assumes a constant rate; the series it is given is the one the rest of the tree exists to prove is not constant

`report.py::_spectral_diagnostics` differences cumulative edges and
hands the deltas to `detect_periodicity`, whose null (`fisher_g_pvalue`)
is i.i.d. exponential periodogram ordinates — Gaussian white noise, i.e.
a *constant* discovery rate. `coverage_regime.py`, `critical_slowing.py`
and `garch.py` all exist on the premise that this rate is
non-stationary.

Measured false-positive rate at nominal alpha=0.05, 2000 replicates, on
a Cox process (Poisson counts, slowly drifting OU rate — the honest
model of discovery):

| n | white (control) | Cox, drifting rate | median peak bin |
|---|---|---|---|
| 128 | 0.049 | 0.176 | 2 |
| 256 | 0.048 | 0.378 | 2 |
| 512 | 0.050 | **0.540** | 3 |

At n=512 sync intervals a campaign with no periodicity whatsoever is
told `PERIODIC — possible corpus-sync artifact` more than half the
time, and that message is an active mis-steer toward a cause that is
not there. The existing `peak_bin >= 2` guard is the reason the median
lands at 2–3: it catches bin 1 and nothing above it. AR(1) shows the
same shape and worse — n=512, phi=0.7 → **0.942**.

**Two things this audit expected and falsified, recorded so they are not
re-proposed:**

- **Volatility clustering is not the problem.** GARCH(1,1) work with
  zero mean-level autocorrelation gives 0.044 against a 0.049 control;
  piecewise-constant variance gives 0.043. Variance clustering keeps
  the ordinates exchangeable in expectation, so `garch.py`'s premise is
  **not** in conflict with Fisher's g here. The conflict is mean-level
  rate drift only.
- **Pure 1/f is already handled.** beta=1 and beta=2 both give 0.000,
  because the argmax collapses into bin 1 and the existing guard
  rejects it. The gap is the mid-band, not the red end.

**Fix:** pre-whiten before the periodogram (fit AR(1) and test the
residuals, or difference a second time) or replace the white null with
an autocorrelation-aware one. Ships with the table above as its test.

**Ordered last in P0** because it corrupts a *display* and not a
decision — the same rule that puts `shapley._prune_edges` last in the
companion document's P0.

**Not a defect, checked:** the second scan in the same function, over
`_exec_time_tracker._times`, is fine. Real exec-time series captured
from live in-process runs against `zlib_read` and `png_read` are close
to white (|acf| ≤ 0.13 at lags 1–5), and `detect_periodicity` returned
not-significant on them. No action for that scan.

---

## P2 — shipped but unwired

### P2-T4. `--fluctuation` computes `L*log(L)` and nothing else

The 09-12 revision rejected thermal equilibrium and equipartition as
metaphors with "no natural target" and recommended no action. It did
not notice that **`core/fluctuation.py` already exists** — a
`WorkFunctional` with an inverse temperature `beta`, a Jarzynski
estimator and a `crooks_forward_reverse`, behind `--fluctuation`,
persisted through `state_store` and displayed by `stats.py:1169`. So
the concept is not absent; it shipped, and the question worth asking is
whether its premise holds. It does not, in three independent ways.

**(a) The quantity is not a free energy. It is a Rényi entropy, exactly.**
With `W(tau) = sum_i -log p_i` and a trajectory drawn from those same
`p_i`, `e^{-beta W} = P(tau)^beta`, so

    -log( E[e^{-beta W}] ) / beta  ==  H_{1+beta}(trajectory distribution)

the Rényi entropy of order `1+beta`. Verified numerically against
ground truth on a known product distribution (6 operators, length-3
trajectories, 400k samples per point):

| beta | `jarzynski_estimator` | Rényi_{1+beta} truth |
|---|---|---|
| 0.01 | 5.07781 | 5.07593 |
| 0.25 | 5.01439 | 5.01264 |
| 0.50 | 4.95057 | 4.95016 |
| 1.00 | 4.83393 | 4.83573 |
| 2.00 | 4.64610 | 4.64641 |

The default `beta=1.0` is therefore collision entropy, `beta -> 0` is
Shannon. `core/renyi.py` already exists and already computes Rényi
spectra — on the edge hit-count distribution rather than the
operator-path distribution, so it is an adjacent object, not a
duplicate. This also settles the equipartition question on its own
terms rather than by analogy: `beta` here is an **entropy order**, not
a temperature, so there is no ensemble to equipartition and no
fluctuation–dissipation relation to exploit. That conclusion matches
the 09-12 revision's; the argument does not.

**(b) In production it is not even that.** `self._operators._available`
(`fuzzer.py:3054`) **does not exist anywhere in the tree** — the only
references are the `list(...)`/`hasattr(...)` pair at `:3054-3056`
itself; the sole `_available` assignments in `src/` are on unrelated
objects (`adapters/lbr_trace.py`, `adapters/perf_event.py`,
`core/smt_solver.py`), none of them the operators service. So the
guard always falls through to
`list(self._last_ops_used)`, `available` becomes the trajectory itself,
`_op_probability` returns `1/L` for every step, and

    W = sum_{i=1..L} -log(1/L) = L * log(L)

Confirmed on a real run (`png_read`, 6000 execs, `--fluctuation`):
5,697 work samples, **7 distinct values** — 0, 1.38629, 3.29584,
5.54518, 10.75056, 16.63553, 29.81880 — which are exactly `n*log(n)`
for n ∈ {1,2,3,4,6,8,12}. The recorded signal is the mutation-stack
depth. It carries no operator identity, no scheduler state and no
coverage information, and `jarzynski_delta_f` is a deterministic
function of the distribution of stack depths.

**(c) The scheduler-aware branches are dead, and Crooks is not Crooks.**
Independently of (b): `_op_probability`'s mopt branch (`:3029`) and elo
branch (`:3045`) both `return max(1.0 / max(len(available), 1), 1e-12)`
— byte-identical to the fallback at `:3046`. Only the
`mc.bandit_stats()` branch (`:3019-3023`) returns anything
operator-dependent, and those Beta posterior means `(a+1)/(a+b+2)` are
not normalised across operators, so `-sum log p` is not the
log-probability of anything and the identity in (a) does not hold there
either. And `crooks_forward_reverse` returns a ratio of *mean works*
between two arbitrary state buffers: there is no reverse protocol, no
matched-`W` density ratio, and no crossing at ΔF. Its comment
("for identical work distributions the ratio is centered at 1.0") is
true of any ratio of equal means and tests nothing.

**This is P2 because leaving it in the middle is the worst of the three
states.** Two honest routes:

1. **Wire and rename.** Pass the real normalised selection
   probabilities from the active scheduler (this is work at the call
   site, not in the module — the same shape as `invasion_select`'s
   never-passed `frontier_edges`, P2-3 in the companion document),
   delete the dead branches and the `_available` guard, drop
   `crooks_forward_reverse`, and rename `jarzynski_delta_f` to the
   Rényi-`1+beta` quantity it provably is.
2. **Retire the module.** Defensible: the quantity it would compute is
   the Rényi entropy of the operator-path distribution, which has no
   consumer beyond a stats line, and `core/renyi.py` is the natural
   home if one is ever wanted.

Either is cheap. The reason to do it soon is not the payoff but that
the current state publishes a thermodynamic-sounding number that is
`L*log(L)`.

---

## P3 — design work, gated on a stated question

### P3-T5. Brownian motion / quadratic variation on the seed-distance trajectory

The 09-12 revision's stronger proposal, and it is a real question: the
AFLGo directed-distance signal feeds the `aflgo` power schedule, and
nothing checks whether directed scheduling produces directed
*progress*. The three-way classification it proposes — slope ≈ 1
ballistic, ≈ 0.5 Brownian, < 0.5 sub-diffusive/trapped — is the right
shape, and its falsifier (recover 0.5 on a synthetic random walk over a
reachability graph before trusting real traces) is the right falsifier.

**Three corrections that gate it, all of which the 09-12 text gets
wrong:**

1. **It is blocked by P0-T1.** The proposal is to "reuse the existing
   log-log least-squares slope machinery from `allan_variance.py`". That
   machinery is the misnamed estimator, with the unweighted five-point
   fit and the 33%-discard at tau=64. Building a second consumer on it
   ships the white-vs-flicker confusion into a new decision path. Do
   P0-T1 first; that is the whole reason it is ordered first below.
2. **There is no distance trajectory to apply it to.** `avg_distance`
   is a per-seed scalar in `seed_meta`, written at `fuzzer.py:4618` and
   `:4636` and read at `seed_picker.py:457` and `:1021`. It is
   overwritten, not appended. The series the proposal assumes exists
   has to be built first, and where it is recorded (per seed over its
   own fuzzing history, or per campaign tick over the corpus mean) is a
   design choice that changes what the slope means. Answer that on
   paper before writing code.
3. **The signal is conditional and can be constant.** A live run in
   this container printed `No target functions found, using distance=1
   for all` — with no `--target-functions` the distance channel is a
   constant, the variogram is identically zero, and every tau point is
   excluded as non-positive. Add the feasibility gate that the
   companion document's P2-1 lesson demands: define the behaviour for
   the no-graph case *before* shipping, or this lands, passes its tests,
   and is never read — the `core/target_difficulty.py` failure mode.

---

## P4 — genuine, blocking nothing

### P4-T6. Dynamic equilibrium: gross corpus flux, not net corpus size — **CLOSED `eddb9a0`, see §6** — **CLOSED, see §6**

Confirmed sound as written on 09-12. `corpus_manager.py` tracks
additions and prunes but nothing distinguishes a corpus that has
plateaued because both processes went to ~0 (stalled) from one that has
plateaued because additions and prunes are active and balanced (healthy,
still finding redundant-but-covering seeds and evicting them at the same
rate). Net size cannot tell these apart; gross flux —
`|additions| + |prunes|` tracked alongside `additions - prunes` — can.
Reuses `DispersionIndex` / `RunningMoments`, already in the tree.

Falsifier, as stated: on a synthetic run with additions and prunes both
forced to zero, the two cases must be distinguishable from the flux log
even though final corpus size is identical.

~~One note the 09-12 text does not make: `_maybe_prune` already computes
the exact surviving-owner count locally, so the prune side of the counter
has a natural home.~~ **Wrong, and it was my note.** `_maybe_prune` is
`EdgeTracker`'s, and it evicts *tracked seeds from the edge tracker* —
a different operation from corpus eviction. The corpus side has no
`_maybe_prune` at all; its two eviction paths are
`CorpusManager.auto_minimize_corpus` (`f.corpus = unique`) and
`deprioritize_near_duplicates` (`f.corpus = [s for s in ...]`), and both
needed a new hook. Two same-named methods on different objects, and I
pointed at the wrong one.

---

## 2. Implementation order, and why this order

1. **P0-T1** — `allan_variance.py`. The only item that changes what a
   default-on campaign does today, and P3-T5 is blocked on it. Do the
   route decision on paper first; ship the beta sweep as the test.
2. **P2-T4** — `--fluctuation`. Second because it is cheap and carries
   no measurement risk: the identity is proved, the degeneracy is
   measured, and both routes (wire-and-rename, or retire) are small and
   reversible. Doing it before P0-T3 keeps the two statistical-framing
   corrections from landing in one reviewer's lap.
3. **P0-T3** — Fisher's g null. Display-only, so it waits behind the
   decision paths, but it is bounded work with its test table already
   written.
4. **P4-T6** — corpus gross flux. Small, independent, no gate.
5. **P3-T5** — distance trajectory. Last: gated on P0-T1 landing, on
   the series-definition question, and on the no-graph behaviour.

Not on this list, deliberately: the PSD machinery itself, the
least-squares *existence* question, and binomial CIs. All three are
genuinely closed and re-proposing them is the failure this document
now exists to prevent.

---

## 3. Environment notes for whoever picks this up

- `pip install -e . --break-system-packages` works; numpy 2.4.4 and
  scipy 1.17.1 are present in the container. The measurements above
  need neither beyond numpy.
- `gcc` builds 26 targets. `clang` is absent, so `tools/build_targets.sh`
  aborts at its trace-cmp stage **after** those 26 are already built —
  the non-zero exit is not a failure of the simple targets.
- `--inprocess-direct` engages correctly against
  `~/fuzzing/builds/png_read_noasan.so` and `zlib_read_noasan.so`.
- **Series lengths are the binding constraint on validating P0-T1 and
  P0-T3 against real data.** `_exec_time_tracker._times` is a deque
  capped at 200. `_discovery_edges` reached 13 entries in a 20k-exec
  run, and `_spectral_diagnostics` needs 51. The n=512 regime where
  P0-T3's false-positive rate hits 0.540 needs a long campaign to reach
  at all, so the synthetic harnesses are the practical test, not a
  substitute that should later be replaced by a live measurement.
- Another Claude surface was operating on the same clone during this
  audit — the reflog shows a `pull --quiet: Fast-forward` this session
  did not issue, which is how the base moved `77d3ab8` → `94d1741`
  mid-work. That commit is docs-only
  (`handover_non_ucb_schedulers_2026-09-13.md`) and touches none of the
  measured files, so no measurement was invalidated. Re-check the base
  before applying patches from this line of work.

## 4. Implementation audit (2026-09-13, later the same day)

Four commits landed on top of the audit revision. `f13f7a7` is this
document itself (identical to the audited version — `git patch-id
--stable` matches). The other three are by `Grok Agent <agent@x.ai>`:
`6cdba43` (P0-T1 route 1 plus the documentation half of P2-T4),
`29f0d60` (the rename), `0e11fd2` (P0-T1 route 2). Both routes to
P0-T1 were applied in sequence; the final state is route 2.

Everything below was measured in a container at `0e11fd2` before the
follow-up commits.

### 4.1 P0-T1 — closed, and the acceptance test passes

`adev(tau)` is now the real overlapping Allan deviation, the second
difference of the cumulative series divided by `2*tau^2*(N-2tau)`. The
variogram survives as `sdev()`, documented as a secondary diagnostic.
`noise_slope` uses weighted OLS with weights ∝ `(N-2tau)/tau^2`. The
module is `core/structure_function.py` and the analyzer spec is
`structure_function`; the detector attribute is `_structure_fn`.

The stated acceptance test, 200 replicates, N=256:

| series | slope | `noise_type()` |
|---|---|---|
| white i.i.d. | −0.516 ± 0.101 | `active` 200/200 |
| flicker 1/f | −0.017 ± 0.032 | `active` **200/200** |
| random walk | +0.451 ± 0.113 | `fatiguing` 197/200 |
| linear downtrend | +0.591 ± 0.040 | `fatiguing` 200/200 |

And the beta sweep, 400 replicates, against the 0.593 that opened this
item:

| beta | mean slope | fraction `fatiguing` |
|---|---|---|
| 0.00 | −0.502 | 0.000 |
| 0.75 | −0.142 | 0.000 |
| **1.00** | **−0.016** | **0.000** |
| 1.25 | +0.106 | 0.022 |
| 1.50 | +0.225 | 0.945 |

The white-noise slope recovers the theoretical −0.5, which is the
falsifier this document asked for. Exact 1/f now sits at −0.016 against
a 0.15 threshold, and the crossover has moved out to beta ≈ 1.4. Real
margin where there was none.

**Two things the implementation did not do, and what measurement says
about each.**

*The magnitude thresholds were not re-derived*, against the stated
criterion. Measured, `adev(2)` is uniformly 0.707× the old `sdev(2)` —
exactly 1/√2, as theory predicts — so `_ADEV_STALL_THRESHOLD = 0.01`
now fires at a 41% higher discovery rate than the value was chosen
for. **It does not bite, and the reason is the argument worth keeping:**
edge counts are integers, and a single discovery anywhere in a 256-tick
window already gives `adev(2) = 0.0445`, 4.4× the threshold. Nothing
lives in the shifted gap; only an all-zero window is below it. Verified
across Poisson rates 0.02–1.0/tick. `_ADEV_ACTIVE_THRESHOLD = 0.1` is
now reachable only from the two fallback branches (`len(points) < 2`,
`slope is None`), so it matters even less. **No action — recorded so
nobody spends a day re-deriving it.**

*`update()` rebuilds the whole cumulative deque in a Python loop on
every call once the buffer is full.* The comment justifies it as keeping
`_cum[0] == 0`, which is not needed: second differences are invariant
under an additive constant, so a plain append preserves both alignment
and correctness. It is one call per stats tick (~10 s of work), so this
is cosmetic, not hot. **No action.**

### 4.1a E-T7 (new) — the cost nobody measured

Neither this document nor `0e11fd2` measured what the estimator change
cost, and it cost something. Stepping the true rate down and counting
ticks until `noise_type()` leaves `"active"`, 1024-tick budget, 40
trials:

| step (edges/tick) | median ticks (of those that reacted) | never left `active` |
|---|---|---|
| 20 → 1 | 13 | 0/40 |
| 5 → 0 | 26 | 0/40 |
| 5 → 1 | 23 | **34/40** |
| 3 → 1 | 17 | **39/40** |

A drop to zero is caught reliably at any pre-step rate, because the
Allan deviation collapses under the stall threshold. A *partial*
slowdown at a low pre-step rate is mostly not caught at all, where the
old estimator caught `5 → 1` in a median 6 ticks.

**Read this carefully before calling it a regression.** `5 → 1` means
discovery is continuing at 1 edge/tick, which is not a stall; and
`noise == "active"` only declines to *pre-emptively halve* the stall
threshold (`fuzzer.py`, the `fatiguing` branch). It does not suppress a
genuine stall, since a true stop is caught in a median 26 ticks over
40/40. And the sensitivity the old estimator had at low rates was the
same mechanism as its `p10 = 1` dead-time tail: firing on Poisson noise.
So the trade is a large false-alarm rate for less pre-emptive fatigue
warning at low absolute rates, and it is very likely worth it.

**Why this is an E item and not a P-anything:** deciding it needs
campaign data on how often a low-rate partial slowdown actually
precedes a stall, which no synthetic harness can supply. The before and
after tables are in
`docs/handover/handover_control_theory_loops_2026-09-12.md` §2.1, whose
dead-time characterisation had to be re-measured for the same reason.

### 4.2 P2-T4 — the first pass documented it; the second closed it

`6cdba43` corrected the docstrings, marked `crooks_forward_reverse` as
not Crooks, and exposed the value under a second key, `renyi_entropy`,
while leaving the call site untouched. So `--fluctuation` went on
publishing the same degenerate number, now labelled with the precise
identity it fails to satisfy in production. That is further into the
middle state this tier exists to get out of, not out of it.

`155bb54` closes it via route 1 (wire and rename). The shape of the fix
is a contract rather than a better fallback, which is the part worth
carrying to the next item like this:

- A scheduler opts in by exposing `last_selection_probs()`.
  `TrajectoryRecord.probs_are_true` defaults to **False**, so a caller
  has to assert the property rather than remember to deny it.
  `WorkFunctional` counts but never pools records without it, so
  `jarzynski_estimator` returns None instead of a number shaped like an
  entropy.
- `exp3` is the only scheduler that can answer today; it already keeps
  that mixture for its own importance-weighted estimator. The UCB family
  selects by deterministic argmax and has **no selection law at all**,
  which is why no fallback is the correct answer and not a gap.
- Measured end to end: the default scheduler reports
  `fluc: W=0.00 n=0 unpooled=3919`; with `--exp3`,
  `fluc: W=10.06 n=1000 H2=7.94` — a real Rényi-2 entropy of the
  operator-path distribution.
- `_SELECTION_PROB_SOURCES` holds the opt-in list, and
  `tests/test_regression_fluctuation_probs.py` asserts every name in it
  is a real attribute. That guard is the point: the defect it replaces
  was a `hasattr` on a name nothing assigned, which disabled the feature
  silently instead of failing.

**Two further defects found in the same module while fixing it**, both
in `155bb54`:

1. `state_key` hashed the operator tuple with the builtin `hash()`,
   which is salted per process. Verified: three `PYTHONHASHSEED` values
   give three different keys for the same trajectory, so state restored
   from disk was orphaned under a key the new process cannot reproduce,
   and `--seed` did not determine the keys. Same defect class as the LSH
   banding removed from crash clustering. Now xxhash/sha256, matching
   what the edge branch already did, with no sign leak from the hex of a
   negative int.
2. The two scheduler-aware branches of `_op_probability` (mopt, elo)
   returned the fallback's value verbatim. They went with the function.

`crooks_forward_reverse` is retired rather than kept as a diagnostic:
a ratio of mean works between two arbitrary state buffers has no reverse
protocol, no matched-`W` density ratio and no crossing at ΔF, and its own
comment — "for identical work distributions the ratio is centered at
1.0" — is true of any ratio of equal means.

### 4.3 Collateral from the rename, fixed in `73a4996`

`29f0d60` updated every `.py` and left **nine references** across five
docs plus `docs/architecture.dot` and the generated
`docs/images/architecture.svg`. One was an inlined code block whose
stated purpose is that its numbers are "reproducible from this document
alone", and it no longer imported. Two needed re-measurement rather than
renaming, because the estimator changed underneath their numbers; see
§4.1a. One turned out not to be rename drift at all but a pre-existing
false claim — `handover_ports_pending.md:16` said `allan_variance.py`
feeds `core/seed_quality.py`, which contains no reference to it and never
did.

The analyzer spec name also changed, `allan` → `structure_function`.
Checked: spec names are internal to `AnalyzerRegistry`, not persisted in
state and not exposed as a CLI flag, so the rename crosses no
compatibility boundary. Recorded because it is the kind of thing that
looks like one.

### 4.4 Lesson, for the next document like this

The 09-12 revision answered by inventory and was wrong four times. The
09-13 audit of *that* was right about the defects and still incomplete
in the same direction: it specified an acceptance test for P0-T1 and did
not ask what the fix would cost, so the low-rate sensitivity in §4.1a
had to be found after the fact — and it invalidated a measured table in
a neighbouring document that neither the fix nor the audit thought to
check. **When a recommendation replaces an estimator, the obligation is
not just to state the acceptance test but to name what currently
depends on the old estimator's numbers.** `grep` for the class name
finds the callers; it does not find the measurements.

---

## 5. P0-T3 — closed, partially, and the residual is stated

`728bdc3`. An AR(p) background fit (Yule-Walker, order by AIC) flattens
the spectrum before the periodogram is scored, so Fisher's g gets the
white-noise series its null assumes. Measured, 500–1000 replicates,
nominal alpha = 0.05:

| null | n | raw | pre-whitened |
|---|---|---|---|
| Gaussian white | 256 | 0.049 | 0.048 |
| Gaussian white | 512 | 0.046 | 0.046 |
| Poisson, drifting OU rate | 256 | 0.367 | 0.062 |
| Poisson, drifting OU rate | 512 | 0.526 | **0.102** |
| AR(1) phi=0.7 | 512 | 0.936 | 0.069 |
| AR(1) phi=0.9 | 512 | 0.848 | 0.055 |
| AR(1) phi=−0.6 | 512 | 0.980 | 0.046 |
| 1/f and 1/f² | 512 | 0.000 | 0.000 |

**This is a 5× improvement and not a repair to nominal.** 0.102 against
0.05 at n=512. A Poisson count series with a drifting rate has a
Lorentzian-plus-flat-floor spectrum and an order-8 AR fit cannot flatten
both halves of it; order 12 does not help the null (0.104) and pushes
white noise to 0.068, which is the fit starting to model the noise.
Recorded in the docstring and reflected in the report wording: the
`PERIODIC` verdict on a drifting series is a lead, not a finding.

### 5.1 The trap this fix shipped into first

Worth reading before touching any spectral background estimate here,
because the failure is silent and produces a *confident wrong answer*
rather than a miss.

A periodic component is itself strongly autocorrelated. So an AR model
fitted to the raw series **models the tone**, and the filter then cancels
the very signal the test exists to find. Measured on the first version:
a bin-64 sinusoid at amplitude 2.0 over unit white noise came back
`significant=True` at bin 27. Not a false negative — a false *location*.

Three approaches were measured before one worked:

| approach | white null | drifting null | tone survives |
|---|---|---|---|
| raw (the defect) | 0.046 | 0.526 | yes |
| AR(p) on the raw series | 0.035 | **0.035** | **no — cancelled** |
| normalise by a median-filtered local background | **0.156** | 0.309 | yes |
| AR(p) on a locally-clipped periodogram | 0.046 | 0.102 | yes |

The second row is the tempting one: it fixes the null *better* than what
shipped. It is also the one that silently destroys the signal. The third
fails the other way — dividing each ordinate by a noisy background
estimate inflates the tail of the maximum, so the white-noise rate triples.
The fourth uses the median filter only to *identify* which ordinates to
clip out of the fit, and lets the smooth AR spectrum do the whitening; a
tone survives at every amplitude tested (1.0 to 8.0), always at the
correct bin.

Clipping against a *global* median instead of a local one was also tried
and fails a fourth way: a red background legitimately sits far above the
global median, so the clip flattens the structure that needs modelling
and the null barely moves (0.560 → 0.532).

### 5.2 The cost, and why it does not bite the motivating case

The filter attenuates what it flattens, so periodicity at very low
frequency — a handful of cycles across the whole window — gets harder to
see. That is the same confound the `peak_bin >= 2` gate already exists
for, one bin further out.

At moderate and high frequencies pre-whitening **gains** power, because
removing the background is what lets a modest peak stand out. A
corpus-sync artifact — the motivating hypothesis for the discovery-rate
scan — has a period of order the sync interval and therefore a high bin.
On a synthetic drift-plus-bin-51 series:

| series | raw | pre-whitened |
|---|---|---|
| drift only | p = 3.1e-20, saved only by the bin gate | p = 1.00 |
| drift + bin-51 sync | found at bin 51, p = 4.6e-62 | found at bin 51, p = **3.1e-69** |

The drift-only row is the one to notice: on that draw the raw test
produced a p-value of 3e-20 and was rescued purely by `peak_bin >= 2`.
The gate was doing all the work, which is exactly why the item was filed.

### 5.3 Two suspicions falsified, now pinned as tests

Both are in `tests/test_regression_periodicity_null.py` so they are not
re-proposed:

- **Volatility clustering was never the problem.** GARCH(1,1) work with
  no mean-level autocorrelation gives 0.044 against a 0.049 control, and
  piecewise-constant variance gives 0.043. Variance clustering leaves the
  ordinates exchangeable in expectation, so `garch.py`'s premise is *not*
  in conflict with Fisher's g. It is mean-level rate drift only. This is
  the plausible-and-wrong hypothesis worth recording: the tree has a
  GARCH module, so it looks like it should be the explanation.
- **Pure 1/f was already handled**, and not by the null: its peak
  collapses into bin 1, which `peak_bin >= 2` rejects. Both 1/f and 1/f²
  measure 0.000 raw.

### 5.4 Falsification

23 new tests, falsified two ways. Defaulting `prewhiten_series` to False
fails 6 of them. Making the background fit peak-blind — removing the clip
— fails exactly the 5 that guard §5.1. `prewhiten_series=False` is kept
as a parameter for that reason: without a way to reproduce the raw
periodogram, nothing in the suite distinguishes "the null was repaired"
from "the test went blind".

231 tests pass across the `detect_periodicity` consumers
(`test_periodicity`, `test_quasiperiodicity`, `test_report`,
`test_stats_reporter`, `test_berlekamp_massey`, the four
`test_regression_bugreport_*`); ruff clean on the three touched files.

### 5.5 Note for whoever does P3-T5

P3-T5 wants a log-log slope over a Brownian-motion hypothesis. §5.1 is
the same class of hazard one level over: an estimator fitted to a series
that contains the thing being measured will absorb it. The distance
trajectory will contain whatever directed-progress signal the item is
looking for, so if any background or trend is removed before the slope is
taken, check the peak-survival property first — measure that the fit does
*not* absorb a synthetic signal of known strength before trusting any
slope it produces.

---

## 6. Outcome: P4-T6 implemented, P3-T5 withdrawn

A separate session had already implemented and unit-tested both
`core/scaling_exponent.py` (P3-T5's proposed estimator) and
`core/corpus_flux.py` (P4-T6) against the *09-12* revision, before this
audit landed. On pulling this revision mid-work:

- **P4-T6 shipped as planned.** `core/corpus_flux.py` (the binomial
  ±1-per-event null, `z = net/sqrt(gross)`) is unchanged by anything in
  this audit — confirmed sound above — and is now wired into
  `corpus_manager.py` (`_corpus_added_count`, paired with the existing
  `_pruned_count`) and `services/fuzzer.py`/`stats.py`.
- **P3-T5's wiring was withdrawn, not shipped.** The already-written
  `core/scaling_exponent.py` module and its `services/fuzzer.py` /
  `services/stats.py` wiring (sampling `avg_distance` once per tick into
  the estimator) were deleted rather than merged, because this revision
  explicitly gates P3-T5 on three unresolved questions — the reuse of
  the misnamed `allan_variance` estimator (moot now that P0-T1 has
  landed, but the module itself never called into `allan_variance.py`,
  so that specific coupling was never present), the undecided
  per-seed-vs-per-tick trajectory definition, and the no-`--target-
  functions` constant-signal case — none of which had been decided on
  paper before the wiring was written. Shipping it anyway would have
  been exactly the failure mode P2-1 of the companion document warns
  about: code that lands, passes its tests, and is never trusted because
  the design question underneath it was never actually settled.


---

## 7. Unrelated, and worse: a persisted enum made the whole state file unloadable

Found while wiring P4-T6. Nothing to do with thermodynamics or stochastic
processes; recorded here because this is where the trail starts and because
the test file for it landed in the tree without its fix.

`CoverageRegimeDetector.save()` converts `self._regime` to `.value` but
passed `self._regime_history` through untouched, and that history is a list
of `(exec_count, CoverageRegime)` tuples. The state file is read back through
`state_store._SafeUnpickler`, whose allowlist is containers and scalars only,
so the reference to `fuzzer_tool.core.percolation` made the **entire file**
rejected. Measured on a state file written by a 2000-exec png campaign:

```
ERROR  refusing to load untrusted state file .../state.pkl.gz
SECTIONS RECOVERED: []
```

Ten sections went with it: `corpus`, `edge_tracker`, `markov`,
`seed_quality`, `crash_mi`, `length_tracker`, `sensitivity`,
`coverage_contract`, `regime`, and `corpus_flux` — so P4-T6's own
persistence was among the casualties on the day it shipped. `--resume`
silently started from zero for every component as soon as the regime
detector had recorded any history, which is after the first stats tick.

**Two things let it survive**, and both are the interesting part:

1. The failure is a `log.error` and an empty dict, not an exception, so a
   resume is externally indistinguishable from a cold start. Same shape as
   the forkserver lesson already in `handover_done`: a mode that silently
   falls back to the other mode cannot be detected from outside.
2. The message blames tampering — "refusing to load untrusted state file" —
   for a payload this codebase wrote itself, which sends anyone who does
   notice it looking for a corrupted corpus directory.

Fixed in `c469398` by persisting `.value` in the history, as `save()`
already did one line above for the scalar field, and accepting either shape
on load. **Deliberately not** fixed by allowlisting `CoverageRegime` in
`_ALLOWED_GLOBALS`: that trades a broken resume for a widened unpickling
surface on a file that lives in a directory a campaign writes constantly and
a user might copy between machines. A test pins that direction — the old
shape must keep raising `UnsafeStateError`.

### 7.1 A test that could not fail

`tests/test_regression_state_enum_payload.py` was already in the tree from
`8bdf52e` ("WIP: corpus flux") **without the fix it was written against**,
so 6 of its 8 cases were red on `388df9f`. Worth knowing if the history
looks odd.

More useful: the first version of one of those cases passed against the
broken code. It scanned the pickle opcode stream for `GLOBAL`/`STACK_GLOBAL`
arguments — and protocol 4 emits `STACK_GLOBAL` with *no* argument, taking
the module and name off the stack as separate string opcodes, so the set it
checked was always empty. An always-empty set trivially contains no
offenders. Replaced with a recursive type audit of the payload, which does
fail. Hard Rule 39's lesson reached from a new direction: the oracle was not
weak, it was structurally incapable of failing, and only falsification
exposed it.

### 7.2 Worth a sweep, not done here

`save()` converting one field and missing another in the same dict is a
shape, not an accident. The other `save()` implementations persisting
anything beyond plain containers and scalars have the same exposure, and the
failure mode is total rather than local — one bad field costs every section.
A mechanical check is cheap: walk each component's `save()` output and assert
every leaf is a primitive. The recursive audit in
`tests/test_regression_state_enum_payload.py` is the helper; pointing it at
all of them is the work. Filed as a P0-class item for the companion
document, not started.
