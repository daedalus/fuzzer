# Handover — control-theory framing of the fuzzer's feedback loops

**Date:** 2026-09-12
**Base (fuzzer):** `972470b` (HEAD at time of analysis)
**Sources:** four Wikipedia articles brought in for integration analysis —
[Active disturbance rejection control](https://en.wikipedia.org/wiki/Active_disturbance_rejection_control),
[PID controller](https://en.wikipedia.org/wiki/PID_controller),
[Nyquist stability criterion](https://en.wikipedia.org/wiki/Nyquist_stability_criterion),
[Closed-loop transfer function](https://en.wikipedia.org/wiki/Closed-loop_transfer_function).

This is an analysis document with two measured results and a prioritised
proposal. Nothing here is implemented. Tier 1 items have falsifiers named;
Tier 2 does not yet, and producing one is the first step for whoever picks it
up, not something done in advance for them.

**Negative result first, so nobody re-derives it:** there is no
control-theory machinery in the tree. `grep -rniE
'(setpoint|proportional gain|integral term|Kp|Ki|Kd|anti-?windup|ADRC|Nyquist|
transfer function|bode|phase margin)'` over `src/`, `tools/`, `docs/` and
`tests/` returns only false positives — `KiB` in size comments, `waitpid`,
`os.getpid`, and the FFT `Nyquist` bin in `structured.py:701` /
`DEEP_DIVE.md`. No prior survey considered and rejected this either; the four
merged port surveys in `docs/port-backlog.md` do not mention feedback control.

The finding is not that control theory is missing. It is that **the fuzzer
already runs four feedback loops**, three of them unnamed, and one of them
structurally broken in a way the vocabulary names precisely.

---

## 1. The four loops that already exist

### L1 — An open-loop knob that should be closed (`seed_picker.py:460`)

```python
if f._anneal_budget > 0:
    f._temperature = max(0.1, 1.0 - f.exec_count / f._anneal_budget)
else:
    f._temperature = 1.0
```

Pure feed-forward on a clock. `_anneal_budget` defaults to 10000 when any
annealing flag is set (`cli/commands.py:1821-1824`). The exploration
temperature — consumed at `seed_picker.py:551`, `:1227`, `:1505`, `:1584` and
in `monte_carlo.py::add_elite` — never reads whether the campaign is
discovering anything.

The measured signal it would need already exists and already has consumers:
`RobustKF` over discovery rate (`core/kalman.py`, 2D constant-velocity with
Huber innovation gating and adaptive R), `AllanVarianceDetector`,
`DispersionIndex`, `CriticalSlowingDown`, `coverage_growth_model()`. Every one
of them feeds a *detector*. None feeds an *actuator*. That asymmetry is the
whole finding: the sensing half of a control loop is built and the actuation
half is a clock.

### L2 — A bang-bang relay with asymmetric hysteresis

Engage: `_maybe_trigger_stall_recovery` (`services/fuzzer.py:5626`), after
`execs_since_edge >= _stall_threshold` (default 1000), gated by the
entropy/Allan/dispersion stack and with gain-scheduled thresholds
(`:5690-5706`: halved on `fatiguing`, `//4` or `//8` on `stalled`).

Release: `services/fuzzer.py:4475-4480`, on **one** new edge. No dwell, no
minimum on-time.

So engaging costs 1000 execs of silence and releasing costs a single edge —
hysteresis asymmetric by a factor of the threshold. See §2 for what that does
under a bursty arrival process.

This is a relay controller, and it matters that it is: the Åström–Hägglund
relay method *deliberately* drives a plant with bang-bang control in order to
measure the ultimate gain and period, which then feed Ziegler–Nichols. The
fuzzer is running that experiment in every campaign and discarding the data.
`_stall_recovery_count` and `_stall_recovery_execs` are already collected
(`fuzzer.py:5718`, `:3796`) and already printed (`stats.py:312-315`,
`:1122-1123`) — what is missing is the per-cycle *period* and *amplitude*.

### L3 — Positive feedback with latch-up (the saturation gate)

`SeedPicker._saturation_gate` (`services/seed_picker.py:1164-1210`):

- `sat >= SATURATION_GATE` (0.99) ⟹ gated
- gated ⟹ `_compute_weights` skips `classify_seeds()` and the whole phase-2
  analysis; `seed_picker.py:792-796` cuts subsumption / diversity /
  Wasserstein / proximity to neutral multipliers
- neutral multipliers ⟹ the picker loses the rareness and diversity signal
  that drives it toward new coverage
- less new coverage ⟹ Chao2 keeps reporting saturation at or near 1.0
- ⟹ gate stays on

That is a positive-feedback path with loop gain ≥ 1. The escapes currently in
the code are two watchdog timers — `SATURATION_REFRESH_EXECS = 2000` and the
`SATURATION_STALL_EXECS = 20000` override — added when the gate was found to
latch. They work, but they are timers, not a loop fix.

This is the one place the Nyquist material says something non-decorative.
Instability comes from excess gain, particularly in the presence of
significant lag; on a self-reinforcing path there is *no* bounded steady
state, so no amount of retuning `SATURATION_GATE` helps. Only two things do:
break the loop (estimate saturation from a channel the gate does not
influence), or give the relay a real dead band with a release threshold below
the engage threshold.

Related, and already known: Chao2 returns 1.0 for *any* plateau, not only for
a genuinely closed universe (measured previously: 60 seeds over a closed
500-edge universe → 1.0; a single starved seed → 1.0). So the sensor feeding
this loop is saturated in the instrument sense as well as the coverage sense.

### L4 — Dead time in the sensing chain

The stall detector is fed once per stats tick
(`fuzzer.py:6786-6797`), and the tick is
`_stats_effective_interval()` = `max(1, int(10 * last_avg_eps))`
(`fuzzer.py:5471-5489`) — roughly **10 seconds of work**, and *variable*,
because it tracks EPS. The value fed is a delta count over that window
(`current_edges - _last_allan_edge_count`), so when EPS moves, both the
sample period and the signal's scale move with it.

§2 measures how long the detector takes to react. The short version: θ lands
somewhere between ~20 s and ~6 minutes of wall-clock, and is not even
approximately constant.

Two consequences for anything built on top:

- **A variable sampling period breaks a discrete PID.** `integral += error *
  dt` with `dt` drifting as a function of the plant's own throughput is the
  anti-pattern. Either give the control loop its own fixed exec-count tick, or
  normalise the process variable to edges-per-*exec* rather than
  edges-per-tick.
- **Ziegler–Nichols is the wrong tuner.** The article says so directly: it
  does not work well with time-delay processes. Use L2's relay data instead
  (§3, Tier 1.3).

There is a second, smaller transport delay already documented in the tree:
`core/schedulers/ducb.py:37` notes that with `mutations_per_input = 8` the
effective horizon of `gamma` is ~8× shorter than it looks, because a batch of
8 operator selections shares one binary reward. That is a phase-lag statement
and it is already written down; no action needed, but it is the reason §4
argues against putting a controller on top of the bandits.

---

## 2. Measurements

Both scripts are self-contained and run against the installed package at
`972470b`. They are inlined here rather than added under `tools/` because
they are one-shot characterisations, not campaign instrumentation — but they
*are* inlined, so the numbers below are reproducible from this document
alone. (Previous handovers have cited raw-output TSVs that were never
committed; this avoids repeating that.)

### 2.1 Sensing-chain dead time of the stall detector

Step the true discovery rate down and count ticks until the real
`AllanVarianceDetector.noise_type()` stops saying `"active"`.

```python
import numpy as np
from fuzzer_tool.core.allan_variance import AllanVarianceDetector

rng = np.random.default_rng(42)

def dead_time(rate_hi, rate_lo, warm=64, maxticks=256):
    d = AllanVarianceDetector(max_buffer_pow=8, min_samples=8)   # repo defaults
    for _ in range(warm):
        d.update(float(rng.poisson(rate_hi)))
    pre = d.noise_type()
    for k in range(1, maxticks + 1):
        d.update(float(rng.poisson(rate_lo)))
        nt = d.noise_type()
        if nt != "active":
            return pre, nt, k
    return pre, d.noise_type(), None

for hi, lo in [(20, 0), (20, 1), (20, 2), (5, 0), (5, 1), (50, 0)]:
    ks, labels = [], []
    for _ in range(20):
        pre, nt, k = dead_time(hi, lo)
        if k is not None:
            ks.append(k); labels.append(nt)
    print(hi, lo, int(np.median(ks)), int(np.percentile(ks, 10)),
          int(np.percentile(ks, 90)), max(set(labels), key=labels.count))
```

| step (edges/tick) | classified as | median ticks | p10 | p90 |
|---|---|---|---|---|
| 20 → 0 | `fatiguing` | 34 | 1 | 35 |
| 20 → 1 | `fatiguing` | 16 | 1 | 36 |
| 20 → 2 | `fatiguing` | 2 | 1 | 36 |
| 5 → 0 | `fatiguing` | 31 | 1 | 39 |
| 5 → 1 | `fatiguing` | 6 | 1 | 38 |
| 50 → 0 | `fatiguing` | 34 | 1 | 35 |

Read the spread, not the median: p10 = 1 and p90 ≈ 36 in every row. The
dead time is not a constant with noise around it, it is a distribution
covering the whole measurable range. At ~10 s per tick that is ~20 s to ~6 min
before the detector reacts to a step.

**Caveat on the p10 = 1 column:** some of those are the detector reacting to
Poisson noise rather than to the step, so the low tail is optimistic. That
makes the useful θ *longer* than the table suggests, not shorter.

### 2.2 Relay switching under bursty discovery

Engage at `execs_since_edge >= 1000`; release after `dwell + 1` edges.
Arrival process is clustered — bursts of ~4 edges separated by ~2500 execs,
which is the regime `AllanVarianceDetector.is_overdispersed()` exists to
identify, and the one where a 1000-exec engage threshold and a 1-edge release
interact worst.

```python
import numpy as np
rng = np.random.default_rng(7)

def run(n_execs=400_000, threshold=1000, burst_mean=4, gap_mean=2500,
        release_dwell=0):
    arrivals, t = [], 0
    while t < n_execs:
        t += rng.exponential(gap_mean)
        for _ in range(1 + rng.poisson(burst_mean - 1)):
            if t < n_execs:
                arrivals.append(int(t))
            t += rng.exponential(30)
    arrivals = sorted(set(arrivals))

    active, switches, recovery, last_edge, ai, since = \
        False, 0, 0, 0, 0, 0
    for e in range(n_execs):
        if ai < len(arrivals) and arrivals[ai] == e:
            ai += 1; last_edge = e
            if active:
                since += 1
                if since > release_dwell:
                    active = False; switches += 1; since = 0
        elif not active and e - last_edge >= threshold:
            active = True; switches += 1; since = 0
        recovery += active
    return switches // 2, recovery / n_execs, len(arrivals)

for dwell in (0, 1, 3, 8):
    print(dwell + 1, *run(release_dwell=dwell))
```

| release after | engage/release cycles per 400k execs | execs in recovery |
|---|---|---|
| **1 edge (current)** | **106** | **65.9%** |
| 2 edges | 99 | 65.9% |
| 4 edges | 79 | 73.9% |
| 9 edges | 51 | 84.5% |

Current behaviour switches every ~3,800 execs, and spends two thirds of the
campaign in "random mode" with `n_mutations` floored at 16.

Note the direction of the trade, because it is the classic relay one and it
is easy to get backwards: **adding dwell reduces switching and increases
duty.** More dwell is not unambiguously better. Which end to pick depends on
whether recovery mode is cheaper or dearer per exec than normal mode — and
that is measurable today from `_stall_recovery_execs` against edges found
while `_stall_recovery_active`, neither of which is currently attributed.

**Scope of this simulation:** it models the relay's *switching logic only*,
with an exogenous arrival process. It assumes recovery mode does not change
the discovery rate, which is certainly false — the loop's actual gain is the
unmeasured quantity. So take the cycle counts as a property of the hysteresis
asymmetry, not as a prediction of campaign behaviour. Measuring the real gain
is Tier 1.3.

---

## 3. Proposal

### Tier 1 — small, falsifiable, no new subsystem

**1.1 Dead band on the saturation gate.** Engage at 0.99, release at ~0.95,
replacing reliance on the two watchdog timers as the only escape (keep the
timers; they also cover the Chao2-plateau sensor problem, which a dead band
does not).

*Falsifier:* construct a plateau over a closed edge universe, assert the
current code stays gated until `SATURATION_STALL_EXECS` and that the
dead-band version releases on the estimate itself. The existing measurement
(60 seeds / 500-edge closed universe → saturation 1.0) is the fixture.

**1.2 Release dwell on the stall relay.** Require k sustained edges, or n
execs of sustained discovery, before dropping out of recovery — not one edge.

*Falsifier:* the §2.2 harness, promoted to a test over the real predicate
rather than a reimplementation of it, asserting the cycle count falls. Do
**not** pick k from the cycle count alone; pick it from 1.3.

**1.3 Log the relay's period and amplitude.** Per cycle: engage exec, release
exec, edges found while active, edges found in the preceding inactive
stretch. This is three counters and it buys two separate things — the
per-exec productivity comparison that 1.2 needs to choose k, and the ultimate
gain and period that Tier 2 needs to tune. Every campaign already runs the
experiment.

### Tier 2 — one real feature: close L1

A **PI** controller on the temperature knob:

- **PV:** Kalman-filtered edges **per exec** (not per tick — see L4), from the
  existing discovery-rate `RobustKF`.
- **SP:** a target discovery rate. Open question: absolute, or a fraction of
  the campaign's own running maximum. The latter is self-normalising across
  targets; the former is tunable but target-specific.
- **MV:** `_temperature`, clamped to [0.1, 1.0] — the actuator bound is
  already there, for free.
- **Sampling:** the control loop gets its **own fixed exec-count tick**. It
  must not inherit `_stats_effective_interval`.
- **No D term.** The PV is a Poisson count and derivative action amplifies
  higher-frequency measurement noise. PI controllers are standard exactly
  where derivative action would be noise-sensitive but the integral term is
  needed to reach the target at all.
- **Anti-windup is mandatory, not a refinement.** The actuator saturates at
  both ends and the PV can sit at zero for an entire plateau — the textbook
  windup case, where the integral accumulates an error larger than the
  regulation variable's maximum and the system then overshoots until it
  unwinds. Clamp the accumulator, or back-calculate it to keep the output
  feasible.
- **Keep `--anneal-budget` as feed-forward.** The PI runs as a *correction on
  top of* the clock schedule, not as a replacement. This is the standard
  feed-forward-plus-feedback arrangement: the feed-forward term is not
  affected by the process feedback, so it cannot contribute to oscillation,
  and it means the current clock behaviour remains exactly what happens when
  the loop is off. Gate behind an opt-in flag, unmeasured, same policy as
  `--tang` / `--continuum`.

Then, and only then: **extend the discovery-rate KF by one state for the
lumped disturbance** — the ADRC piece. Its premise is the right one for this
plant: extend the model with a fictitious state standing for everything the
user did not put in the mathematical description, estimate it online with an
extended state observer, and use it in the control signal to decouple the
system from the actual perturbation. There is no model of
d(edges)/d(temperature) and there will not be one. In fuzzing terms the
lumped disturbance is: a new region unlocking, a region exhausting,
`_maybe_prune` evicting seeds, an SHM resize, corpus reload on resume.

Two thirds of the ESO is already built. `RobustKF` is 2D constant-velocity
with Huber gating and adaptive measurement-noise covariance — the third state
is the addition. And ADRC's tracking differentiator has a second purpose that
the same filter already serves: it avoids amplifying noise in a derivative
term by integrating rather than differentiating, which is exactly why the 2D
KF reads the rate off the state instead of differencing counts.

**Sensitivity function, once, as analysis and not as code.** After 1.3 yields
Ku/Tu, compute `S = 1/(1 + CG)` for the closed L1. It answers a question that
is directly meaningful here: when a region exhausts and the true rate halves,
what fraction of that disturbance survives as wasted budget before the loop
compensates? Worth one paragraph in `DEEP_DIVE.md`. Not worth a module.

### Tier 3 — considered and rejected

Recorded as rejected, not merely absent, so they are not re-proposed
(`docs/port-backlog.md` convention):

- **Nyquist plots, Bode plots, phase margin as artifacts.** A frequency
  response requires injecting sinusoidal setpoint variation over hours per
  campaign, and the plant is non-stationary by construction, so any
  measurement expires before it is useful. The *criterion* still earns its
  place as the argument in L3; the *plot* does not. Use the relay experiment.
- **Ziegler–Nichols.** Explicitly poor on time-delay processes, and §2.1 says
  θ is both large and badly conditioned.
- **Full nonlinear ADRC** — NESO plus nonlinear state error feedback. The
  nonlinear error feedback buys disturbance rejection without overshoot on a
  plant whose dynamics can at least be bounded. Here they cannot, and the
  extra tuning parameters have no data to set them from. Take the ESO idea,
  leave the nonlinear half.
- **The D term, anywhere in this codebase.**

---

## 4. Where *not* to put a controller

**Not on the operator schedulers.** They are already closed-loop learners
with their own loop gains — `cucb.py` `gamma=0.9995`, `ducb.py`
`gamma=0.9999`, `epsilon_greedy.py` `decay=0.9995`, EXP3's `gamma`, MOpt's
inertia weight. A PI on top of a bandit is a cascade with two integrators,
which is a stability problem rather than a feature. The genuinely useful
control observation about them is the transport delay already noted in
`ducb.py:37`, and it needs no new machinery.

**Actuator candidates, ranked.** A controller needs a manipulated variable
with a known sign and a bounded range:

1. `_temperature` — best. Monotone in exploration by construction and already
   clamped to [0.1, 1.0]. Its *effect on discovery rate* is not monotone, but
   that is what the disturbance state is for.
2. `mutations_per_input` — cleanest mechanically: integer, bounded, directly
   proportional to budget per seed. Currently a static CLI constant (default
   8, `fuzzer.py:741`) scaled per seed by `_last_perf_score`
   (`operators.py:4814-4815`). Worth noting that
   `operators.py:4816-4817` already floors it to 16 during recovery —
   i.e. there is *already* a hand-written bang-bang override on exactly the
   knob a regulator would own. If L1 lands, that floor should become the
   controller's output, not a separate mechanism.
3. Anything inside a scheduler — no, per above.

---

## 5. Open questions gating the work

- **Setpoint semantics for L1**: absolute edges/exec, or a fraction of the
  running maximum? Nothing in the tree answers this and the choice decides
  whether the loop is portable across targets.
- **Is recovery mode cheaper or dearer per exec?** Tier 1.2 cannot pick a
  dwell without this, and it is a three-counter measurement (1.3).
- **What is the real sign and magnitude of d(edges)/d(temperature)?** §2.2
  assumes zero and says so. If it is near zero over the operating range, L1
  is not worth closing at all and Tier 2 should be dropped — that is the
  honest kill criterion for this whole line of work, and it should be
  measured before the controller is written, not after.
- **Does the Chao2 sensor need fixing before or instead of L3's dead band?**
  A dead band on a sensor that reads 1.0 for every plateau moves the latch
  point without removing the sensor's blind spot. The two fixes are
  independent and only one is proposed here.
