# Handover — QEA as a Hilbert-space object: what `core/qea.py` actually is

**Date:** 2026-08-31 (updated with §7 implementation)
**Base:** `de26c1c` (analysis); implementation in this repo state rebased onto `2edbce1`
**Status:** §§1–3 analysis, unchanged. §4's bug fixed upstream in `2edbce1`
(independently of this doc). §5's sketch implemented on request — see §7.

---

## 1. What's actually there

Each bit is a single-qubit state `(α, β)` with `α² + β² = 1`, but only `α` is
ever stored — `β` is implicit as `√(1-α²)`, and there is no phase term. That
places every bit on the real, non-negative quarter of the unit circle, not
the full Bloch sphere: a bit's state is one real number in `[0, 1]`, not a
point on a 2-sphere.

An individual of `n` bits (`QEAIndividual`, `qea.py:115`) stores this as a
flat `amplitudes` array of length `8 * num_bytes` — one `α` per bit
(`qea.py:124`, `num_bytes` property at `qea.py:157`). `collapse()`
(`qea.py:251`) samples each bit independently: `P(bit=0) = α_i²`, drawn via
a single vectorized `np.random.random` call against the whole array
(`qea.py:267`).

## 2. The structural fact: product state, not entangled state

This is a mean-field / product-state approximation, not a point in a genuine
n-qubit Hilbert space. A real n-qubit register has dimension `2ⁿ` and its
general state is a vector over all `2ⁿ` basis strings — that's what lets it
represent entanglement, where the distribution over bit `i` depends on the
value sampled for bit `j`.

This code instead tracks `n` independent single-qubit states — the tensor
product `|ψ₁⟩⊗|ψ₂⟩⊗...⊗|ψₙ⟩` — which is `O(n)` in both storage and update
cost, not `O(2ⁿ)`. `collapse()` makes that independence explicit: every bit's
draw depends only on its own `α`, never on any other bit's outcome. That's
the deliberate tradeoff in Han & Kim's original QEA (see module docstring,
`qea.py:20`) — it's the only reason this representation is tractable at all
for byte-length individuals. A true entangled encoding of a 64-byte seed
would need `2⁵¹²` amplitudes.

## 3. Consequence for this fuzzer: QEA cannot represent cross-bit correlation

Because each bit's amplitude evolves independently in `rotation_gate()`
(`qea.py:274`), QEA can bias the population toward individually-likely bit
values, but structurally cannot bias toward jointly-valid combinations —
e.g. "these 4 bytes are a CRC of the rest," or "byte 0 must be `0x89` **and**
byte 1 must be `0x50`" (PNG magic). Every `α` update in `rotation_gate` is a
function of that bit's own value in `collapsed` and the scalar `improved`
flag; there's no term anywhere that couples one bit's update to another
bit's value.

That gap is exactly what the structural/grammar mutators are already
compensating for. `structural_constraints.py` names the same failure mode
independently of this investigation: mutating a value invalidates dependent
fields (length prefixes, TLV nesting, wraparound arithmetic) unless something
tracks the dependency and repairs it — which is what `field_constraints.py`'s
topological repair does. That repair machinery is the entanglement-like
correlation QEA's representation has no way to hold internally. QEA and the
structural mutators aren't redundant; QEA owns per-bit likelihood, the
structural layer owns joint validity, and the split is forced by QEA's
`O(n)` design choice, not incidental.

## 4. A correctness bug found in the process: `rotation_gate` is not a rotation

**Status: fixed upstream, independently of this document, in commit
`2edbce1` ("fix(qea): rotation_gate performs a true angular rotation, not a
linear walk").** The description below is kept as the original finding for
context; `rotation_gate()` now does the `cos(Δθ)/sin(Δθ)` rotation described
here rather than the `α ± delta` walk.

Checking the Hilbert-space framing against the code surfaced a separate,
smaller issue — worth fixing on its own regardless of anything above.

The literature version of QEA rotates by angle `θ` (`α = cos θ`), updating as
`α ← cos(Δθ)·α − sin(Δθ)·β`. That has a built-in deceleration near the poles:
`dα/dθ → 0` as `α → 0` or `1`, so convergence naturally slows as a bit
approaches certainty — an annealing effect that falls out of the trig, not
an added schedule.

`rotation_gate()` (`qea.py:274`, at the time of writing) instead did
`α ± delta`, clamped to `[ALPHA_MIN, ALPHA_MAX] = [0.01, 0.99]`. That's a
linear walk in amplitude space, not angle space, and it does not decelerate
near saturation — it walks at a constant `delta` per step until it hits the
hard clamp and stops dead. So the practical difference from the paper's
version isn't cosmetic: near-certain bits in that implementation were one
`improved` flip away from a full swing back into contention, where the
angle-based version would already have `dα/dθ` shrunk to near zero and made
that bit sticky. Whether that's desirable is a separate question — this
fuzzer's `mutate_amplitudes()` already exists as an explicit diversity-reset
mechanism, so the paper's implicit deceleration may be partially redundant
here. But it should be a chosen tradeoff, not an unnoticed side effect of
`α ± delta` versus `cos(Δθ)`.

## 5. If genuine (partial) entanglement is wanted later

**Status: implemented — see §7.** The sketch below is kept as the original
proposal; §7 documents what was actually shipped and where it diverges from
this sketch.

Model `α` per byte-block with a small pairwise correlation matrix (coupling
within a byte, say — 8×8) instead of a scalar per bit. Cost goes from `O(n)`
to `O(n·k)` for block size `k`, which stays tractable, and lets QEA itself
learn some local joint structure instead of relying entirely on the grammar
mutators for it.

**Not recommended as *unprompted* next work** (see §7 for why it was built
anyway). `elite_reset_every` already exists as a diversity-preservation
knob, and the structural mutators already own correlation successfully
(§3) — a block-correlation QEA duplicates that coverage at real
implementation cost, for a representation whose whole reason to exist is
staying cheap. This was flagged originally only as the direct answer to
"how would more Hilbert-space-like expressivity apply to this module," not
as something the fuzzer needed.

## 6. What to read this alongside

- `core/qea.py` — the module analyzed here (`_bias_amplitudes_from`,
  `collapse`, `rotation_gate`, `mutate_amplitudes`, `QEALifecycle._evolve`,
  and — as of §7 — `collapse_correlated`, `update_couplings`).
- `core/structural_constraints.py` and `core/field_constraints.py` — the
  correlation-repair machinery that compensates for §3.
- `core/ga.py` — the committed-value alternative representation; useful
  contrast for what QEA is trading away for its `O(n)` cost.
- Han & Kim, "Quantum-inspired evolutionary algorithm for a class of
  combinatorial optimization," IEEE Trans. Evol. Comp., 2002 — cited in the
  module docstring, source of the rotation-gate formulation in §4.
- `tests/test_qea_correlation.py` — tests for the §7 implementation.

## 7. Implementation: intra-byte coupling (§5, built)

Shipped as an opt-in extension to `core/qea.py`, `core/cli/commands.py`, and
`core/services/fuzzer.py`. Default is off (`use_correlation=False` /
no `--qea-correlation` flag); with it off, every code path in this section
is unreached and behavior is identical to before this commit.

**Representation.** `QEAIndividual` gains an optional `coupling` field:
`ndarray | None`, shape `(num_bytes, 8, 8)`, `None` unless correlation is
enabled. This is the 8×8-per-byte matrix from §5's sketch — scoped to
within a byte deliberately, for the reason §5 gives (cross-byte correlation
is the structural mutators' job, and extending the tensor across byte
boundaries reintroduces cost this representation exists to avoid).

**Sampling — `collapse_correlated()`.** §5 didn't specify how a coupling
matrix turns into a joint sample; the implementation uses a classical
pairwise (Ising-like) model, sampled by Gibbs sweeps (default 3 full sweeps
over the byte's 8 bits). Per-bit amplitudes convert to a local field via
`_alpha_to_field()`: `h = ln(α² / (1 - α²))`, chosen so `sigmoid(h) = α²`
exactly — i.e. with all-zero coupling, `collapse_correlated()` reduces to
independent per-bit sampling from the field alone and reproduces
`collapse()`'s marginals bit-for-bit (verified statistically in
`test_zero_coupling_matches_marginals`; this is also why a freshly bred or
freshly discovered individual, which starts at zero coupling, behaves
identically to the uncorrelated representation until it accumulates
signal). Each Gibbs step draws bit `i` from
`sigmoid(h_i + Σ_j J_ij · s_j)`, `s ∈ {+1, -1}` for `{bit=0, bit=1}` — a
positive `J_ij` biases bits `i, j` toward agreement, negative toward
disagreement.

One implementation pitfall worth flagging for anyone extending this: the
textbook Ising conditional is `sigmoid(2·(h + ΣJs))`, and the first version
of this code used that factor of 2 out of habit. It's wrong here — that
doubling belongs to a Hamiltonian written as `-h·s - ΣJ·s·s'`, not to a
field defined so `sigmoid(h) = α²` directly, and applying it anyway silently
shifted every marginal (α=0.9 collapsed to `P(bit=0)≈0.94` instead of
`0.81`). Caught by a statistical test before it shipped, not by inspection —
worth remembering if this model is extended further.

**Learning — `update_couplings()`.** Hebbian rule, applied alongside
`rotation_gate()` in `on_fuzz_result()` using the same parent and same
collapsed bytes: `ΔJ_ij = ±delta · s_i · s_j`, sign matching `improved`.
Reinforces whatever pairwise relationship (agreement or disagreement) the
collapsed bytes actually had, when that outcome was good; pushes away from
it when it wasn't. Diagonal is kept at zero and every entry is clamped to
`±correlation_max` (default 2.0), mirroring `ALPHA_MIN`/`ALPHA_MAX`'s role
for amplitudes.

**Lifecycle wiring.** `QEALifecycle` gained `use_correlation`,
`correlation_delta`, `correlation_max`, `correlation_sweeps`.
`initialize()` and new-coverage individuals in `on_fuzz_result()` get a
zero coupling matrix when enabled; `pick_seed()` and `_evolve()`'s parent
collapses use `collapse_correlated()` instead of `collapse()` when a
parent has one. Offspring from `_evolve()`'s crossover deliberately do
**not** inherit either parent's coupling matrix — crossover already
scrambles which bytes came from which parent, so a carried-over matrix
would describe correlations for bytes that may no longer sit at those
positions; the child starts at zero and learns its own, matching how
`_bias_amplitudes_from()` already discards the parents' amplitude nuance
in favor of a fresh bias. CLI flags: `--qea-correlation`,
`--qea-correlation-delta`, `--qea-correlation-max`,
`--qea-correlation-sweeps`.

**Cost.** A coupling tensor is `64` floats/byte vs. `8` floats/byte for
amplitudes — an 8× memory increase over the already-documented amplitude
cost, only paid when `use_correlation=True`. `collapse_correlated()` is
`O(sweeps · 8² · num_bytes)` vs. `collapse()`'s `O(num_bytes)` — still
linear in `num_bytes`, with a constant-factor cost tied to `sweeps`
(default 3), not to population size or generation count.

**What this doesn't change.** §5's recommendation against building this
unprompted stands — the structural mutators still own cross-byte
correlation, and this only reaches within a byte. This was implemented on
explicit request, not because a benchmark showed the structural mutators
under-covering intra-byte correlation; no A/B evidence is claimed here that
`use_correlation=True` improves fuzzing outcomes on any target. Treat it the
same way `elite_reset_every` and the coupling-magnitude CLI flags are
treated in commit `9657454`: an arm to test, not a default.
