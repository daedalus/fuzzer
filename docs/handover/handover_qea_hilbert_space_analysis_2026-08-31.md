# Handover — QEA as a Hilbert-space object: what `core/qea.py` actually is

**Date:** 2026-08-31
**Base:** `de26c1c`
**Status: ANALYSIS ONLY. No code changed.** This documents the structural
gap between the Hilbert-space framing QEA is named after and what
`core/qea.py` implements, plus a correctness bug in `rotation_gate()` found
while checking that framing against the code. No fix is proposed here beyond
the extension sketched in §4, which is flagged as low-value and not built.

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

Checking the Hilbert-space framing against the code surfaced a separate,
smaller issue — worth fixing on its own regardless of anything above.

The literature version of QEA rotates by angle `θ` (`α = cos θ`), updating as
`α ← cos(Δθ)·α − sin(Δθ)·β`. That has a built-in deceleration near the poles:
`dα/dθ → 0` as `α → 0` or `1`, so convergence naturally slows as a bit
approaches certainty — an annealing effect that falls out of the trig, not
an added schedule.

`rotation_gate()` (`qea.py:274`) instead does `α ± delta`, clamped to
`[ALPHA_MIN, ALPHA_MAX] = [0.01, 0.99]` (`qea.py:316-327`). That's a linear
walk in amplitude space, not angle space, and it does not decelerate near
saturation — it walks at a constant `delta` per step until it hits the hard
clamp and stops dead. So the practical difference from the paper's version
isn't cosmetic: near-certain bits in this implementation are one `improved`
flip away from a full swing back into contention, where the angle-based
version would already have `dα/dθ` shrunk to near zero and made that bit
sticky. Whether that's desirable is a separate question — this fuzzer's
`mutate_amplitudes()` (`qea.py:334`) already exists as an explicit
diversity-reset mechanism, so the paper's implicit deceleration may be
partially redundant here. But it should be a chosen tradeoff, not an
unnoticed side effect of `α ± delta` versus `cos(Δθ)`.

## 5. If genuine (partial) entanglement is wanted later

Model `α` per byte-block with a small pairwise correlation matrix (coupling
within a byte, say — 8×8) instead of a scalar per bit. Cost goes from `O(n)`
to `O(n·k)` for block size `k`, which stays tractable, and lets QEA itself
learn some local joint structure instead of relying entirely on the grammar
mutators for it.

**Not recommending this as next work.** `elite_reset_every` already exists
as a diversity-preservation knob, and the structural mutators already own
correlation successfully (§3) — a block-correlation QEA would duplicate that
coverage at real implementation cost, for a representation whose whole
reason to exist is staying cheap. Flagging it here only because it's the
direct answer to "how would more Hilbert-space-like expressivity apply to
this module," not because the fuzzer needs it.

## 6. What to read this alongside

- `core/qea.py` — the module analyzed here (`_bias_amplitudes_from`,
  `collapse`, `rotation_gate`, `mutate_amplitudes`, `QEALifecycle._evolve`).
- `core/structural_constraints.py` and `core/field_constraints.py` — the
  correlation-repair machinery that compensates for §3.
- `core/ga.py` — the committed-value alternative representation; useful
  contrast for what QEA is trading away for its `O(n)` cost.
- Han & Kim, "Quantum-inspired evolutionary algorithm for a class of
  combinatorial optimization," IEEE Trans. Evol. Comp., 2002 — cited in the
  module docstring, source of the rotation-gate formulation in §4.
