# Handover — QEA as a Hilbert-space object

**Original:** 2026-08-31. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_qea_hilbert_space_analysis_2026-08-31.md`.
**Verified against:** `4c021daa`. Analysis (product state, no cross-bit
correlation), the `rotation_gate` fix, intra-byte coupling and cooling are all
in `core/qea.py`.

---

## 1. A/B `--qea-correlation` and `--qea-cooling`

**What.** Both shipped opt-in on request, with no evidence they help:

- `--qea-correlation`: per-byte 8×8 Ising coupling, Gibbs-sampled
  (`core/qea.py::collapse_correlated`, Hebbian `update_couplings`).
- `--qea-cooling`: Δθ decay keyed to `elite_reset_every`'s cycle
  (`QEALifecycle._effective_rotation_angle`).

Both are also turned on by `--hail-mary` (`cli/commands.py` `_HAIL_MARY_FLAGS`),
so they change that preset's behaviour unmeasured.

**Gap.** `tools/lib/bench_paired.py` has `qea`, `qea-elite-reset`,
`qea-no-rotation`, `qea-no-bias` arms but none for correlation or cooling.

**To do.** Register `qea-correlation` and `qea-cooling` arms (baseline `qea`),
run paired on the `direct_lite` sets per the Boltzmann protocol
(`docs/learnings/2026-08-30-boltzmann-ab-result.md`).

**Acceptance.** Per-target W/L and median Δ edges with a power statement. On a
null or loss, drop both from `--hail-mary`, as `--continuum-reward` was
(`docs/learnings/2026-09-22-continuum-reward-ab-result.md`).
