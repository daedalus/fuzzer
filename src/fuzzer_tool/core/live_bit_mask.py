"""Online liveness estimation via an OR-accumulator over XOR-diff samples.

Port of `xoreaxeaxeax/skitter-creek-bath-salts`'s
`analysis/gather_aliases.py:363-400` (`MaskState`, `observe_hit`), per
`docs/handover/handover_skittercreek_tailslayer_port.md` item 4.

What this answers, and how it differs from item 1
---------------------------------------------------
:mod:`fuzzer_tool.core.xor_map_solver` (item 1) answers "what exactly is
the map" -- precise, exact, elimination- or z3-backed, real cost per bit.
This module answers a cheaper, different, *prior* question: "which bit
positions matter at all, to anything" -- O(1) work per sample, no solver.

Model each of the ``n_bits`` candidate positions as an unknown binary
latent -- *live* (actually participates in whatever relation is being
observed) or *dead* (never does). Before any samples, that's ``n_bits``
bits of uncertainty over the support set. Each sample's ``baseline ^
mutant`` is a noisy query that reveals a subset of the live bits --
whichever ones happened to differ on that draw -- and
``mask |= (baseline ^ mutant)`` is a monotone, one-sided estimator of the
true live-bit set: it can only grow, never shrink, and converges to
exactly the true support as samples accumulate, because a truly-live bit
has vanishing probability of *never once* appearing in the diff over many
draws (coupon-collector tail).

Convergence semantics (deliberately stricter than the source)
----------------------------------------------------------------
The source's `MASK_SWITCH_AFTER = 200` is a fixed stopping rule on
*cumulative hit count*: once 200 total observations have been made,
switch from broad sampling to exhaustive search restricted to the
converged mask. :attr:`LiveBitMaskEstimator.is_converged` here is a
different, stricter condition: true once *`switch_after` consecutive*
`observe()` calls have passed *without the mask growing* -- i.e. recent
evidence, not just accumulated evidence. This can also flip back to
`False`: if the mask grows again after having satisfied that condition,
the estimator correctly reports itself as no-longer-converged rather than
keeping a one-way "switched" flag that a stale early guess could leave
stuck true. Which threshold is actually safe is a real empirical
question, not assumed here -- see the sensitivity-sweep discipline in
`tests/test_live_bit_mask.py`.

What `.mask` does NOT claim
-----------------------------
A set bit in `.mask` means "this position moved the observed diff at
least once, in `.samples_seen` samples so far." It never means "moves on
every mutation" -- no code path here computes or claims per-mutation
trigger rate, only accumulated ever-moved-once evidence. Same
fail-closed discipline as `int_checksum_solver.py`'s modulus recovery:
absence of evidence is unresolved, not a negative claim. In particular,
a *dead* verdict is only warranted once `.is_converged` is true, not
merely because `.mask` doesn't yet contain a given bit -- a rare-but-live
bit that hasn't fired yet is indistinguishable, before convergence, from
a genuinely dead one.

Fuzzer-domain mapping and non-goals
--------------------------------------
`target`/`alias` (two addresses known to alias) in the source become
`baseline`/`mutant` (two coverage bitmaps from two runs of the same seed
differing in one byte/mutation) here. No mode-switching CLI/sampling
logic is ported (`samples_for`, `choose_initial_mask`, `pick_pa`,
`RunLog`) -- those govern *how DRAM addresses get chosen to probe*, which
has no fuzzer analog; the fuzzer already has its own input-selection
machinery (`schedules.py`, `ga.py`) and this module only needs to consume
`(baseline, mutant)` pairs that machinery already produces, not decide
what to try next.

This module has no dependents yet -- it's a leaf utility, landed ahead of
its intended consumers (`schedules.py` byte down-weighting,
`format_learner.py` padding/dead-region inference). Wiring it in is a
separate, later step, gated on a real-corpus convergence-threshold
sensitivity sweep per the handover doc's Sequencing section, not done
here.


Validation status: a genuinely coverage-dead region now exists to test
against. Four real campaigns (zlib, png twice, jpeg) produced none, for
structural reasons -- compressed data has no padding, and any CRC-covered
format rules out coverage-dead bytes outright. `tools/gen_synthetic_target.py`
supplies one by construction (0/60 dead-region mutations move coverage,
60/60 live-region ones do; see
docs/sweeps/synthetic_target_ground_truth_2026-08-19.md). The false-negative
rate has now been measured against it via
`tools/sweep_liveness_thresholds.py --synthetic-target`: the dead region
earns a DEAD verdict at every switch_after tested and the live region never
does (false-negative and false-positive rate both 0). What that run cannot
measure is a real cold-but-live region -- one that emits a long no-growth
run before its first edge -- so `_LIVENESS_DEAD_WEIGHT` stays a soft
down-weight rather than a hard exclusion. See
docs/sweeps/synthetic_liveness_calibration_2026-08-29.md.
"""

from __future__ import annotations


class LiveBitMaskEstimator:
    """OR-accumulator over ``baseline ^ mutant`` samples, with a
    consecutive-no-growth convergence detector.

    Args:
        n_bits: Width of the observed bit-vector space. `observe()`
            rejects any `baseline`/`mutant` with bits set at or above
            this width -- guilty until proven in-range, same discipline
            as `gf2_common.invert_bitmask_map`'s row-width check.
        switch_after: Number of consecutive `observe()` calls with no
            mask growth required before `.is_converged` reports `True`.
            Default `200`, matching the source's `MASK_SWITCH_AFTER`,
            though the semantics differ (see module docstring).
    """

    def __init__(self, n_bits: int, switch_after: int = 200) -> None:
        if n_bits <= 0:
            raise ValueError(f"n_bits must be positive, got {n_bits}")
        if switch_after <= 0:
            raise ValueError(f"switch_after must be positive, got {switch_after}")
        self.n_bits = n_bits
        self.switch_after = switch_after
        self._mask = 0
        self._samples_seen = 0
        self._consecutive_no_growth = 0

    @property
    def mask(self) -> int:
        """Accumulated live-bit mask: bit ``i`` is set iff some observed
        sample had a nonzero XOR at position ``i``, at least once."""
        return self._mask

    @property
    def samples_seen(self) -> int:
        """Total number of `observe()` calls made so far."""
        return self._samples_seen

    @property
    def is_converged(self) -> bool:
        """True iff the last `switch_after` consecutive `observe()` calls
        produced no mask growth. See module docstring for why this is
        stricter than -- and can toggle back to `False` unlike -- a
        one-way cumulative-count switch."""
        return self._consecutive_no_growth >= self.switch_after

    def observe(self, baseline: int, mutant: int) -> int:
        """Fold one `(baseline, mutant)` sample into the estimator.

        Returns the diff (``baseline ^ mutant``) this call revealed, in
        case a caller wants the raw per-sample signal (e.g. to attribute
        it to a specific input byte -- see the per-byte usage note in the
        module docstring) in addition to the accumulated state.

        Raises:
            ValueError: if `baseline` or `mutant` has any bit set at
                position `>= n_bits`.
        """
        full = (1 << self.n_bits) - 1
        if baseline & ~full:
            raise ValueError(f"baseline {baseline:#x} has bits outside the {self.n_bits}-bit field")
        if mutant & ~full:
            raise ValueError(f"mutant {mutant:#x} has bits outside the {self.n_bits}-bit field")

        diff = baseline ^ mutant
        self._samples_seen += 1
        new_mask = self._mask | diff
        if new_mask != self._mask:
            self._mask = new_mask
            self._consecutive_no_growth = 0
        else:
            self._consecutive_no_growth += 1
        return diff
