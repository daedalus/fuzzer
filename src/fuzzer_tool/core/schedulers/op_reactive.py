"""ReactiveGradientScheduler: gradient bandit with forgetting and a spike guard.

Port of the "reactive optimizer" ideas onto ``GradientBanditScheduler``
(the static one). Two add-ons, both applied before the parent's update:

    w ← min(w, spike · EMA(w))    # spike guard: cut this step's reward
    H ← (1 − forget) · H          # AdamW decoupled decay, success-clocked

Decay runs on successes only, so forgetting follows information arrival,
not pull count. Per-pull decay must be slower than the yield rate to learn
rare arms and faster than it to recover from a dead one; no constant does
both. Measured on ``tests/support/bandit_env.py`` (forget=0.003):

    env             gradient      reactive
    stationary      0.938         0.938    (best-arm tail share, min, 40 seeds)
    DecayingBest    0.003         0.946    (revived-arm share, min, 20 seeds)
    rare (0.015)    collapses 2/12 collapses 1/12

Per-pull decay (1e-4..1e-3) failed DecayingBest or rare yields. Not ported:
Muon (normalized Nesterov steps erase per-arm magnitude: 20% of seeds
collapsed), Adam moments (no change), cooldown (campaigns have no horizon).
"""

from __future__ import annotations

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_gradient import GradientBanditScheduler

# EMA rate of the success-reward tracker the spike guard compares against.
_REWARD_EMA_DECAY = 0.99


class ReactiveGradientScheduler(GradientBanditScheduler):
    """Gradient bandit that forgets on success and clips reward spikes.

    Args:
        forget: Fraction of every preference dropped per success, in [0, 1).
            0 disables forgetting.
        spike: Reward cap as a multiple of the success-reward EMA. 0 disables.
        rng: Shared RandPool (Hard Rule 16).
        **kw: Forwarded to ``GradientBanditScheduler``.
    """

    supports_priors = False  # init_arm() takes no (alpha, beta) override.

    def __init__(
        self,
        forget: float = 0.003,
        spike: float = 3.0,
        rng: RandPool | None = None,
        **kw,
    ):
        if not (0.0 <= forget < 1.0):
            raise ValueError(f"forget must be in [0, 1), got {forget!r}")
        if spike < 0.0:
            raise ValueError(f"spike must be >= 0, got {spike!r}")

        super().__init__(rng=rng, **kw)
        self.forget = forget
        self.spike = spike
        self._reward_ema = 0.0
        self._spike_cuts = 0

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Clip and forget on an on-policy success, then the parent update."""
        if success and name == self._last_op and self._last_probs:
            weight = self._clip(weight)
            self._decay()
        super().record(name, success, weight)

    def _clip(self, weight: float) -> float:
        """Cut *weight* to spike · EMA; the EMA tracks the clipped value."""
        ema = self._reward_ema
        if not ema:
            self._reward_ema = weight
            return weight

        cap = self.spike * ema
        if self.spike and weight > cap:
            weight = cap
            self._spike_cuts += 1

        self._reward_ema = _REWARD_EMA_DECAY * ema + (1.0 - _REWARD_EMA_DECAY) * weight
        return weight

    def _decay(self) -> None:
        """Shrink every preference toward 0 (uniform policy)."""
        if not self.forget:
            return
        keep = 1.0 - self.forget
        prefs = self.preferences
        for op in prefs:
            prefs[op] *= keep

    def bandit_stats(self) -> dict:
        """Parent diagnostics plus spike-guard counters."""
        stats = super().bandit_stats()
        stats["reactive_spike_cuts"] = self._spike_cuts
        stats["reactive_reward_ema"] = self._reward_ema
        return stats
