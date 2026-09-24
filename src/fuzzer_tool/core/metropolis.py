"""Metropolis energy for corpus admission of non-improving mutants.

``--metropolis`` used ``exp(-1/T)``: a constant ΔE, so acceptance was a clock,
blind to the candidate (order_theory P0-3). Here, with ``M`` the mutant's
edges and ``P`` its parent's::

    ΔE = |P| / |M ∪ P|        p = min(1, exp(-ΔE / T))

ΔE is the share of the combined path the parent already explains:

    M = P or M ⊂ P (early exit)         ΔE = 1      p = exp(-1/T)  (legacy floor)
    5-edge error path, 2 new, |P|=100   ΔE ≈ 0.98   ≈ floor
    half of a 100-edge path rerouted    ΔE ≈ 0.67

``1 - Jaccard`` was rejected (identical mutant -> p=1), as were edge rarity
(no incumbent anchor) and hit counts (SHM-only). Full reasoning: the P0-3
section of ``docs/handover/handover_order_theory_ecc_ising_2026-09-20.md``.
"""

from __future__ import annotations

import math

__all__ = ["MIN_TEMPERATURE", "WORST_ENERGY", "accept_prob", "path_energy"]

# Floor on T, kept from the legacy expression (avoids exp(-x/0)).
MIN_TEMPERATURE = 0.01

# ΔE when the mutant adds nothing to the parent's path, or P is unknown.
WORST_ENERGY = 1.0


def path_energy(mutant: set[int], parent: set[int]) -> float:
    """``|P| / |M ∪ P|`` in ``[0, 1]``; ``WORST_ENERGY`` when ``P`` is empty.

    An untracked parent carries no information, so it gets the legacy rate
    rather than ``ΔE = 0`` (which would admit every mutant).
    """
    if not parent:
        return WORST_ENERGY

    union = len(mutant) + len(parent) - len(mutant & parent)
    return len(parent) / union


def accept_prob(delta_e: float, temperature: float) -> float:
    """``min(1, exp(-ΔE / max(T, MIN_TEMPERATURE)))``."""
    return min(1.0, math.exp(-delta_e / max(temperature, MIN_TEMPERATURE)))
