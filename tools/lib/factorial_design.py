"""Two-level orthogonal screening designs for hyperparameter sweeps.

Screening ``k`` open knobs (PLL ``kp``/``ki``, lock thresholds,
``explore_floor``) with a full grid costs ``2**k`` runs. A Plackett-Burman
design (a Hadamard matrix minus its all-ones column) estimates every main
effect in the next multiple of 4 above ``k``: 12 runs for 11 knobs, not 2048.

    ranked = screen({"kp": (0.1, 0.9), "ki": (0.01, 0.5)}, run_cfg)
    # run_cfg({"kp": 0.9, "ki": 0.01}) -> float; ranked[0] is the knob that
    # moves the response most.

Main effects are aliased with two-factor interactions; ``fold=True`` doubles
the runs and cancels that aliasing (resolution IV). Screening only: effects
say which knobs matter, not their optimum or their interactions.

Hadamard orders built: powers of 2 (Sylvester), ``q+1`` for prime
``q = 3 mod 4`` (Paley I: 12, 20, 24, 32, 44, 48, 60, 64), and doublings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

Row = tuple[int, ...]

_MAX_ORDER = 64
_PALEY_MOD = 4
_PALEY_RESIDUE = 3


def _is_prime(q: int) -> bool:
    return q > 1 and all(q % d for d in range(2, int(q**0.5) + 1))


def _paley(q: int) -> list[list[int]]:
    """Paley I Hadamard matrix of order q+1 (q prime, q = 3 mod 4)."""
    squares = {(x * x) % q for x in range(1, q)}

    def chi(a: int) -> int:
        a %= q
        if a == 0:
            return 0
        return 1 if a in squares else -1

    # S = [[0, 1..1], [-1, Q]] is skew; H = S + I.
    h = [[1] + [1] * q]
    for i in range(q):
        h.append([-1] + [chi(j - i) + (1 if i == j else 0) for j in range(q)])
    return h


def _sylvester_double(h: list[list[int]]) -> list[list[int]]:
    return [r + r for r in h] + [r + [-v for v in r] for r in h]


def _hadamard(n: int) -> list[list[int]] | None:
    """Hadamard matrix of order *n*, or None if none is constructed here."""
    if n == 1:
        return [[1]]
    if _is_prime(n - 1) and (n - 1) % _PALEY_MOD == _PALEY_RESIDUE:
        return _paley(n - 1)
    if n % 2:
        return None
    half = _hadamard(n // 2)
    return _sylvester_double(half) if half is not None else None


def _order_for(k: int) -> int:
    """Smallest constructible order n with n >= k+1, n a multiple of 4."""
    n = 4
    while n <= _MAX_ORDER:
        if n >= k + 1 and _hadamard(n) is not None:
            return n
        n += 4
    raise ValueError(f"no design built for {k} factors (max {_MAX_ORDER - 1})")


def design_matrix(k: int) -> list[Row]:
    """Rows of a balanced orthogonal ``+/-1`` design with *k* columns."""
    if k < 1:
        raise ValueError(f"need at least 1 factor, got {k}")
    n = _order_for(k)
    h = _hadamard(n)
    assert h is not None

    # Flip rows so column 0 is all +1, then drop it: the rest are balanced.
    return [tuple(r[j] * r[0] for j in range(1, k + 1)) for r in h]


def fold_over(rows: Sequence[Row]) -> list[Row]:
    """Append the sign-reversed runs: cancels two-factor aliasing of main effects."""
    return [*rows, *(tuple(-v for v in r) for r in rows)]


def main_effects(rows: Sequence[Row], y: Sequence[float]) -> list[float]:
    """Mean response at +1 minus mean at -1, per column."""
    if not rows or len(rows) != len(y):
        raise ValueError(f"need equal, non-zero rows and responses ({len(rows)} vs {len(y)})")

    scale = 2.0 / len(rows)
    return [
        scale * sum(r[j] * v for r, v in zip(rows, y, strict=True)) for j in range(len(rows[0]))
    ]


@dataclass(frozen=True)
class Effect:
    name: str
    effect: float  # high-level mean minus low-level mean, in response units


def screen(
    factors: Mapping[str, tuple[float, float]],
    run: Callable[[dict[str, float]], float],
    *,
    fold: bool = False,
    replicates: int = 1,
) -> list[Effect]:
    """Run *run* over the design; return factors ranked by ``|effect|``.

    *factors* maps name -> ``(low, high)``. Each config is run *replicates*
    times and averaged.
    """
    if not factors:
        raise ValueError("no factors to screen")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    for name, (lo, hi) in factors.items():
        if lo == hi:
            raise ValueError(f"factor {name!r} has identical levels")

    names = list(factors)
    rows = design_matrix(len(names))
    if fold:
        rows = fold_over(rows)

    ys = []
    for r in rows:
        cfg = {n: factors[n][1 if v > 0 else 0] for n, v in zip(names, r, strict=True)}
        ys.append(sum(run(cfg) for _ in range(replicates)) / replicates)

    effects = [Effect(n, e) for n, e in zip(names, main_effects(rows, ys), strict=True)]
    return sorted(effects, key=lambda e: -abs(e.effect))
