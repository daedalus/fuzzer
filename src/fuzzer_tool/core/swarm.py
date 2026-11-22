"""Swarm testing over mutation operators (Groce et al., ISSTA'12; Csmith).

Csmith gets its bug yield from running each generated program with a random
subset of language features switched off. A feature that is always on can
mask a bug that needs another feature absent (``-O2`` folds the construct
away, an op overwrites the byte another op just set). The fuzzer has the same
shape: every round draws from all available operators, so a crash that needs
"op A without op B" is reached only when B happens not to be drawn.

    ops = [flip, arith, splice, dict_ins, havoc]     (from build_ops)
    mask   1    0      1       0         1           (one coin per op)
    ops' = [flip,        splice,         havoc]      (select_op sees these)

One mask per round = one test case, as in the paper. Schedulers still pick
within the subset, so bandit credit lands on ops that actually ran.
"""

from __future__ import annotations

from fuzzer_tool.core.rand_pool import RandPool

# Per-op inclusion probability. 0.5 is the paper's default: it maximises the
# number of distinct subsets, and every pair of ops is jointly absent 1/4 of
# rounds.
DEFAULT_SWARM_PROB = 0.5

# Coins are drawn as bytes (one vectorised randint_list call), so the
# probability is quantised to 1/256.
_COIN_RANGE = 256

# Fewer candidates than this are returned untouched: a 1-op list has no
# subset worth drawing, and 2 ops would idle one half the rounds.
MIN_SWARM_OPS = 3


def check_prob(prob: float) -> float:
    """Return *prob* if it lies in (0, 1); else ValueError (0 or 1 is not a swarm)."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"swarm_prob must be in (0, 1), got {prob}")
    return float(prob)


def swarm_filter(ops: list[str], rng: RandPool, prob: float = DEFAULT_SWARM_PROB) -> list[str]:
    """Random subset of *ops*, each kept with probability *prob*; never empty.

    Order is preserved. When every coin fails, one op chosen by *rng* is
    kept, so ``select_op`` always has a candidate.
    """
    if len(ops) < MIN_SWARM_OPS:
        return ops

    threshold = int(prob * _COIN_RANGE)
    coins = rng.randint_list(0, _COIN_RANGE - 1, len(ops))
    kept = [op for op, coin in zip(ops, coins, strict=True) if coin < threshold]
    if kept:
        return kept

    return [rng.choice(ops)]
