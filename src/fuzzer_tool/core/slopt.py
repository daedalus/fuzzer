"""SLOPT batch-exponent bandit (Koike et al., ACSAC 2022, arXiv:2211.03285).

SLOPT's mutation scheme (Algorithm 2) picks ONE operator per round and
applies it ``2**t`` times at random positions. The batch exponent ``t`` is
not a constant: the paper found the batch size that finds new paths depends
on the seed's size and on the operator, and in ways that differ between
targets, so it is learned online. Per section 4.2:

* one bandit instance per (seed-size group, operator);
* size groups ``[0, 1e2), [1e2, 1e3), [1e3, 1e4), [1e4, 1e5), [1e5, inf)``;
* seven arms per instance, arm ``t`` meaning batch size ``2**t``
  (``t`` in 1..7; AFL++'s batch exponent never exceeds 7);
* the pulled arm is rewarded when the round's input finds a new path.

The bandit is Thompson sampling. The paper's comparison (Table 2) ranked
TS, discounted TS and ADS-TS together at the top and found no evidence of
non-stationarity worth the discount; the adversarial algorithms (EXP3-IX,
EXP3++) ranked last.

Rewards in [0, 1] use the fractional-Bernoulli update: ``alpha += w``,
``beta += 1 - w``. A round that found nothing adds 1 to beta.
"""

from __future__ import annotations

import bisect

from fuzzer_tool.core.rand_pool import RandPool

#: Upper edges of the seed-size groups (bytes); len >= 1e5 is the last group.
SIZE_GROUP_EDGES = (100, 1_000, 10_000, 100_000)

#: Batch exponents; arm t applies the operator 2**t times.
EXPONENTS = tuple(range(1, 8))


def size_group(seed_len: int) -> int:
    """Index of the paper's size group for a seed of ``seed_len`` bytes."""
    return bisect.bisect_right(SIZE_GROUP_EDGES, seed_len)


def _clamp_reward(success: bool, weight: float) -> float:
    if not success:
        return 0.0
    return min(1.0, max(0.0, float(weight)))  # NaN -> 0.0


class SloptBatchBandit:
    """Thompson sampling over batch exponents, one instance per (group, op).

    Instances are created on first use: ~200 operators x 5 groups would be
    1,000 instances up front, of which a campaign touches a fraction.

    Args:
        rng: The fuzzer's ``RandPool`` (Hard Rule 16), so ``--seed``
            reproduces every exponent drawn.
    """

    def __init__(self, rng: RandPool | None = None) -> None:
        self._rng = rng if rng is not None else RandPool()
        # (group, op) -> [alpha per arm], [beta per arm]
        self._alpha: dict[tuple[int, str], list[float]] = {}
        self._beta: dict[tuple[int, str], list[float]] = {}
        self._pulls = 0

    def _instance(self, key: tuple[int, str]) -> tuple[list[float], list[float]]:
        alpha = self._alpha.get(key)
        if alpha is None:
            alpha = self._alpha[key] = [1.0] * len(EXPONENTS)
            self._beta[key] = [1.0] * len(EXPONENTS)
        return alpha, self._beta[key]

    def choose(self, op: str, seed_len: int) -> int:
        """Draw a batch exponent for applying ``op`` to a seed of this size."""
        alpha, beta = self._instance((size_group(seed_len), op))
        betav = self._rng.betavariate
        draws = [betav(a, b) for a, b in zip(alpha, beta, strict=True)]
        return EXPONENTS[draws.index(max(draws))]

    def record(
        self, op: str, seed_len: int, exponent: int, success: bool, weight: float = 1.0
    ) -> None:
        """Credit arm ``exponent`` of the (group, op) instance with one round."""
        arm = EXPONENTS.index(exponent)
        alpha, beta = self._instance((size_group(seed_len), op))
        reward = _clamp_reward(success, weight)
        alpha[arm] += reward
        beta[arm] += 1.0 - reward
        self._pulls += 1

    def posterior_means(self, op: str, seed_len: int) -> list[float]:
        """Posterior mean reward per exponent (diagnostics and tests)."""
        alpha, beta = self._instance((size_group(seed_len), op))
        return [a / (a + b) for a, b in zip(alpha, beta, strict=True)]

    def stats(self) -> dict:
        """Pull count, instances in use, and each instance's leading exponent."""
        best = {}
        for (group, op), alpha in self._alpha.items():
            beta = self._beta[(group, op)]
            means = [a / (a + b) for a, b in zip(alpha, beta, strict=True)]
            low = SIZE_GROUP_EDGES[group - 1] if group else 0
            best[f"{op}@{low}+"] = EXPONENTS[means.index(max(means))]
        return {
            "slopt_pulls": self._pulls,
            "slopt_instances": len(self._alpha),
            "slopt_best_exponent": best,
        }
