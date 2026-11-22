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

import zlib
from collections import OrderedDict
from hashlib import blake2b

from fuzzer_tool.core.rand_pool import RandPool

# --swarm-scope values: one mask per round (test case) or one per parent seed.
SCOPE_ROUND = "round"
SCOPE_SEED = "seed"
SWARM_SCOPES = (SCOPE_ROUND, SCOPE_SEED)

# Rounds a seed keeps one mask before it is redrawn. A seed that stays
# productive therefore cycles through several operator configurations, as
# Csmith runs one configuration per generated program.
DEFAULT_SEED_ROUNDS = 64

# Seeds whose mask is remembered (LRU); older seeds restart at epoch 0.
MASK_CACHE_SIZE = 1024

_SALT_BYTES = 8

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


class SeedMasks:
    """Per-seed swarm masks: one operator subset held for a seed's rounds.

    A coin is a pure function of (run salt, seed key, epoch, op name), so the
    mask is stable when ``build_ops`` returns a different list for the same
    seed, and is reproducible for a given RNG seed. Only ``(rounds, epoch,
    coins drawn)`` per seed is stored, in an LRU of ``MASK_CACHE_SIZE``.
    """

    def __init__(
        self,
        rng: RandPool,
        prob: float = DEFAULT_SWARM_PROB,
        seed_rounds: int = DEFAULT_SEED_ROUNDS,
    ) -> None:
        if seed_rounds < 1:
            raise ValueError(f"swarm seed_rounds must be >= 1, got {seed_rounds}")
        self._threshold = int(check_prob(prob) * _COIN_RANGE)
        self._seed_rounds = seed_rounds
        self._salt = bytes(rng.randint_list(0, _COIN_RANGE - 1, _SALT_BYTES))
        # key -> [rounds used in this epoch, epoch, {op: coin}]
        self._state: OrderedDict[int, list] = OrderedDict()

    def _coin(self, key: int, epoch: int, op: str) -> int:
        h = blake2b(
            op.encode(),
            digest_size=1,
            key=self._salt,
            salt=key.to_bytes(4, "little") + epoch.to_bytes(4, "little"),
        )
        return h.digest()[0]

    def filter(self, ops: list[str], data: bytes) -> list[str]:
        """Subset of *ops* for the seed *data*; never empty, order kept."""
        if len(ops) < MIN_SWARM_OPS:
            return ops

        key = zlib.crc32(data)
        entry = self._state.pop(key, None)
        if entry is None:
            entry = [0, 0, {}]
        elif entry[0] >= self._seed_rounds:
            entry = [0, entry[1] + 1, {}]
        self._state[key] = entry
        if len(self._state) > MASK_CACHE_SIZE:
            self._state.popitem(last=False)

        entry[0] += 1
        _, epoch, coins = entry
        for op in ops:
            if op not in coins:
                coins[op] = self._coin(key, epoch, op)

        kept = [op for op in ops if coins[op] < self._threshold]
        if kept:
            return kept

        # No coin passed: keep the lowest-coin op (stable for this epoch).
        return [min(ops, key=coins.__getitem__)]
