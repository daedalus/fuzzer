"""SeedSFQScheduler: stochastic fair queueing (McKenney 1990, Linux sch_sfq).

Seeds are hashed by *flow* into buckets; buckets are served round robin and
seeds round robin inside a bucket. A flow is the seed's lineage parent
(``parent_key`` under ``--lineage``), else the seed itself, so one prolific
parent's children share one slot instead of crowding out the corpus::

    bucket 17  [f0 f1 ... f99]   <- 100 siblings: 1/2 of picks together
    bucket 902 [solo]            <- lone seed:    1/2 of picks

The hash salt is re-drawn every ``perturb_period`` picks (RandPool), so an
unlucky collision does not last. The bucket hash is crc32, not ``hash()``:
Python salts ``str`` hashes per process, which would move flows between
runs and break ``--seed`` reproducibility. One bucket is exactly
``seed_round_robin``. ``record`` only feeds the Elo match.
"""

from __future__ import annotations

import zlib
from collections import deque
from collections.abc import Callable, Hashable

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

SFQ_BUCKETS = 1024
#: Picks between hash-salt perturbations.
PERTURB_PERIOD = 1024
SALT_MAX = 2**32 - 1


def bucket_of(salt: int, flow: Hashable, buckets: int) -> int:
    """Process-stable bucket of *flow*: ``crc32(b"salt:flow") % buckets``."""
    raw = flow if isinstance(flow, bytes) else str(flow).encode()
    return zlib.crc32(f"{salt}:".encode() + raw) % buckets


def _self(key: str) -> Hashable:
    return key


class SeedSFQScheduler(ArmCounts):
    """Stochastic fair queueing for seed selection; flow per seed."""

    #: No informative priors: selection reads flows only.
    supports_priors = False

    def __init__(self, rng, buckets: int = SFQ_BUCKETS, perturb_period: int = PERTURB_PERIOD):
        if rng is None:
            raise ValueError("SeedSFQScheduler requires a RandPool (Hard Rule 16)")
        super().__init__()
        self._rng = rng
        self._buckets = max(1, int(buckets))
        self._period = max(1, int(perturb_period))
        self._salt = self._rng.randint(0, SALT_MAX)
        self._queues: dict[int, deque[str]] = {}
        self._order: list[int] = []
        self._cursor = 0
        self._picks = 0
        self._seen: list[str] | None = None

    def select_seed(
        self, seed_ids: list[str], flow_fn: Callable[[str], Hashable | None] = _self
    ) -> str:
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]

        self._picks += 1
        if self._picks % self._period == 0:
            self._salt = self._rng.randint(0, SALT_MAX)
            self._seen = None
        if self._seen is None or seed_ids != self._seen:
            self._sync(seed_ids, flow_fn)

        bucket = self._order[self._cursor % len(self._order)]
        self._cursor += 1
        queue = self._queues[bucket]
        seed = queue[0]
        queue.rotate(-1)
        return seed

    def _sync(self, seed_ids: list[str], flow_fn: Callable[[str], Hashable | None]) -> None:
        """Re-bucket the corpus; buckets ordered by first appearance."""
        queues: dict[int, deque[str]] = {}
        for seed_id in seed_ids:
            flow = flow_fn(seed_id)
            key = seed_id if flow is None else flow
            queues.setdefault(bucket_of(self._salt, key, self._buckets), deque()).append(seed_id)

        self._queues = queues
        self._order = list(queues)
        self._seen = list(seed_ids)
