"""Field segmentation analyzer: owns the FieldHMM, serves spans (``--field-hmm``).

Lifecycle::

    activate ──► load saved model ──or──► fit on corpus
    admit ─────► every RETRAIN_EVERY admits: refit, rename, drop caches
    op/pos ────► segments(seed): Viterbi spans, cached per seed
                 donor(state):   a same-state span from a decoded seed

Spans are a pure function of (model, seed), so they are recomputed after a
refit or resume instead of being persisted per seed. State names come from
Weizz tags when ``--weizz-tags`` left any on the training seeds.
"""

from collections import deque
from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from fuzzer_tool.core.field_hmm import FieldHMM, FieldKind
from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.weizz_tags import TagFlags, load_tags_from_meta

# Admits between refits.
RETRAIN_EVERY = 64
# Most recent corpus entries a refit trains on.
TRAIN_SEEDS = 256
# Seeds whose spans stay cached (FIFO; Hard Rule 54).
SPAN_CACHE_MAX = 512
# Donor spans kept per state for splicing.
DONORS_MAX = 64
# Fewer spans than this: the seed shows no structure, op not offered.
MIN_SPANS = 2

# Weizz flag -> FieldKind, checked in this order (IS_LEN wins, as in weizz_tags).
_FLAG_KINDS = (
    (TagFlags.IS_LEN, FieldKind.LENGTH),
    (TagFlags.IS_CHECKSUM, FieldKind.CHECKSUM),
    (TagFlags.IS_MAGIC, FieldKind.MAGIC),
)

Segments = tuple[np.ndarray, np.ndarray, np.ndarray]
SeedMeta = dict[bytes, dict[str, Any]]


class FieldHmmAnalyzer:
    """Viterbi spans per seed from a periodically refit FieldHMM.

    Args:
        n_states: Hidden states of the model.
        rng: Shared RandPool (Hard Rule 16), used by propose().
    """

    def __init__(self, n_states: int = 8, rng: RandPool | Any = None):
        self._hmm = FieldHMM(n_states=n_states)
        self._rng = rng or get_default_rand_pool()
        self._admits = 0
        self._cache: dict[bytes, Segments] = {}
        self._donors: dict[int, deque[bytes]] = {}

    def kind(self, state: int) -> FieldKind:
        return self._hmm.kind(state)

    def cached(self) -> int:
        """Seeds with cached spans."""
        return len(self._cache)

    def fit_corpus(self, corpus: Sequence[bytes], seed_meta: SeedMeta) -> None:
        """Refit on the newest TRAIN_SEEDS entries, rename states, drop caches."""
        sample = list(corpus[-TRAIN_SEEDS:])
        self._hmm.fit(sample)
        self._hmm.name_states(_weizz_labels(sample, seed_meta))

        # State ids changed meaning: every cached span is stale.
        self._cache.clear()
        self._donors.clear()

    def on_admit(self, corpus: Sequence[bytes], seed_meta: SeedMeta) -> None:
        """Count an admitted seed; refit every RETRAIN_EVERY admits."""
        self._admits += 1
        if self._admits % RETRAIN_EVERY == 0:
            self.fit_corpus(corpus, seed_meta)

    def segments(self, data: bytes) -> Segments | None:
        """``(starts, ends, states)`` arrays of *data*'s spans; None when empty."""
        segs = self._cache.get(data)
        if segs is not None:
            return segs
        if not data:
            return None

        spans = self._hmm.spans(data)
        starts = np.array([s for s, _, _ in spans], np.int32)
        ends = np.array([e for _, e, _ in spans], np.int32)
        states = np.array([st for _, _, st in spans], np.uint8)
        segs = (starts, ends, states)
        self._remember(data, segs)
        return segs

    def has_spans(self, data: bytes) -> bool:
        """True when *data* splits into at least MIN_SPANS spans."""
        segs = self.segments(data)
        return segs is not None and segs[0].size >= MIN_SPANS

    def donor(self, state: int, rng: Any) -> bytes | None:
        """A cached span of *state* from any decoded seed, or None."""
        pool = self._donors.get(state)
        return rng.choice(pool) if pool else None

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """A span start inside *buf_len*: field boundaries are where formats branch."""
        segs = self.segments(data)
        if segs is None or segs[0].size < MIN_SPANS:
            return None

        starts: list[int] = segs[0][segs[0] < buf_len].tolist()
        return cast(int, self._rng.choice(starts)) if starts else None

    def to_state(self) -> dict[str, list[Any]]:
        return self._hmm.to_state()

    def load_state(self, state: dict[str, list[Any]]) -> bool:
        """Restore a saved model; False when it does not fit this K."""
        try:
            self._hmm.load_state(state)
        except ValueError:
            return False
        return True

    def _remember(self, data: bytes, segs: Segments) -> None:
        # FIFO caches keep memory bounded (Hard Rule 54).
        self._cache[data] = segs
        if len(self._cache) > SPAN_CACHE_MAX:
            del self._cache[next(iter(self._cache))]

        for start, end, state in zip(*(a.tolist() for a in segs), strict=True):
            pool = self._donors.setdefault(state, deque(maxlen=DONORS_MAX))
            pool.append(data[start:end])


def _weizz_labels(seeds: Sequence[bytes], seed_meta: SeedMeta) -> list[tuple[bytes, np.ndarray]]:
    # Per-byte FieldKind values from Weizz tags; seeds without tags skipped.
    out = []
    for seed in seeds:
        smap = load_tags_from_meta(seed_meta.get(seed) or {})
        if smap is None:
            continue

        flags = np.array([t.flags for t in smap.tags], np.int64)
        kinds = np.zeros(flags.size, np.uint8)
        for flag, kind in reversed(_FLAG_KINDS):
            kinds[(flags & int(flag)) != 0] = kind.value
        out.append((seed, kinds))
    return out
