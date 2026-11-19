"""Online categorical naive Bayes gate: skip mutants unlikely to find coverage.

Estimates P(new coverage | op, position bucket, seed-size bucket) from
per-exec outcomes, Laplace-smoothed (alpha = 1)::

    P(y)       = (N_y + 1) / (N + 2)
    P(x_i | y) = (c[y, i, x_i] + 1) / (N_y + K_i)
    P(1 | x)   = P(1) prod P(x_i|1) / sum_y P(y) prod P(x_i|y)

Gate (``--exec-gate nb``)::

    warmup? ──yes──► run
       │no
    P(1|x) >= lift * base rate? ──yes──► run
       │no
    rng < eps? ──yes──► run (explore: the model keeps learning, no op starves)
       │no
    skip (caller re-rolls the mutant)

Memory is fixed: one int64 row per class over the concatenated feature
values. Counts halve when N reaches ``cap`` so old evidence decays.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any

import numpy as np

POS_BUCKETS = 8  # buffer eighths; +1 slot for delocalised ops
SIZE_BUCKETS = 24  # log2 size classes, clamped: >= 8 MiB share the last
OP_SLOTS = 256  # op-name ids; overflow shares the last slot
CARDS = (OP_SLOTS, POS_BUCKETS + 1, SIZE_BUCKETS)

DEFAULT_LIFT = 0.25
DEFAULT_EPS = 0.05
_WARMUP = 1024
_CAP = 1 << 24


class GateMode(Enum):
    """Pre-exec mutant gate (``--exec-gate``)."""

    OFF = "off"
    NB = "nb"


class Outcome(IntEnum):
    """Exec outcome; the value is the class row."""

    BORING = 0
    NOVEL = 1


def pos_bucket(offset: int | None, size: int) -> int:
    """Eighth of the buffer *offset* falls in; ``POS_BUCKETS`` for None."""
    if offset is None:
        return POS_BUCKETS
    if size <= 0:
        return 0
    return min(offset * POS_BUCKETS // size, POS_BUCKETS - 1)


def size_bucket(size: int) -> int:
    """log2 size class: 0 -> 0, 1 -> 1, 2..3 -> 2, ..."""
    return min(size.bit_length(), SIZE_BUCKETS - 1)


class OpIds:
    """Bounded op-name -> slot map; names past ``slots - 1`` share the last."""

    def __init__(self, slots: int = OP_SLOTS) -> None:
        self._ids: dict[str, int] = {}
        self._overflow = slots - 1

    def __len__(self) -> int:
        return len(self._ids)

    def names(self) -> list[str]:
        """Names in slot order: ``OpIds.from_names`` restores the same slots."""
        return list(self._ids)

    @classmethod
    def from_names(cls, names: list[str], slots: int = OP_SLOTS) -> OpIds:
        ids = cls(slots)
        for name in names:
            ids.of(name)
        return ids

    def of(self, name: str) -> int:
        slot = self._ids.get(name)
        if slot is not None:
            return slot
        if len(self._ids) >= self._overflow:
            return self._overflow
        slot = self._ids[name] = len(self._ids)
        return slot


def _check(lift: float, eps: float, warmup: int, cap: int) -> None:
    if lift < 0 or not 0 <= eps <= 1 or warmup < 0 or cap < 2:
        raise ValueError(f"bad gate args: lift={lift} eps={eps} warmup={warmup} cap={cap}")


class NBGate:
    """Categorical naive Bayes over fixed-cardinality features.

    Feature values are not range-checked on the hot path: the encoders
    above clamp them, and a value >= its cardinality would alias the next
    feature's counts.

    Args:
        cards: Cardinality of each feature.
        rng: ``RandPool`` (or a scripted stand-in) for the eps draw.
        lift: Skip below ``lift`` times the base novelty rate.
        eps: Floor probability of running a mutant the model would skip.
        warmup: Observations before the gate may skip anything.
        cap: N at which every count halves.
    """

    def __init__(
        self,
        cards: tuple[int, ...],
        rng: Any,
        lift: float = DEFAULT_LIFT,
        eps: float = DEFAULT_EPS,
        warmup: int = _WARMUP,
        cap: int = _CAP,
    ) -> None:
        if not cards or min(cards) < 1:
            raise ValueError(f"need >= 1 feature of cardinality >= 1, got {cards}")
        _check(lift, eps, warmup, cap)
        self._rng = rng
        self.cards = tuple(cards)
        self.lift = lift
        self.eps = eps
        self.warmup = warmup
        self.cap = cap
        self.skipped = 0
        self.explored = 0
        self._offs = np.cumsum((0,) + self.cards[:-1]).astype(np.int64)
        self._card_arr = np.asarray(self.cards, dtype=np.float64)
        self._counts = np.zeros((2, sum(self.cards)), dtype=np.int64)
        self._classes = np.zeros(2, dtype=np.int64)
        self._bind_views()

    def _bind_views(self) -> None:
        # Scalar hot path over memoryviews of the numpy counts: posterior
        # 0.86 us vs 10.6 us fancy-indexed, record 0.49 us vs 3.9 us.
        self._feat = tuple(zip(self._offs.tolist(), self.cards, strict=True))
        self._rows = (memoryview(self._counts[0]), memoryview(self._counts[1]))
        self._cls = memoryview(self._classes)

    @property
    def total(self) -> int:
        return int(self._classes.sum())

    @property
    def nbytes(self) -> int:
        return int(self._counts.nbytes + self._classes.nbytes)

    def posterior(self, x: tuple[int, ...]) -> float:
        """P(NOVEL | x), Laplace-smoothed."""
        m0, m1 = self._rows
        n0, n1 = self._cls
        # Odds times one likelihood ratio per feature; float64 holds the
        # product for any realistic feature count.
        odds = (n1 + 1) / (n0 + 1)
        for (off, card), v in zip(self._feat, x, strict=True):
            j = off + v
            odds *= (m1[j] + 1) * (n0 + card) / ((m0[j] + 1) * (n1 + card))
        return odds / (1.0 + odds)

    def posterior_batch(self, xs: np.ndarray) -> np.ndarray:
        """``posterior`` over rows of *xs* (n, features), in log space."""
        idx = self._offs + np.asarray(xs, dtype=np.int64)
        c0 = self._counts[0, idx] + 1
        c1 = self._counts[1, idx] + 1
        n0, n1 = self._classes
        log_lr = np.log(c1 * (n0 + self._card_arr)) - np.log(c0 * (n1 + self._card_arr))
        log_odds = np.log((n1 + 1) / (n0 + 1)) + log_lr.sum(axis=1)
        return 1.0 / (1.0 + np.exp(-log_odds))

    def retune(self, lift: float, eps: float) -> None:
        """New thresholds over the learned counts (flags over a resumed state)."""
        _check(lift, eps, self.warmup, self.cap)
        self.lift = lift
        self.eps = eps

    def _base(self) -> float:
        n0, n1 = self._cls
        return (n1 + 1) / (n0 + n1 + 2)

    def admit(self, x: tuple[int, ...]) -> bool:
        """True to run the mutant with features *x*; False to skip it."""
        n0, n1 = self._cls
        if n0 + n1 < self.warmup:
            return True
        if self.posterior(x) >= self.lift * self._base():
            return True
        if self._rng.random() < self.eps:
            self.explored += 1
            return True
        self.skipped += 1
        return False

    def record(self, x: tuple[int, ...], outcome: Outcome) -> None:
        """Learn one executed mutant's outcome."""
        row = self._rows[outcome]
        for (off, _), v in zip(self._feat, x, strict=True):
            row[off + v] += 1
        cls = self._cls
        cls[outcome] += 1
        if cls[0] + cls[1] < self.cap:
            return
        # In place: the memoryviews stay bound.
        self._counts >>= 1
        self._classes >>= 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": self.cards,
            "lift": self.lift,
            "eps": self.eps,
            "warmup": self.warmup,
            "cap": self.cap,
            # Bytes, not ndarray: see CountMin.to_dict.
            "counts": self._counts.tobytes(),
            "classes": self._classes.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], rng: Any) -> NBGate:
        gate = cls(
            tuple(data["cards"]),
            rng,
            lift=data["lift"],
            eps=data["eps"],
            warmup=data["warmup"],
            cap=data["cap"],
        )
        raw = data["counts"]
        classes = data["classes"]
        if len(raw) != gate._counts.nbytes or len(classes) != len(gate._classes):
            raise ValueError(f"nb-gate state does not fit cards {gate.cards}")
        # In place: the hot-path memoryviews stay bound.
        gate._counts[:] = np.frombuffer(raw, dtype=np.int64).reshape(gate._counts.shape)
        gate._classes[:] = classes
        return gate
