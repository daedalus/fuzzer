from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

_EPS = 1e-12


@dataclass(frozen=True)
class TrajectoryRecord:
    """One observed mutation trajectory and its terminal coverage state."""

    ops: tuple[str, ...]
    probs: tuple[float, ...]
    outcome: str
    state_key: str = ""
    hit_edges: frozenset[int] = field(default_factory=frozenset)
    new_edges: int = 0
    probs_are_true: bool = False
    """Whether ``probs`` are genuine normalised selection probabilities.

    The Rényi identity in :class:`WorkFunctional` holds only when the
    trajectory was drawn from the very distribution recorded in ``probs``.
    A caller that cannot supply that distribution must leave this False;
    the work value is then not a path log-probability and is not pooled.
    Default False so a caller has to assert the property, not forget to
    deny it.
    """


class WorkFunctional:
    """Maps a mutation trajectory to a scalar work value that is the Rényi
    entropy of the operator-path distribution (order 1+β).

    With ``W(τ) = Σ_i -log(max(p_i, ε))`` and trajectories drawn from the
    same ``p_i``, the Jarzynski-style estimator

        -log(E[e^{-β W}]) / β

    is exactly the Rényi entropy H_{1+β} of the trajectory distribution.
    Default β=1 yields collision entropy; β→0 recovers Shannon.  This is
    not a free-energy difference and there is no fluctuation–dissipation
    relation to exploit (β here is an entropy order, not a temperature).

    The identity holds only if the trajectory was drawn from the same
    ``p_i`` that are recorded.  Callers assert that with
    ``TrajectoryRecord.probs_are_true``; records without it are counted
    but never pooled, so the estimator returns None rather than a number
    that looks like an entropy and is not one.  Deterministic (argmax)
    schedulers have no selection distribution at all, so most of this
    tree's schedulers cannot supply one — see the call site in
    ``services/fuzzer.py::_record_fluctuation_observation``.
    """

    def __init__(self, beta: float = 1.0, window: int = 1000) -> None:
        self.beta = beta
        self.window = window
        self._states: dict[str, list[float]] = {}
        self._last_work: float = 0.0
        self._last_state_key: str = ""
        self._last_outcome: str = ""
        self._unpooled: int = 0

    @staticmethod
    def state_key(record: TrajectoryRecord) -> str:
        if record.hit_edges:
            try:
                import xxhash
            except ImportError:
                xxhash = None  # type: ignore[assignment]
            data = b"".join(
                (int(e) & 0xFFFFFFFF).to_bytes(4, "little", signed=False)
                for e in sorted(record.hit_edges)
            )
            if xxhash is not None:
                return f"e_{xxhash.xxh3_64_intdigest(data):x}"
            return f"e_{hashlib.sha256(data).hexdigest()[:16]}"
        if record.ops:
            # Not the builtin hash(): PYTHONHASHSEED salts str hashing per
            # process, so the same trajectory keyed differently on every run
            # and any state restored from disk was orphaned under a key this
            # process can no longer produce. Same defect class as the LSH
            # banding that was removed from crash clustering.
            data = "\x00".join(record.ops).encode("utf-8", "surrogatepass")
            try:
                import xxhash
            except ImportError:
                return f"o_{hashlib.sha256(data).hexdigest()[:16]}"
            return f"o_{xxhash.xxh3_64_intdigest(data):016x}"
        return "_"

    def _append(self, state_key: str, work: float) -> None:
        buf = self._states.setdefault(state_key, [])
        buf.append(work)
        if len(buf) > self.window:
            del buf[: len(buf) - self.window]

    @staticmethod
    def _step_work(prob: float) -> float:
        if prob <= _EPS:
            return -math.log(_EPS)
        return -math.log(prob)

    def observe(self, record: TrajectoryRecord) -> float:
        if not record.ops:
            self._last_work = 0.0
            self._last_state_key = record.state_key or self.state_key(record)
            self._last_outcome = record.outcome
            return 0.0
        if not record.probs:
            probs = tuple(1.0 / max(len(record.ops), 1) for _ in record.ops)
        else:
            probs = tuple(max(p, _EPS) for p in record.probs)
        work = sum(self._step_work(p) for p in probs)
        state_key = record.state_key or self.state_key(record)
        if record.probs_are_true:
            self._append(state_key, work)
        else:
            # Counted, not pooled. Pooling would make jarzynski_estimator
            # return a number with the shape of an entropy and none of its
            # meaning -- which is what the uniform fallback above produces:
            # W = L*log(L), a function of trajectory length alone.
            self._unpooled += 1
        self._last_work = work
        self._last_state_key = state_key
        self._last_outcome = record.outcome
        return work

    def trajectory_work(self) -> float:
        return self._last_work

    def _stable_exp_mean(self, values: Sequence[float]) -> float:
        if not values:
            return 0.0
        shifted = [-self.beta * w for w in values]
        m = max(shifted)
        exps = [math.exp(x - m) for x in shifted]
        return sum(exps) / len(exps) * math.exp(m)

    def jarzynski_estimator(self, state_key: str) -> float | None:
        buf = self._states.get(state_key, [])
        if not buf:
            return None
        mean_exp = self._stable_exp_mean(buf)
        if mean_exp <= 0.0:
            return None
        return -math.log(mean_exp) / max(self.beta, _EPS)

    def stats(self, state_key: str) -> dict:
        buf = self._states.get(state_key, [])
        if not buf:
            return {"samples": 0, "unpooled": self._unpooled}
        est = self.jarzynski_estimator(state_key)
        return {
            "samples": len(buf),
            "mean_work": sum(buf) / len(buf),
            "last_work": buf[-1],
            # Historical key name; value is Rényi entropy H_{1+β}.
            "jarzynski_delta_f": est,
            "renyi_entropy": est,
            "unpooled": self._unpooled,
        }

    def snapshot(self) -> dict:
        return {
            "beta": self.beta,
            "window": self.window,
            "last_work": self._last_work,
            "last_state_key": self._last_state_key,
            "last_outcome": self._last_outcome,
            "unpooled": self._unpooled,
            "states": {k: v[:] for k, v in self._states.items()},
        }

    def restore(self, data: dict) -> None:
        self.beta = float(data.get("beta", self.beta))
        self.window = int(data.get("window", self.window))
        self._last_work = float(data.get("last_work", 0.0))
        self._last_state_key = str(data.get("last_state_key", ""))
        self._last_outcome = str(data.get("last_outcome", ""))
        self._unpooled = int(data.get("unpooled", 0))
        self._states = {str(k): list(v) for k, v in data.get("states", {}).items()}
