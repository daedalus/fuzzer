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

    Probability sources are delegated to the caller.  When the caller
    cannot supply true selection probabilities the work collapses to
    L·log(L) (uniform over the trajectory itself) — that degeneracy is
    a call-site defect, not a property of this module.
    """

    def __init__(self, beta: float = 1.0, window: int = 1000) -> None:
        self.beta = beta
        self.window = window
        self._states: dict[str, list[float]] = {}
        self._last_work: float = 0.0
        self._last_state_key: str = ""
        self._last_outcome: str = ""

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
            return f"o_{hash(tuple(record.ops)):x}"
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
        self._append(state_key, work)
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

    def crooks_forward_reverse(self, state_a: str, state_b: str) -> dict:
        """Ratio of mean works between two state buffers.

        This is *not* Crooks' theorem: there is no reverse protocol, no
        matched-W density ratio, and no crossing at ΔF.  Retained only as a
        diagnostic of work-distribution shift between two arbitrary keys.
        """
        fwd = self._states.get(state_a, [])
        rev = self._states.get(state_b, [])
        if not fwd or not rev:
            return {"forward": len(fwd), "reverse": len(rev), "ratio": None}
        fwd_mean = sum(fwd) / len(fwd)
        rev_mean = sum(rev) / len(rev)
        ratio = None if fwd_mean <= _EPS or rev_mean <= _EPS else rev_mean / fwd_mean
        return {
            "forward": len(fwd),
            "reverse": len(rev),
            "forward_mean_work": fwd_mean,
            "reverse_mean_work": rev_mean,
            "ratio": ratio,
        }

    def stats(self, state_key: str) -> dict:
        buf = self._states.get(state_key, [])
        if not buf:
            return {"samples": 0}
        est = self.jarzynski_estimator(state_key)
        return {
            "samples": len(buf),
            "mean_work": sum(buf) / len(buf),
            "last_work": buf[-1],
            # Historical key name; value is Rényi entropy H_{1+β}.
            "jarzynski_delta_f": est,
            "renyi_entropy": est,
        }

    def snapshot(self) -> dict:
        return {
            "beta": self.beta,
            "window": self.window,
            "last_work": self._last_work,
            "last_state_key": self._last_state_key,
            "last_outcome": self._last_outcome,
            "states": {k: v[:] for k, v in self._states.items()},
        }

    def restore(self, data: dict) -> None:
        self.beta = float(data.get("beta", self.beta))
        self.window = int(data.get("window", self.window))
        self._last_work = float(data.get("last_work", 0.0))
        self._last_state_key = str(data.get("last_state_key", ""))
        self._last_outcome = str(data.get("last_outcome", ""))
        self._states = {str(k): list(v) for k, v in data.get("states", {}).items()}
