"""PRNG state learner: recovers a weak xor-combined-LFSR generator's internal
state from per-execution cmplog observations, and exposes predicted future
draws to the fuzzer's mutation operators.

Attaches to the fuzzer instance as ``f.prng_state_learner`` (see
``analyzer_registry.py``), the same wiring shape as ``f.checksum_learner``.

Targets CWE-338 (use of a predictable/weak PRNG). A fuzz target that seeds a
``taus88``-family generator (Boost's taus88, or a hand-rolled xor-combine /
combined-LFSR variant -- see ``core/prng_state_recovery.py`` for the family
this covers) for nonces, session tokens, or sequence numbers leaks its full
internal state the moment 3 consecutive draws are observed. Once recovered,
every future draw is known in advance, which lets a mutation operator write
the *actual* next value into the input instead of guessing -- turning an
otherwise-unreachable "does this match the token I generated" check into a
one-shot pass.

Signal source
-------------
Scans ``f._cmplog.pairs`` -- the same per-execution comparison log
``ChecksumLearner.extract_cmplog_pairs`` reads -- for 4-byte operands that do
NOT occur anywhere in the input buffer. An operand echoed from the input is
attacker-controlled data, not the target's internal state; one that appears
nowhere in the input but shows up in a comparison is the target's own value
leaking out through that comparison. Several such "magic" operands observed
within a single execution, taken in cmplog encounter order, are treated as a
candidate run of consecutive PRNG outputs from one generator instance.

This is deliberately the mirror image of
``ChecksumLearner.extract_cmplog_pairs``: that one wants the operand found
IN the input (the checksummed "data"); this one wants the operand found
NOWHERE in the input (pure, unexplained internal state).

Recovery cost and false-positive control
-----------------------------------------
``recover_taus88_state`` is a fixed-cost 96x96 GF(2) elimination per attempt
-- cheap, but not free enough to run unconditionally on every execution, so
candidates are capped at ``_MAX_SAMPLES`` per attempt (mirrors
``CHECKSUM_PAIRS_MAX``'s rationale exactly: bound the cost of an attempt,
not just how often one is made). A recovered state is only kept once
``verify_recovery`` confirms it reproduces every sample it was derived
from, including ones beyond the 3 needed to pin the state -- an accidental
96-bit fit to non-taus88 "random" data is not impossible, and extra
confirming samples are what tells it apart from a real one. When later
samples disagree with an already-cached state (a false-positive site, or a
different generator instance), the cache is dropped rather than kept stale.
"""

from __future__ import annotations

from typing import Any

from fuzzer_tool.core.prng_state_recovery import (
    LFSRParams,
    TAUS88_PARAMS,
    predict_next,
    recover_taus88_state,
    taus88_step,
    verify_recovery,
)

_MIN_SAMPLES = 3  # exactly what pins a 96-bit taus88 state
_MAX_SAMPLES = 16  # cap on candidates fed to one recovery attempt

__all__ = ["PRNGStateLearner"]


class PRNGStateLearner:
    """Learns and caches a recovered taus88-family PRNG state."""

    def __init__(
        self,
        fuzzer: Any,
        params: tuple[LFSRParams, LFSRParams, LFSRParams] = TAUS88_PARAMS,
    ) -> None:
        self.f = fuzzer
        self.params = params
        self._state: tuple[int, int, int] | None = None
        # Samples the currently cached _state was confirmed against, in
        # order -- kept so a later observation can re-verify or invalidate it.
        self._confirmed_samples: list[int] = []
        self.attempts = 0
        self.successes = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_state(self) -> bool:
        """True when a recovered, verified state is cached."""
        return self._state is not None

    def observe_execution(self, input_data: bytes) -> bool:
        """Extract candidate consecutive draws from one execution and
        attempt (re)recovery.

        Returns:
            True when a state is cached after this call (whether newly
            recovered, re-confirmed, or already cached and untouched
            because this execution had too few candidates to matter).
        """
        candidates = self._extract_candidates(input_data)
        if len(candidates) < _MIN_SAMPLES:
            return self.has_state()

        if self._state is not None and verify_recovery(self._state, candidates, self.params):
            # Still consistent with everything seen so far -- nothing to redo.
            return True

        if self._try_recover(candidates):
            return True

        # Either never had a state, or the cached one just failed against
        # fresh evidence (a different generator instance, or the earlier
        # fit was a coincidence) -- drop it rather than keep serving
        # predictions that are now known to be wrong.
        self._state = None
        self._confirmed_samples = []
        return False

    def predict(self, n: int = 1) -> list[int] | None:
        """Predict the *n* draws after the last confirmed sample, or None.

        ``self._state`` is kept origin-aligned (its own, zero-step output
        equals ``self._confirmed_samples[0]`` -- see :func:`recover_taus88_state`
        and :meth:`_try_recover`), because that's the alignment
        ``verify_recovery`` needs to cheaply re-confirm the cache in
        :meth:`observe_execution`. Prediction wants the opposite end, so
        walk forward to the state whose own output is the *last* confirmed
        sample before asking for what comes after it.
        """
        if self._state is None:
            return None
        aligned = self._state
        for _ in range(len(self._confirmed_samples) - 1):
            aligned = taus88_step(aligned, self.params)
        return predict_next(aligned, n)

    def next_value_bytes(self, byteorder: str = "little") -> bytes | None:
        """Convenience for mutators: the single next predicted draw, packed."""
        predicted = self.predict(1)
        if not predicted:
            return None
        return predicted[0].to_bytes(4, byteorder)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_candidates(self, input_data: bytes) -> list[int]:
        """Distinct 4-byte cmplog operands absent from *input_data*."""
        cmplog = getattr(self.f, "_cmplog", None)
        if not cmplog or not cmplog.pairs:
            return []
        seen: set[int] = set()
        candidates: list[int] = []
        for op_a, op_b in cmplog.pairs:
            for op in (op_a, op_b):
                if len(op) != 4 or input_data.find(op) >= 0:
                    continue
                value = int.from_bytes(op, "little")
                if value in seen:
                    continue
                seen.add(value)
                candidates.append(value)
            if len(candidates) >= _MAX_SAMPLES:
                return candidates
        return candidates

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _try_recover(self, candidates: list[int]) -> bool:
        self.attempts += 1
        try:
            state = recover_taus88_state(candidates, params=self.params)
        except ValueError:
            return False
        if state is None or not verify_recovery(state, candidates, self.params):
            return False
        self._state = state
        self._confirmed_samples = list(candidates)
        self.successes += 1
        return True

    # ------------------------------------------------------------------
    # Persistence (mirrors ChecksumLearner.to_dict/from_dict)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": list(self._state) if self._state is not None else None,
            "confirmed_samples": self._confirmed_samples,
            "attempts": self.attempts,
            "successes": self.successes,
        }

    @classmethod
    def from_dict(cls, fuzzer: Any, data: dict[str, Any] | None) -> PRNGStateLearner:
        learner = cls(fuzzer)
        if data:
            state = data.get("state")
            if state and len(state) == 3:
                learner._state = (int(state[0]), int(state[1]), int(state[2]))
            learner._confirmed_samples = list(data.get("confirmed_samples") or [])
            learner.attempts = int(data.get("attempts", 0))
            learner.successes = int(data.get("successes", 0))
        return learner
