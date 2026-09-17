"""PRNG state learner: recovers a weak GF(2)-linear generator's internal
state from per-execution cmplog observations, and exposes predicted future
draws to the fuzzer's mutation operators.

Attaches to the fuzzer instance as ``f.prng_state_learner`` (see
``analyzer_registry.py``), the same wiring shape as ``f.checksum_learner``.

Targets CWE-338 (use of a predictable/weak PRNG). A fuzz target that seeds a
combined-LFSR (taus88/taus113/LFSR258), Marsaglia xorshift, or other
GF(2)-linear generator (see ``core/prng_state_recovery.py`` for the family
this covers) for nonces, session tokens, or sequence numbers leaks its output
stream through comparison operands. A handful of observed consecutive draws
pin the generator's whole future output (how many is family-specific -- see
``_CANDIDATE_FAMILIES`` and ``confident_samples``), after which every future
draw is known in advance, which lets a mutation operator write the *actual*
next value into the input instead of guessing -- turning an otherwise-
unreachable "does this match the token I generated" check into a one-shot
pass.

Family is not assumed. ``_try_recover`` tries every 4-byte-output candidate
family smallest-state-first (cheapest elimination first) and keeps whichever
verifies -- the "reversing the target's binary just to learn which family it
uses" step ``core/prng_state_recovery.py``'s module docstring says this class
avoids. 8-byte-output families (``xorshift64``, ``lfsr258``) are shipped in
``core/prng_state_recovery.py`` but not tried here yet: the extraction below
and ``operators.py::_op_prng_predict`` both hard-code a 4-byte operand width,
and widening either is a separate change (touching the mutator's placement
search too) left as follow-up.

Signal source
-------------
Reads ``f._cmplog.last_conds`` -- the *ordered* ``CondStmt`` list from the
most recent drain, each record carrying its comparison PC -- and groups
operands by PC. It deliberately does NOT read ``f._cmplog.pairs``, which an
earlier version of this module used: that pool is campaign-wide,
first-seen-only, capped with eviction, and drained on an interval once
saturated, so its iteration order is first-sighting order pooled across many
executions rather than the order values were compared in, and repeated draws
are deduplicated away entirely. Recovery needs consecutive outputs of one
stream, so it needs exactly the order and per-PC grouping ``pairs`` discards.

Two properties separate a generator draw from every other operand, and both
are used:

1. **Ordered and per-site.** A generator is drained at one call site, so its
   draws appear at one comparison PC, one per query, in stream order. The
   longest such per-PC run is the candidate sequence.

2. **Nondeterministic across identical inputs.** An operand echoed from the
   input repeats when the same input is replayed; a generator's value does
   not. ``_run_history`` accumulates, per input hash, the values seen at each
   PC, and a PC whose value set grows across replays of one input is
   preferred as a state site. (``CmplogCollector`` declares a ``_run_history``
   field for exactly this purpose and never populates it; this is that idea,
   kept local to the consumer that needs it.)

A candidate operand must also be absent from the input buffer -- an operand
found verbatim in the input is data the target read back, not state it
generated -- which is the mirror image of
``ChecksumLearner.extract_cmplog_pairs`` (that one wants the operand found
IN the input).

Applicability
-------------
Only in-process execution (``direct_lite``/persistent) is observed, gated on
``f._inprocess_runner``. There one generator instance survives across
iterations, so draws collected over a drain are genuinely consecutive outputs
of one stream -- the regime where prediction beats the existing
input-to-state machinery, which is always one draw stale against a
still-advancing generator. Under a forkserver with a fixed ``--seed`` every
execution redraws the identical stream and I2S already covers it; under a
time/pid seed there is no cross-execution stream to predict at all.

Cross-drain continuation
------------------------
A later drain's draws are further along the same stream, not a repeat of the
samples the state was recovered from, and the gap is unknown: draws the
target made without comparing them are invisible here. So re-confirmation
searches forward from the cached frontier for the new sequence within
``_MAX_ADVANCE_SEARCH`` steps (the "skip math" of the Factorio RNG writeup
this module's method comes from, in its cheapest form) and advances the
frontier when it matches. Verifying the new samples against the origin-
aligned state instead -- which is what an earlier version did -- can only
succeed when the stream happens to repeat, so it discarded a good state on
almost every drain and re-ran recovery from scratch.

Recovery cost and false-positive control
-----------------------------------------
Each family's elimination is a fixed-cost ``state_bits x state_bits`` GF(2)
solve -- cheap (at most 320x320, for lfsr258), but not free enough to run
unconditionally on every execution, so candidates are capped at
``_MAX_SAMPLES`` per attempt (mirrors ``CHECKSUM_PAIRS_MAX``'s rationale
exactly: bound the cost of an attempt, not just how often one is made). A
recovered state is only kept once ``verify_state`` confirms it reproduces
every sample it was derived from, including ones beyond the family's own
``min_samples`` -- an accidental fit to non-generator "random" data is not
impossible at exactly the pinning count, and ``confident_samples``'s extra
word is what tells it apart from a real one (see that function's docstring
for the measured false-positive rates). ``attempts``/``successes`` count
*window* attempts, not the per-family probes inside one: trying all of
xorshift32/taus88/taus113/xorshift128 against one window and failing every
one is a single recorded attempt, matching the cost model above (one
elimination-sized decision per execution, not four). When later samples
disagree with an already-cached state (a false-positive site, or a
different generator instance), the cache is dropped rather than kept stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from fuzzer_tool.core.prng_state_recovery import (
    FAMILIES,
    LinearPRNG,
    confident_samples,
    family,
    output_word,
    predict_words,
    recover_state,
    step_state,
    verify_state,
)

_OPERAND_BYTES = 4  # width this learner extracts and the mutator writes back
# The 4-byte-output shipped families, in FAMILIES' smallest-state-first
# order (cheapest elimination tried first). 8-byte families (xorshift64,
# lfsr258) are excluded -- see the module docstring's "Family is not
# assumed" section for why.
_CANDIDATE_FAMILIES: tuple[LinearPRNG, ...] = tuple(
    spec for spec in FAMILIES.values() if spec.out_bytes == _OPERAND_BYTES
)
# Below this floor, no family has enough of a margin to be worth an attempt.
# Individual families gate at their own (larger) confident_samples() inside
# _try_recover; this is just the cheapest one, so windows never even reach
# _try_recover before any family could possibly verify.
_MIN_SAMPLES = min(confident_samples(spec) for spec in _CANDIDATE_FAMILIES)
_MAX_SAMPLES = 16  # cap on candidates fed to one recovery attempt
# How far ahead of the cached frontier to look for a later drain's samples.
# Bounds the cost of continuation (one step is a few integer ops) while
# tolerating draws the target made without comparing them.
_MAX_ADVANCE_SEARCH = 64
# Cap on per-input PC histories retained for the nondeterminism filter.
_RUN_HISTORY_CAP = 256
# Sentinel for "no pending window at all", so a legitimate None PC bucket
# (a shim build that logs no PCs) is not confused with the empty case.
_NO_CANDIDATE = object()

__all__ = ["PRNGStateLearner"]


@dataclass
class _PCHistory:
    """Per-(input, PC) record backing the nondeterminism test."""

    values: set[int] = field(default_factory=set)
    max_drain_width: int = 0


class PRNGStateLearner:
    """Learns and caches a recovered GF(2)-linear PRNG state, family included."""

    def __init__(self, fuzzer: Any) -> None:
        self.f = fuzzer
        self._spec: LinearPRNG | None = None
        self._state: tuple[int, ...] | None = None
        # Samples the currently cached _state was confirmed against, in
        # order -- kept so a later observation can re-verify or invalidate it.
        self._confirmed_samples: list[int] = []
        # The frontier: walked forward so its own zero-step output equals the
        # LAST confirmed sample, i.e. the state to predict the future from.
        # Derived from _state, so it is not persisted -- from_dict rebuilds it.
        self._frontier: tuple[int, ...] | None = None
        # input hash -> {pc -> (values ever seen, widest single drain)}. A pc
        # is nondeterministic when it has produced MORE distinct values for one
        # input than any single drain of that input contained: that is a value
        # that changed between replays, the signature of internal state. The
        # comparison against drain width is what separates it from a site that
        # merely compares several different constants every time.
        self._run_history: dict[int, dict[int, _PCHistory]] = {}
        # pc -> the run of draws accumulated at that site but not yet folded
        # into a confirmed state. A drain typically contributes one value per
        # site, so a run is assembled across executions; bounded to
        # _MAX_SAMPLES, oldest dropped first.
        self._pending: dict[int | None, list[int]] = {}
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

        A drain usually carries ONE draw, not a run: the fuzz loop calls the
        shim's ``__cmplog_reset`` after every collection, which truncates the
        record file, and a target typically draws once per execution. Measured
        against ``targets/prng_token_read.c``: 5 records per drain, exactly one
        of them the token. So candidates accumulate in a per-PC window across
        drains -- consecutive executions of one in-process generator are
        consecutive outputs of one stream -- and recovery runs once a window
        reaches ``_MIN_SAMPLES``. Requiring the run inside a single drain
        (which an earlier version did) never fires on a real target.

        Returns:
            True when a state is cached after this call (whether newly
            recovered, re-confirmed, or already cached and untouched
            because this execution added too little to matter).
        """
        # Only in-process execution keeps one generator instance alive across
        # iterations; see the Applicability section above. Elsewhere there is
        # no cross-execution stream, so don't spend the elimination or, worse,
        # cache a fit to values that will never recur.
        if getattr(self.f, "_inprocess_runner", None) is None:
            return self.has_state()

        fresh = self._extract_by_pc(input_data)
        for pc, values in fresh.items():
            window = self._pending.setdefault(pc, [])
            window.extend(values)
            if len(window) > _MAX_SAMPLES:
                del window[:-_MAX_SAMPLES]

        pc = self._best_pc()
        if pc is _NO_CANDIDATE:
            return self.has_state()
        window = self._pending[pc]

        if self._state is not None and self._continues_from_frontier(window):
            # Same stream, further along: advance rather than re-recover.
            window.clear()
            return True

        if len(window) < _MIN_SAMPLES:
            return self.has_state()

        if self._try_recover(window):
            window.clear()
            return True

        # This window is not a run of this generator's outputs: either it
        # never was one, or it straddles a reseed. Slide it by one rather
        # than clearing, so a window that merely starts in the wrong place
        # recovers on the next draw instead of waiting for a whole new one.
        del window[0]
        # A cached state that fresh evidence contradicts is worse than none:
        # every prediction it serves is known-wrong.
        self._clear_state()
        return False

    def predict(self, n: int = 1) -> list[int] | None:
        """Predict the *n* draws after the last confirmed sample, or None.

        ``self._state`` is kept origin-aligned (its own, zero-step output
        equals ``self._confirmed_samples[0]`` -- see :func:`recover_state`
        and :meth:`_try_recover`), because that's the alignment
        ``verify_state`` needs to cheaply re-confirm the cache in
        :meth:`observe_execution`. Prediction wants the opposite end, so
        walk forward to the state whose own output is the *last* confirmed
        sample before asking for what comes after it.
        """
        if self._frontier is None or self._spec is None:
            return None
        return predict_words(self._frontier, n, self._spec)

    def next_value_bytes(self, byteorder: Literal["little", "big"] = "little") -> bytes | None:
        """Convenience for mutators: the single next predicted draw, packed."""
        predicted = self.predict(1)
        if not predicted:
            return None
        return predicted[0].to_bytes(_OPERAND_BYTES, byteorder)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_by_pc(self, input_data: bytes) -> dict[int | None, list[int]]:
        """This drain's 4-byte operands absent from *input_data*, grouped by
        comparison PC and kept in encounter order within each group.

        ``None`` is the no-PC bucket, so a shim build that logs no PC still
        yields a usable sequence rather than nothing.
        """
        cmplog = getattr(self.f, "_cmplog", None)
        conds = getattr(cmplog, "last_conds", None) if cmplog is not None else None
        if not conds:
            return {}

        history = self._note_input(input_data)
        by_pc: dict[int | None, list[int]] = {}
        for cond in conds:
            base = cond.base
            pc = base.pc
            for op in (base.op_a, base.op_b):
                if len(op) != _OPERAND_BYTES or input_data.find(op) >= 0:
                    continue
                value = int.from_bytes(op, "little")
                bucket = by_pc.setdefault(pc, [])
                if value in bucket:
                    # A repeat inside one drain is not the next draw: a
                    # generator returning the same word twice running is
                    # indistinguishable from one comparison logged twice, and
                    # the latter is overwhelmingly more likely.
                    continue
                bucket.append(value)
        for pc, values in by_pc.items():
            if pc is None:
                continue
            entry = history.get(pc)
            if entry is None:
                entry = history[pc] = _PCHistory()
            entry.values.update(values)
            entry.max_drain_width = max(entry.max_drain_width, len(values))
        return by_pc

    def _best_pc(self) -> Any:
        """The PC whose pending window is the most promising candidate run.

        Prefers a site already proven nondeterministic across replays of one
        input -- the signature of internal state rather than echoed input --
        and breaks ties on how much of a run has accumulated.
        """
        if not self._pending:
            return _NO_CANDIDATE
        varied = self._varied_pcs()

        def rank(pc: int | None) -> tuple[bool, int]:
            return (pc in varied, len(self._pending[pc]))

        return max(self._pending, key=rank)

    def _varied_pcs(self) -> set[int]:
        """PCs whose value changed between replays of one input.

        More distinct values than the widest single drain of that input means
        a replay produced something new. A site logging the same five
        constants every execution never qualifies, however many they are.
        """
        return {
            pc
            for history in self._run_history.values()
            for pc, entry in history.items()
            if len(entry.values) > entry.max_drain_width
        }

    def _note_input(self, input_data: bytes) -> dict[int, _PCHistory]:
        """Return (creating if needed) the per-PC value history for this input."""
        key = hash(input_data)
        history = self._run_history.get(key)
        if history is None:
            if len(self._run_history) >= _RUN_HISTORY_CAP:
                # Oldest insertion first (dicts preserve order). This is a
                # coincidence filter, not a ledger, so plain FIFO is enough.
                self._run_history.pop(next(iter(self._run_history)))
            history = {}
            self._run_history[key] = history
        return history

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _set_state(
        self, spec: LinearPRNG, state: tuple[int, ...], samples: list[int]
    ) -> None:
        """Cache *state* (its own output == samples[0]) and derive the frontier
        whose own output == samples[-1]."""
        self._spec = spec
        self._state = state
        self._confirmed_samples = list(samples)
        frontier = state
        for _ in range(len(samples) - 1):
            frontier = step_state(frontier, spec)
        self._frontier = frontier

    def _clear_state(self) -> None:
        self._spec = None
        self._state = None
        self._frontier = None
        self._confirmed_samples = []

    def _continues_from_frontier(self, candidates: list[int]) -> bool:
        """True when *candidates* are the stream continuing past the frontier.

        Searches forward up to ``_MAX_ADVANCE_SEARCH`` draws for the candidate
        run, because draws the target made without comparing them leave gaps
        this side cannot see. On a match the frontier advances past the run, so
        the next prediction is relative to the newest confirmed draw.
        """
        if self._frontier is None or self._spec is None or not candidates:
            return False
        spec = self._spec
        probe = self._frontier
        for _ in range(_MAX_ADVANCE_SEARCH):
            probe = step_state(probe, spec)
            if output_word(probe, spec) != candidates[0]:
                continue
            walk = probe
            for expected in candidates[1:]:
                walk = step_state(walk, spec)
                if output_word(walk, spec) != expected:
                    break
            else:
                self._frontier = walk
                self._confirmed_samples.extend(candidates)
                return True
        return False

    def _try_recover(self, candidates: list[int]) -> bool:
        """Try every candidate family, smallest-state first, on *candidates*.

        One call is one recorded attempt regardless of how many families are
        probed inside it -- see the module docstring's cost-model note.
        """
        self.attempts += 1
        for spec in _CANDIDATE_FAMILIES:
            if len(candidates) < confident_samples(spec):
                continue
            try:
                state = recover_state(candidates, spec)
            except ValueError:
                continue
            if state is None or not verify_state(state, candidates, spec):
                continue
            self._set_state(spec, state, candidates)
            self.successes += 1
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence (mirrors ChecksumLearner.to_dict/from_dict)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": list(self._state) if self._state is not None else None,
            # Which family the state belongs to, so from_dict rebuilds with
            # the right step/output maps instead of guessing. Absent in dicts
            # written before this field existed -- from_dict falls back to
            # taus88 then, the only family the learner ever recovered.
            "family": self._spec.name if self._spec is not None else None,
            "confirmed_samples": self._confirmed_samples,
            "attempts": self.attempts,
            "successes": self.successes,
        }

    @classmethod
    def from_dict(cls, fuzzer: Any, data: dict[str, Any] | None) -> PRNGStateLearner:
        learner = cls(fuzzer)
        if data:
            state = data.get("state")
            if state:
                spec = family(data.get("family") or "taus88") or FAMILIES["taus88"]
                restored = tuple(int(x) for x in state)
                samples = [int(x) for x in data.get("confirmed_samples") or []]
                # The frontier is derived, not stored, so rebuild it here --
                # otherwise a resumed campaign predicts from the origin-aligned
                # state and hands out draws it has already seen. With no
                # samples recorded (a state dict written before this field
                # existed), the state's own output is the only anchor there is.
                learner._set_state(
                    spec, restored, samples or [output_word(restored, spec)]
                )
            learner.attempts = int(data.get("attempts", 0))
            learner.successes = int(data.get("successes", 0))
        return learner
