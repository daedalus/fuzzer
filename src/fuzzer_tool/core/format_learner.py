"""Format structure learner — schema-harness methodology for fuzzing.

Induces an executable world model of a binary format from fuzzing observations.
Maps the schema-harness control loop to the fuzzer's observation space:

  observe:   (input_bytes, coverage_transition, sanitizer_output)
  state:     inferred format fields, boundaries, and dependencies
  action:    mutation operators applied at specific positions
  mechanism: how mutations in positions affect coverage paths

The learner maintains:
- A Timeline of (input_hash, mutation_op, position, coverage_delta, sanitizer)
- Candidate field hypotheses with confidence scores
- A backtested format model that predicts coverage consequences of mutations

Core loop:
  1. Mutate → observe coverage transition
  2. Hypothesize field boundaries from transition patterns
  3. Backtest hypotheses against full Timeline
  4. Only trust hypotheses that survive full backtest
"""

import hashlib
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

BACKTEST_INTERVAL = 500  # run backtest every N recorded transitions


@dataclass
class FieldHypothesis:
    """A candidate format field with boundaries and properties."""
    offset: int
    width: int
    field_type: str  # "magic", "length", "crc", "flags", "data", "unknown"
    confidence: float = 0.0
    observations: int = 0
    # Which mutations at this offset reliably change coverage
    sensitive_ops: dict = field(default_factory=dict)
    # What coverage transitions this field controls
    controlled_edges: set = field(default_factory=set)
    # Dependencies: "if I change this field, these other fields must also change"
    dependencies: list = field(default_factory=list)


@dataclass
class TimelineEntry:
    """One observation in the append-only Timeline.

    Stores input_hash (16 bytes) instead of full input to save memory.
    """
    input_hash: str  # SHA256 prefix for identification
    mutation_op: str
    mutation_offset: int
    mutation_width: int
    coverage_before: int
    coverage_after: int
    new_edges: set
    lost_edges: set


class FormatLearner:
    """Induces format structure from fuzzing observations.

    Follows the schema-harness methodology:
    - State grounding: infers field boundaries from mutation sensitivity
    - Mechanism discovery: finds how fields control coverage paths
    - Backtesting: validates hypotheses against full Timeline
    - Action for discovery: selects mutations that discriminate hypotheses
    """

    def __init__(self, max_timeline: int = 5000):
        self.timeline: list[TimelineEntry] = []
        self.max_timeline = max_timeline
        self.hypotheses: list[FieldHypothesis] = []
        self.field_map: dict[int, FieldHypothesis] = {}
        self.backtest_passes: int = 0
        self.backtest_fails: int = 0
        self.format_model_version: int = 0
        self._transitions_since_backtest: int = 0

    def record_transition(
        self,
        input_bytes: bytes,
        mutation_op: str,
        mutation_offset: int,
        mutation_width: int,
        coverage_before: int,
        coverage_after: int,
        new_edges: set,
        lost_edges: set,
    ):
        """Append a real transition to the Timeline.

        Stores only input hash, not full bytes, to save memory.
        """
        input_hash = hashlib.sha256(input_bytes).hexdigest()[:16]

        entry = TimelineEntry(
            input_hash=input_hash,
            mutation_op=mutation_op,
            mutation_offset=mutation_offset,
            mutation_width=mutation_width,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            new_edges=new_edges,
            lost_edges=lost_edges,
        )
        self.timeline.append(entry)
        if len(self.timeline) > self.max_timeline:
            self.timeline = self.timeline[-self.max_timeline:]

        self._update_hypotheses(entry)

        # Periodic backtest
        self._transitions_since_backtest += 1
        if self._transitions_since_backtest >= BACKTEST_INTERVAL:
            self._transitions_since_backtest = 0
            ok, desc = self.backtest()
            if not ok:
                log.debug("Backtest failed: %s", desc)

    def _update_hypotheses(self, entry: TimelineEntry):
        """Update field hypotheses based on a new observation."""
        offset = entry.mutation_offset
        width = entry.mutation_width
        delta = entry.coverage_after - entry.coverage_before
        has_effect = delta != 0 or entry.new_edges or entry.lost_edges

        existing = None
        for h in self.hypotheses:
            if h.offset <= offset < h.offset + h.width:
                existing = h
                break

        if existing:
            existing.observations += 1
            if has_effect:
                existing.sensitive_ops[entry.mutation_op] = (
                    existing.sensitive_ops.get(entry.mutation_op, 0) + 1
                )
                existing.controlled_edges.update(entry.new_edges)
                if len(existing.sensitive_ops) >= 2:
                    existing.confidence = min(1.0, existing.confidence + 0.1)
            else:
                existing.confidence = max(0.0, existing.confidence - 0.02)
        elif has_effect:
            h = FieldHypothesis(
                offset=offset,
                width=max(width, 1),
                field_type="unknown",
                confidence=0.3,
                observations=1,
                sensitive_ops={entry.mutation_op: 1},
                controlled_edges=set(entry.new_edges),
            )
            self.hypotheses.append(h)
            self.field_map[offset] = h

        self._classify_fields()

    def _classify_fields(self):
        """Classify hypotheses into field types based on evidence patterns."""
        for h in self.hypotheses:
            if h.observations < 3:
                continue

            if h.offset == 0 and h.confidence > 0.5:
                h.field_type = "magic"
                continue
            if len(h.controlled_edges) > 5 and h.confidence > 0.4:
                h.field_type = "length"
                continue
            if len(h.sensitive_ops) > 3 and h.confidence > 0.3:
                h.field_type = "crc"
                continue
            if h.observations > 10 and h.confidence > 0.2:
                h.field_type = "data"
                continue
            h.field_type = "unknown"

    def backtest(self) -> tuple[bool, Optional[str]]:
        """Replay the ENTIRE Timeline through the current format model."""
        if not self.hypotheses:
            return True, None

        for i, entry in enumerate(self.timeline):
            predicted_effect = self._predict_effect(entry)
            actual_effect = (
                entry.coverage_after != entry.coverage_before
                or bool(entry.new_edges)
                or bool(entry.lost_edges)
            )

            if predicted_effect is None:
                continue

            if predicted_effect != actual_effect:
                self.backtest_fails += 1
                desc = (
                    f"Transition {i}: mutation {entry.mutation_op} "
                    f"at offset {entry.mutation_offset} — "
                    f"predicted {'effect' if predicted_effect else 'no effect'}, "
                    f"got {'effect' if actual_effect else 'no effect'} "
                    f"(edges: {entry.coverage_before} → {entry.coverage_after})"
                )
                return False, desc

        self.backtest_passes += 1
        self.format_model_version += 1
        return True, None

    def _predict_effect(self, entry: TimelineEntry):
        """Predict whether a mutation should affect coverage."""
        offset = entry.mutation_offset
        for h in self.hypotheses:
            if h.offset <= offset < h.offset + h.width and h.confidence > 0.3:
                return True
        for h in self.hypotheses:
            if entry.mutation_op in h.sensitive_ops and h.confidence > 0.3:
                return True
        return None

    def suggest_discriminating_mutation(self, candidates: list[str]) -> Optional[tuple[str, int]]:
        """Suggest a mutation that would discriminate between hypotheses."""
        if len(self.hypotheses) < 2:
            return None

        for i, h1 in enumerate(self.hypotheses):
            for h2 in self.hypotheses[i + 1:]:
                if h1.field_type != h2.field_type:
                    for op in candidates:
                        if op in h1.sensitive_ops and op not in h2.sensitive_ops:
                            return (op, h1.offset)
                        if op not in h1.sensitive_ops and op in h2.sensitive_ops:
                            return (op, h2.offset)
        return None

    def get_format_summary(self) -> dict:
        """Return a summary of the inferred format structure."""
        sorted_hyps = sorted(self.hypotheses, key=lambda h: h.offset)
        fields = []
        for h in sorted_hyps:
            fields.append({
                "offset": h.offset,
                "width": h.width,
                "type": h.field_type,
                "confidence": round(h.confidence, 3),
                "observations": h.observations,
                "sensitive_ops": dict(h.sensitive_ops),
                "controlled_edges": len(h.controlled_edges),
            })

        return {
            "timeline_size": len(self.timeline),
            "hypotheses": len(self.hypotheses),
            "classified": sum(1 for h in self.hypotheses if h.field_type != "unknown"),
            "backtest_passes": self.backtest_passes,
            "backtest_fails": self.backtest_fails,
            "model_version": self.format_model_version,
            "fields": fields,
        }

    def get_state(self) -> dict:
        """Serialize for persistence."""
        return {
            "timeline": [
                {
                    "input_hash": e.input_hash,
                    "op": e.mutation_op,
                    "offset": e.mutation_offset,
                    "width": e.mutation_width,
                    "cov_before": e.coverage_before,
                    "cov_after": e.coverage_after,
                    "new_edges": list(e.new_edges),
                    "lost_edges": list(e.lost_edges),
                }
                for e in self.timeline[-1000:]
            ],
            "hypotheses": [
                {
                    "offset": h.offset,
                    "width": h.width,
                    "type": h.field_type,
                    "confidence": h.confidence,
                    "observations": h.observations,
                    "sensitive_ops": dict(h.sensitive_ops),
                    "controlled_edges": list(h.controlled_edges),
                }
                for h in self.hypotheses
            ],
            "backtest_passes": self.backtest_passes,
            "backtest_fails": self.backtest_fails,
            "model_version": self.format_model_version,
        }

    def load_state(self, state: dict):
        """Restore from persistence."""
        self.timeline = []
        for e in state.get("timeline", []):
            self.timeline.append(TimelineEntry(
                input_hash=e.get("input_hash", ""),
                mutation_op=e["op"],
                mutation_offset=e["offset"],
                mutation_width=e["width"],
                coverage_before=e["cov_before"],
                coverage_after=e["cov_after"],
                new_edges=set(e.get("new_edges", [])),
                lost_edges=set(e.get("lost_edges", [])),
            ))
        self.hypotheses = []
        for h in state.get("hypotheses", []):
            hyp = FieldHypothesis(
                offset=h["offset"],
                width=h["width"],
                field_type=h["type"],
                confidence=h["confidence"],
                observations=h["observations"],
                sensitive_ops=h.get("sensitive_ops", {}),
                controlled_edges=set(h.get("controlled_edges", [])),
            )
            self.hypotheses.append(hyp)
            self.field_map[hyp.offset] = hyp
        self.backtest_passes = state.get("backtest_passes", 0)
        self.backtest_fails = state.get("backtest_fails", 0)
        self.format_model_version = state.get("model_version", 0)
