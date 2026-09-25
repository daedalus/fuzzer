"""Format structure learner — schema-harness methodology for fuzzing.

Induces an executable world model of a binary format from fuzzing observations.
Maps the schema-harness control loop to the fuzzer's observation space:

  observe:   (input_bytes, coverage_transition, sanitizer_output)
  state:     inferred format fields, boundaries, and dependencies
  action:    mutation operators applied at specific positions
  mechanism: how mutations in positions affect coverage paths

Multi-format targets (an ffmpeg-style demuxer probing RIFF/MP4/MKV/... in
turn, any parser that dispatches on a magic number) give the *same* byte
offset different meanings depending on which format an input actually is —
offset 4 is a RIFF chunk size in a WAV file and part of an `ftyp` box in an
MP4. A single global field map conflates these into noise. So the learner
doesn't maintain one field map: it maintains a `FormatCluster` per observed
input signature (by default, the first couple of bytes), each with its own
independent Timeline, field hypotheses, and backtest history. Growing the
corpus with a new container type grows a new cluster instead of corrupting
the existing ones, and each cluster can be exploited as its own model.

The learner (per cluster) maintains:
- A Timeline of (input_hash, mutation_op, position, coverage_delta, sanitizer)
- Candidate field hypotheses with confidence scores
- A backtested format model that predicts coverage consequences of mutations

Core loop, per cluster:
  1. Mutate → observe coverage transition
  2. Hypothesize field boundaries from transition patterns
  3. Backtest hypotheses against that cluster's full Timeline
  4. Only trust hypotheses that survive full backtest

`FormatLearner` itself stays call-compatible with the single-format version:
`.hypotheses`, `.timeline`, `.field_map`, `.backtest_passes/fails`, and
`.format_model_version` are views onto the "primary" cluster (the one with
the most recorded observations), so existing callers that don't care about
multi-format tracking keep working unchanged. `get_format_summary()` adds a
`formats` list with one summary per cluster for callers that do.
"""

import hashlib
import logging
from array import array
from dataclasses import dataclass, field

from fuzzer_tool.core.running_stats import RunningMoments

log = logging.getLogger(__name__)

BACKTEST_INTERVAL = 500  # run backtest every N recorded transitions
DEFAULT_SIG_LEN = 2  # bytes of prefix used to cluster inputs into format hypotheses
DEFAULT_MAX_FORMATS = 6  # cap on concurrently tracked format clusters (bounds memory)
DEFAULT_SIGNATURE = ""  # cluster key used before any real signature is known
DEFAULT_PROMOTE_THRESHOLD = 3  # times a signature must recur before it gets its own cluster
MAX_TRACKED_SIGNATURES = 4096  # cap on candidate signatures awaiting promotion
_FENWICK_INIT_SIZE = 64  # initial byte-offset capacity; doubles on demand


_MIN_CLASSIFY_OBS = 3  # observations before a field is typed or stride-boosted


def _rank_value(items: list[tuple[float, int]], rank: int) -> float:
    """Value at 0-based *rank* in a sorted (value, count) histogram."""
    seen = 0
    for v, cnt in items:
        seen += cnt
        if seen > rank:
            return v
    return items[-1][0]


class _CoverageFenwick:
    """Range-add / point-query Fenwick tree over byte offsets.

    Each field [offset, offset+width) is one edge: +1 at offset, -1 at end.
    Index 0 is the null-frame root; depth(pos) = fields covering pos.

        field (2, 3):   +1 at 2, -1 at 5
        depth:          pos 0 1 2 3 4 5
                            0 0 1 1 1 0

    O(log n) per add/query. Zero depth lets callers skip the O(H) scan.
    Spans are clipped at 0; callers scan for negative positions.

    `sparse` is False once span widths sum past the highest indexed byte:
    fields then blanket the input, depth is almost never 0, and the query
    is pure overhead (measured ~6% slower on a 256-byte fully-covered frame).
    """

    def __init__(self, size: int = _FENWICK_INIT_SIZE):
        self._n = size
        self._tree = array("i", bytes(4 * (size + 1)))
        self._mass = 0
        self._hi = 0
        self.sparse = True

    def add(self, offset: int, width: int) -> None:
        """Add field [offset, offset+width) as one range edge."""
        end = offset + width
        if end <= 0 or width <= 0:
            return

        if end >= self._n:
            self._grow(end + 1)

        self._bump(max(offset, 0) + 1, 1)
        self._bump(end + 1, -1)

        self._mass += end - max(offset, 0)
        self._hi = max(self._hi, end)
        self.sparse = self._mass < self._hi

    def depth(self, pos: int) -> int:
        """Fields covering *pos* — prefix sum from the null-frame root."""
        if pos < 0 or pos >= self._n:
            return 0

        tree = self._tree
        i = pos + 1
        total = 0
        while i:
            total += tree[i]
            i &= i - 1
        return total

    def _bump(self, i: int, delta: int) -> None:
        tree = self._tree
        n = self._n
        while i <= n:
            tree[i] += delta
            i += i & -i

    def _grow(self, need: int) -> None:
        """Double capacity and rebuild in O(n) from the point diffs."""
        old_n = self._n
        diffs = [self.depth(p) - self.depth(p - 1) for p in range(old_n)]

        n = old_n
        while n < need:
            n *= 2
        tree = array("i", bytes(4 * (n + 1)))

        # Linear Fenwick build: seed leaves, push each node into its parent.
        for p, d in enumerate(diffs):
            tree[p + 1] = d
        for i in range(1, n + 1):
            parent = i + (i & -i)
            if parent <= n:
                tree[parent] += tree[i]

        self._n = n
        self._tree = tree


@dataclass
class FieldHypothesis:
    """A candidate format field with boundaries and properties."""

    offset: int
    width: int
    field_type: str  # "magic", "length", "crc", "flags", "data", "padding", "unknown"
    confidence: float = 0.0
    observations: int = 0
    # Which mutations at this offset reliably change coverage
    sensitive_ops: dict = field(default_factory=dict)
    # What coverage transitions this field controls
    controlled_edges: set = field(default_factory=set)
    # Dependencies: "if I change this field, these other fields must also change"
    dependencies: list = field(default_factory=list)
    # Per-position byte value frequency: relative_position -> byte_value -> count
    value_counts: dict = field(default_factory=dict)


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


@dataclass
class FormatCluster:
    """One candidate format's independent field map, Timeline, and backtest
    history — everything `FormatLearner` used to hold as a single flat
    model, now scoped to inputs sharing one signature.

    Deliberately isolated from every other cluster: an observation routed
    here never updates another cluster's hypotheses, and its backtest only
    ever replays its own Timeline. That isolation is the whole point — it's
    what lets the learner hold "offset 4 is a chunk length" and "offset 4 is
    a codec tag" as two live, uncontradicted hypotheses at once.
    """

    signature: str
    z_score_threshold: float = 2.0
    timeline: list[TimelineEntry] = field(default_factory=list)
    hypotheses: list[FieldHypothesis] = field(default_factory=list)
    field_map: dict[int, FieldHypothesis] = field(default_factory=dict)
    backtest_passes: int = 0
    backtest_fails: int = 0
    format_model_version: int = 0
    # Byte-level record stride inferred from this format's own seeds
    # (estimate_record_size) — a structural prior for field classification.
    record_stride: int | None = None
    total_observations: int = 0
    _transitions_since_backtest: int = 0
    _delta_moments: RunningMoments = field(default_factory=RunningMoments)
    # delta -> count; exact MAD in O(distinct deltas) instead of sorting history.
    _delta_counts: dict[float, int] = field(default_factory=dict, repr=False, compare=False)
    # Coverage index over `hypotheses`; rebuilt if the list is swapped or
    # appended to from outside (`_synced_cover`).
    _cover: _CoverageFenwick = field(default_factory=_CoverageFenwick, repr=False, compare=False)
    _cover_list: list | None = field(default=None, repr=False, compare=False)
    _cover_len: int = field(default=0, repr=False, compare=False)

    def set_record_stride(self, stride: int | None):
        """Set the record-stride structural prior from periodicity detection.

        A stride-aligned hypothesis (width equal to the stride, or starting at
        a stride boundary) is more likely a real format field — classification
        boosts its confidence slightly.
        """
        self.record_stride = stride

    def _add_hypothesis(self, h: FieldHypothesis) -> None:
        """Append *h* to the frame and index its span."""
        cover = self._synced_cover()
        self.hypotheses.append(h)
        self.field_map[h.offset] = h
        cover.add(h.offset, h.width)
        self._cover_len += 1

    def _synced_cover(self) -> _CoverageFenwick:
        """Coverage index, rebuilt when `hypotheses` changed behind it."""
        hyps = self.hypotheses
        if hyps is self._cover_list and len(hyps) == self._cover_len:
            return self._cover

        cover = _CoverageFenwick()
        for h in hyps:
            cover.add(h.offset, h.width)
        self._cover = cover
        self._cover_list = hyps
        self._cover_len = len(hyps)
        return cover

    def _covering(self, pos: int) -> FieldHypothesis | None:
        """First hypothesis (list order) covering *pos*, or None."""
        hyps = self.hypotheses
        cover = self._cover
        if hyps is not self._cover_list or len(hyps) != self._cover_len:
            cover = self._synced_cover()
        if cover.sparse and pos >= 0 and not cover.depth(pos):
            return None

        for h in hyps:
            if h.offset <= pos < h.offset + h.width:
                return h
        return None

    def record(self, entry: TimelineEntry, input_bytes: bytes, max_timeline: int):
        """Append a real transition to this cluster's Timeline and update
        its hypotheses, periodically backtesting against its own history."""
        self.timeline.append(entry)
        # In place: a slice copy would copy all max_timeline entries per record.
        excess = len(self.timeline) - max_timeline
        if excess > 0:
            del self.timeline[:excess]
        self.total_observations += 1

        self._update_hypotheses(entry, input_bytes)

        self._transitions_since_backtest += 1
        if self._transitions_since_backtest >= BACKTEST_INTERVAL:
            self._transitions_since_backtest = 0
            ok, desc = self.backtest()
            if not ok:
                log.debug("Backtest failed for format %r: %s", self.signature, desc)

    def _update_hypotheses(self, entry: TimelineEntry, input_bytes: bytes | None = None):
        """Update field hypotheses based on a new observation."""
        offset = entry.mutation_offset
        width = entry.mutation_width
        delta = entry.coverage_after - entry.coverage_before

        has_effect = self._has_effect(entry, delta)

        # Delocalised ops set `mutation_offset=None`; they have no single byte
        # to attribute, so skip field-hypothesis updates rather than crashing.
        if offset is None:
            return

        existing = self._covering(offset)

        touched = existing
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
            self._add_hypothesis(h)
            touched = h

        if offset is not None and input_bytes is not None:
            self._track_values(offset, width, input_bytes)

        # Only the field this record is evidence about; O(1), not O(H).
        if touched is not None and touched.observations >= _MIN_CLASSIFY_OBS:
            self._boost_stride(touched)
            self._classify(touched)

    def _has_effect(self, entry: TimelineEntry, delta: int) -> bool:
        """Whether *entry*'s coverage delta is field-sensitive evidence."""
        # Z-score gate: a mutation only counts as "field-sensitive" if
        # its effect is a statistical outlier relative to ambient noise,
        # not just nonzero.  Under high excess kurtosis (zero-inflated
        # coverage deltas), fall back to MAD-based z-score for robustness.
        self._observe_delta(float(delta))
        if self._delta_moments.count >= 3 and self._delta_moments.stddev > 0:
            if self._delta_moments.kurtosis > 3.0:
                # Heavy-tailed: use MAD-based robust z-score
                mad = self._median_absolute_deviation()
                if mad > 0:
                    z = abs(delta - self._delta_moments.mean) / (mad * 1.4826)
                else:
                    z = abs(self._delta_moments.z_score(delta))
            else:
                z = abs(self._delta_moments.z_score(delta))
            has_effect = (
                z > self.z_score_threshold or bool(entry.new_edges) or bool(entry.lost_edges)
            )
        else:
            # Too few observations for z-score — fall back to nonzero check
            has_effect = delta != 0 or bool(entry.new_edges) or bool(entry.lost_edges)
        return has_effect

    def _track_values(self, offset: int, width: int, input_bytes: bytes):
        """Track per-position byte values inside each covered hypothesis.

        value_counts maps relative position -> byte value -> count.  The seed
        generator uses this to emit structurally valid seeds from learned
        field content.  Only bytes inside an existing hypothesis are tracked
        — values outside known fields are ignored.
        """
        if offset < 0 or offset >= len(input_bytes):
            return

        end = min(offset + width, len(input_bytes))
        for pos in range(offset, end):
            byte_val = input_bytes[pos]
            # Find which hypothesis owns this byte position
            h = self._covering(pos)
            if h is None:
                continue

            pos_counts = h.value_counts.setdefault(pos - h.offset, {})
            pos_counts[byte_val] = pos_counts.get(byte_val, 0) + 1

    def record_liveness(self, offset: int, width: int, confirmed_dead: bool) -> None:
        """Corroborating evidence from item 4's `LiveBitMaskEstimator`
        (`core/live_bit_mask.py`, per
        `docs/handover/handover_done_2026-09-06.md`).

        A byte range whose liveness estimator has *converged* with an
        empty mask -- i.e. many consecutive mutations touching it never
        once moved coverage -- is a cheap signal that the range is
        padding, alignment filler, or an unparsed/ignored region. This is
        corroborating evidence only, not a replacement for the
        coverage-delta-driven classification `_update_hypotheses` already
        does:

        - If a hypothesis already exists for this offset, it was created
          because a mutation there *did* show a coverage effect
          (hypotheses are only created `elif has_effect` in
          `_update_hypotheses`). A later confirmed-dead verdict for the
          same range therefore *contradicts* that hypothesis rather than
          confirming it -- treated here as a small confidence penalty,
          not silently overwritten, since the two signals disagreeing is
          itself useful information (e.g. the field was live early in the
          run and stopped mattering, or the two evidence sources are
          looking at genuinely different byte spans that happen to
          overlap).
        - If no hypothesis exists yet, coverage-delta evidence alone
          cannot distinguish "genuinely dead" from "hasn't been tried
          enough yet" -- exactly the gap `LiveBitMaskEstimator.is_converged`
          is designed to close (see that module's docstring). Only then is
          a new, low-confidence `field_type="padding"` hypothesis created.

        `confirmed_dead=False` is a no-op: this method only ever reports
        positive convergence, never "still unresolved" (that's simply not
        calling it).
        """
        if not confirmed_dead:
            return

        h = self._covering(offset)
        if h is not None:
            if h.field_type != "padding":
                h.confidence = max(0.0, h.confidence - 0.05)
                # Re-type now; no stride boost, liveness is not field evidence.
                if h.observations >= _MIN_CLASSIFY_OBS:
                    self._classify(h)
            return

        h = FieldHypothesis(
            offset=offset,
            width=max(width, 1),
            field_type="padding",
            confidence=0.2,
            observations=0,
        )
        self._add_hypothesis(h)

    def _classify_fields(self):
        """Full pass: boost and classify every field with enough observations."""
        for h in self.hypotheses:
            if h.observations < _MIN_CLASSIFY_OBS:
                continue

            self._boost_stride(h)
            self._classify(h)

    def _boost_stride(self, h: FieldHypothesis) -> None:
        """Structural prior: stride-aligned hypotheses are likelier real fields."""
        stride = self.record_stride
        if stride and (h.width == stride or h.offset % stride == 0):
            h.confidence = min(1.0, h.confidence + 0.05)

    @staticmethod
    def _classify(h: FieldHypothesis) -> None:
        """Derive *h*'s field type from its own evidence."""
        if h.offset == 0 and h.confidence > 0.5:
            h.field_type = "magic"
            return
        if len(h.controlled_edges) > 5 and h.confidence > 0.4:
            h.field_type = "length"
            return
        if len(h.sensitive_ops) > 3 and h.confidence > 0.3:
            h.field_type = "crc"
            return
        if h.observations > 10 and h.confidence > 0.2:
            h.field_type = "data"
            return
        h.field_type = "unknown"

    def _observe_delta(self, delta: float) -> None:
        """Feed one coverage delta to the moments and the MAD histogram."""
        self._delta_moments.update(delta)
        counts = self._delta_counts
        counts[delta] = counts.get(delta, 0) + 1

    def _median_absolute_deviation(self) -> float:
        """MAD of coverage deltas — robust alternative to stddev under heavy tails.

        Upper median (rank n//2) of the deltas, then of |delta - median|,
        read off the histogram: O(D log D) for D distinct deltas, not
        O(n log n) over the whole history. E.g. {0: 40, 1: 1, 12: 1}:
        median 0, deviations {0: 40, 1: 1, 12: 1}, MAD 0.
        """
        counts = self._delta_counts
        n = sum(counts.values())
        if n < 3:
            return 0.0

        rank = n // 2
        median = _rank_value(sorted(counts.items()), rank)

        devs: dict[float, int] = {}
        for v, cnt in counts.items():
            d = abs(v - median)
            devs[d] = devs.get(d, 0) + cnt
        return _rank_value(sorted(devs.items()), rank)

    def backtest(self) -> tuple[bool, str | None]:
        """Replay this cluster's ENTIRE Timeline through its current format model."""
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
        # Zero depth: no field covers offset, skip the span scan.
        cover = self._synced_cover()
        covered = offset is None or offset < 0 or not cover.sparse or cover.depth(offset)
        for h in self.hypotheses if covered else ():
            if h.offset <= offset < h.offset + h.width and h.confidence > 0.3:
                return True
        for h in self.hypotheses:
            if entry.mutation_op in h.sensitive_ops and h.confidence > 0.3:
                return True
        return None

    def suggest_discriminating_mutation(self, candidates: list[str]) -> tuple[str, int] | None:
        """Suggest a mutation that would discriminate between hypotheses."""
        if len(self.hypotheses) < 2:
            return None

        for i, h1 in enumerate(self.hypotheses):
            for h2 in self.hypotheses[i + 1 :]:
                if h1.field_type != h2.field_type:
                    for op in candidates:
                        if op in h1.sensitive_ops and op not in h2.sensitive_ops:
                            return (op, h1.offset)
                        if op not in h1.sensitive_ops and op in h2.sensitive_ops:
                            return (op, h2.offset)
        return None

    def get_format_summary(self) -> dict:
        """Return a summary of this cluster's inferred format structure."""
        sorted_hyps = sorted(self.hypotheses, key=lambda h: h.offset)
        fields = []
        for h in sorted_hyps:
            # Overall most common byte value across all positions in this field
            most_common_val = None
            if h.value_counts:
                all_counts: dict[int, int] = {}
                for pos_counts in h.value_counts.values():
                    for bv, cnt in pos_counts.items():
                        all_counts[bv] = all_counts.get(bv, 0) + cnt
                if all_counts:
                    most_common_val = max(all_counts.items(), key=lambda kv: kv[1])[0]
            fields.append(
                {
                    "offset": h.offset,
                    "width": h.width,
                    "type": h.field_type,
                    "confidence": round(h.confidence, 3),
                    "observations": h.observations,
                    "sensitive_ops": dict(h.sensitive_ops),
                    "controlled_edges": len(h.controlled_edges),
                    "most_common_value": most_common_val,
                }
            )

        return {
            "signature": self.signature,
            "sample_count": self.total_observations,
            "timeline_size": len(self.timeline),
            "hypotheses": len(self.hypotheses),
            "classified": sum(1 for h in self.hypotheses if h.field_type != "unknown"),
            "backtest_passes": self.backtest_passes,
            "backtest_fails": self.backtest_fails,
            "model_version": self.format_model_version,
            "record_stride": self.record_stride,
            "fields": fields,
        }

    def get_learned_value(self, offset: int, width: int) -> bytes | None:
        """Return the most commonly observed bytes for a field range.

        Returns bytes of length `width` with the most frequent byte at
        each position within the range, or None if no value data exists.
        """
        if not self.hypotheses:
            return None

        # Find hypothesis covering this range
        covering = None
        for h in self.hypotheses:
            if h.offset <= offset and offset + width <= h.offset + h.width:
                covering = h
                break

        if covering is None or not covering.value_counts:
            return None

        # For each position within the field, pick the most common byte value
        result = bytearray()
        for rel_pos in range(width):
            counts = covering.value_counts.get(rel_pos)
            if not counts:
                result.append(0)
            else:
                most_common = max(counts.items(), key=lambda kv: kv[1])[0]
                result.append(most_common)
        return bytes(result)

    def get_state(self) -> dict:
        """Serialize this cluster for persistence."""
        return {
            "signature": self.signature,
            "sample_count": self.total_observations,
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
                    "value_counts": dict(h.value_counts),
                }
                for h in self.hypotheses
            ],
            "backtest_passes": self.backtest_passes,
            "backtest_fails": self.backtest_fails,
            "model_version": self.format_model_version,
            "record_stride": self.record_stride,
        }

    @classmethod
    def from_state(cls, signature: str, state: dict, z_score_threshold: float = 2.0):
        """Reconstruct a cluster from `get_state()` output."""
        cluster = cls(signature=signature, z_score_threshold=z_score_threshold)
        for e in state.get("timeline", []):
            cluster.timeline.append(
                TimelineEntry(
                    input_hash=e.get("input_hash", ""),
                    mutation_op=e["op"],
                    mutation_offset=e["offset"],
                    mutation_width=e["width"],
                    coverage_before=e["cov_before"],
                    coverage_after=e["cov_after"],
                    new_edges=set(e.get("new_edges", [])),
                    lost_edges=set(e.get("lost_edges", [])),
                )
            )
        for h in state.get("hypotheses", []):
            hyp = FieldHypothesis(
                offset=h["offset"],
                width=h["width"],
                field_type=h["type"],
                confidence=h["confidence"],
                observations=h["observations"],
                sensitive_ops=h.get("sensitive_ops", {}),
                controlled_edges=set(h.get("controlled_edges", [])),
                value_counts=dict(h.get("value_counts", {})),
            )
            cluster._add_hypothesis(hyp)
        cluster.backtest_passes = state.get("backtest_passes", 0)
        cluster.backtest_fails = state.get("backtest_fails", 0)
        cluster.format_model_version = state.get("model_version", 0)
        cluster.record_stride = state.get("record_stride")
        cluster.total_observations = state.get("sample_count", len(cluster.timeline))
        return cluster


class FormatLearner:
    """Induces format structure from fuzzing observations.

    Follows the schema-harness methodology:
    - State grounding: infers field boundaries from mutation sensitivity
    - Mechanism discovery: finds how fields control coverage paths
    - Backtesting: validates hypotheses against full Timeline
    - Action for discovery: selects mutations that discriminate hypotheses

    Observations are routed into per-format `FormatCluster`s, keyed by an
    input signature (`sig_len` bytes of prefix, by default). A target that
    demuxes several container formats therefore accumulates several
    concurrent, mutually uncontradicted field maps instead of one map that
    silently averages incompatible formats together. `max_formats` bounds
    how many clusters are tracked at once; the least-observed cluster is
    evicted to make room for a new signature once the cap is hit, so a
    stream of one-off garbage prefixes can't crowd out real formats.

    A signature only gets its own cluster once it has recurred
    `promote_threshold` times; until then, observations under it fall
    into the shared default cluster. This matters because the signature
    window can itself be the byte range a mutation (or synthetic value)
    is varying — a real recurring format (repeatedly re-mutated from the
    same seed, header usually untouched) clears the threshold easily,
    while a prefix that never repeats twice never earns its own cluster
    and behaves exactly like the old single-format learner.

    For callers that only care about "the" format, `.hypotheses`,
    `.timeline`, `.field_map`, `.backtest_passes`/`.backtest_fails`, and
    `.format_model_version` are read/write views onto the *primary*
    cluster — the one with the most recorded observations — so this stays
    drop-in compatible with single-format use.
    """

    def __init__(
        self,
        max_timeline: int = 5000,
        z_score_threshold: float = 2.0,
        max_formats: int = DEFAULT_MAX_FORMATS,
        sig_len: int = DEFAULT_SIG_LEN,
        promote_threshold: int = DEFAULT_PROMOTE_THRESHOLD,
    ):
        self.max_timeline = max_timeline
        self.z_score_threshold = z_score_threshold
        self.max_formats = max_formats
        self.sig_len = sig_len
        self.promote_threshold = promote_threshold
        self.clusters: dict[str, FormatCluster] = {}
        # Signature of the most recently recorded transition — used as the
        # routing target for calls (like record_liveness) that don't carry
        # their own input_bytes.
        self._last_signature: str | None = None
        # Raw-signature recurrence counts, for promotion — not full
        # clusters, just a cheap int per candidate signature.
        self._signature_counts: dict[str, int] = {}
        # Small buffer of (entry, input_bytes) per not-yet-promoted raw
        # signature, capped at promote_threshold — replayed into a fresh
        # dedicated cluster the moment that signature is promoted, so a
        # recurring format's *first* few observations aren't stranded in
        # the shared default cluster once it earns its own.
        self._pending: dict[str, list[tuple[TimelineEntry, bytes]]] = {}

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    @staticmethod
    def format_signature(input_bytes: bytes, sig_len: int = DEFAULT_SIG_LEN) -> str:
        """Cluster key for an input: hex of its first `sig_len` bytes.

        This mirrors how a real demuxer dispatches — on a magic number at
        (or near) offset 0 — so inputs that a target would itself route to
        different format-specific parsing code end up in different
        clusters. It's a heuristic, not ground truth: a mutation that
        happens to land inside the signature window can still split one
        real format across clusters, which is exactly why clusters are
        cheap and eviction-bounded rather than assumed authoritative.
        """
        if not input_bytes:
            return DEFAULT_SIGNATURE
        return input_bytes[:sig_len].hex()

    def _signature_for(self, input_bytes: bytes | None) -> str | None:
        if input_bytes is not None:
            return self.format_signature(input_bytes, self.sig_len)
        return self._last_signature

    def _bound_signature_tracking(self):
        """Cheap, approximate forgetting so `_signature_counts`/`_pending`
        can't grow without limit over a long campaign full of one-off
        prefixes that never recur."""
        if len(self._signature_counts) < MAX_TRACKED_SIGNATURES:
            return
        stalest = sorted(self._signature_counts.items(), key=lambda kv: kv[1])
        for sig, _ in stalest[: len(stalest) // 4 or 1]:
            del self._signature_counts[sig]
            self._pending.pop(sig, None)

    def _known_cluster_for(self, raw_signature: str) -> str:
        """Route a signature that's only being *read* (or is a secondary
        signal, not itself a Timeline observation): use its own cluster if
        one already exists, otherwise the shared default — never triggers
        promotion, since there's no observation here to buffer."""
        if raw_signature in self.clusters:
            return raw_signature
        return DEFAULT_SIGNATURE

    def _get_or_create_cluster(self, signature: str) -> FormatCluster:
        cluster = self.clusters.get(signature)
        if cluster is not None:
            return cluster
        if len(self.clusters) >= self.max_formats:
            # Never evict the shared default cluster — it's the fallback
            # every not-yet-promoted signature depends on.
            candidates = [c for c in self.clusters.values() if c.signature != DEFAULT_SIGNATURE]
            evicted = min(candidates or self.clusters.values(), key=lambda c: c.total_observations)
            del self.clusters[evicted.signature]
            log.debug(
                "format_learner: evicting cluster %r (%d obs) for new signature %r",
                evicted.signature,
                evicted.total_observations,
                signature,
            )
        cluster = FormatCluster(signature=signature, z_score_threshold=self.z_score_threshold)
        self.clusters[signature] = cluster
        return cluster

    @property
    def primary_cluster(self) -> FormatCluster | None:
        """The cluster with the most recorded observations, or None."""
        if not self.clusters:
            return None
        return max(self.clusters.values(), key=lambda c: c.total_observations)

    # ------------------------------------------------------------------
    # Backward-compatible single-format view (reads/writes primary cluster)
    # ------------------------------------------------------------------
    @property
    def hypotheses(self) -> list[FieldHypothesis]:
        # Returns the primary cluster's live list (not a copy), so
        # `fl.hypotheses.append(...)` on a fresh learner persists — it
        # lands in (and creates, if needed) the default cluster.
        c = self.primary_cluster or self._get_or_create_cluster(
            self._last_signature or DEFAULT_SIGNATURE
        )
        return c.hypotheses

    @hypotheses.setter
    def hypotheses(self, value: list[FieldHypothesis]):
        self._get_or_create_cluster(self._last_signature or DEFAULT_SIGNATURE).hypotheses = value

    @property
    def field_map(self) -> dict[int, FieldHypothesis]:
        c = self.primary_cluster or self._get_or_create_cluster(
            self._last_signature or DEFAULT_SIGNATURE
        )
        return c.field_map

    @field_map.setter
    def field_map(self, value: dict[int, FieldHypothesis]):
        self._get_or_create_cluster(self._last_signature or DEFAULT_SIGNATURE).field_map = value

    @property
    def timeline(self) -> list[TimelineEntry]:
        c = self.primary_cluster
        return c.timeline if c is not None else []

    @property
    def format_model_version(self) -> int:
        c = self.primary_cluster
        return c.format_model_version if c is not None else 0

    @property
    def backtest_passes(self) -> int:
        c = self.primary_cluster
        return c.backtest_passes if c is not None else 0

    @property
    def backtest_fails(self) -> int:
        c = self.primary_cluster
        return c.backtest_fails if c is not None else 0

    @property
    def record_stride(self) -> int | None:
        c = self.primary_cluster
        return c.record_stride if c is not None else None

    @property
    def _delta_moments(self) -> RunningMoments:
        c = self.primary_cluster
        return c._delta_moments if c is not None else RunningMoments()

    def _classify_fields(self):
        c = self.primary_cluster
        if c is not None:
            c._classify_fields()

    def backtest(self) -> tuple[bool, str | None]:
        c = self.primary_cluster
        if c is None:
            return True, None
        return c.backtest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_record_stride(self, stride: int | None, input_bytes: bytes | None = None):
        """Set the record-stride structural prior for one format cluster.

        Routes by `input_bytes`'s signature when given (each format keeps
        its own stride), otherwise applies to the most recently recorded
        cluster.
        """
        if input_bytes is not None:
            sig = self._known_cluster_for(self.format_signature(input_bytes, self.sig_len))
        else:
            sig = self._last_signature or DEFAULT_SIGNATURE
        self._get_or_create_cluster(sig).set_record_stride(stride)

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
        """Append a real transition to the appropriate format cluster's Timeline.

        Stores only the input hash, not full bytes, to save memory. Which
        cluster this lands in is decided by `input_bytes`'s signature (see
        `format_signature`) — except a signature seen for the first time
        isn't immediately trusted with a cluster of its own: it's buffered
        and, if it recurs `promote_threshold` times, promoted to a fresh
        cluster built by replaying the buffer. Until (and unless) that
        happens, it's folded into the shared default cluster, so a
        signature that never recurs still contributes its observation
        instead of being lost.
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
        raw_signature = self.format_signature(input_bytes, self.sig_len)

        if raw_signature == DEFAULT_SIGNATURE or raw_signature in self.clusters:
            self._last_signature = raw_signature
            self._get_or_create_cluster(raw_signature).record(entry, input_bytes, self.max_timeline)
            return

        self._bound_signature_tracking()
        pending = self._pending.setdefault(raw_signature, [])
        pending.append((entry, input_bytes))
        if len(pending) > self.promote_threshold:
            pending.pop(0)
        count = self._signature_counts.get(raw_signature, 0) + 1
        self._signature_counts[raw_signature] = count

        if count >= self.promote_threshold:
            # Promote: this signature has now recurred enough to earn its
            # own cluster. Replay everything buffered for it (in order)
            # into a fresh cluster, rather than starting from just this
            # one observation.
            del self._pending[raw_signature]
            cluster = self._get_or_create_cluster(raw_signature)
            for pend_entry, pend_bytes in pending:
                cluster.record(pend_entry, pend_bytes, self.max_timeline)
            self._last_signature = raw_signature
        else:
            # Still unproven — fold into the shared default cluster so
            # the observation isn't lost if this signature never recurs.
            self._get_or_create_cluster(DEFAULT_SIGNATURE).record(
                entry, input_bytes, self.max_timeline
            )
            self._last_signature = DEFAULT_SIGNATURE

    def record_liveness(
        self, offset: int, width: int, confirmed_dead: bool, input_bytes: bytes | None = None
    ) -> None:
        """Corroborating dead-region evidence, routed to a format cluster.

        See `FormatCluster.record_liveness` for the reasoning. Routes by
        `input_bytes`'s signature when given (only into an *already
        promoted* cluster — this alone never promotes a signature),
        otherwise by whichever cluster last recorded a transition
        (falling back to the default cluster if none has yet).
        """
        if not confirmed_dead:
            return
        if input_bytes is not None:
            sig = self._known_cluster_for(self.format_signature(input_bytes, self.sig_len))
        else:
            sig = self._last_signature or DEFAULT_SIGNATURE
        self._get_or_create_cluster(sig).record_liveness(offset, width, confirmed_dead=True)

    def suggest_discriminating_mutation(self, candidates: list[str]) -> tuple[str, int] | None:
        """Suggest a mutation that would discriminate between hypotheses
        within the primary format cluster."""
        c = self.primary_cluster
        if c is None:
            return None
        return c.suggest_discriminating_mutation(candidates)

    def get_format_summary(self) -> dict:
        """Return a summary of the inferred format structure.

        Top-level keys mirror the single-format shape (and describe the
        primary cluster, for backward compatibility). `format_count` and
        `formats` additionally expose every tracked cluster, ranked by
        observation count, for callers that want to reason about — or
        seed from — more than one live format hypothesis.
        """
        if not self.clusters:
            return {
                "timeline_size": 0,
                "hypotheses": 0,
                "classified": 0,
                "backtest_passes": 0,
                "backtest_fails": 0,
                "model_version": 0,
                "record_stride": None,
                "fields": [],
                "format_count": 0,
                "formats": [],
            }

        ranked = sorted(self.clusters.values(), key=lambda c: -c.total_observations)
        summary = ranked[0].get_format_summary()
        summary["format_count"] = len(self.clusters)
        summary["formats"] = [c.get_format_summary() for c in ranked]
        return summary

    def get_learned_value(
        self, offset: int, width: int, input_bytes: bytes | None = None
    ) -> bytes | None:
        """Return the most commonly observed bytes for a field range.

        When `input_bytes` is given, reads from the cluster matching its
        signature (so a caller building a seed for a specific format gets
        that format's learned values, not whichever cluster happens to be
        primary). Falls back to the primary cluster otherwise.
        """
        sig = self._signature_for(input_bytes)
        cluster = self.clusters.get(sig) if sig is not None else None
        if cluster is None:
            cluster = self.primary_cluster
        if cluster is None:
            return None
        return cluster.get_learned_value(offset, width)

    def get_state(self) -> dict:
        """Serialize for persistence.

        Top-level keys mirror the primary cluster's own `get_state()` for
        backward compatibility with consumers (e.g. `format_seed_generator`)
        that only ever knew about one format. `clusters` carries every
        tracked format's full state for anyone that wants it.
        """
        primary = self.primary_cluster
        base = (
            primary.get_state()
            if primary is not None
            else FormatCluster(signature=DEFAULT_SIGNATURE).get_state()
        )
        base["sig_len"] = self.sig_len
        base["max_formats"] = self.max_formats
        base["primary_signature"] = primary.signature if primary is not None else None
        base["clusters"] = {sig: c.get_state() for sig, c in self.clusters.items()}
        return base

    def load_state(self, state: dict):
        """Restore from persistence.

        Accepts both the current multi-cluster shape (a `clusters` key)
        and a legacy single-format dump (no `clusters` key), which loads
        entirely into the default cluster.
        """
        self.clusters = {}
        clusters_state = state.get("clusters")
        if clusters_state:
            self.sig_len = state.get("sig_len", self.sig_len)
            self.max_formats = state.get("max_formats", self.max_formats)
            for sig, cstate in clusters_state.items():
                self.clusters[sig] = FormatCluster.from_state(sig, cstate, self.z_score_threshold)
            self._last_signature = state.get("primary_signature") or next(iter(self.clusters), None)
        else:
            cluster = FormatCluster.from_state(DEFAULT_SIGNATURE, state, self.z_score_threshold)
            self.clusters[DEFAULT_SIGNATURE] = cluster
            self._last_signature = DEFAULT_SIGNATURE
