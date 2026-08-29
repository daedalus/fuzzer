"""Regression tests for the deterministic fuzzing stage.

Before this stage existed, AFL's deterministic operators (bitflip,
byte-flip, arithmetic, interesting-value substitution) were one-shot
handlers in operators.py, and core/skipdet.py's SkipDetector was fully
implemented but never called from anywhere -- nothing walked the operators
systematically across a seed, and nothing decided which seeds deserved that
treatment.

A first pass at wiring SkipDetector in landed directly in fuzzer.py as
Fuzzer._run_deterministic_stage(): a standalone blocking loop that ran
executions and updated edge-tracking bookkeeping on its own. These tests
pin the merged, corrected version instead:

- OperatorEngine.maybe_deterministic_mutation() (operators.py) queues a
  real per-position walk (_deterministic_mutation_stream: true bitflip
  1/1 across every bit, not one random bit per byte) and drains it from
  inside mutate() -- so deterministic-stage mutants go through the exact
  same execution/coverage/save_to_corpus path as every other mutation,
  instead of a parallel loop that finds new coverage and then discards the
  mutant that found it (_run_deterministic_stage never called
  save_to_corpus).
- core.skipdet.trace_mini_from_edges() folds edge_id % map_size instead of
  indexing by edge_id // 8 directly -- the original indexed almost every
  real (hash-valued) edge_id out of bounds and silently dropped it via the
  `idx < len(trace_mini)` guard, so should_det_fuzz rarely saw any
  undetermined bits regardless of a seed's actual coverage.
"""

import tempfile
from pathlib import Path

from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS, INTERESTING_UNSIGNED_8
from fuzzer_tool.core.skipdet import SkipDetector, trace_mini_from_edges
from fuzzer_tool.services.operators import _deterministic_mutation_stream

TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


def _build_fuzzer(seed: bytes, deterministic: bool = True):
    """Build a Fuzzer with one seed on disk."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.mkdtemp()
    corpus = Path(tmp) / "corpus"
    crashes = Path(tmp) / "crashes"
    (corpus / "seeds").mkdir(parents=True)
    crashes.mkdir()
    (corpus / "seeds" / "seed1").write_bytes(seed)
    return Fuzzer(
        target=TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        deterministic=deterministic,
    )


def _mark_favored(fuzzer, seed: bytes, edge_ids=frozenset({1, 2, 3})):
    seed_key = fuzzer._seed_key(seed)
    fuzzer._favored = {seed_key}
    fuzzer._edge_tracker.seed_edges[seed_key] = set(edge_ids)
    return seed_key


def _expected_mutation_count(data: bytes) -> int:
    return (
        len(data) * 8  # bitflip 1/1
        + len(data)  # byte flip 8/8
        + len(data) * len(ARITHMETIC_DELTAS) * 2  # arithmetic +/-
        + len(data) * len(INTERESTING_UNSIGNED_8)  # interesting 8-bit
    )


class TestDeterministicMutationStream:
    """The generator that walks operators systematically across a seed."""

    def test_mutation_count_matches_afl_schedule(self):
        data = b"ABCD"
        muts = list(_deterministic_mutation_stream(data, max_mutations=10_000))
        assert len(muts) == _expected_mutation_count(data)

    def test_every_mutant_differs_from_original(self):
        data = b"ABCD"
        muts = list(_deterministic_mutation_stream(data, max_mutations=10_000))
        assert all(m != data for m in muts)
        assert all(len(m) == len(data) for m in muts)

    def test_respects_max_mutations_cap(self):
        data = b"A" * 100
        muts = list(_deterministic_mutation_stream(data, max_mutations=50))
        assert len(muts) == 50

    def test_empty_data_yields_nothing(self):
        assert list(_deterministic_mutation_stream(b"", max_mutations=1000)) == []


class TestTraceMiniFromEdges:
    """Sparse edge_id set -> positional bitmap for SkipDetector.

    Regression coverage for the indexing bug in the first landed version
    (Fuzzer._get_seed_trace_mini): it used `idx = edge_id // 8` directly and
    dropped anything landing outside the bitmap, which for hash-valued
    edge_ids meant nearly everything. trace_mini_from_edges folds by
    map_size instead, so no edge is ever silently dropped.
    """

    def test_bitmap_size_matches_map_size(self):
        sd = SkipDetector(map_size=65536)
        tm = trace_mini_from_edges({1, 2, 3}, sd.map_size)
        assert len(tm) == sd.map_size // 8

    def test_large_hash_valued_edge_ids_still_set_bits(self):
        """A first-cut `edge_id // 8` indexer would silently drop these."""
        sd = SkipDetector(map_size=65536)
        big_edge_ids = {0xDEADBEEF, 0x1234_5678_9ABC, 2**40 + 7}
        tm = trace_mini_from_edges(big_edge_ids, sd.map_size)
        assert any(byte != 0 for byte in tm)

    def test_gate_fires_once_then_suppresses_same_edges(self):
        sd = SkipDetector(map_size=65536)
        tm = trace_mini_from_edges({100, 200, 300}, sd.map_size)
        assert sd.should_det_fuzz(tm, True, False, 0) is True
        # Same edges again: those bits are now marked explored.
        assert sd.should_det_fuzz(tm, True, False, 0) is False

    def test_unfavored_seed_never_runs(self):
        sd = SkipDetector(map_size=65536)
        tm = trace_mini_from_edges({1, 2, 3}, sd.map_size)
        assert sd.should_det_fuzz(tm, False, False, 0) is False

    def test_already_passed_det_never_reruns(self):
        sd = SkipDetector(map_size=65536)
        tm = trace_mini_from_edges({1, 2, 3}, sd.map_size)
        assert sd.should_det_fuzz(tm, True, True, 0) is False


class TestDeterministicStageWiring:
    """End-to-end: OperatorEngine draining a real seed's queue."""

    def test_enabled_by_default(self):
        """Matches already-shipped behavior: on unless --no-deterministic."""
        fuzzer = _build_fuzzer(b"ABCDEFGH")
        assert fuzzer._skip_detector is not None

    def test_disabled_via_deterministic_false(self):
        fuzzer = _build_fuzzer(b"ABCDEFGH", deterministic=False)
        assert fuzzer._skip_detector is None
        assert fuzzer._operators.maybe_deterministic_mutation(b"ABCDEFGH") is None

    def test_non_favored_seed_never_runs_deterministic_stage(self):
        fuzzer = _build_fuzzer(b"ABCDEFGH")
        seed_key = fuzzer._seed_key(b"ABCDEFGH")
        fuzzer._edge_tracker.seed_edges[seed_key] = {1, 2, 3}
        # Deliberately not adding seed_key to fuzzer._favored.
        assert fuzzer._operators.maybe_deterministic_mutation(b"ABCDEFGH") is None
        assert fuzzer.seed_meta[b"ABCDEFGH"]["seed_passed_det"] is True

    def test_favored_seed_drains_full_queue_then_stops(self):
        data = b"ABCDEFGH"
        fuzzer = _build_fuzzer(data)
        _mark_favored(fuzzer, data)

        mutants = []
        while True:
            m = fuzzer._operators.maybe_deterministic_mutation(data)
            if m is None:
                break
            mutants.append(m)

        expected = _expected_mutation_count(data)
        assert len(mutants) == expected
        assert all(m != data for m in mutants)
        assert fuzzer.seed_meta[data]["seed_passed_det"] is True
        # Once passed_det, further calls are a no-op, not a fresh pass.
        assert fuzzer._operators.maybe_deterministic_mutation(data) is None

    def test_mutate_drains_deterministic_queue_before_havoc(self):
        data = b"ABCDEFGH"
        fuzzer = _build_fuzzer(data)
        _mark_favored(fuzzer, data)
        expected = _expected_mutation_count(data)

        det_calls = 0
        havoc_calls = 0
        for _ in range(expected + 20):
            fuzzer.mutate(data)
            if fuzzer._last_ops_used == []:
                det_calls += 1
            else:
                havoc_calls += 1

        assert det_calls == expected
        assert havoc_calls == 20
        assert fuzzer._det_execs == expected

    def test_seed_meta_keyed_by_raw_bytes_not_seed_key(self):
        """seed_meta is keyed by the raw seed bytes, not the seed_key hash
        string that _favored / _edge_tracker.seed_edges use. Looking it up
        by seed_key instead would make every lookup miss and silently
        disable the whole gate."""
        data = b"ABCDEFGH"
        fuzzer = _build_fuzzer(data)
        seed_key = fuzzer._seed_key(data)
        assert seed_key not in fuzzer.seed_meta
        assert data in fuzzer.seed_meta

    def test_deterministic_mutant_reaches_corpus_via_fuzz_one(self):
        """Regression for the corpus-loss bug in the first landed version:
        Fuzzer._run_deterministic_stage ran real executions and updated
        edge-tracking bookkeeping directly, but never called
        save_to_corpus (only fuzz_one's own branches do), so any mutant it
        found interesting was discarded on the spot.

        Routing deterministic mutants through mutate() -> fuzz_one() means
        they hit the same save_to_corpus branch as every other mutation.
        Forcing _is_interesting to always return True isolates that wiring
        without depending on this target's (uninstrumented) coverage.
        """
        data = b"ABCDEFGH"
        fuzzer = _build_fuzzer(data)
        _mark_favored(fuzzer, data)
        fuzzer._is_interesting = lambda returncode, stderr: True

        before = len(fuzzer.corpus)
        fuzzer.fuzz_one(data)
        after = len(fuzzer.corpus)

        assert fuzzer._last_ops_used == [], (
            "first fuzz_one() call should draw a deterministic mutant"
        )
        assert after > before, "the deterministic mutant fuzz_one() found interesting must be saved"
