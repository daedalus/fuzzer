"""The byteflip 8/8 pass must build an effector map, and it must fail open.

AFL gates the arithmetic and interesting-value passes -- 24 of the 33 mutants
per byte -- on a map built during the byteflip pass it has already paid for.
This tree ran all 33 unconditionally while observing, on every one of those
executions, the coverage that answers the question.

The dangerous failure here is not under-skipping, it is over-skipping: a map
that reads "inert" for a byte that was simply never probed deletes two whole
passes at that position on no evidence. Half the cases below exist to pin
that direction.
"""

import tempfile
from pathlib import Path

from fuzzer_tool.core.mutations import ARITHMETIC_DELTAS, INTERESTING_UNSIGNED_8
from fuzzer_tool.services.operators import (
    _DET_EFF_INERT,
    _DET_EFF_LIVE,
    _DET_EFF_UNKNOWN,
    DeterministicEffectorMap,
    _deterministic_mutation_stream,
)

N_ARITH = len(ARITHMETIC_DELTAS) * 2
N_INTERESTING = len(INTERESTING_UNSIGNED_8)
PER_BYTE = 8 + 1 + N_ARITH + N_INTERESTING  # 33

# Bytes chosen so that no mutant can coincide with the original: 0x41..0x41+n
# is disjoint from INTERESTING_UNSIGNED_8, and every arithmetic delta is
# non-zero, so "the index where mutant differs from data" is always defined.
def _seed(length: int) -> bytes:
    return bytes((0x41 + i) & 0xFF for i in range(length))


def _diff_index(data: bytes, mutant: bytes) -> int:
    for i, (a, b) in enumerate(zip(data, mutant, strict=True)):
        if a != b:
            return i
    raise AssertionError("mutant is identical to the seed")


def _drive(data: bytes, live: set[int], cap: int = 10**9, drop: set[int] | None = None):
    """Consume the stream the way the provider does.

    *live* is ground truth: byteflipping one of those positions changes the
    trace. *drop* names byteflip positions whose mutant is discarded before
    execution, which is what _dedup_mutate does when the exec bloom fires --
    those must never be recorded.

    Returns (all mutants, mutants attributed to the arithmetic and
    interesting passes, the effector map).
    """
    effector = DeterministicEffectorMap(len(data))
    stream = _deterministic_mutation_stream(data, cap, effector=effector)
    drop = drop or set()
    mutants, tail = [], []
    n_head = 0
    for mutant in stream:
        pending = effector.pending
        effector.pending = -1
        mutants.append(mutant)
        if pending >= 0:
            n_head += 1
            if pending not in drop:
                effector.eff[pending] = _DET_EFF_LIVE if pending in live else _DET_EFF_INERT
        elif n_head:
            # Past the byteflip pass: everything left is arith/interesting.
            tail.append(mutant)
    return mutants, tail, effector


class TestGating:
    def test_arith_and_interesting_only_touch_live_bytes(self):
        data = _seed(64)
        live = {3, 17, 40}
        _mutants, tail, _eff = _drive(data, live)
        assert {_diff_index(data, m) for m in tail} == live

    def test_total_cost_is_exactly_nine_per_byte_plus_gated_tail(self):
        data = _seed(64)
        live = {3, 17, 40}
        mutants, _tail, _eff = _drive(data, live)
        expected = (8 + 1) * len(data) + (N_ARITH + N_INTERESTING) * len(live)
        assert len(mutants) == expected
        # For reference: the ungated schedule is 33 per byte.
        assert len(list(_deterministic_mutation_stream(data))) == PER_BYTE * len(data)

    def test_map_records_both_verdicts(self):
        data = _seed(16)
        _m, _t, eff = _drive(data, {2, 5})
        assert [i for i, v in enumerate(eff.eff) if v == _DET_EFF_LIVE] == [2, 5]
        assert all(v == _DET_EFF_INERT for i, v in enumerate(eff.eff) if i not in (2, 5))


class TestFailsOpen:
    def test_no_effector_reproduces_the_ungated_schedule(self):
        # The parameter must be inert when absent: a seeded run without an
        # effector has to be byte-identical to one from before it existed.
        for length in (1, 7, 64, 129):
            for cap in (10**9, 500, 47, 3):
                data = _seed(length)
                assert list(_deterministic_mutation_stream(data, cap)) == list(
                    _deterministic_mutation_stream(data, cap, effector=None)
                )

    def test_effector_that_is_never_told_anything_changes_nothing(self):
        # Every position stays UNKNOWN, so no position may be skipped. This
        # is the shape of a campaign where the shim has no rolling path hash
        # or the seed has no calibrated baseline.
        for length in (8, 64):
            data = _seed(length)
            eff = DeterministicEffectorMap(length)
            gated = list(_deterministic_mutation_stream(data, 10**9, effector=eff))
            assert gated == list(_deterministic_mutation_stream(data, 10**9))
            assert all(v == _DET_EFF_UNKNOWN for v in eff.eff)

    def test_all_inert_is_treated_as_a_broken_measurement(self):
        # A map claiming no byte of the seed is read is evidence of an
        # unstable trace, not of a seed the parser ignores. Deleting both
        # remaining passes on that basis is the expensive mistake.
        data = _seed(32)
        mutants, _tail, _eff = _drive(data, live=set())
        assert len(mutants) == PER_BYTE * len(data)

    def test_positions_dropped_before_execution_keep_their_schedule(self):
        # _dedup_mutate draws a mutant, finds it in the exec bloom, and draws
        # another. The discarded position was never executed, so it must stay
        # UNKNOWN rather than being recorded as inert.
        data = _seed(32)
        live = {4}
        dropped = {9, 21}
        _mutants, tail, eff = _drive(data, live, drop=dropped)
        assert all(eff.eff[i] == _DET_EFF_UNKNOWN for i in dropped)
        assert {_diff_index(data, m) for m in tail} == live | dropped

    def test_positions_the_byteflip_quota_never_reached_keep_their_schedule(self):
        # With a cap that stops the byteflip pass early, the unprobed tail of
        # the seed must not be gated away.
        data = _seed(128)
        effector = DeterministicEffectorMap(len(data))
        cap = 8 * len(data) + 20 + 400  # byteflip reaches ~20 of 128 positions
        stream = _deterministic_mutation_stream(data, cap, effector=effector)
        tail = []
        n_head = 0
        for mutant in stream:
            pending = effector.pending
            effector.pending = -1
            if pending >= 0:
                n_head += 1
                effector.eff[pending] = _DET_EFF_INERT  # every probed byte inert
            elif n_head:
                tail.append(mutant)
        probed = [i for i, v in enumerate(effector.eff) if v != _DET_EFF_UNKNOWN]
        assert probed  # the pass did run, partially
        assert len(probed) < len(data)  # and did not finish
        touched = {_diff_index(data, m) for m in tail}
        assert touched  # the gated passes still ran
        assert not touched & set(probed)  # only on bytes never probed


class TestBudget:
    def test_never_exceeds_the_cap_and_never_costs_more_than_ungated(self):
        data = _seed(96)
        live = set(range(0, 96, 8))  # 12 live bytes
        for cap in (10**9, 3000, 1500, 964, 300, 47, 1):
            mutants, _tail, _eff = _drive(data, live, cap=cap)
            assert len(mutants) <= cap
            assert len(mutants) <= len(list(_deterministic_mutation_stream(data, cap)))

    def test_head_truncation_is_decided_before_the_map_exists(self):
        # Honest limitation, pinned so it is not mistaken for a bug later.
        # The up-front quota split is computed from the *ungated* cost,
        # because the effector map does not exist until the byteflip pass has
        # run. So a cap below the ungated schedule truncates the bitflip and
        # byteflip passes even when the gated schedule would have fitted
        # whole, and a truncated byteflip pass in turn leaves part of the map
        # UNKNOWN. Only the arithmetic / interesting split is re-decided.
        data = _seed(96)
        live = set(range(0, 96, 8))
        gated_total = 9 * len(data) + (N_ARITH + N_INTERESTING) * len(live)
        cap = gated_total + 200  # comfortably fits the gated schedule
        assert cap < PER_BYTE * len(data)  # but not the ungated one
        mutants, _tail, eff = _drive(data, live, cap=cap)
        assert len(mutants) <= cap
        assert any(v == _DET_EFF_UNKNOWN for v in eff.eff)

    def test_all_four_passes_survive_a_tight_cap(self):
        # The per-pass quota property the up-front split exists to hold must
        # survive the re-split: a prefix cap would delete later passes.
        data = _seed(96)
        live = set(range(0, 96, 8))
        cap = 8 * 96 + 96 + 40
        _mutants, tail, _eff = _drive(data, live, cap=cap)
        deltas = set(ARITHMETIC_DELTAS)
        saw_arith = any(
            (m[_diff_index(data, m)] - data[_diff_index(data, m)]) & 0xFF in deltas
            or (data[_diff_index(data, m)] - m[_diff_index(data, m)]) & 0xFF in deltas
            for m in tail
        )
        saw_interesting = any(m[_diff_index(data, m)] in INTERESTING_UNSIGNED_8 for m in tail)
        assert saw_arith
        assert saw_interesting


# ── engine-level plumbing ────────────────────────────────────────────────
#
# The stream is only half of it. The other half is that the verdict reaching
# note_deterministic_result belongs to the mutant that was actually executed.

TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


def _build_fuzzer(seed: bytes):
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmp = tempfile.mkdtemp()
    corpus = Path(tmp) / "corpus"
    crashes = Path(tmp) / "crashes"
    (corpus / "seeds").mkdir(parents=True)
    crashes.mkdir()
    (corpus / "seeds" / "seed1").write_bytes(seed)
    f = Fuzzer(
        target=TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        deterministic=True,
    )
    seed_key = f._seed_key(seed)
    f._favored = {seed_key}
    f._edge_tracker.seed_edges[seed_key] = {1, 2, 3}
    return f, seed_key


class TestEnginePlumbing:
    def test_verdicts_gate_the_drained_queue(self):
        data = _seed(16)
        f, seed_key = _build_fuzzer(data)
        live = {2, 11}
        mutants = []
        while True:
            m = f._operators.maybe_deterministic_mutation(data)
            if m is None:
                break
            mutants.append(m)
            pending = f._operators._det_pending
            if pending is not None:
                f._operators.note_deterministic_result(pending[1] in live)
        assert len(mutants) == 9 * len(data) + (N_ARITH + N_INTERESTING) * len(live)

    def test_a_second_draw_reassigns_the_pending_slot(self):
        # _dedup_mutate draws again when the exec bloom fires; only the last
        # mutant drawn is executed, so only its position may take the verdict.
        data = _seed(16)
        f, seed_key = _build_fuzzer(data)
        for _ in range(8 * len(data)):  # drain the bitflip pass
            f._operators.maybe_deterministic_mutation(data)
        f._operators.maybe_deterministic_mutation(data)
        first = f._operators._det_pending
        f._operators.maybe_deterministic_mutation(data)
        second = f._operators._det_pending
        assert first is not None and second is not None and first[1] != second[1]
        f._operators.note_deterministic_result(False)
        eff = f._operators._det_eff[seed_key].eff
        assert eff[second[1]] == _DET_EFF_INERT
        assert eff[first[1]] == _DET_EFF_UNKNOWN

    def test_note_without_a_pending_draw_is_a_no_op(self):
        data = _seed(8)
        f, seed_key = _build_fuzzer(data)
        f._operators.maybe_deterministic_mutation(data)  # a bitflip: no position
        assert f._operators._det_pending is None
        f._operators.note_deterministic_result(True)
        assert all(v == _DET_EFF_UNKNOWN for v in f._operators._det_eff[seed_key].eff)

    def test_effector_map_is_dropped_with_the_queue(self):
        data = _seed(8)
        f, seed_key = _build_fuzzer(data)
        while f._operators.maybe_deterministic_mutation(data) is not None:
            pending = f._operators._det_pending
            if pending is not None:
                f._operators.note_deterministic_result(True)
        assert seed_key not in f._operators._det_eff
        assert seed_key not in f._operators._det_queues

    def test_no_baseline_means_no_verdict(self):
        # _note_det_effector must stay silent when the seed has no recorded
        # path hash: an unfillable map leaves every byte UNKNOWN, which keeps
        # the full schedule.
        data = _seed(16)
        f, seed_key = _build_fuzzer(data)
        for _ in range(8 * len(data) + 1):
            f._operators.maybe_deterministic_mutation(data)
        assert f._operators._det_pending is not None
        assert f._edge_tracker.get_seed_path_hash(seed_key) == 0
        f._note_det_effector()
        assert all(v == _DET_EFF_UNKNOWN for v in f._operators._det_eff[seed_key].eff)
