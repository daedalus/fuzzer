"""Tests for the runtime K-Scheduler wiring (WS-I).

KatzChannel is exercised against a synthetic ICFG — no ELF, no SHM.
The coverage contract helpers are pure functions over a stub Fuzzer.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.core.horizon import HorizonGraph
from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.schedulers.katz import KatzResult
from fuzzer_tool.services.corpus_manager import check_coverage_contract, current_coverage_contract
from fuzzer_tool.services.katz_channel import KatzChannel


def _icfg(n=4):
    src = np.array([0, 1, 2], dtype=np.int64)
    dst = np.array([1, 2, 3], dtype=np.int64)
    addrs = [0x100 * i for i in range(n)]
    return InterproceduralCFG(addrs, [f"f{i}" for i in range(n)], src, dst, {})


def _channel():
    ch = KatzChannel(_icfg(), {k: k for k in range(4)})
    # No SHM attached: sample() must degrade to None rather than crash.
    return ch


def _bits(idxs, n=4):
    b = np.zeros(n, dtype=bool)
    for i in idxs:
        b[i] = True
    return b


class TestSamplingAndMasks:
    def test_sample_without_shm_is_none(self):
        assert _channel().sample() is None

    def test_record_accumulates_hits_and_masks(self):
        ch = _channel()
        ch.record(_bits({0, 1}), seed_key="s1")
        ch.record(_bits({1}), seed_key="s1")
        assert ch.hit_counts.tolist() == [1.0, 2.0, 0.0, 0.0]
        mask = np.unpackbits(np.frombuffer(ch._masks["s1"], dtype=np.uint8), bitorder="little")[:4]
        assert mask.tolist() == [1, 1, 0, 0]

    def test_mask_or_merges_not_replaces(self):
        ch = _channel()
        ch.record(_bits({0}), seed_key="s")
        ch.record(_bits({3}), seed_key="s")
        mask = np.unpackbits(np.frombuffer(ch._masks["s"], dtype=np.uint8), bitorder="little")[:4]
        assert mask.tolist() == [1, 0, 0, 1]

    def test_no_seed_key_skips_mask_but_counts(self):
        ch = _channel()
        ch.record(_bits({2}), seed_key=None)
        assert ch._masks == {}
        assert ch.hit_counts[2] == 1.0


class TestScores:
    def test_ensure_scores_caches_until_dirty_and_due(self):
        ch = _channel()
        ch.record(_bits({0}), seed_key="s")
        first = ch.ensure_scores()
        again = ch.ensure_scores()
        assert first is again  # not dirty / interval not elapsed
        ch.exec_count += 1000
        ch.record(_bits({1}), seed_key="s2")  # new mask -> dirty
        second = ch.ensure_scores()
        assert second is not first
        assert isinstance(second, KatzResult)

    def test_seed_energy_ranks_rare_over_hit(self):
        """Oracle from W4: seed whose path holds the rare sink outscores
        the one pinned to the saturated sink."""
        ch = _channel()
        for _ in range(10):
            ch.record(_bits({1}))  # saturate u1 only
        ch.record(_bits({3}), seed_key="rare")  # touches rare sink u3
        ch.record(_bits({1}), seed_key="hit")  # touches hit sink u1
        ch.ensure_scores(force=True)
        e_rare = ch.seed_energy("rare")
        e_hit = ch.seed_energy("hit")
        assert max(e_rare, e_hit) > 0
        assert e_rare > e_hit

    def test_unknown_seed_scores_zero(self):
        ch = _channel()
        ch.record(_bits({0}), seed_key="known")
        ch.ensure_scores(force=True)
        assert ch.seed_energy("never-seen") == 0.0


class TestPersistence:
    def test_state_round_trip(self):
        ch = _channel()
        ch.exec_count = 77
        ch.record(_bits({0, 1}), seed_key="s")
        state = ch.state_dict()
        revived = _channel()
        revived.load_state_dict(state)
        assert revived.hit_counts.tolist() == ch.hit_counts.tolist()
        assert set(revived._masks) == {"s"}
        assert revived.exec_count == ch.exec_count

    def test_load_marks_dirty(self):
        ch = _channel()
        ch.load_state_dict({"hit_counts": [1, 0, 0, 0], "masks": {}, "exec_count": 5})
        first = ch.ensure_scores()
        ch.load_state_dict({"hit_counts": [9, 0, 0, 0], "masks": {}, "exec_count": 6})
        second = ch.ensure_scores()
        assert second is not first


class TestCoverageContract:
    def _stub(self, katz=None):
        return SimpleNamespace(target="/bin/true", _katz_channel=katz)

    def test_current_contract_detects_legacy_k2(self):
        c = current_coverage_contract(self._stub())
        assert c == {"ngram_k": 2, "node_channel": False}

    def test_missing_saved_section_resumes_freely(self):
        check_coverage_contract(None, {"ngram_k": 3, "node_channel": True})

    def test_matching_contract_passes(self):
        cur = {"ngram_k": 3, "node_channel": True}
        check_coverage_contract(cur, cur)

    def test_k_mismatch_refuses(self):
        with pytest.raises(RuntimeError, match="ngram_k"):
            check_coverage_contract(
                {"ngram_k": 2, "node_channel": False}, {"ngram_k": 3, "node_channel": False}
            )

    def test_node_channel_mismatch_refuses(self):
        with pytest.raises(RuntimeError, match="node_channel"):
            check_coverage_contract(
                {"ngram_k": 3, "node_channel": False}, {"ngram_k": 3, "node_channel": True}
            )

    def test_horizon_graph_type_still_importable_for_wiring(self):
        # Guards against accidental circular-import breakage of the seam
        # between channel and graph layers.
        assert HorizonGraph is not None
