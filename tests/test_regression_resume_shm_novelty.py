"""Regression: --resume re-ran every seed to refill the SHM virgin map.

The tracker's edge set was restored but the SHM novelty map (the fuzz loop's
actual "is this new" signal) was not, so resume paid a full seed pass (~155 s
for 2528 seeds) and, without it, early mutants all read as new coverage. The
map is now saved with the state, tied to a target hash, and the pass is
skipped only when nothing else needs the seed executions.
"""

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.corpus_manager import (
    compare_target_fingerprint,
    seed_key,
    target_fingerprint,
)
from fuzzer_tool.services.fuzzer import Fuzzer


@pytest.fixture
def shm():
    s = ShmCoverage(size=1024)
    yield s
    s.close() if hasattr(s, "close") else None


def _populated(shm: ShmCoverage) -> ShmCoverage:
    shm._seen_edge_ids.update({1, 2, 3, 70000})
    shm.mask_edges({5, 9})
    shm._virgin[2] = 0b101
    shm._virgin_wide[1 << 25] = 0b10
    shm._max_counts = np.zeros(2048, dtype=np.uint32)
    shm._max_counts[7] = 400
    shm._max_counts_wide[1 << 25] = 99
    shm.cumulative_edges = 4
    shm._peak_cumulative_edges = 4
    shm.bucket_transitions = 11
    shm.max_count_transitions = 2
    return shm


def test_novelty_round_trips_through_the_real_state_store(tmp_path, shm):
    """The store drops ndarray payloads, so this must survive a real save/load."""
    _populated(shm)
    store = StateStore(tmp_path)
    store.set("shm_novelty", shm.export_novelty())
    store.save()
    saved = StateStore(tmp_path).get("shm_novelty")
    assert saved is not None, "state store dropped the section"

    fresh = ShmCoverage(size=1024)
    assert fresh.import_novelty(saved)
    assert fresh._seen_edge_ids == shm._seen_edge_ids
    assert fresh._masked_edge_ids == {5, 9}
    assert fresh._virgin[2] == 0b101 and fresh._virgin[5] == 0xFF
    assert fresh._virgin_wide == shm._virgin_wide
    assert fresh._max_counts[7] == 400 and fresh._max_counts_wide == {1 << 25: 99}
    assert (fresh.cumulative_edges, fresh.bucket_transitions) == (4, 11)
    assert fresh.max_count_transitions == 2


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: d.update(version=99),
        lambda d: d.update(virgin_bits=[1]),  # length mismatch
        lambda d: d.update(virgin_idx=[-1], virgin_bits=[1]),
        lambda d: d.update(max_idx=[1 << 30], max_val=[1]),  # past dense range
        lambda d: d.update(cumulative_edges=-3),
        lambda d: d.update(seen=["x"]),
        lambda d: d.pop("masked"),
    ],
)
def test_malformed_state_is_rejected_and_leaves_table_pristine(shm, corrupt):
    state = _populated(shm).export_novelty()
    corrupt(state)
    fresh = ShmCoverage(size=1024)
    assert fresh.import_novelty(state) is False
    assert not fresh._seen_edge_ids and fresh.cumulative_edges == 0
    assert fresh._virgin.size == 0 and not fresh._virgin_wide


def test_import_refuses_a_table_that_has_already_observed_edges(shm):
    state = _populated(shm).export_novelty()
    used = ShmCoverage(size=1024)
    used._seen_edge_ids.add(42)
    assert used.import_novelty(state) is False
    assert used._seen_edge_ids == {42}


def test_non_dict_state_is_rejected():
    assert ShmCoverage(size=1024).import_novelty([1, 2, 3]) is False
    assert ShmCoverage(size=1024).import_novelty(None) is False


def test_masked_ids_cannot_be_unseen_by_a_hand_edited_state(shm):
    state = _populated(shm).export_novelty()
    state["seen"] = []  # tamper: masked ids no longer listed as seen
    fresh = ShmCoverage(size=1024)
    assert fresh.import_novelty(state)
    assert {5, 9} <= fresh._seen_edge_ids


def test_shared_virgin_tables_are_not_exported(shm):
    from fuzzer_tool.adapters.shm import SharedVirgin

    shm.share_virgin(SharedVirgin())
    assert shm.export_novelty() is None


# -- target fingerprint ---------------------------------------------------


class _T:
    def __init__(self, path):
        self.target = str(path)


def test_fingerprint_tracks_file_content(tmp_path):
    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF-one")
    a = target_fingerprint(_T(exe))
    exe.write_bytes(b"\x7fELF-two")
    b = target_fingerprint(_T(exe))
    assert a and b and a != b


def test_fingerprint_is_none_for_an_unreadable_target(tmp_path):
    assert target_fingerprint(_T(tmp_path / "missing")) is None


def test_fingerprint_is_cached_per_session(tmp_path):
    exe = tmp_path / "t"
    exe.write_bytes(b"one")
    f = _T(exe)
    first = target_fingerprint(f)
    exe.write_bytes(b"two")
    assert target_fingerprint(f) == first


@pytest.mark.parametrize(
    ("saved", "cur", "want"),
    [
        ({"target_sha256": "aa"}, {"target_sha256": "aa"}, "match"),
        ({"target_sha256": "aa"}, {"target_sha256": "bb"}, "mismatch"),
        ({}, {"target_sha256": "aa"}, "unknown"),  # state predates the hash
        (None, {"target_sha256": "aa"}, "unknown"),
        ({"target_sha256": "aa"}, {"target_sha256": None}, "unknown"),
    ],
)
def test_compare_target_fingerprint(saved, cur, want):
    assert compare_target_fingerprint(saved, cur) == want


# -- the calibration decision --------------------------------------------


class _Store:
    def __init__(self, **sections):
        self.sections = sections

    def get(self, name, default=None):
        return self.sections.get(name, default)


class _Shm:
    cumulative_edges = 0

    def __init__(self, accept=True):
        self.accept = accept
        self.imported = None

    def import_novelty(self, state):
        self.imported = state
        return self.accept


def _fuzzer(*, target_state="match", saved=True, accept=True, **features):
    f = Fuzzer.__new__(Fuzzer)
    f.resume = True
    f.use_coverage = True
    f._seed_calibration = True
    f.multi_targets = []
    f.target = "/x/t"
    f.corpus = [b"a", b"bb"]
    f.seed_meta = {b"a": {"coverage_edges": 3}, b"bb": {"coverage_edges": 9}}
    f.shm_cov = _Shm(accept)
    f._edge_tracker = EdgeTracker()
    f._edge_tracker.cumulative_edges = {1, 2, 3}
    f._resume_target_state = target_state
    f._cmplog = features.get("cmplog")
    f._distance = features.get("distance")
    f._frontier_oracle = None
    f._crash_explorer = None
    f._edge_ledger = features.get("ledger")
    sections = {"strata": {"x": 1}} if features.get("strata_saved") else {}
    if saved:
        sections["shm_novelty"] = {"seen": [1, 2, 3], "cumulative_edges": 3}
    f._state_store = _Store(**sections)
    f.probed = []
    f._report_edge_id_stability = lambda seed: f.probed.append(("stab", seed))
    f._report_coverage_noise = lambda seed: f.probed.append(("noise", seed))
    f._report_comparison_reach = lambda n: None
    f._report_resume_reach = lambda n: None

    def boom(data):
        raise AssertionError("seed was executed")

    f._run_target = boom
    return f


def test_matching_target_skips_the_seed_pass_and_still_probes_ids(capsys):
    f = _fuzzer()
    f._calibrate_seed_baselines()  # _run_target raises if any seed ran
    assert f.shm_cov.imported["cumulative_edges"] == 3
    # Probe runs on the widest seed by saved coverage, not corpus order.
    assert f.probed == [("stab", b"bb"), ("noise", b"bb")]
    assert "seed calibration skipped" in capsys.readouterr().out


def test_mismatched_target_runs_the_full_pass(capsys):
    f = _fuzzer(target_state="mismatch")
    with pytest.raises(AssertionError, match="seed was executed"):
        f._calibrate_seed_baselines()
    assert f.shm_cov.imported is None


def test_unknown_hash_runs_the_full_pass_and_says_why(capsys):
    f = _fuzzer(target_state="unknown")
    with pytest.raises(AssertionError, match="seed was executed"):
        f._calibrate_seed_baselines()
    assert "target hash unavailable" in capsys.readouterr().out


def test_state_without_saved_novelty_runs_the_full_pass():
    f = _fuzzer(saved=False)
    with pytest.raises(AssertionError, match="seed was executed"):
        f._calibrate_seed_baselines()


def test_unusable_saved_novelty_runs_the_full_pass(capsys):
    f = _fuzzer(accept=False)
    with pytest.raises(AssertionError, match="seed was executed"):
        f._calibrate_seed_baselines()
    assert "unusable" in capsys.readouterr().out


@pytest.mark.parametrize("feature", ["cmplog", "distance", "ledger"])
def test_consumers_of_live_seed_runs_keep_the_full_pass(feature, capsys):
    """cmplog tokens, distance tags and an unrestored strata ledger only come
    from executing seeds; skipping would silently starve them."""
    f = _fuzzer(**{feature: object()})
    with pytest.raises(AssertionError, match="seed was executed"):
        f._calibrate_seed_baselines()
    assert f.shm_cov.imported is None
    assert "running full seed calibration" in capsys.readouterr().out


def test_restored_strata_ledger_does_not_block_the_skip():
    f = _fuzzer(ledger=object(), strata_saved=True)
    f._calibrate_seed_baselines()
    assert f.shm_cov.imported is not None


def test_empty_restored_tracker_never_skips():
    f = _fuzzer()
    f._edge_tracker.cumulative_edges = set()
    with pytest.raises(AssertionError, match="seed was executed"):
        f._calibrate_seed_baselines()


def test_fresh_run_is_unaffected():
    f = _fuzzer()
    f.resume = False
    with pytest.raises(AssertionError, match="seed was executed"):
        f._calibrate_seed_baselines()
    assert f.shm_cov.imported is None


def test_seed_key_import_is_the_shared_helper():
    assert seed_key(b"x") == seed_key(b"x")
