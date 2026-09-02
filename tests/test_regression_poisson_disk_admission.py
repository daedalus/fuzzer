"""Regression: Poisson-disk admission is wired correctly end-to-end.

Tests:
1. PoissonDiskAdmission admits the first seed (no prior neighbors).
2. A seed that owns unique edges is admitted even when Jaccard-similar.
3. A seed with no unique edges is rejected when Jaccard-similar.
4. Fuzzer.__init__ accepts the new kwargs without error.
5. CLI --poisson-disk-admission and --poisson-disk-min-jaccard are accepted.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

from fuzzer_tool.core.edge_tracker import MinHashLSH
from fuzzer_tool.services.corpus_manager import (
    PoissonAdmissionDecision,
    PoissonDiskAdmission,
)
from fuzzer_tool.services.fuzzer import Fuzzer


class TestPoissonDiskAdmissionUnit:
    """Unit tests for the PoissonDiskAdmission class."""

    def _mock_fuzzer(self) -> MagicMock:
        """Minimal fuzzer mock with the attributes PoissonDiskAdmission needs."""
        f = MagicMock()
        f._edge_tracker = MagicMock()
        # Minimal MinHashLSH mock that find_similar can interrogate.
        f._edge_tracker._minhash = MagicMock(spec=MinHashLSH)
        # Seed-edges for unique-edge safety-valve tests.
        f._edge_tracker.seed_edges = {}
        f._admitted_keys = set()
        f._poisson_occupied_buckets = set()
        f._poisson_last_new_bucket_exec = 0
        f.exec_count = 0
        return f

    def test_first_seed_admitted(self):
        """The first seed ever is always admitted (disk has no occupants)."""
        f = self._mock_fuzzer()
        minhash = f._edge_tracker._minhash
        minhash.find_similar.return_value = set()  # no similar neighbors
        minhash.num_bands = 4
        minhash.band_size = 4
        minhash.signatures = {}

        pda = PoissonDiskAdmission(f, min_jaccard=0.25)
        decision = pda.check(b"first seed", "key_first")
        assert decision == PoissonAdmissionDecision.ADMIT
        assert "key_first" in f._admitted_keys

    def test_similar_seed_with_unique_edges_is_admitted(self):
        """A Jaccard-similar seed that owns unique edges bypasses rejection."""
        f = self._mock_fuzzer()
        minhash = f._edge_tracker._minhash
        # find_similar returns an admitted neighbor.
        minhash.find_similar.return_value = {"key_existing"}
        minhash.num_bands = 4
        minhash.band_size = 4
        minhash.signatures = {}

        # Candidate owns edge 99; neighbor owns only edges 1-10.
        f._edge_tracker.seed_edges = {
            "key_candidate": {99},  # unique
            "key_existing": {1, 2, 3, 4, 5},
        }
        # key_existing must be in admitted_keys for the neighbor to count.
        f._admitted_keys = {"key_existing"}

        pda = PoissonDiskAdmission(f, min_jaccard=0.25)
        decision = pda.check(b"candidate", "key_candidate")
        assert decision == PoissonAdmissionDecision.ADMIT_NEAR_DUP
        assert "key_candidate" in f._admitted_keys

    def test_similar_seed_with_no_unique_edges_is_rejected(self):
        """A Jaccard-similar seed with no unique edges is rejected."""
        f = self._mock_fuzzer()
        minhash = f._edge_tracker._minhash
        minhash.find_similar.return_value = {"key_existing"}
        minhash.num_bands = 4
        minhash.band_size = 4
        minhash.signatures = {}

        # Candidate's edges are a subset of neighbor's.
        f._edge_tracker.seed_edges = {
            "key_candidate": {1, 2, 3},  # no unique edges
            "key_existing": {1, 2, 3, 4, 5},
        }
        f._admitted_keys = {"key_existing"}

        pda = PoissonDiskAdmission(f, min_jaccard=0.25)
        decision = pda.check(b"candidate", "key_candidate")
        assert decision == PoissonAdmissionDecision.REJECT_NEAR_DUP
        # Admitted keys must not include the rejected seed.
        assert "key_candidate" not in f._admitted_keys

    def test_find_similar_result_is_filtered_by_admitted_keys(self):
        """find_similar is called but the result is filtered: seed must not
        match itself.  This is the core invariant that prevents a seed from
        always being near-duplicate of itself."""
        f = self._mock_fuzzer()
        minhash = f._edge_tracker._mintracker = MagicMock()
        f._edge_tracker.seed_edges = {}

        # find_similar returns a set that INCLUDES the seed itself.
        # After filtering by admitted_keys the candidate should be empty.
        minhash.find_similar.return_value = {"key_self", "key_other"}
        minhash.num_bands = 4
        minhash.band_size = 4
        minhash.signatures = {}
        f._admitted_keys = {"key_other"}  # only key_other is admitted
        f._edge_tracker.seed_edges = {}

        pda = PoissonDiskAdmission(f, min_jaccard=0.25)
        decision = pda.check(b"the_seed", "key_self")
        # key_self is NOT in admitted_keys, so filtering yields only key_other.
        # key_other has no unique edges relative to key_self (key_self has no edges).
        assert decision == PoissonAdmissionDecision.REJECT_NEAR_DUP
        assert "key_self" not in f._admitted_keys


class TestPoissonDiskAdmissionFuzzerIntegration:
    """Verify Fuzzer.__init__ accepts the new kwargs without error."""

    def test_fuzzer_accepts_poisson_kwargs(self, tmp_path):
        target = tmp_path / "target"
        target.write_text("#!/bin/sh\necho ok")
        target.chmod(0o755)

        # Must not raise. Pass the minimum required args to Fuzzer().
        f = Fuzzer(
            target=str(target),
            corpus_dir=tmp_path / "corpus",
            crashes_dir=tmp_path / "crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            poisson_disk_admission=True,
            poisson_disk_min_jaccard=0.3,
        )
        assert f._use_poisson_disk_admission is True
        assert f._poisson_disk_min_jaccard == 0.3
        assert f._poisson_admission is None  # lazy-init
        assert isinstance(f._admitted_keys, set)
        assert isinstance(f._poisson_occupied_buckets, set)

        # Simulate the attributes save_to_corpus reads.
        f._duplicate_reject_count = 0
        f._poisson_reject_count = 0
        f._poisson_near_dup_admit_count = 0
        f.exec_count = 0

    def test_deprecated_near_duplicates_short_circuit(self, tmp_path):
        """deprioritize_near_duplicates skips scan when poisson is active
        and not enough redundant admissions have accumulated."""
        target = tmp_path / "target"
        target.write_text("#!/bin/sh\necho ok")
        target.chmod(0o755)

        f = Fuzzer(
            target=str(target),
            corpus_dir=tmp_path / "corpus",
            crashes_dir=tmp_path / "crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            poisson_disk_admission=True,
        )
        f._redundant_admission_count = 10  # below 50 threshold

        # Mock the edge tracker to verify it's NOT called.
        f._edge_tracker = MagicMock()
        f._edge_tracker.find_near_duplicate_seeds = MagicMock(return_value=[])

        # The method should return immediately without calling find_near_duplicate_seeds.
        from fuzzer_tool.services.corpus_manager import CorpusManager

        cm = CorpusManager(f)
        cm.deprioritize_near_duplicates()

        f._edge_tracker.find_near_duplicate_seeds.assert_not_called()


class TestPoissonDiskAdmissionCLI:
    """Smoke tests: argparse accepts the flags and wires to Fuzzer kwargs."""

    def test_poisson_disk_flag_accepted(self, monkeypatch):
        captured: dict = {}

        def _spy(args):
            captured.update(vars(args))
            return 0

        from fuzzer_tool.cli import commands

        monkeypatch.setattr(commands, "cmd_fuzz", _spy)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "fuzzer-tool",
                "fuzz",
                "/bin/true",
                "--poisson-disk-admission",
                "--poisson-disk-min-jaccard",
                "0.4",
                "-n",
                "1",
                "--no-shm",
            ],
        )
        rc = commands.main()
        assert rc == 0
        assert captured["poisson_disk_admission"] is True
        assert captured["poisson_disk_min_jaccard"] == 0.4
