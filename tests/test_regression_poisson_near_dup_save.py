"""Regression: the ADMIT_NEAR_DUP branch of CorpusManager.save_to_corpus.

The existing Poisson-disk tests only drive PoissonDiskAdmission.check();
nothing exercised the save path that consumes its decision, which is where
the flag was written into seed_meta[data] before that entry existed
(KeyError on the first near-duplicate admission of every campaign).
"""

from __future__ import annotations

import pytest

from fuzzer_tool.services import corpus_manager as cm
from fuzzer_tool.services.fuzzer import Fuzzer


def _fuzzer(tmp_path):
    target = tmp_path / "target"
    target.write_text("#!/bin/sh\necho ok")
    target.chmod(0o755)
    return Fuzzer(
        target=str(target),
        corpus_dir=tmp_path / "corpus",
        crashes_dir=tmp_path / "crashes",
        max_len=256,
        timeout=1,
        mutations_per_input=2,
        poisson_disk_admission=True,
    )


@pytest.fixture
def force_decision(monkeypatch):
    def _force(decision):
        monkeypatch.setattr(cm.PoissonDiskAdmission, "check", lambda self, data, key: decision)

    return _force


def test_near_dup_admission_does_not_raise_and_sets_flag(tmp_path, force_decision):
    f = _fuzzer(tmp_path)
    force_decision(cm.PoissonAdmissionDecision.ADMIT_NEAR_DUP)
    data = b"\x00" * 64
    f.save_to_corpus(data, parent=None)
    assert data in f.corpus
    assert f.seed_meta[data]["_is_near_duplicate"] is True
    # The flag must survive the fresh seed_meta literal, not precede it.
    assert f.seed_meta[data]["fuzz_count"] == 0


def test_plain_admission_does_not_set_flag(tmp_path, force_decision):
    f = _fuzzer(tmp_path)
    force_decision(cm.PoissonAdmissionDecision.ADMIT)
    data = b"\x01" * 64
    f.save_to_corpus(data, parent=None)
    assert data in f.corpus
    assert "_is_near_duplicate" not in f.seed_meta[data]
