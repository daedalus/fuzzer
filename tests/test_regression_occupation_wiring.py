"""Regression test: occupation record hook fires on real fuzz_one() runs.

Phase B1 of docs/handover/handover_RoRd.md — when ``occupation=True``, the
same site that turns a run's sparse SHM edge-count table into an EdgeTracker
update must also feed ``Fuzzer._occupation_rarity`` (LongitudinalRarity) and
populate ``Fuzzer._last_occupation``, without changing existing coverage
bookkeeping when the flag is off.
"""

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

_TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")
requires_test_target = pytest.mark.skipif(
    not Path(_TARGET).exists(), reason="targets/test_target not built"
)


def _build_fuzzer(**kwargs):
    tmp = tempfile.TemporaryDirectory()
    corpus = Path(tmp.name) / "corpus"
    crashes = Path(tmp.name) / "crashes"
    corpus.mkdir()
    crashes.mkdir()
    kwargs.setdefault("max_len", 4096)
    kwargs.setdefault("use_coverage", True)
    f = Fuzzer(target=_TARGET, corpus_dir=str(corpus), crashes_dir=str(crashes), **kwargs)
    f._test_tmp = tmp
    return f


@requires_test_target
class TestOccupationRecordHook:
    def test_off_by_default_no_observation(self):
        f = _build_fuzzer()
        assert f._occupation_rarity is None
        f.fuzz_one(b"AAAAAAAA")
        assert f._occupation_rarity is None
        assert f._last_occupation is None

    def test_enabled_records_occupation_on_new_coverage(self):
        f = _build_fuzzer(occupation=True)
        assert f._occupation_rarity is not None
        assert f._occupation_rarity.n_histories == 0

        # First execution of any input to an uninstrumented-baseline target
        # always yields "new" coverage (nothing observed yet), so the
        # has_new_coverage branch that hosts the occupation hook fires.
        f.fuzz_one(b"AAAAAAAA")

        assert f._occupation_rarity.n_histories >= 1
        assert f._last_occupation is not None
        assert len(f._last_occupation) > 0
