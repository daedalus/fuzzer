"""Regression: ``--secretary`` is diagnostic only (P2-4), and must stay so.

``SecretaryStopping`` documents that nothing calls ``should_stop`` to reweight
or retire anything, but two consumers did: the seed picker cut a stopped seed's
weight 100x, and ``save_to_corpus`` deferred a minimization on every stop. The
rule is broken (rank bounded by 1/(1-decay) = 20 against a threshold of
n/e = 183, and the discovery-rate stream carries a 1/t envelope), so a seed
with a 50% discovery rate is stopped at observation 20 and never recovers.
``--elo all`` enables it. See handover_bandit_stopping_search_2026-09-02.md §1.
"""

import shutil
import tempfile
from pathlib import Path

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.seed_picker import SeedPicker


class _Stopped:
    """Secretary stub whose rule always fires."""

    def __init__(self):
        self.observed = []

    def should_stop(self):
        return True, "stub"

    def observe(self, value):
        self.observed.append(value)


class _Fuzzer:
    def __init__(self, secretary):
        self._edge_tracker = EdgeTracker()
        self.exec_count = 0
        self._last_new_edge_exec = 0
        self._saturation_gated = True  # neutral cached multipliers
        self._cached_weights = {}
        self._secretary = secretary
        self._seed_secretary = {"sk": _Stopped()} if secretary else {}


def _weight(secretary):
    f = _Fuzzer(secretary)
    w, _sub, _spa = SeedPicker(f)._weight_cached("sk", 1.0, {}, f)
    return w


class TestSeedWeight:
    def test_control_disabled_matches_itself(self):
        assert _weight(False) == _weight(False)

    def test_regression_secretary_stop_leaves_seed_weight(self):
        """Falsification: pre-fix, a stopped seed weighed 0.01x the control."""
        assert _weight(True) == _weight(False)

    def test_adversarial_every_seed_stopped(self):
        """Every seed stopped: weights still match the disabled picker."""
        keys = [f"s{i}" for i in range(50)]
        on, off = _Fuzzer(True), _Fuzzer(False)
        on._seed_secretary = {k: _Stopped() for k in keys}
        p_on, p_off = SeedPicker(on), SeedPicker(off)
        w_on = [p_on._weight_cached(k, 2.0, {}, on)[0] for k in keys]
        w_off = [p_off._weight_cached(k, 2.0, {}, off)[0] for k in keys]
        assert w_on == w_off


class TestCorpusMinimize:
    @staticmethod
    def _save(secretary):
        from fuzzer_tool.services.fuzzer import Fuzzer

        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "c" / "seeds").mkdir(parents=True)
            (tmp / "x").mkdir()
            (tmp / "c" / "seeds" / "s").write_bytes(b"seed")
            f = Fuzzer(
                target="/nonexistent",
                corpus_dir=str(tmp / "c"),
                crashes_dir=str(tmp / "x"),
                secretary=secretary,
            )
            stub = _Stopped()
            if secretary:
                f._corpus_secretary = stub
            deferred = []
            f._defer_minimize = lambda: deferred.append(1)
            f.save_to_corpus(b"new input", parent=b"seed")
            return len(deferred), stub.observed
        finally:
            shutil.rmtree(tmp)

    def test_control_disabled_never_defers(self):
        assert self._save(False)[0] == 0

    def test_regression_secretary_stop_does_not_defer_minimize(self):
        """Falsification: pre-fix, a stop scheduled a minimization."""
        assert self._save(True)[0] == self._save(False)[0]

    def test_adversarial_still_observed_for_display(self):
        """Decoupling must not drop the observation the report reads."""
        assert len(self._save(True)[1]) == 1
