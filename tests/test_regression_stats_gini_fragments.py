"""The seed-energy and operator-selection Gini stats fragments must never
take down print_stats.

Same hazard as test_regression_stats_model_fragments.py: a stand-in that
answers every attribute (MagicMock, a partially restored state) must still
degrade to an empty string rather than raising out of print_stats.
"""

from unittest.mock import MagicMock

from fuzzer_tool.services.stats import StatsReporter


def _reporter():
    return StatsReporter.__new__(StatsReporter)


class TestSeedEnergyGiniFragment:
    def test_answers_everything_stand_in_yields_empty(self):
        f = MagicMock()
        assert _reporter()._print_stats_seed_energy_gini_str(f) == ""

    def test_absent_seed_meta_yields_empty(self):
        f = object()
        assert _reporter()._print_stats_seed_energy_gini_str(f) == ""

    def test_empty_seed_meta_yields_empty(self):
        f = MagicMock()
        f.seed_meta = {}
        assert _reporter()._print_stats_seed_energy_gini_str(f) == ""

    def test_single_seed_yields_empty(self):
        f = MagicMock()
        f.seed_meta = {b"a": {"fuzz_count": 7}}
        assert _reporter()._print_stats_seed_energy_gini_str(f) == ""

    def test_even_corpus_renders_zero(self):
        f = MagicMock()
        f.seed_meta = {
            b"a": {"fuzz_count": 5},
            b"b": {"fuzz_count": 5},
        }
        out = _reporter()._print_stats_seed_energy_gini_str(f)
        assert out == " | seed-gini: 0.00"

    def test_skewed_corpus_renders_high(self):
        f = MagicMock()
        f.seed_meta = {
            b"a": {"fuzz_count": 1000},
            b"b": {"fuzz_count": 0},
            b"c": {"fuzz_count": 0},
        }
        out = _reporter()._print_stats_seed_energy_gini_str(f)
        assert out.startswith(" | seed-gini: 0.")
        assert float(out.rsplit(": ", 1)[1]) > 0.5


class TestOpGiniFragment:
    def test_answers_everything_stand_in_yields_empty(self):
        f = MagicMock()
        assert _reporter()._print_stats_op_gini_str(f) == ""

    def test_absent_op_attempts_yields_empty(self):
        f = object()
        assert _reporter()._print_stats_op_gini_str(f) == ""

    def test_single_operator_yields_empty(self):
        f = MagicMock()
        f._op_attempts = {"havoc": 42}
        assert _reporter()._print_stats_op_gini_str(f) == ""

    def test_even_operators_renders_zero(self):
        f = MagicMock()
        f._op_attempts = {"havoc": 10, "splice": 10}
        out = _reporter()._print_stats_op_gini_str(f)
        assert out == " | op-gini: 0.00"

    def test_skewed_operators_renders_high(self):
        f = MagicMock()
        f._op_attempts = {"havoc": 1000, "splice": 1, "arith": 1}
        out = _reporter()._print_stats_op_gini_str(f)
        assert out.startswith(" | op-gini: 0.")
        assert float(out.rsplit(": ", 1)[1]) > 0.5
