"""Tests for Fuzzer service (unit tests, no real target execution)."""

from unittest.mock import patch

from fuzzer_tool.core.markov import MarkovChain
from fuzzer_tool.core.montecarlo import MonteCarloScheduler
from fuzzer_tool.services.fuzzer import Fuzzer


class TestFuzzerUnit:
    def _make_fuzzer(self, **kwargs):
        defaults = dict(
            target="/bin/true",
            corpus_dir="/tmp/fuzz_test_corpus",
            crashes_dir="/tmp/fuzz_test_crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )
        defaults.update(kwargs)
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(**defaults)
        return f

    def test_init(self):
        f = self._make_fuzzer()
        assert f.max_len == 256
        assert f.exec_count == 0
        assert f.crash_count == 0

    def test_mutate_returns_bytes(self):
        f = self._make_fuzzer()
        result = f.mutate(b"AAAA")
        assert isinstance(result, bytes)

    def test_mutate_empty_input(self):
        f = self._make_fuzzer()
        result = f.mutate(b"")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_is_crash_sanitizer(self):
        f = self._make_fuzzer()
        stderr = "==1==ERROR: AddressSanitizer: heap-buffer-overflow\n"
        assert f._is_crash(0, stderr)

    def test_is_crash_signal(self):
        f = self._make_fuzzer()
        assert f._is_crash(-6, "")
        assert f._is_crash(-11, "")

    def test_is_not_crash_timeout(self):
        f = self._make_fuzzer()
        assert not f._is_crash(-1, "timeout")

    def test_is_interesting_signal(self):
        f = self._make_fuzzer()
        assert f._is_interesting(-6, "")
        assert f._is_interesting(-11, "")

    def test_is_interesting_asan(self):
        f = self._make_fuzzer()
        assert f._is_interesting(0, "ASAN detected")

    def test_with_markov(self):
        f = self._make_fuzzer(markov_order=1, markov_generate=True)
        assert isinstance(f.markov, MarkovChain)

    def test_with_mc_bandit(self):
        f = self._make_fuzzer(mc_bandit=True)
        assert isinstance(f.mc, MonteCarloScheduler)
        assert "bit_flip" in f.mc.arm_alpha

    def test_with_mc_cem(self):
        f = self._make_fuzzer(mc_cem=True)
        assert isinstance(f.mc, MonteCarloScheduler)

    def test_save_to_corpus(self):
        f = self._make_fuzzer()
        data = b"test_data_12345"
        f.save_to_corpus(data)
        assert data in f.corpus
        f.save_to_corpus(data)
        assert f.corpus.count(data) == 1

    def test_pick_seed_empty_corpus(self):
        f = self._make_fuzzer()
        f.corpus = []
        seed = f._pick_seed()
        assert seed == b"AAAAAAAA"
