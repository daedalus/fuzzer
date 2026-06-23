"""Tests for Fuzzer service (unit tests, no real target execution)."""

from unittest.mock import patch

from fuzzer_tool.adapters.shim_factory import ShimResult
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

    def test_mutate_includes_splice(self):
        f = self._make_fuzzer(mutations_per_input=64)
        f.corpus = [b"AAAA", b"BBBB"]
        results = {f.mutate(b"AAAA") for _ in range(200)}
        assert any(len(r) >= 2 for r in results)

    def test_splice_mutation_operator(self):
        f = self._make_fuzzer(mutations_per_input=64)
        f.corpus = [b"AAAA", b"BBBB"]
        splice_count = 0
        for _ in range(200):
            result = f.mutate(b"AAAA")
            assert isinstance(result, bytes)
            if len(result) != 4:
                splice_count += 1
        assert splice_count > 0

    def test_seed_metadata_initialized(self):
        f = self._make_fuzzer()
        f.corpus = [b"AAAA", b"BBBB"]
        f._init_seed_metadata()
        assert len(f.seed_meta) == 2
        for meta in f.seed_meta.values():
            assert meta["fuzz_count"] == 0
            assert meta["coverage_edges"] == 0

    def test_pick_seed_weights_less_fuzzed(self):
        f = self._make_fuzzer()
        f.corpus = [b"AAAA", b"BBBB"]
        f._init_seed_metadata()
        f.seed_meta[b"AAAA"]["fuzz_count"] = 100
        f.seed_meta[b"BBBB"]["fuzz_count"] = 0
        counts = {b"AAAA": 0, b"BBBB": 0}
        for _ in range(200):
            seed = f._pick_seed()
            counts[seed] = counts.get(seed, 0) + 1
        assert counts[b"BBBB"] > counts[b"AAAA"]

    def test_pick_seed_weights_coverage(self):
        f = self._make_fuzzer()
        f.corpus = [b"AAAA", b"BBBB"]
        f._init_seed_metadata()
        f.seed_meta[b"AAAA"]["coverage_edges"] = 50
        f.seed_meta[b"BBBB"]["coverage_edges"] = 0
        counts = {b"AAAA": 0, b"BBBB": 0}
        for _ in range(200):
            seed = f._pick_seed()
            counts[seed] = counts.get(seed, 0) + 1
        assert counts[b"AAAA"] > counts[b"BBBB"]

    def test_pick_seed_weights_recency(self):
        import time

        f = self._make_fuzzer()
        f.corpus = [b"AAAA", b"BBBB"]
        f._init_seed_metadata()
        f.seed_meta[b"AAAA"]["added_at"] = time.time() - 1000
        f.seed_meta[b"BBBB"]["added_at"] = time.time()
        counts = {b"AAAA": 0, b"BBBB": 0}
        for _ in range(200):
            seed = f._pick_seed()
            counts[seed] = counts.get(seed, 0) + 1
        assert counts[b"BBBB"] > counts[b"AAAA"]

    def test_save_to_corpus_adds_metadata(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._make_fuzzer(corpus_dir=f"{tmpdir}/corpus", crashes_dir=f"{tmpdir}/crashes")
            initial_count = len(f.seed_meta)
            f.save_to_corpus(b"test_data_5678")
            assert len(f.seed_meta) == initial_count + 1
            meta = f.seed_meta[b"test_data_5678"]
            assert meta["fuzz_count"] == 0
            assert meta["coverage_edges"] == 0

    def test_shm_coverage_none_by_default(self):
        f = self._make_fuzzer()
        assert f.shm_cov is None

    def test_coverage_report_none_by_default(self):
        f = self._make_fuzzer()
        assert f.coverage_report is None

    def test_coverage_report_set(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            f = self._make_fuzzer(coverage_report=f"{tmpdir}/cov.json")
            assert f.coverage_report is not None
            assert f.coverage_report.name == "cov.json"

    def test_dump_coverage_report_no_data(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "cov.json"
            f = self._make_fuzzer(coverage_report=str(report_path))
            f._dump_coverage_report()
            assert not report_path.exists()

    def test_auto_timeout_flag(self):
        f = self._make_fuzzer()
        assert hasattr(f, "coverage_report")

    def test_seed_default(self):
        f = self._make_fuzzer()
        assert f.seed == 42

    def test_seed_custom(self):
        f = self._make_fuzzer(seed=123)
        assert f.seed == 123

    def test_seed_reproducibility(self):
        import random as _random

        f1 = self._make_fuzzer(seed=42)
        _random.seed(42)
        results1 = [f1.mutate(b"AAAA") for _ in range(10)]
        f2 = self._make_fuzzer(seed=42)
        _random.seed(42)
        results2 = [f2.mutate(b"AAAA") for _ in range(10)]
        assert results1 == results2

    def test_grammar_none_by_default(self):
        f = self._make_fuzzer()
        assert f.grammar is None

    def test_persistent_none_by_default(self):
        f = self._make_fuzzer()
        assert f._persistent_runner is None

    def test_inprocess_none_by_default(self):
        f = self._make_fuzzer()
        assert f._inprocess_runner is None


class TestInProcessRunner:
    """Tests for in-process target execution."""

    def _make_runner(self, **kwargs):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        defaults = dict(
            target="/bin/true",
            function_name="LLVMFuzzerTestOneInput",
            timeout=1,
        )
        defaults.update(kwargs)
        with patch("fuzzer_tool.adapters.inprocess.InProcessRunner._start"):
            r = InProcessRunner(**defaults)
        return r

    def test_init_with_mock(self):
        r = self._make_runner()
        assert r.target == "/bin/true"
        assert r.timeout == 1

    def test_no_shim_by_default(self):
        r = self._make_runner()
        assert r._shim is None

    def test_shim_built_with_coverage_env_id(self):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        with patch("fuzzer_tool.adapters.inprocess.build_shim") as mock_build:
            mock_build.return_value = ShimResult(
                shim_path="/tmp/fake.so",
                coverage_type="inline_8bit",
                needs_preload=True,
            )
            with patch("fuzzer_tool.adapters.inprocess.load_shim"):
                with patch("ctypes.CDLL"):
                    r = InProcessRunner(
                        target="/tmp/fake.so",
                        coverage_env_id="12345",
                    )
                    assert r._shim is not None
                    assert r._shim.coverage_type == "inline_8bit"
                    mock_build.assert_called_once()

    def test_read_bitmap_returns_none_without_shim(self):
        r = self._make_runner()
        assert r.read_bitmap() is None

    def test_reset_bitmap_noop_without_shim(self):
        r = self._make_runner()
        r.reset_bitmap()

    def test_run_one_python_func(self):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        with patch.object(InProcessRunner, "_start"):
            r = InProcessRunner.__new__(InProcessRunner)
            r.target = "test"
            r.function_name = "func"
            r.timeout = 1
            r.shm_size = 65536
            r.direct = False
            r.coverage_env_id = None
            r._lib = None
            r._is_c = False
            r._shim = None
            r._shim_handle = None
            r._loader_path = None
            r._bitmap_out = None
            r._func = lambda data: 0

            rc, err = r.run_one(b"hello")
            assert rc == 0
            assert err == ""

    def test_run_one_python_func_exception(self):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        with patch.object(InProcessRunner, "_start"):
            r = InProcessRunner.__new__(InProcessRunner)
            r.target = "test"
            r.function_name = "func"
            r.timeout = 1
            r.shm_size = 65536
            r.direct = False
            r.coverage_env_id = None
            r._lib = None
            r._is_c = False
            r._shim = None
            r._shim_handle = None
            r._loader_path = None
            r._bitmap_out = None
            r._func = lambda data: (_ for _ in ()).throw(ValueError("boom"))

            rc, err = r.run_one(b"hello")
            assert rc == -2
            assert "boom" in err

    def test_run_one_python_func_returns_int(self):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        with patch.object(InProcessRunner, "_start"):
            r = InProcessRunner.__new__(InProcessRunner)
            r.target = "test"
            r.function_name = "func"
            r.timeout = 1
            r.shm_size = 65536
            r.direct = False
            r.coverage_env_id = None
            r._lib = None
            r._is_c = False
            r._shim = None
            r._shim_handle = None
            r._loader_path = None
            r._bitmap_out = None
            r._func = lambda data: 42

            rc, err = r.run_one(b"hello")
            assert rc == 42
            assert err == ""

    def test_stop(self):
        r = self._make_runner()
        r._shim = ShimResult(shim_path="/tmp/fake.so", coverage_type="none")
        with patch("fuzzer_tool.adapters.inprocess.cleanup_shim") as mock_cleanup:
            r.stop()
            mock_cleanup.assert_called_once_with("/tmp/fake.so")
        assert r._func is None
        assert r._lib is None
        assert r._shim is None

    def test_run_c_subprocess_crash_detection(self):
        """Test that subprocess-based C execution detects SIGSEGV in child."""
        import signal
        import subprocess
        import tempfile

        from fuzzer_tool.adapters.inprocess import InProcessRunner

        crash_c = b"""
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size >= 1 && data[0] == 'X') {
        ((void(*)())0)();
    }
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = f"{tmpdir}/crash.so"
            c_path = f"{tmpdir}/crash.c"
            with open(c_path, "wb") as f:
                f.write(crash_c)
            subprocess.run(
                ["gcc", "-shared", "-fPIC", "-o", so_path, c_path],
                check=True,
                capture_output=True,
            )

            r = InProcessRunner(target=so_path, timeout=2)

            rc, err = r.run_one(b"hello")
            assert rc == 0

            rc, err = r.run_one(b"X")
            assert rc == -signal.SIGSEGV

            r.stop()


class TestInProcessFuzzer:
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

    def test_fuzzer_with_inprocess(self):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        with patch.object(InProcessRunner, "_start"):
            f = self._make_fuzzer(
                inprocess=True,
                inprocess_func="my_func",
            )
            assert f._inprocess_runner is not None
            assert f._inprocess_runner.function_name == "my_func"

    def test_fuzzer_inprocess_none_by_default(self):
        f = self._make_fuzzer()
        assert f._inprocess_runner is None
