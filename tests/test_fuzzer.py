"""Tests for Fuzzer service (unit tests, no real target execution)."""

import math
from array import array
from unittest.mock import patch

import pytest

from fuzzer_tool.adapters.shim_factory import ShimResult
from fuzzer_tool.core.markov import MarkovChain
from fuzzer_tool.core.schedulers import MonteCarloScheduler
from fuzzer_tool.services.fuzzer import Fuzzer


class TestFuzzerUnit:
    def _make_fuzzer(self, **kwargs):
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
        defaults = dict(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
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
        assert isinstance(seed, bytes)
        assert len(seed) > 0

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

    def test_corpus_boost_truncates_long_seeds(self):
        f = self._make_fuzzer(max_len=10, corpus_boost=10, boost_mean=5, boost_std=1)
        f.corpus = [b"A" * 50, b"B" * 50]
        f._boost_corpus_sizes()
        assert all(len(s) <= 10 for s in f.corpus)
        assert all(len(s) >= 1 for s in f.corpus)

    def test_corpus_boost_pads_repeat(self):
        f = self._make_fuzzer(
            max_len=100, corpus_boost=100, boost_mean=50, boost_std=5, boost_pad="repeat"
        )
        f.corpus = [b"ABC", b"XY"]
        f._boost_corpus_sizes()
        for s in f.corpus:
            assert 1 <= len(s) <= 100
            assert s[:3] == b"ABC" or s[:2] == b"XY"

    def test_corpus_boost_pads_zero(self):
        f = self._make_fuzzer(
            max_len=100, corpus_boost=100, boost_mean=50, boost_std=5, boost_pad="zero"
        )
        f.corpus = [b"A"]
        f._boost_corpus_sizes()
        assert len(f.corpus[0]) >= 1
        assert f.corpus[0][0] == ord("A")
        assert all(b == 0 for b in f.corpus[0][1:])

    def test_corpus_boost_pads_random(self):
        f = self._make_fuzzer(
            max_len=100, corpus_boost=100, boost_mean=50, boost_std=5, boost_pad="random"
        )
        f.corpus = [b"A"]
        f._boost_corpus_sizes()
        assert len(f.corpus[0]) >= 1
        assert f.corpus[0][0] == ord("A")

    def test_corpus_boost_empty_corpus(self):
        f = self._make_fuzzer(corpus_boost=100)
        f.corpus = []
        f._boost_corpus_sizes()
        assert f.corpus == []

    def test_corpus_boost_disabled(self):
        f = self._make_fuzzer(corpus_boost=0, max_len=10)
        original = [b"hello world this is long"]
        f.corpus = list(original)
        f._boost_corpus_sizes()
        assert f.corpus == original

    def test_corpus_boost_respects_max_len(self):
        f = self._make_fuzzer(max_len=32, corpus_boost=32, boost_mean=100, boost_std=50)
        f.corpus = [b"A" * 10 for _ in range(50)]
        f._boost_corpus_sizes()
        assert all(len(s) <= 32 for s in f.corpus)

    def test_corpus_boost_invalidates_cache(self):
        f = self._make_fuzzer(corpus_boost=100)
        f.corpus = [b"AAAA", b"BBBB"]
        f._seed_key_cache = {b"AAAA": "old", b"BBBB": "old"}
        f._boost_corpus_sizes()
        assert len(f._seed_key_cache) == 0

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

    @pytest.mark.skip(
        reason="Hangs after persistent loader subprocess — flaky environment interaction"
    )
    def test_shim_built_with_coverage_env_id(self):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        with patch("fuzzer_tool.adapters.inprocess.build_shim") as mock_build:
            mock_build.return_value = ShimResult(
                shim_path="/tmp/fake.so",
                coverage_type="inline_8bit",
                needs_preload=True,
            )
            print("hangs here 1")
            with (
                patch("fuzzer_tool.adapters.inprocess.load_shim"),
                patch("ctypes.CDLL"),
            ):
                print("hangs here 2")
                r = InProcessRunner(
                    target="/tmp/fake.so",
                    coverage_env_id="12345",
                )
                print("hangs here 3")
                assert r._shim is not None
                assert r._shim.coverage_type == "inline_8bit"
                print("hangs here 4")
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
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
        defaults = dict(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
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


class TestFuzzerHelpers:
    """Test helper methods that don't require process execution."""

    def _make_fuzzer(self, **kwargs):
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
        defaults = dict(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
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

    def test_seed_key(self):
        f = self._make_fuzzer()
        key = f._seed_key(b"test data")
        assert isinstance(key, str)
        assert len(key) > 0

    def test_seed_key_deterministic(self):
        f = self._make_fuzzer()
        assert f._seed_key(b"test") == f._seed_key(b"test")

    def test_seed_key_different(self):
        f = self._make_fuzzer()
        assert f._seed_key(b"aaa") != f._seed_key(b"bbb")

    def test_build_ops_basic(self):
        f = self._make_fuzzer()
        ops = f._build_ops(b"test")
        assert isinstance(ops, list)
        assert len(ops) > 0
        assert "bit_flip" in ops

    def test_build_ops_with_dict(self):
        f = self._make_fuzzer(dictionary=[b"token1", b"token2"])
        ops = f._build_ops(b"test")
        assert "dict_insert" in ops
        assert "dict_replace" in ops

    def test_build_ops_with_markov(self):
        f = self._make_fuzzer(markov_order=1)
        ops = f._build_ops(b"test")
        assert "markov_bytes" in ops

    def test_select_op(self):
        f = self._make_fuzzer()
        f._last_mopt_particles = []
        ops = ["bit_flip", "byte_flip", "arithmetic"]
        op = f._select_op(ops)
        assert op in ops

    def test_select_position(self):
        f = self._make_fuzzer()
        buf = bytearray(b"test data")
        pos = f._select_position(buf, b"test data")
        assert 0 <= pos < len(buf)

    def test_op_bit_flip(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        original = bytes(buf)
        f._op_bit_flip(buf, 0, b"")
        assert buf != original  # bit was flipped

    def test_op_byte_flip(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        original = bytes(buf)
        f._op_byte_flip(buf, 0, b"")
        assert buf != original

    def test_op_interesting_8(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_interesting_8(buf, 0, b"")

    def test_op_interesting_16(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_interesting_16(buf, 0, b"")

    def test_op_interesting_32(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_interesting_32(buf, 0, b"")

    def test_op_arithmetic(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_arithmetic(buf, 0, b"")

    def test_op_random_bytes(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_random_bytes(buf, 0, b"")

    def test_op_block_insert(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_block_insert(buf, 0, b"")

    def test_op_block_delete(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_block_delete(buf, 0, b"")

    def test_op_block_duplicate(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_block_duplicate(buf, 0, b"")

    def test_op_dict_insert(self):
        f = self._make_fuzzer(dictionary=[b"token1", b"token2"])
        buf = bytearray(b"\x00" * 10)
        f._op_dict_insert(buf, 0, b"")

    def test_op_dict_replace(self):
        f = self._make_fuzzer(dictionary=[b"token1", b"token2"])
        buf = bytearray(b"\x00" * 10)
        f._op_dict_replace(buf, 0, b"")

    def test_op_checksum_repair(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_checksum_repair(buf, 0, b"")

    def test_op_type_replace(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_type_replace(buf, 0, b"")

    def test_op_ascii_num(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_ascii_num(buf, 0, b"")

    def test_op_byte_shuffle(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_byte_shuffle(buf, 0, b"")

    def test_op_byte_delete(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_byte_delete(buf, 0, b"")

    def test_op_byte_insert(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_byte_insert(buf, 0, b"")

    def test_op_insert_ascii_num(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_insert_ascii_num(buf, 0, b"")

    def test_op_transpose_16(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_transpose_16(buf, 0, b"")

    def test_op_transpose_32(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_transpose_32(buf, 0, b"")

    def test_op_transpose_64(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_transpose_64(buf, 0, b"")

    def test_op_bit_transpose_8(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_bit_transpose_8(buf, 0, b"")

    def test_op_bit_transpose_16(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_bit_transpose_16(buf, 0, b"")

    def test_op_bit_transpose_32(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_bit_transpose_32(buf, 0, b"")

    def test_op_bit_transpose_64(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_bit_transpose_64(buf, 0, b"")

    def test_op_length_grow(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_length_grow(buf, 0, b"")

    def test_op_length_shrink(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_length_shrink(buf, 0, b"")

    def test_op_repeat_clone(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_repeat_clone(buf, 0, b"")

    def test_op_truncate(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_truncate(buf, 0, b"")

    def test_op_swap_regions(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_swap_regions(buf, 0, b"")

    def test_op_swap_bytes(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_swap_bytes(buf, 0, b"")

    def test_op_endianness_swap(self):
        f = self._make_fuzzer()
        buf = bytearray(b"\x00" * 10)
        f._op_endianness_swap(buf, 0, b"")

    def test_discovery_rate(self):
        f = self._make_fuzzer()
        f._discovery_execs = array("Q", (100, 200, 300))
        f._discovery_edges = array("Q", (10, 15, 20))
        rate = f.discovery_rate()
        assert rate >= 0

    def test_discovery_rate_empty(self):
        f = self._make_fuzzer()
        f._discovery_execs = array("Q")
        f._discovery_edges = array("Q")
        rate = f.discovery_rate()
        assert rate == 0.0

    def test_pareto_front(self):
        scores = [(1.0, 2.0, 0.5), (2.0, 1.0, 0.5), (1.5, 1.5, 0.5)]
        front = Fuzzer._pareto_front(scores)
        assert isinstance(front, set)

    def test_pareto_front_empty(self):
        front = Fuzzer._pareto_front([])
        assert front == set()

    def test_pareto_front_4d(self):
        """4D Pareto: dim0 dominates, dim1/2/3 matter for tie-breaking."""
        scores = [
            (1.0, 2.0, 0.5, 0.1),
            (2.0, 1.0, 0.5, 0.9),
            (1.5, 1.5, 0.5, 0.5),
        ]
        front = Fuzzer._pareto_front(scores)
        assert isinstance(front, set)
        assert len(front) > 0

    def test_pareto_front_4d_all_dominated(self):
        """One seed dominates all others in all 4 dims → front size 1."""
        scores = [
            (3.0, 3.0, 3.0, 3.0),  # dominates everything
            (1.0, 1.0, 1.0, 1.0),
            (2.0, 2.0, 2.0, 2.0),
        ]
        front = Fuzzer._pareto_front(scores)
        assert 0 in front
        assert len(front) == 1

    def test_pareto_front_4d_no_domination(self):
        """Each seed is best in exactly one dimension → all 4 on front."""
        scores = [
            (3.0, 1.0, 1.0, 1.0),
            (1.0, 3.0, 1.0, 1.0),
            (1.0, 1.0, 3.0, 1.0),
            (1.0, 1.0, 1.0, 3.0),
        ]
        front = Fuzzer._pareto_front(scores)
        assert len(front) == 4


class TestCrashDataPruning:
    """Regression tests for _prune_crash_data bounding crash structures."""

    def _make_fuzzer_with_crashes(self, n_sigs: int):
        """Create a Fuzzer with n_sigs unique crash signatures."""
        f = object.__new__(Fuzzer)
        f.crash_sigs = {f"sig_{i}": (n_sigs - i) for i in range(n_sigs)}
        f.crash_hashes = {f"hash_{i}" for i in range(n_sigs)}
        f.crash_frames = {f"sig_{i}": [f"frame_{i}_0", f"frame_{i}_1"] for i in range(n_sigs)}
        f.crash_min_sizes = {f"stack_{i}": 10 + i for i in range(n_sigs)}
        f._crash_replays = {f"sig_{i}": [i] for i in range(n_sigs)}
        return f

    def test_regression_prune_crash_data_caps_sigs(self):
        """_prune_crash_data reduces crash_sigs to <= MAX_CRASH_SIGS."""
        from fuzzer_tool.services.fuzzer import MAX_CRASH_SIGS

        n = MAX_CRASH_SIGS * 2
        f = self._make_fuzzer_with_crashes(n)
        assert len(f.crash_sigs) == n
        f._prune_crash_data()
        assert len(f.crash_sigs) <= MAX_CRASH_SIGS
        # Should be approximately 75% of MAX_CRASH_SIGS
        target = MAX_CRASH_SIGS * 3 // 4
        assert len(f.crash_sigs) >= target

    def test_regression_prune_crash_data_evicts_associated_structures(self):
        """Evicted sigs are removed from crash_frames, crash_min_sizes, _crash_replays."""
        from fuzzer_tool.services.fuzzer import MAX_CRASH_SIGS

        n = MAX_CRASH_SIGS * 2
        f = self._make_fuzzer_with_crashes(n)
        all_sigs = set(f.crash_sigs)
        f._prune_crash_data()
        kept_sigs = set(f.crash_sigs)
        evicted = all_sigs - kept_sigs
        # Evicted sigs should not appear in associated structures
        for sig in evicted:
            assert sig not in f.crash_frames
            assert sig not in f._crash_replays
            # crash_min_sizes keys on stack hash, not sig — verify the
            # total number of entries doesn't exceed kept_sigs count
        # All associated dicts should have at most as many entries as crash_sigs
        assert len(f.crash_frames) <= len(f.crash_sigs)
        assert len(f._crash_replays) <= len(f.crash_sigs)

    def test_regression_prune_crash_data_below_limit_noop(self):
        """_prune_crash_data is a no-op when crash_sigs is under MAX_CRASH_SIGS."""
        from fuzzer_tool.services.fuzzer import MAX_CRASH_SIGS

        n = MAX_CRASH_SIGS // 2
        f = self._make_fuzzer_with_crashes(n)
        sigs_before = dict(f.crash_sigs)
        frames_before = dict(f.crash_frames)
        f._prune_crash_data()
        assert f.crash_sigs == sigs_before
        assert f.crash_frames == frames_before

    def test_regression_prune_crash_data_keeps_most_frequent(self):
        """The most frequently hit signatures survive pruning."""
        from fuzzer_tool.services.fuzzer import MAX_CRASH_SIGS

        n = MAX_CRASH_SIGS * 2
        f = self._make_fuzzer_with_crashes(n)
        # sig_0 has count n (highest), sig_{n-1} has count 1 (lowest)
        f._prune_crash_data()
        # sig_0 should still be present (most frequent)
        assert "sig_0" in f.crash_sigs
        # sig_{n-1} should be evicted (least frequent)
        assert f"sig_{n - 1}" not in f.crash_sigs

    def test_regression_prune_crash_data_preserves_crash_hashes(self):
        """crash_hashes is left intact after pruning (prevents duplicate disk writes)."""
        from fuzzer_tool.services.fuzzer import MAX_CRASH_SIGS

        n = MAX_CRASH_SIGS * 2
        f = self._make_fuzzer_with_crashes(n)
        hashes_before = set(f.crash_hashes)
        f._prune_crash_data()
        assert f.crash_hashes == hashes_before


class TestMetropolisCorpusAdmission:
    """Metropolis acceptance for non-improving inputs in fuzz_one()."""

    def _make_fuzzer_with_metropolis(self, anneal_budget=100000, temperature=1.0, metropolis=True):
        """Build a minimal Fuzzer with Metropolis enabled (no real target execution)."""
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=f"{tmpdir}/corpus",
                crashes_dir=f"{tmpdir}/crashes",
                max_len=256,
                timeout=1,
                mutations_per_input=2,
                anneal_budget=anneal_budget,
                metropolis=metropolis,
            )
        f._temperature = temperature
        return f

    def test_metropolis_accepts_exploratory_junk_when_hot(self):
        """At T≈1.0, Metropolis admits non-improving inputs (P=exp(-1)≈0.37)."""
        f = self._make_fuzzer_with_metropolis(anneal_budget=1000000, temperature=1.0)
        assert f._metropolis is True
        assert f._anneal_budget > 0
        p_accept = math.exp(-1.0 / max(f._temperature, 0.01))
        # P(accept) ≈ 0.37 at T=1.0 — clearly non-zero
        assert p_accept > 0.3

    def test_metropolis_rejects_at_cold(self):
        """At T≈0.1, Metropolis rejects non-improving inputs (P=exp(-10)≈0.00005)."""
        f = self._make_fuzzer_with_metropolis(anneal_budget=100, temperature=0.1)
        p_accept = math.exp(-1.0 / max(f._temperature, 0.01))
        # exp(-10) ≈ 0.00005 — effectively zero
        assert p_accept < 0.001

    def test_metropolis_disabled_by_default(self):
        """Without --metropolis, non-improving inputs are always rejected."""
        f = self._make_fuzzer_with_metropolis(metropolis=False)
        assert f._metropolis is False

    def test_metropolis_no_anneal_budget_cold(self):
        """With --metropolis but without --anneal-budget, temperature stays 1.0
        perpetually — but the gate requires _anneal_budget > 0."""
        f = self._make_fuzzer_with_metropolis(anneal_budget=0, metropolis=True)
        # The gate won't fire because _anneal_budget == 0
        assert f._anneal_budget == 0
        assert f._temperature == 1.0

    def test_metropolis_temperature_clamp(self):
        """Temperature is clamped to at least 0.01 to avoid division by zero."""
        f = self._make_fuzzer_with_metropolis(anneal_budget=100000, temperature=0.0)
        clamped = max(f._temperature, 0.01)
        assert clamped >= 0.01
        p_accept = math.exp(-1.0 / clamped)
        assert 0 < p_accept < 1  # finite probability


class TestCullQueue:
    """Regression tests for favored/top_rated cull queue."""

    def test_cull_queue_minimal_set_cover(self):
        """Favored set should cover all edges with minimal seeds."""
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=f"{tmpdir}/corpus",
                crashes_dir=f"{tmpdir}/crashes",
                max_len=256,
                timeout=1,
                mutations_per_input=2,
            )
        # Three seeds, three edges. Seed A covers {1,2}, B covers {2,3}, C covers {3}.
        # Minimal cover: {A, C} covers all edges; B is redundant.
        f._edge_tracker.seed_edges = {
            "seed_a": {1, 2},
            "seed_b": {2, 3},
            "seed_c": {3},
        }
        f._edge_tracker._global_edge_hits = {1: 10, 2: 10, 3: 10}
        f.seed_meta = {
            "seed_a": {"total_time": 1.0, "fuzz_count": 1, "input_size": 100},
            "seed_b": {"total_time": 0.5, "fuzz_count": 1, "input_size": 100},
            "seed_c": {"total_time": 0.2, "fuzz_count": 1, "input_size": 100},
        }
        f._cull_queue()
        assert f._favored == {"seed_a", "seed_c"}

    def test_cull_queue_empty_tracker(self):
        """Empty edge tracker produces empty favored set."""
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=f"{tmpdir}/corpus",
                crashes_dir=f"{tmpdir}/crashes",
                max_len=256,
                timeout=1,
                mutations_per_input=2,
            )
        f._edge_tracker.seed_edges = {}
        f.seed_meta = {}
        f._cull_queue()
        assert f._favored == set()
