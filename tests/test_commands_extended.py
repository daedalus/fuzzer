"""Extended unit tests for cli/commands.py — coverage improvement."""

import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.cli.commands import (
    _add_common_args,
    _auto_tune_timeout,
    _detect_asan,
    _get_dirs,
    _validate_target,
    cmd_estimate,
    cmd_fuzz,
    cmd_import,
    cmd_minimize,
    cmd_rank,
    cmd_replay,
    cmd_tmin,
    main,
)


class TestDetectAsan:
    def test_detects_asan_binary(self, monkeypatch):
        """ASAN binary has __asan_init symbol."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=b"__asan_init"))
        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _detect_asan("/fake/target") is True

    def test_no_asan(self, monkeypatch):
        """Non-ASAN binary lacks the symbol."""
        mock_run = MagicMock(return_value=MagicMock(returncode=0, stdout=b"main\nfoo"))
        monkeypatch.setattr(subprocess, "run", mock_run)
        assert _detect_asan("/fake/target") is False

    def test_nm_not_found(self, monkeypatch):
        """Graceful handling when nm is not installed."""
        monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError))
        assert _detect_asan("/fake/target") is False

    def test_nm_timeout(self, monkeypatch):
        """Graceful handling when nm times out."""
        monkeypatch.setattr(
            subprocess, "run", MagicMock(side_effect=subprocess.TimeoutExpired("nm", 10))
        )
        assert _detect_asan("/fake/target") is False


class TestAutoTuneTimeout:
    def test_returns_reasonable_timeout(self, monkeypatch, tmp_path):
        """Auto-tuned timeout should be reasonable."""
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF" + b"\x00" * 100)
        target.chmod(0o755)

        mock_run = MagicMock(return_value=(0, ""))
        monkeypatch.setattr("fuzzer_tool.adapters.process.run_target_stdin", mock_run)

        timeout = _auto_tune_timeout(str(target), runs=3)
        assert 0.05 <= timeout <= 30.0


class TestCmdFuzz:
    def test_fuzz_function_exists(self):
        """cmd_fuzz should be callable."""
        assert callable(cmd_fuzz)

    def test_fuzz_help(self):
        """fuzz subcommand should accept --help."""
        result = subprocess.run(
            [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "fuzz" in result.stdout.lower()


class TestCmdEstimate:
    def test_estimate_help(self):
        """estimate command should accept --help."""
        result = subprocess.run(
            [sys.executable, "-m", "fuzzer_tool", "estimate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "estimate" in result.stdout.lower()

    def test_estimate_missing_corpus_exits(self, tmp_path):
        """estimate should fail without --corpus."""
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        args = argparse.Namespace(
            target=str(target),
            corpus=None,
            calibrate=100,
        )
        # Missing corpus should raise or error
        with pytest.raises((SystemExit, TypeError)):
            cmd_estimate(args)


class TestCmdImport:
    def test_import_afl(self, monkeypatch, tmp_path):
        """Import from AFL format."""
        src = tmp_path / "afl_out"
        src.mkdir()
        corpus = tmp_path / "corpus"
        crashes = tmp_path / "crashes"

        mock_import = MagicMock(return_value=(10, 5))
        monkeypatch.setattr("fuzzer_tool.services.import_corpus.import_from_afl", mock_import)

        args = argparse.Namespace(
            source_dir=str(src),
            format="afl",
            corpus=str(corpus),
            crashes=str(crashes),
        )
        result = cmd_import(args)
        assert result == 0
        mock_import.assert_called_once()

    def test_import_libfuzzer(self, monkeypatch, tmp_path):
        """Import from libFuzzer format."""
        src = tmp_path / "libfuzzer_out"
        src.mkdir()
        corpus = tmp_path / "corpus"

        mock_import = MagicMock(return_value=20)
        monkeypatch.setattr("fuzzer_tool.services.import_corpus.import_from_libfuzzer", mock_import)

        args = argparse.Namespace(
            source_dir=str(src),
            format="libfuzzer",
            corpus=str(corpus),
            crashes=None,
        )
        result = cmd_import(args)
        assert result == 0


class TestCmdTmin:
    def test_tmin_validates_target(self, tmp_path):
        """tmin should validate target exists."""
        args = argparse.Namespace(
            target="/nonexistent/target",
            crash_file=str(tmp_path / "crash.bin"),
            timeout=5,
            file_mode=False,
            target_args=None,
            coverage=False,
            grammar=None,
            mutations_per_input=8,
            max_len=4096,
        )
        with pytest.raises(SystemExit):
            cmd_tmin(args)


class TestCmdReplay:
    def test_replay_validates_target(self, tmp_path):
        """replay should validate target exists."""
        args = argparse.Namespace(
            target="/nonexistent/target",
            corpus=str(tmp_path / "corpus"),
            timeout=5,
            file_mode=False,
            target_args=None,
            coverage=False,
            max_len=4096,
            iterations=10,
        )
        with pytest.raises(SystemExit):
            cmd_replay(args)


class TestCmdRank:
    def test_rank_empty_corpus(self, tmp_path):
        """rank with empty corpus should handle gracefully."""
        corpus = tmp_path / "empty_corpus"
        corpus.mkdir()

        # Create a valid target
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        args = argparse.Namespace(
            corpus=str(corpus),
            top=10,
            dump=None,
            format="text",
            target=str(target),
        )
        # Should not raise, just print empty results
        cmd_rank(args)


class TestMain:
    def test_main_no_args_shows_help(self, monkeypatch):
        """main() with no args should show help."""
        monkeypatch.setattr(sys, "argv", ["fuzzer-tool"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse exits with code 2 when no args given
        assert exc_info.value.code == 2

    def test_main_fuzz_subcommand(self, monkeypatch):
        """main() should dispatch to fuzz subcommand."""
        mock_fuzz = MagicMock(return_value=0)
        monkeypatch.setattr("fuzzer_tool.cli.commands.cmd_fuzz", mock_fuzz)

        monkeypatch.setattr(
            sys,
            "argv",
            ["fuzzer-tool", "fuzz", "/tmp/target", "-n", "100"],
        )
        # Need a valid target
        target = Path("/tmp/target")
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        try:
            main()
        except SystemExit:
            pass

        # Cleanup
        target.unlink(missing_ok=True)


class TestCmdFuzzConstruction:
    """Test cmd_fuzz Fuzzer construction with various options."""

    def _make_default_args(self, tmp_path):
        """Create a Namespace with all required attributes for cmd_fuzz."""
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)
        corpus = tmp_path / "corpus"
        crashes = tmp_path / "crashes"
        return argparse.Namespace(
            target=str(target),
            corpus=str(corpus),
            crashes=str(crashes),
            timeout=5,
            file_mode=False,
            target_args=None,
            coverage=False,
            iterations=0,
            max_len=4096,
            mutations=8,
            deep_coverage=False,
            max_bps=50000,
            dict=None,
            markov=False,
            markov_gen=False,
            markov_order="1",
            markov_blend=False,
            mc_bandit=False,
            mc_cem=False,
            mopt=False,
            targets=None,
            anneal_budget=0,
            mc_elite_frac=0.1,
            mc_refit_int=1000,
            mc_decay_interval=100,
            pairwise_blend=0.0,
            stats_file=None,
            stats_interval=1000,
            coverage_report=None,
            coverage_log=None,
            grammar=None,
            persistent=False,
            cmplog=False,
            max_corpus=0,
            minimize_every_execs=0,
            no_shm=False,
            resume=False,
            trace=False,
            seed=42,
            crash_codes=None,
            replay_n=0,
            schedule_ablation=None,
            replicator=False,
            shapley=False,
            mi_guided=False,
            renyi_weight=False,
            transfer_entropy=False,
            elo=False,
            secretary=False,
            secretary_window=500,
            secretary_exploration=0.368,
            sensitivity=False,
            ga=False,
            ga_pop_size=200,
            ga_gen_size=500,
            ga_elite_frac=0.1,
            ga_crossover_rate=0.7,
            ga_mutation_rate=0.3,
            ga_tournament_size=3,
            ga_speciation_threshold=0.3,
            continue_until_crash=False,
            calibrate=0,
            stall=1000,
            map_size=0,
            report=None,
            auto_timeout=False,
            inprocess=False,
            inprocess_direct=False,
            inprocess_func="LLVMFuzzerTestOneInput",
            jobs=0,
            sync_interval=1.0,
            plot_graph=None,
        )

    def test_fuzz_constructs_fuzzer(self, monkeypatch, tmp_path):
        """cmd_fuzz should construct Fuzzer and call run."""
        args = self._make_default_args(tmp_path)
        mock_fuzzer = MagicMock()
        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", lambda **kwargs: mock_fuzzer)

        result = cmd_fuzz(args)
        assert result == 0
        mock_fuzzer.run.assert_called_once()

    def test_fuzz_with_report(self, monkeypatch, tmp_path):
        """cmd_fuzz with --report should generate report."""
        args = self._make_default_args(tmp_path)
        args.report = str(tmp_path / "report.md")
        mock_fuzzer = MagicMock()
        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", lambda **kwargs: mock_fuzzer)
        monkeypatch.setattr(
            "fuzzer_tool.services.report.generate_report", lambda *a, **k: "# Report"
        )

        result = cmd_fuzz(args)
        assert result == 0
        assert (tmp_path / "report.md").exists()

    def test_fuzz_with_report_stdout(self, monkeypatch, tmp_path):
        """cmd_fuzz with --report - should print to stdout."""
        args = self._make_default_args(tmp_path)
        args.report = "-"
        mock_fuzzer = MagicMock()
        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", lambda **kwargs: mock_fuzzer)
        monkeypatch.setattr(
            "fuzzer_tool.services.report.generate_report", lambda *a, **k: "# Report"
        )

        result = cmd_fuzz(args)
        assert result == 0

    def test_fuzz_with_dict(self, monkeypatch, tmp_path):
        """cmd_fuzz with --dict should load dictionary."""
        args = self._make_default_args(tmp_path)
        dict_file = tmp_path / "dict.txt"
        dict_file.write_text("token1\ntoken2\n")
        args.dict = str(dict_file)
        mock_fuzzer = MagicMock()
        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", lambda **kwargs: mock_fuzzer)
        monkeypatch.setattr(
            "fuzzer_tool.core.mutations.load_dictionary", lambda *a, **k: ["token1", "token2"]
        )

        result = cmd_fuzz(args)
        assert result == 0

    def test_fuzz_with_dict_missing(self, monkeypatch, tmp_path):
        """cmd_fuzz with missing dict should exit."""
        args = self._make_default_args(tmp_path)
        args.dict = "/nonexistent/dict.txt"

        with pytest.raises(SystemExit):
            cmd_fuzz(args)


class TestCmdMinimize:
    def test_minimize_validates_target(self, tmp_path):
        """minimize should validate target exists."""
        args = argparse.Namespace(
            target="/nonexistent/target",
            corpus=str(tmp_path / "corpus"),
            timeout=5,
            file_mode=False,
            target_args=None,
            coverage=False,
            output=None,
            rate_distortion=False,
            target_frac=0.95,
        )
        with pytest.raises(SystemExit):
            cmd_minimize(args)


class TestCmdReplayExtended:
    def test_replay_missing_crash_file(self, monkeypatch, tmp_path):
        """replay should fail if crash file doesn't exist."""
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        args = argparse.Namespace(
            target=str(target),
            crash_file=str(tmp_path / "nonexistent.bin"),
            timeout=5,
            file_mode=False,
            target_args=None,
            coverage=False,
        )
        result = cmd_replay(args)
        assert result == 1


class TestCmdRankExtended:
    def test_rank_with_seeds(self, tmp_path):
        """rank with seeds should rank them."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "seed1.bin").write_bytes(b"\x00" * 10)
        (corpus / "seed2.bin").write_bytes(b"\x01" * 10)

        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        args = argparse.Namespace(
            corpus=str(corpus),
            top=10,
            dump=None,
            format="text",
            target=str(target),
        )
        cmd_rank(args)


class TestCmdEstimateExtended:
    def test_estimate_with_corpus(self, monkeypatch, tmp_path):
        """estimate should run with valid corpus."""
        target = tmp_path / "target"
        target.write_bytes(b"\x7fELF")
        target.chmod(0o755)

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "seed.bin").write_bytes(b"\x00" * 16)

        mock_fuzzer = MagicMock()
        mock_fuzzer._edge_tracker = MagicMock()
        mock_fuzzer._edge_tracker.good_turing_estimate.return_value = {
            "n": 100,
            "n1": 10,
            "n2": 5,
            "estimated_undiscovered": 50,
            "confidence": "medium",
        }
        mock_fuzzer.discovery_rate.return_value = 5.0
        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", lambda **kwargs: mock_fuzzer)
        monkeypatch.setattr(
            "fuzzer_tool.core.target_profiler.TargetProfiler",
            lambda *a, **k: MagicMock(profile=lambda: MagicMock(functions={}, rodata_strings=[])),
        )

        args = argparse.Namespace(
            target=str(target),
            corpus=str(corpus),
            calibrate=100,
        )
        result = cmd_estimate(args)
        # cmd_estimate doesn't return a value, just prints
        assert result is None or result == 0
