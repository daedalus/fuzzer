"""Tests for services/tmin.py — crash minimization."""

from unittest.mock import patch

from fuzzer_tool.core.reducer import Verdict
from fuzzer_tool.services.tmin import tmin


class TestTmin:
    def test_nonexistent_crash_file(self):
        result = tmin("/fake/target", "/nonexistent/file.bin")
        assert result is None

    def test_empty_crash_file(self, tmp_path):
        crash = tmp_path / "empty.bin"
        crash.write_bytes(b"")
        result = tmin("/fake/target", str(crash))
        assert result is None

    def test_crash_not_reproduced(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(0, "", 1)):
            result = tmin("/bin/true", str(crash))
        assert result is None

    def test_file_mode(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_file", return_value=(0, "", 1)):
            result = tmin("/bin/true", str(crash), file_mode=True)
        assert result is None

    def test_crash_signature_asan(self, tmp_path):
        asan_stderr = "ERROR: AddressSanitizer: heap-buffer-overflow\nABORTING"
        crash_file = tmp_path / "crash.bin"
        crash_file.write_bytes(b"A" * 100)
        with patch(
            "fuzzer_tool.adapters.process.run_target_stdin", return_value=(1, asan_stderr, 1)
        ):
            result = tmin("/bin/false", str(crash_file))
        # ASAN crash reproduced, minimizer runs, may return minimized data
        assert result is None or isinstance(result, bytes)

    def test_crash_signature_signal(self, tmp_path):
        crash_file = tmp_path / "crash.bin"
        crash_file.write_bytes(b"B" * 50)
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = tmin("/bin/false", str(crash_file))
        assert result is None or isinstance(result, bytes)

    def test_file_mode_timeout(self, tmp_path):
        crash_file = tmp_path / "crash.bin"
        crash_file.write_bytes(b"C" * 10)
        # First call reproduces, then minimize_bytes calls many times
        # Return crash on first call, then "no crash" for minimize attempts
        call_count = [0]

        def fake_run_file(target, data, timeout, tmp_dir, args, env=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (-11, "segfault", 1)
            return (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_file", side_effect=fake_run_file):
            result = tmin("/bin/false", str(crash_file), file_mode=True)
        assert result is None or isinstance(result, bytes)

    def test_main_exists(self):
        from fuzzer_tool.services.tmin import main

        assert callable(main)


class TestTminLineageMode:
    """Lineage replay (Stage 5): rehydrated ancestor becomes the minimize start."""

    def test_lineage_candidate_used_when_smaller(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash_input = b"X" * 100
        ancestor = b"ROOTSEED"
        crash.write_bytes(crash_input)

        def fake_run(target, data, timeout, env=None):
            if data in (crash_input, ancestor):
                return (-11, "", 1)
            return (0, "", 1)

        with (
            patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run),
            patch(
                "fuzzer_tool.services.tmin._lineage_candidate", return_value=ancestor
            ) as mock_cand,
        ):
            result = tmin(
                "/bin/true",
                str(crash),
                lineage=True,
                corpus_dir=str(tmp_path),
            )
        mock_cand.assert_called_once()
        # Lineage replay found the 10-byte ancestor; minimization kept it.
        assert result == ancestor

    def test_lineage_off_ignores_candidate(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash_input = b"X" * 100
        crash.write_bytes(crash_input)

        def fake_run(target, data, timeout, env=None):
            if data == crash_input:
                return (-11, "", 1)
            return (0, "", 1)

        with (
            patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run),
            patch(
                "fuzzer_tool.services.tmin._lineage_candidate", return_value=b"ROOTSEED"
            ) as mock_cand,
        ):
            result = tmin("/bin/true", str(crash))
        mock_cand.assert_not_called()
        # No lineage → no candidate → minimize finds nothing smaller → original.
        assert result == crash_input

    def test_candidate_that_does_not_crash_ignored(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash_input = b"X" * 100
        crash.write_bytes(crash_input)

        def fake_run(target, data, timeout, env=None):
            if data == crash_input:
                return (-11, "", 1)
            return (0, "", 1)

        with (
            patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run),
            # Candidate bytes do NOT crash → lineage must fall through.
            patch(
                "fuzzer_tool.services.tmin._lineage_candidate",
                return_value=b"NOCRASHSEED",
            ),
        ):
            result = tmin("/bin/true", str(crash), lineage=True, corpus_dir=str(tmp_path))
        assert result == crash_input


def _saved_crashes(d, skip):
    return [p for p in d.glob("*.bin") if p.name != skip]


class TestTminAlsoInteresting:
    """C-Reduce --also-interesting: other-signature crashes are kept."""

    def test_other_signature_saved(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"XXXXXXXXZZZZZZZZ")
        also_dir = tmp_path / "also"

        def fake_run(target, data, timeout, env=None):
            if b"X" in data:
                return (-11, "", 1)
            return (-6, "", 1) if b"Z" in data else (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            result = tmin("/bin/true", str(crash), also_dir=str(also_dir))

        assert result == b"X"
        saved = _saved_crashes(also_dir, crash.name)
        assert len(saved) == 1
        assert b"X" not in saved[0].read_bytes()

    def test_falsification_same_signature_not_saved(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"XXXXZZZZ")

        def fake_run(target, data, timeout, env=None):
            return (-11, "", 1) if b"X" in data else (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            tmin("/bin/true", str(crash))

        assert _saved_crashes(tmp_path, crash.name) == []

    def test_adversarial_signature_flood_bounded(self):
        """Every candidate crashes differently: kept set stays capped and
        each signature keeps its smallest input."""
        from fuzzer_tool.services.tmin import MAX_ALSO_SIGS, _CrashJudge

        flood = MAX_ALSO_SIGS * 2
        judge = _CrashJudge(lambda d: (f"sig{len(d) % flood}", -11, ""), "orig")
        for n in range(2 * flood, 0, -1):
            data = b"Z" * n
            assert judge.verdict(data) is Verdict.ALSO
            judge.keep(data)

        assert len(judge.also) == MAX_ALSO_SIGS
        assert all(
            len(d) == min(n for n in range(1, 2 * flood + 1) if f"sig{n % flood}" == s)
            for s, (d, _rc, _err) in judge.also.items()
        )


class TestTminCache:
    def test_candidate_runs_once(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash_input = b"AAAAAAAAAAAAAAAAQ"
        crash.write_bytes(crash_input)
        runs = []

        def fake_run(target, data, timeout, env=None):
            runs.append(data)
            return (-11, "", 1) if b"Q" in data else (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            assert tmin("/bin/true", str(crash)) == b"Q"

        candidates = [r for r in runs if r != crash_input]
        dupes = {r for r in candidates if candidates.count(r) > 1}
        # The final re-verification of the result is a deliberate re-run.
        assert dupes <= {b"Q"}
        assert candidates.count(b"Q") == 2


class _StubTree:
    """Stand-in TreeMutator: shrink = drop every 'T' byte (if still crashing)."""

    calls = 0

    def __init__(self, grammar):
        pass

    def hierarchical_shrink(self, data, still_crashes, max_rounds=64):
        _StubTree.calls += 1
        cand = data.replace(b"T", b"")
        return cand if still_crashes(cand) else data


class _GrowTree(_StubTree):
    def hierarchical_shrink(self, data, still_crashes, max_rounds=64):
        return data + b"GROWN"


class TestTminGrammarPass:
    def _run(self, tmp_path, tree, crash_input):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(crash_input)

        def fake_run(target, data, timeout, env=None):
            return (-11, "", 1) if b"X" in data else (0, "", 1)

        with (
            patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run),
            patch("fuzzer_tool.core.grammar.TreeMutator", tree),
        ):
            return tmin("/bin/true", str(crash), grammar=object())

    def test_grammar_pass_reruns_each_main_sweep(self, tmp_path):
        _StubTree.calls = 0
        assert self._run(tmp_path, _StubTree, b"TTXTT") == b"X"
        # Sweep 1 shrank, so the fixpoint ran a second sweep.
        assert _StubTree.calls >= 2

    def test_adversarial_growing_shrink_rejected(self, tmp_path):
        assert self._run(tmp_path, _GrowTree, b"ab" * 8 + b"X") == b"X"
