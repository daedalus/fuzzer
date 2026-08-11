"""Tests for core/root_cause.py (edit-script ddmin) and
services/root_cause.py (crash reproduction + baseline selection)."""

from unittest.mock import patch

from fuzzer_tool.core.root_cause import (
    apply_edit_subset,
    build_edit_script,
    ddmin_edits,
    edit_indices,
    format_root_cause_report,
)
from fuzzer_tool.services.root_cause import root_cause


class TestEditScript:
    def test_identical_inputs_produce_no_edits(self):
        script = build_edit_script(b"hello", b"hello")
        assert edit_indices(script) == []

    def test_reconstructs_target_from_full_subset(self):
        base = b"hello world this is fine"
        crash = b"hello wOrld this iz fine EXTRA"
        script = build_edit_script(base, crash)
        idx = set(edit_indices(script))
        assert apply_edit_subset(base, script, idx) == crash

    def test_empty_subset_reconstructs_base(self):
        base = b"hello world"
        crash = b"HELLO WORLD"
        script = build_edit_script(base, crash)
        assert apply_edit_subset(base, script, set()) == base


class TestDdminEdits:
    def test_isolates_single_causal_replace(self):
        """8 total edits, but only replacing byte 7 ('o'->'O') matters."""
        base = b"hello world this is fine"
        crash = b"hello wOrld this iz fine EXTRA"
        script = build_edit_script(base, crash)
        assert len(edit_indices(script)) > 1

        def interesting(candidate: bytes) -> bool:
            return len(candidate) > 7 and candidate[7] == ord("O")

        minimal_bytes, minimal_idx = ddmin_edits(base, script, interesting, max_stages=200)
        assert len(minimal_idx) == 1
        op, pos, data = script[minimal_idx[0]]
        assert (op, pos, data) == ("replace", 7, b"O")
        assert interesting(minimal_bytes)

    def test_requires_both_edits_together(self):
        """Crash needs the first AND last byte changed simultaneously —
        ddmin must not shrink below either one."""
        base = b"AAAABBBBCCCCDDDD"
        crash = b"XAAABBBBCCCCDDDY"
        script = build_edit_script(base, crash)
        changes = edit_indices(script)
        assert len(changes) == 2

        def interesting(candidate: bytes) -> bool:
            return candidate[:1] == b"X" and candidate[-1:] == b"Y"

        minimal_bytes, minimal_idx = ddmin_edits(base, script, interesting, max_stages=200)
        assert sorted(minimal_idx) == sorted(changes)
        assert minimal_bytes == crash

    def test_no_edits_returns_base_unchanged(self):
        base = b"same"
        script = build_edit_script(base, base)
        minimal_bytes, minimal_idx = ddmin_edits(base, script, lambda c: True, max_stages=50)
        assert minimal_bytes == base
        assert minimal_idx == []

    def test_nothing_interesting_falls_back_to_full_diff(self):
        """If interesting_fn never returns True for any subset, ddmin can't
        reduce and the result is the full edit set (still reconstructs crash
        bytes since that's what the caller already verified separately)."""
        base = b"AAAA"
        crash = b"BBBB"
        script = build_edit_script(base, crash)
        minimal_bytes, minimal_idx = ddmin_edits(base, script, lambda c: False, max_stages=50)
        assert sorted(minimal_idx) == edit_indices(script)

    def test_respects_max_stages_budget(self):
        """A pathological interesting_fn that's always False should not
        cause unbounded target executions."""
        base = bytes(range(50))
        crash = bytes((b + 1) % 256 for b in base)
        script = build_edit_script(base, crash)
        calls = {"n": 0}

        def counting_fn(candidate: bytes) -> bool:
            calls["n"] += 1
            return False

        ddmin_edits(base, script, counting_fn, max_stages=10)
        assert calls["n"] <= 10


class TestFormatReport:
    def test_empty_minimal_set_reports_no_isolation(self):
        base, crash = b"abc", b"abd"
        script = build_edit_script(base, crash)
        report = format_root_cause_report(base, crash, script, [])
        assert "no subset" in report

    def test_report_lists_replace_offsets(self):
        base, crash = b"abc", b"abd"
        script = build_edit_script(base, crash)
        idx = edit_indices(script)
        report = format_root_cause_report(base, crash, script, idx)
        assert "REPLACE" in report
        assert "0x0002" in report


class TestRootCauseService:
    def test_crash_file_missing(self):
        assert root_cause("/bin/true", "/nonexistent/crash.bin", corpus_dir="/tmp") is None

    def test_empty_crash_file(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"")
        assert root_cause("/bin/true", str(crash), corpus_dir=str(tmp_path)) is None

    def test_crash_not_reproduced(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(0, "", 1)):
            result = root_cause("/bin/true", str(crash), corpus_dir=str(tmp_path))
        assert result is None

    def test_no_baseline_source_given(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = root_cause("/bin/true", str(crash))
        assert result is None

    def test_empty_corpus_dir(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        empty_corpus = tmp_path / "corpus"
        empty_corpus.mkdir()
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = root_cause("/bin/true", str(crash), corpus_dir=str(empty_corpus))
        assert result is None

    def test_explicit_baseline_that_also_crashes_rejected(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        baseline = tmp_path / "baseline.bin"
        baseline.write_bytes(b"BBBB")

        # Everything crashes -> baseline is rejected as not-actually-safe
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = root_cause("/bin/true", str(crash), baseline_file=str(baseline))
        assert result is None

    def test_end_to_end_isolates_causal_byte(self, tmp_path):
        """Full pipeline: explicit baseline, real ddmin round-trip through
        a fake target that only crashes when byte 2 is 'X'."""
        crash_input = b"AAXAA"
        baseline_input = b"AAAAA"
        crash = tmp_path / "crash.bin"
        crash.write_bytes(crash_input)
        baseline = tmp_path / "baseline.bin"
        baseline.write_bytes(baseline_input)

        def fake_run(target, data, timeout, env=None):
            if len(data) > 2 and data[2] == ord("X"):
                return (-11, "", 1)
            return (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            result = root_cause(
                "/bin/true",
                str(crash),
                baseline_file=str(baseline),
                max_stages=100,
            )

        assert result is not None
        assert len(result["minimal_indices"]) == 1
        op, pos, data = result["script"][result["minimal_indices"][0]]
        assert (op, pos, data) == ("replace", 2, b"X")
        assert "REPLACE" in result["report"]

    def test_picks_nearest_corpus_seed_as_baseline(self, tmp_path):
        """Two corpus seeds: pick whichever is textually nearer to the crash,
        confirm it as non-crashing, and diff against it."""
        crash_input = b"hello wOrld"
        crash = tmp_path / "crash.bin"
        crash.write_bytes(crash_input)

        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        near = corpus_dir / "near.bin"
        near.write_bytes(b"hello world")
        far = corpus_dir / "far.bin"
        far.write_bytes(b"completely unrelated text of a different shape entirely")

        def fake_run(target, data, timeout, env=None):
            if b"O" in data:
                return (-11, "", 1)
            return (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_stdin", side_effect=fake_run):
            result = root_cause(
                "/bin/true",
                str(crash),
                corpus_dir=str(corpus_dir),
                max_stages=100,
            )

        assert result is not None
        assert result["baseline_name"] == "near.bin"
        assert len(result["minimal_indices"]) == 1
