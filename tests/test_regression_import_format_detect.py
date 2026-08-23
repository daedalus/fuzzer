"""Regression: corpus format auto-detect was unreachable.

``services/import_corpus.py:main`` gated on
``args.format == "afl" or (args.format == "afl" and ...)`` -- a tautology whose
second disjunct can only be true when the first already is, and whose default
is ``"afl"``. Every invocation without an explicit ``--format`` took the AFL
branch, so a libFuzzer corpus imported 0 seeds and printed a success line.
"""

import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.services.import_corpus import _detect_source_format


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory(prefix="fuzz_import_") as d:
        yield Path(d)


# ── 3. import format detection ────────────────────────────────────────


class TestSourceFormatDetection:
    def test_afl_layout(self, tmp_root):
        (tmp_root / "queue").mkdir()
        (tmp_root / "crashes").mkdir()
        assert _detect_source_format(tmp_root) == "afl"

    def test_afl_by_fuzzer_stats(self, tmp_root):
        (tmp_root / "fuzzer_stats").write_text("execs_done : 1\n")
        assert _detect_source_format(tmp_root) == "afl"

    def test_libfuzzer_flat_corpus(self, tmp_root):
        for i in range(3):
            (tmp_root / f"{i:040x}").write_bytes(b"seed")
        assert _detect_source_format(tmp_root) == "libfuzzer"

    def test_honggfuzz_layout(self, tmp_root):
        (tmp_root / "cases_honggfuzz").mkdir()
        assert _detect_source_format(tmp_root) == "honggfuzz"


class TestSourceFormatFalsification:
    def test_libfuzzer_is_reachable_at_all(self, tmp_root):
        # Falsification of the tautology: under the old gate NO input could
        # ever produce anything but "afl". A flat corpus must not be AFL.
        for i in range(5):
            (tmp_root / f"seed{i}").write_bytes(b"x")
        assert _detect_source_format(tmp_root) != "afl"

    def test_afl_is_not_over_claimed(self, tmp_root):
        # And the complement: detection must not simply return libfuzzer for
        # everything now.
        (tmp_root / "queue").mkdir()
        assert _detect_source_format(tmp_root) == "afl"


class TestSourceFormatAdversarial:
    def test_afl_wins_over_flat_files(self, tmp_root):
        # An AFL output dir also has top-level files (fuzzer_stats, plot_data).
        # Most-specific-first ordering must keep it AFL.
        (tmp_root / "queue").mkdir()
        (tmp_root / "plot_data").write_text("x\n")
        (tmp_root / "fuzzer_stats").write_text("y\n")
        assert _detect_source_format(tmp_root) == "afl"

    def test_honggfuzz_wins_over_flat_files(self, tmp_root):
        (tmp_root / "cases_honggfuzz").mkdir()
        (tmp_root / "loose").write_bytes(b"x")
        assert _detect_source_format(tmp_root) == "honggfuzz"

    def test_empty_directory_falls_back_without_raising(self, tmp_root):
        assert _detect_source_format(tmp_root) in ("afl", "libfuzzer")

    def test_nonexistent_directory_does_not_raise(self, tmp_root):
        # A typo'd path must produce a format, not a traceback; the importer
        # reports the empty result.
        assert _detect_source_format(tmp_root / "missing") == "afl"

    def test_hangs_dir_is_treated_as_afl_output(self, tmp_root):
        # findings/ and hangs/ mark an AFL-family output tree even without
        # queue/, and must not be mistaken for a flat libFuzzer corpus.
        (tmp_root / "hangs").mkdir()
        assert _detect_source_format(tmp_root) != "libfuzzer"
