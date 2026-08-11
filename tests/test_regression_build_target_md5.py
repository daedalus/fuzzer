"""Target checksum tracking in tools/build_targets.sh.

The `removed` branch shipped dead: `.target.md5` was only ever appended to
or replaced in place, never truncated, so the record was permanently a
superset of the `.prev` baseline and nothing could be missing from it. That
is not the sort of thing a reader catches, so it gets a test that actually
runs the shell.

The functions are extracted from the script and sourced, which keeps them
under test without building anything.
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(not BUILD_SCRIPT.exists(), reason="build_targets.sh not present")


def _extract_functions() -> str:
    """Pull the md5 helpers out of the build script."""
    text = BUILD_SCRIPT.read_text()
    start = text.index("# Escape a path for use")
    end = text.index("# ── Vendored libpng / zlib selection")
    body = text[start:end]
    for name in (
        "_md5_re()",
        "snapshot_target_md5()",
        "record_target_md5()",
        "verify_target_md5()",
    ):
        assert name in body, f"{name} not found; extraction range is stale"
    return body


PRELUDE = textwrap.dedent(
    """
    TARGETS_MD5=".target.md5"
    ok() { echo "OK: $1"; }
    warn() { echo "WARN: $1"; }
    # Calls the script's own snapshot_target_md5 rather than reimplementing
    # the snapshot-and-truncate, so reverting the truncation fails a test
    # instead of being masked by the harness.
    build() {
        snapshot_target_md5
        for f in "$@"; do record_target_md5 "$f"; done
        verify_target_md5
    }
    """
)


def run(tmp_path: Path, script: str) -> str:
    full = PRELUDE + _extract_functions() + "\n" + textwrap.dedent(script)
    proc = subprocess.run(
        ["bash", "-c", full], cwd=tmp_path, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


class TestChangeDetection:
    def test_first_build_reports_new(self, tmp_path):
        out = run(tmp_path, "echo a > png; echo b > jpeg; build png jpeg")
        assert "OK: 2 new targets" in out

    def test_identical_rebuild_reports_unchanged(self, tmp_path):
        out = run(tmp_path, "echo a > png; build png >/dev/null; build png")
        assert "OK: 1 targets unchanged" in out
        assert "checksum changed" not in out

    def test_modified_binary_reports_changed(self, tmp_path):
        out = run(tmp_path, "echo a > png; build png >/dev/null; echo NEW > png; build png")
        assert "WARN: png: checksum changed" in out


class TestRemovedDetection:
    """The branch that could not fire before the record was truncated."""

    def test_vanished_binary_is_reported(self, tmp_path):
        out = run(
            tmp_path,
            """
            echo a > png; echo b > jpeg
            build png jpeg >/dev/null
            rm -f jpeg
            build png
            """,
        )
        assert "WARN: jpeg: was built previously, binary is gone" in out
        assert "WARN: 1 targets removed from build" in out

    def test_partial_build_does_not_report_removed(self, tmp_path):
        """Not rebuilt is not gone.

        A partial build (--asan alone) legitimately skips most targets. If
        those were reported as removed the warning would fire on every
        normal partial build and be trained away.
        """
        out = run(
            tmp_path,
            """
            echo a > png; echo b > jpeg
            build png jpeg >/dev/null
            build png
            """,
        )
        assert "removed" not in out
        assert "1 not rebuilt this run" in out

    def test_partial_build_keeps_the_rest_of_the_record(self, tmp_path):
        out = run(
            tmp_path,
            """
            echo a > png; echo b > jpeg; echo c > zlib
            build png jpeg zlib >/dev/null
            build png >/dev/null
            cat "$TARGETS_MD5"
            """,
        )
        recorded = {line.split()[1] for line in out.strip().splitlines() if line.strip()}
        assert recorded == {"png", "jpeg", "zlib"}

    def test_record_has_no_duplicate_paths(self, tmp_path):
        out = run(
            tmp_path,
            """
            echo a > png; echo b > jpeg
            build png jpeg >/dev/null
            build png >/dev/null
            build png jpeg >/dev/null
            cat "$TARGETS_MD5"
            """,
        )
        paths = [line.split()[1] for line in out.strip().splitlines() if line.strip()]
        assert len(paths) == len(set(paths)), paths


class TestActualVerification:
    def test_binary_modified_after_the_build_is_caught(self, tmp_path):
        """The recorded checksum must be compared against the file.

        Before, the stored md5 was parsed and never used: the function only
        diffed disk against the previous build, so it was a change detector
        wearing the name of a verifier.
        """
        out = run(
            tmp_path,
            """
            echo a > png
            build png >/dev/null
            echo TAMPERED > png
            verify_target_md5
            """,
        )
        assert "WARN: png: on-disk checksum does not match the recorded one" in out
        assert "WARN: 1 targets modified after build" in out


class TestPathHandling:
    def test_regex_metacharacters_in_filenames(self, tmp_path):
        """Unescaped dots in "png_read.so" would match any character."""
        out = run(
            tmp_path,
            """
            echo a > png_readXso
            echo b > png_read.so
            # Dotted name second: record_target_md5 deletes matching entries
            # before appending, so an unescaped "." only eats a line that is
            # already there.
            build png_readXso png_read.so >/dev/null
            cat "$TARGETS_MD5"
            """,
        )
        paths = {line.split()[1] for line in out.strip().splitlines() if line.strip()}
        assert paths == {"png_read.so", "png_readXso"}

    def test_stale_prev_is_cleaned_when_record_is_absent(self, tmp_path):
        out = run(
            tmp_path,
            """
            touch "$TARGETS_MD5.prev"
            verify_target_md5
            [ -f "$TARGETS_MD5.prev" ] && echo LEFTOVER || echo CLEANED
            """,
        )
        assert "CLEANED" in out


class TestGitignore:
    def test_prev_file_is_ignored(self):
        """An interrupted build leaves .target.md5.prev behind."""
        ignore = (BUILD_SCRIPT.parent.parent / ".gitignore").read_text()
        assert re.search(r"^\.target\.md5\.prev$", ignore, re.M)
