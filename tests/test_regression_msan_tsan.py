"""Regression tests: MSAN / TSAN detection and report parsing.

`sanitizer.py` has parsed MemorySanitizer and ThreadSanitizer reports since
early on, but `build_targets.sh` only ever emitted -fsanitize=address and
-fsanitize=undefined — the detection half existed while the build half did
not, so two bug classes were unreachable in practice:

  * MSAN finds use-of-uninitialized-value, which ASAN cannot detect at all.
  * TSAN finds data races.

These tests compile tiny targets exercising each bug class and pin both that
the sanitizer fires and that SanitizerReport extracts the *full* error type.
The TSAN case is a regression guard specifically: TSAN spells its error
"data race" with a space, but the pattern matched "data-race" with a hyphen,
so it never fired and the generic fallback truncated the error type — and
therefore the dedup signature — to just "data".
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.sanitizer import SanitizerReport

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang required")

MSAN_SRC = """\
#include <stdio.h>
#include <unistd.h>
int main(void) {
    char buf[64];
    ssize_t n = read(0, buf, 8);
    if (n <= 0) return 0;
    int uninit;                 /* deliberately never initialized */
    if (buf[0] == 'M') {
        if (uninit == 12345) { printf("hit\\n"); }
    }
    return 0;
}
"""

TSAN_SRC = """\
#include <pthread.h>
#include <unistd.h>
static int shared = 0;
static void *worker(void *arg) { (void)arg; shared++; return 0; }
int main(void) {
    char buf[8];
    if (read(0, buf, 1) <= 0) return 0;
    if (buf[0] != 'T') return 0;
    pthread_t a, b;
    pthread_create(&a, 0, worker, 0);
    pthread_create(&b, 0, worker, 0);
    pthread_join(a, 0); pthread_join(b, 0);
    return 0;
}
"""


def _build(tmp_path, name: str, src: str, flags: list[str]):
    c = tmp_path / f"{name}.c"
    c.write_text(src)
    out = tmp_path / name
    proc = subprocess.run(
        ["clang", *flags, "-fno-omit-frame-pointer", "-g", "-O0", "-o", str(out), str(c)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"{name} build unavailable: {proc.stderr[:200]}")
    return out


def _run(binary, stdin: bytes) -> str:
    return subprocess.run([str(binary)], input=stdin, capture_output=True).stderr.decode(
        errors="replace"
    )


class TestMsan:
    def test_msan_detects_uninitialized_read(self, tmp_path):
        binary = _build(
            tmp_path,
            "msan_bug",
            MSAN_SRC,
            ["-fsanitize=memory", "-fsanitize-memory-track-origins=2", "-fPIE", "-pie"],
        )
        report = SanitizerReport.parse(_run(binary, b"MMMMMMMM"))
        assert report is not None, "MSAN produced no parseable report"
        assert report.sanitizer == "MemorySanitizer"
        assert report.error_type == "use-of-uninitialized-value"

    def test_asan_cannot_see_the_same_bug(self, tmp_path):
        """The reason MSAN is worth building: ASAN is blind to this bug."""
        binary = _build(tmp_path, "asan_same", MSAN_SRC, ["-fsanitize=address"])
        proc = subprocess.run([str(binary)], input=b"MMMMMMMM", capture_output=True)
        assert proc.returncode == 0
        assert SanitizerReport.parse(proc.stderr.decode(errors="replace")) is None


class TestTsan:
    def test_tsan_detects_data_race(self, tmp_path):
        binary = _build(tmp_path, "tsan_bug", TSAN_SRC, ["-fsanitize=thread"])
        report = SanitizerReport.parse(_run(binary, b"T"))
        assert report is not None, "TSAN produced no parseable report"
        assert report.sanitizer == "ThreadSanitizer"

    def test_tsan_error_type_is_not_truncated(self, tmp_path):
        """Regression: "data race" has a space; matching "data-race" failed
        and the generic \\S+ fallback truncated it to "data", collapsing
        distinct TSAN bug classes into one dedup bucket."""
        binary = _build(tmp_path, "tsan_bug2", TSAN_SRC, ["-fsanitize=thread"])
        report = SanitizerReport.parse(_run(binary, b"T"))
        assert report is not None
        assert report.error_type == "data race", f"truncated to {report.error_type!r}"
        assert report.signature == "ThreadSanitizer:data race"


class TestErrorTypeExtraction:
    """Multi-word error types must survive parsing for every sanitizer."""

    @pytest.mark.parametrize(
        "stderr,sanitizer,error_type",
        [
            (
                "WARNING: ThreadSanitizer: data race (pid=1)",
                "ThreadSanitizer",
                "data race",
            ),
            (
                "WARNING: ThreadSanitizer: thread leak (pid=1)",
                "ThreadSanitizer",
                "thread leak",
            ),
            (
                "==1==WARNING: MemorySanitizer: use-of-uninitialized-value",
                "MemorySanitizer",
                "use-of-uninitialized-value",
            ),
            (
                "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1",
                "AddressSanitizer",
                "heap-buffer-overflow",
            ),
        ],
    )
    def test_full_error_type_parsed(self, stderr, sanitizer, error_type):
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert report.sanitizer == sanitizer
        assert report.error_type == error_type
