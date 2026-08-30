"""Checks for the in-process GNU Grep target.

`targets/grep_read.c` used to fork/exec `/usr/bin/grep` once per execution.
That had two consequences, and neither one was visible as a failure:

- The code under test ran in a different, uninstrumented process image, so
  every edge the target reported was an edge of the wrapper itself. The
  campaign looked healthy -- edges climbed, the corpus grew -- while
  covering none of grep.
- A process spawn per execution reintroduced exactly the cost that the
  `direct_lite` set exists to avoid (see
  docs/handover/handover_boltzmann_ab_2026-08-30.md section 1).

Both failure modes are silent, so the first test here is a revert guard: if
the target ever goes back to spawning a process, the suite says so instead
of reporting a healthy campaign over fake coverage. The remaining tests
compile the real harness and run it, which is the only layer that can
detect the engines no longer being driven.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "targets", "grep_read.c")
VENDOR = os.path.join(ROOT, "vendor", "grep")
ARCHIVE = os.path.join(VENDOR, "lib", "libgreputils.a")


def _source() -> str:
    with open(TARGET, encoding="utf-8") as fh:
        return fh.read()


def _strip_comments(text: str) -> str:
    """Drop comments so prose about the old design does not read as code.

    The header comment explains the exec wrapper it replaced, and names
    execlp and /usr/bin/grep while doing so. Matching raw source text would
    make this test fail on its own documentation.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    """Compile the real harness against the vendored engines.

    grep_read.c supplies its own main(), which reads a testcase from stdin;
    the only thing missing outside the real build is __afl_map_edge, which
    afl_shim.c normally provides. Stubbing just that keeps this test
    independent of the shim while still exercising the harness itself.
    """
    out = tmp_path_factory.mktemp("grep_target") / "grep_read"
    stub = tmp_path_factory.mktemp("grep_stub") / "stub.c"
    stub.write_text("void __afl_map_edge(unsigned int c) { (void)c; }\n")
    cmd = [
        "cc",
        "-O1",
        "-g",
        "-o",
        str(out),
        "-include",
        os.path.join(VENDOR, "config.h"),
        f"-I{VENDOR}",
        f"-I{os.path.join(VENDOR, 'lib')}",
        f"-I{os.path.join(VENDOR, 'src')}",
        TARGET,
        str(stub),
        os.path.join(VENDOR, "lib", "dfa.c"),
        os.path.join(VENDOR, "lib", "localeinfo.c"),
        os.path.join(VENDOR, "src", "kwset.c"),
        ARCHIVE,
    ]
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.returncode != 0:
        pytest.skip(f"grep target did not build here: {rc.stderr[-500:]}")
    return str(out)


class TestNoProcessSpawn:
    """The target must drive grep in-process, not spawn it."""

    def test_no_exec_of_a_grep_binary(self):
        code = _strip_comments(_source())
        for bad in ("execlp", "execvp", "execl", "execv", "posix_spawn"):
            assert bad not in code, (
                f"{bad} in grep_read.c: the target is spawning a process again. "
                "Coverage would be of the wrapper only."
            )

    def test_no_fork(self):
        code = _strip_comments(_source())
        assert not re.search(r"\bfork\s*\(", code), "grep_read.c forks again"

    def test_no_system_grep_path(self):
        code = _strip_comments(_source())
        assert "/usr/bin/grep" not in code
        assert "/bin/grep" not in code

    def test_links_the_vendored_engines(self):
        """The harness must call grep's own matchers, not libc regex.

        Falling back to POSIX regcomp/regexec would still compile and still
        produce coverage, but it would be coverage of glibc rather than of
        the vendored grep -- the same class of mistake as the exec wrapper,
        one layer down.
        """
        code = _source()
        for sym in ("dfacomp", "dfaexec", "kwsexec"):
            assert sym in code, f"{sym} missing: the grep engines are not driven"


@pytest.mark.skipif(
    not os.path.exists(ARCHIVE),
    reason="vendor/grep not built (run tools/vendor_grep.sh)",
)
@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
class TestHarnessRuns:
    """Compile the real harness and drive it, including the error paths."""

    def _run(self, binary, payload: bytes):
        return subprocess.run([binary], input=payload, capture_output=True, timeout=60)

    def test_valid_pattern_runs_clean(self, binary):
        # mode 1 (extended DFA), pattern "a(b|c)+d", then subject text.
        payload = b"\x01\x08" + b"a(b|c)+d" + b"xxabcbcdyy\n"
        assert self._run(binary, payload).returncode == 0

    def test_malformed_pattern_is_trapped_not_fatal(self, binary):
        """A bad pattern must not take the process down.

        The DFA engine calls dfaerror() for most fuzzer-generated patterns
        and dfaerror is _Noreturn -- grep exits there. A persistent target
        cannot, so the harness longjmps out. If that trap is removed this
        aborts, which is a signal, not a clean exit.
        """
        payload = b"\x00\x02" + b"a[" + b"some text"
        assert self._run(binary, payload).returncode == 0

    def test_fixed_string_mode_runs(self, binary):
        payload = b"\x02\x03" + b"bcd" + b"hello bcd world"
        assert self._run(binary, payload).returncode == 0

    def test_every_mode_survives_a_hostile_input(self, binary):
        """Sweep the mode byte over its whole range.

        Modes are taken modulo MODE_MAX, so a byte outside the table must
        still land on a real engine rather than falling through.
        """
        for mode in range(0, 12):
            payload = bytes([mode, 4]) + b"[[:a" + b"\x00\xff\n zzz"
            rc = self._run(binary, payload)
            assert rc.returncode == 0, f"mode {mode} exited {rc.returncode}"

    def test_pattern_may_contain_nul(self, binary):
        """Patterns are passed as (pointer, length), not as C strings.

        The exec wrapper had to NUL-terminate for argv, which truncated
        every pattern at its first zero byte and made that region of the
        input space unreachable.
        """
        payload = b"\x02\x03" + b"a\x00c" + b"xxa\x00cyy"
        assert self._run(binary, payload).returncode == 0
