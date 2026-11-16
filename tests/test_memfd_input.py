"""In-memory input file (wtf fshooks port, part A).

The input lives in a memfd named ``/proc/<pid>/fd/<n>``: no disk writes per
exec, a path that never changes (so file-mode targets can use the forkserver,
whose argv is fixed at exec), and a path in-process harnesses can open
(``FUZZ_INPUT_PATH``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = str(REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c")

# Exit code = first byte of the file after "-i"; 200 open failed, 201 empty, 202 no -i.
ARGFILE_SRC = r"""
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "-i") != 0) continue;
        FILE *f = fopen(argv[i + 1], "rb");
        if (!f) return 200;
        int c = fgetc(f);
        fclose(f);
        return c == EOF ? 201 : c;
    }
    return 202;
}
"""

# In-process harness: returns the first byte of $FUZZ_INPUT_PATH, -1 if unset.
INPROC_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    const char *p = getenv("FUZZ_INPUT_PATH");
    if (!p) return -1;
    FILE *f = fopen(p, "rb");
    if (!f) return -2;
    int c = fgetc(f);
    fclose(f);
    return c == EOF ? -3 : c;
}
"""

OPEN_FAILED, EMPTY = 200, 201


def _clang(tmp: Path, name: str, src: str, *flags: str) -> str:
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    c = tmp / f"{name}.c"
    c.write_text(src)
    out = tmp / name
    proc = subprocess.run(
        ["clang", "-O1", *flags, "-o", str(out), str(c)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        pytest.skip(f"{name} build failed: {proc.stderr[-400:]}")
    return str(out)


@pytest.fixture(scope="module")
def plain_exe(tmp_path_factory) -> str:
    return _clang(tmp_path_factory.mktemp("memfd"), "argfile", ARGFILE_SRC)


@pytest.fixture(scope="module")
def shim_exe(tmp_path_factory) -> str:
    """Same program with the shim: the loader runs it as a forkserver."""
    return _clang(
        tmp_path_factory.mktemp("memfd_fs"),
        "argfile_fs",
        ARGFILE_SRC,
        "-include",
        SHIM,
    )


class TestMemfdInput:
    def test_roundtrip_and_stable_path(self):
        from fuzzer_tool.adapters.memfd_input import MemfdInput

        m = MemfdInput()
        try:
            path = m.path
            m.write(b"hello")
            assert Path(path).read_bytes() == b"hello"
            m.write(b"world!")
            assert m.path == path
            assert Path(path).read_bytes() == b"world!"
        finally:
            m.close()

    def test_shrink_leaves_no_tail(self):
        """Falsification: a shorter input must not keep the previous one's tail."""
        from fuzzer_tool.adapters.memfd_input import MemfdInput

        m = MemfdInput()
        try:
            m.write(b"AAAAAAAA")
            m.write(b"B")
            assert Path(m.path).read_bytes() == b"B"
            m.write(b"")
            assert Path(m.path).read_bytes() == b""
        finally:
            m.close()

    def test_visible_to_other_process(self):
        """The path is /proc/<pid>/fd/<n>: a child opens it without inheriting the fd."""
        from fuzzer_tool.adapters.memfd_input import MemfdInput

        m = MemfdInput()
        try:
            m.write(b"\x00\x01payload")
            got = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.stdout.buffer.write(open({m.path!r},'rb').read())",
                ],
                capture_output=True,
                close_fds=True,
            ).stdout
            assert got == b"\x00\x01payload"
            assert m.path.startswith(f"/proc/{os.getpid()}/fd/")
        finally:
            m.close()

    def test_closed_path_gone(self):
        """Adversarial: after close the path must not resolve (no stale reads)."""
        from fuzzer_tool.adapters.memfd_input import MemfdInput

        m = MemfdInput()
        path = m.path
        m.write(b"x")
        ino = os.stat(path).st_ino
        m.close()
        m.close()  # idempotent

        # The fd number may be reused by another open file, never by this one.
        assert not os.path.exists(path) or os.stat(path).st_ino != ino


class TestRunTargetFile:
    def test_memfd_path_used_no_disk_file(self, plain_exe, tmp_path):
        from fuzzer_tool.adapters.memfd_input import MemfdInput
        from fuzzer_tool.adapters.process import run_target_file

        m = MemfdInput()
        try:
            rcs = [
                run_target_file(
                    plain_exe, bytes([b]), 5.0, str(tmp_path), ["-i", "{file}"], input_file=m
                )[0]
                for b in (7, 9, 7)
            ]
        finally:
            m.close()

        assert rcs == [7, 9, 7]
        assert list(tmp_path.iterdir()) == []

    def test_control_tempfile_path(self, plain_exe, tmp_path):
        """Control: the legacy temp-file path gives the same answers."""
        from fuzzer_tool.adapters.process import run_target_file

        rcs = [
            run_target_file(plain_exe, bytes([b]), 5.0, str(tmp_path), ["-i", "{file}"])[0]
            for b in (7, 9, 7)
        ]

        assert rcs == [7, 9, 7]


def _loader_ok():
    from fuzzer_tool.adapters.forkserver import _ensure_compiled

    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")


def _runner(exe: str, argv):
    from fuzzer_tool.adapters.forkserver import ForkserverRunner
    from fuzzer_tool.adapters.shm import ShmCoverage

    shm = ShmCoverage(size=8192)
    env = {"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size)}
    return ForkserverRunner(exe, timeout=5.0, env=env, argv=argv), shm


class TestForkserverArgv:
    @pytest.mark.parametrize("which", ["plain_exe", "shim_exe"])
    def test_file_arg_template(self, which, request):
        """{file} in argv reaches the target in both exec and forkserver modes."""
        _loader_ok()
        exe = request.getfixturevalue(which)
        r, shm = _runner(exe, ["-i", "{file}"])
        try:
            assert r.start()
            rcs = [r.run_one(bytes([b]))[0] for b in (3, 11, 3)]
            mode = r.exec_mode
        finally:
            r.stop()
            shm.cleanup()

        assert rcs == [3, 11, 3]
        assert mode == ("forkserver" if which == "shim_exe" else "exec")

    def test_input_is_memory_backed(self, plain_exe):
        _loader_ok()
        r, shm = _runner(plain_exe, ["-i", "{file}"])
        try:
            assert r.start()
            assert r.input_path.startswith(f"/proc/{os.getpid()}/fd/")
        finally:
            r.stop()
            shm.cleanup()

    def test_control_default_argv(self, plain_exe):
        """Control: no template keeps `<target> <input>`; the probe sees no -i."""
        _loader_ok()
        r, shm = _runner(plain_exe, None)
        try:
            assert r.start()
            rc = r.run_one(b"\x05")[0]
        finally:
            r.stop()
            shm.cleanup()

        assert rc == 202

    def test_arg_with_spaces_kept(self, tmp_path):
        """Adversarial: one argv entry with spaces stays one entry."""
        _loader_ok()
        src = r"""
        #include <string.h>
        int main(int argc, char **argv) { return argc == 3 && !strcmp(argv[1], "a b  c") ? 42 : 1; }
        """
        exe = _clang(tmp_path, "spaces", src)
        r, shm = _runner(exe, ["a b  c", "{file}"])
        try:
            assert r.start()
            rc = r.run_one(b"x")[0]
        finally:
            r.stop()
            shm.cleanup()

        assert rc == 42

    def test_newline_arg_rejected(self, plain_exe):
        """Adversarial: a newline would split the line protocol; refuse loudly."""
        from fuzzer_tool.adapters.forkserver import ForkserverRunner

        with pytest.raises(ValueError):
            ForkserverRunner(plain_exe, argv=["-i", "bad\narg"])


# Child process: in-process loads share the interpreter's environment and handles.
_INPROC = """
import sys
from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.adapters.memfd_input import MemfdInput

so, use = sys.argv[1], sys.argv[2]
m = MemfdInput() if use == "memfd" else None
r = InProcessRunner(so, "fuzz_test", timeout=5.0, direct_lite=True, input_file=m)
print(repr([r.run_one(bytes([b]))[0] for b in (4, 8, 4)]))
r.stop()
"""


@pytest.fixture(scope="module")
def inproc_so(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("memfd_so")
    return _clang(tmp, "inproc.so", INPROC_SRC, "-fPIC", "-shared")


class TestInProcess:
    def _run(self, so: str, use: str) -> list[int]:
        env = {k: v for k, v in os.environ.items() if k != "FUZZ_INPUT_PATH"}
        proc = subprocess.run(
            [sys.executable, "-c", _INPROC, so, use],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        return eval(proc.stdout.strip().splitlines()[-1])

    def test_harness_reads_input_path(self, inproc_so):
        assert self._run(inproc_so, "memfd") == [4, 8, 4]

    def test_control_without_memfd(self, inproc_so):
        """Control: no memfd, no FUZZ_INPUT_PATH; the harness must notice."""
        assert self._run(inproc_so, "none") == [-1, -1, -1]


class TestFuzzerWiring:
    def test_file_mode_not_excluded_from_forkserver(self):
        """Wiring: file mode and target args no longer veto the forkserver."""
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer._setup_forkserver)

        assert "or self.file_mode" not in src
        assert "(self.target_args and not self.file_mode)" in src
        assert "argv=argv" in src

    def test_runner_passes_memfd(self):
        import inspect

        from fuzzer_tool.services.runner import TargetRunner

        assert "input_file=f._input_file" in inspect.getsource(TargetRunner._run_file)
