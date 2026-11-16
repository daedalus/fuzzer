"""Shim open() hooks for an aliased input path (wtf fshooks port, part B).

A library that opens a hardcoded path gets the fuzzer's in-memory input:
``FUZZ_INPUT_ALIAS`` names the path, ``FUZZ_INPUT_PATH`` the memfd. The hooks
are hidden in the target DSO, so they also catch library objects compiled
without the shim.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = str(REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c")
ALIAS = "/nonexistent/fuzz_alias.bin"
OPEN_FAILED, EMPTY, NULL_REJECTED = 200, 201, 203

# Library object, built WITHOUT the shim: it must still hit the hidden hooks.
LIB_SRC = r"""
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int first_byte(int fd) {
    if (fd < 0) return 200;
    unsigned char c;
    ssize_t n = read(fd, &c, 1);
    close(fd);
    return n == 1 ? c : 201;
}

static int first_byte_f(FILE *f) {
    if (!f) return 200;
    int c = fgetc(f);
    fclose(f);
    return c == EOF ? 201 : c;
}

int lib_read(const char *api, const char *path) {
    struct stat st;
    struct stat64 st64;
    if (!strcmp(api, "open")) return first_byte(open(path, O_RDONLY));
    if (!strcmp(api, "open64")) return first_byte(open64(path, O_RDONLY));
    if (!strcmp(api, "openat")) return first_byte(openat(AT_FDCWD, path, O_RDONLY));
    if (!strcmp(api, "openat64")) return first_byte(openat64(AT_FDCWD, path, O_RDONLY));
    if (!strcmp(api, "nofollow")) return first_byte(open(path, O_RDONLY | O_NOFOLLOW));
    if (!strcmp(api, "fopen")) return first_byte_f(fopen(path, "rb"));
    if (!strcmp(api, "fopen64")) return first_byte_f(fopen64(path, "rb"));
    if (!strcmp(api, "stat")) return stat(path, &st) ? 200 : (int)st.st_size;
    if (!strcmp(api, "stat64")) return stat64(path, &st64) ? 200 : (int)st64.st_size;
    if (!strcmp(api, "lstat")) return lstat(path, &st) ? 200 : (S_ISREG(st.st_mode) ? (int)st.st_size : 202);
    if (!strcmp(api, "access")) return access(path, R_OK) ? 200 : 0;
    if (!strcmp(api, "null")) return open(NULL, O_RDONLY) < 0 ? 203 : 204;
    return 255;
}
"""

MAIN_SRC = r"""
int lib_read(const char *api, const char *path);
int main(int argc, char **argv) {
    return lib_read(argc > 1 ? argv[1] : "open", argc > 2 ? argv[2] : "/nonexistent/fuzz_alias.bin");
}
"""

SO_SRC = r"""
#include <stddef.h>
int lib_read(const char *api, const char *path);
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    return lib_read("fopen", "/nonexistent/fuzz_alias.bin");
}
"""

APIS = ["open", "open64", "openat", "openat64", "nofollow", "fopen", "fopen64"]
STAT_APIS = ["stat", "stat64", "lstat"]


def _cc(*args: str) -> None:
    if shutil.which("clang") is None:
        pytest.skip("clang not available")
    proc = subprocess.run(["clang", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"build failed: {proc.stderr[-400:]}")


@pytest.fixture(scope="module")
def build(tmp_path_factory) -> dict[str, str]:
    tmp = tmp_path_factory.mktemp("alias")
    (tmp / "lib.c").write_text(LIB_SRC)
    (tmp / "main.c").write_text(MAIN_SRC)
    (tmp / "so.c").write_text(SO_SRC)
    _cc("-O1", "-fPIC", "-c", str(tmp / "lib.c"), "-o", str(tmp / "lib.o"))
    _cc("-O1", "-include", SHIM, str(tmp / "main.c"), str(tmp / "lib.o"), "-o", str(tmp / "exe"))
    _cc(
        "-O1", "-fPIC", "-shared", "-include", SHIM,
        str(tmp / "so.c"), str(tmp / "lib.o"), "-o", str(tmp / "t.so"),
    )  # fmt: skip
    return {"exe": str(tmp / "exe"), "so": str(tmp / "t.so")}


@pytest.fixture
def memfd():
    from fuzzer_tool.adapters.memfd_input import MemfdInput

    m = MemfdInput()
    m.write(b"\x2aHELLO")
    yield m
    m.close()


def _env(**extra: str) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("FUZZ_INPUT_")}
    return {**base, **extra}


def _run(exe: str, api: str, env: dict[str, str], path: str = ALIAS) -> int:
    return subprocess.run([exe, api, path], env=env, timeout=30).returncode


class TestExecutable:
    @pytest.mark.parametrize("api", APIS)
    def test_alias_reads_memfd(self, build, memfd, api):
        env = _env(FUZZ_INPUT_ALIAS=ALIAS, FUZZ_INPUT_PATH=memfd.path)

        assert _run(build["exe"], api, env) == 0x2A

    @pytest.mark.parametrize("api", STAT_APIS)
    def test_stat_sees_regular_file(self, build, memfd, api):
        """lstat follows the memfd's /proc symlink: the alias is a regular file."""
        env = _env(FUZZ_INPUT_ALIAS=ALIAS, FUZZ_INPUT_PATH=memfd.path)

        assert _run(build["exe"], api, env) == len(b"\x2aHELLO")

    def test_access(self, build, memfd):
        env = _env(FUZZ_INPUT_ALIAS=ALIAS, FUZZ_INPUT_PATH=memfd.path)

        assert _run(build["exe"], "access", env) == 0

    @pytest.mark.parametrize("api", APIS + ["access"])
    def test_control_no_alias(self, build, api):
        """Control: without the env pair the hardcoded path does not exist."""
        assert _run(build["exe"], api, _env()) == OPEN_FAILED

    def test_other_path_untouched(self, build, memfd):
        """Adversarial: only the exact alias string is swapped."""
        env = _env(FUZZ_INPUT_ALIAS=ALIAS, FUZZ_INPUT_PATH=memfd.path)

        assert _run(build["exe"], "fopen", env, ALIAS + "x") == OPEN_FAILED
        assert _run(build["exe"], "fopen", env, ALIAS[:-1]) == OPEN_FAILED

    def test_alias_without_target_passes_through(self, build):
        """Adversarial: alias set but no FUZZ_INPUT_PATH: hooks stay inert."""
        assert _run(build["exe"], "fopen", _env(FUZZ_INPUT_ALIAS=ALIAS)) == OPEN_FAILED

    def test_null_path_not_crashing(self, build, memfd):
        """Adversarial: open(NULL) still fails with EFAULT, not a shim SIGSEGV."""
        env = _env(FUZZ_INPUT_ALIAS=ALIAS, FUZZ_INPUT_PATH=memfd.path)

        assert _run(build["exe"], "null", env) == NULL_REJECTED

    def test_hooks_hidden(self, build):
        """The hooks must not be exported: other DSOs keep libc's open."""
        if shutil.which("nm") is None:
            pytest.skip("nm not available")
        out = subprocess.run(
            ["nm", "-D", "--defined-only", build["so"]], capture_output=True, text=True
        ).stdout
        exported = {line.split()[-1] for line in out.splitlines() if line.strip()}

        assert not exported & {"open", "open64", "openat", "fopen", "stat", "lstat", "access"}


_INPROC = textwrap.dedent(
    """
    import os, sys
    from fuzzer_tool.adapters.inprocess import InProcessRunner
    from fuzzer_tool.adapters.memfd_input import MemfdInput

    so, alias = sys.argv[1], sys.argv[2]
    if alias:
        os.environ["FUZZ_INPUT_ALIAS"] = alias
    m = MemfdInput()
    r = InProcessRunner(so, "fuzz_test", timeout=5.0, direct_lite=True, input_file=m)
    print(repr([r.run_one(bytes([b]))[0] for b in (5, 9, 5)]))
    r.stop()
    """
)


class TestInProcess:
    def _run(self, so: str, alias: str) -> list[int]:
        proc = subprocess.run(
            [sys.executable, "-c", _INPROC, so, alias],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
            env=_env(),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        return ast.literal_eval(proc.stdout.strip().splitlines()[-1])

    def test_library_open_follows_input(self, build):
        """Falsification: each exec's bytes, not the first exec's, reach the library."""
        assert self._run(build["so"], ALIAS) == [5, 9, 5]

    def test_control_no_alias(self, build):
        assert self._run(build["so"], "") == [OPEN_FAILED] * 3


def _fuzz_kwargs(monkeypatch, tmp_path, extra: list[str]) -> dict:
    from fuzzer_tool.cli import commands

    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    argv = ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    commands.main()
    return captured


class TestWiring:
    def test_cli_alias_implies_file_mode(self, monkeypatch, tmp_path):
        kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--input-alias", ALIAS])

        assert kw["input_alias"] == ALIAS
        assert kw["file_mode"] is True

    def test_cli_default_none(self, monkeypatch, tmp_path):
        kw = _fuzz_kwargs(monkeypatch, tmp_path, [])

        assert kw["input_alias"] is None

    def test_alias_env_reaches_every_backend(self, tmp_path):
        """One os.environ site feeds spawn (environ copy), loader and in-process."""
        from fuzzer_tool.services import fuzzer as fz
        from fuzzer_tool.services.runner import environ_copy

        exe = tmp_path / "t"
        exe.write_bytes(b"\x7fELF")
        exe.chmod(0o755)
        f = fz.Fuzzer.__new__(fz.Fuzzer)
        fz._snapshot_environ_once()
        try:
            f.file_mode, f.input_alias = True, ALIAS
            f._input_file = None
            fz.Fuzzer._export_input_env(f)
            env = environ_copy()
            assert env["FUZZ_INPUT_ALIAS"] == ALIAS
            assert env["FUZZ_INPUT_PATH"] == f._input_file.path
        finally:
            f._input_file.close()
            fz._restore_environ()

        assert "FUZZ_INPUT_ALIAS" not in os.environ


class TestForkserver:
    def test_regression_alias_reads_loader_input(self, build):
        """The loader must stage into the memfd FUZZ_INPUT_PATH names.

        Measured before the fix: the loader wrote its own memfd while the
        target's alias pointed at the Fuzzer's, which nothing wrote, so every
        exec read the empty file (fuzzgoat: 2 edges instead of 28).
        """
        from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
        from fuzzer_tool.adapters.memfd_input import MemfdInput
        from fuzzer_tool.adapters.shm import ShmCoverage

        if _ensure_compiled() is None:
            pytest.skip("fuzz_loader failed to compile")
        m = MemfdInput()
        shm = ShmCoverage(size=8192)
        env = {
            "__AFL_SHM_ID": shm.env_id,
            "AFL_MAP_SIZE": str(shm.size),
            "FUZZ_INPUT_ALIAS": ALIAS,
            "FUZZ_INPUT_PATH": m.path,
        }
        r = ForkserverRunner(build["exe"], timeout=5.0, env=env, argv=["fopen", ALIAS], input_file=m)
        try:
            assert r.start()
            rcs = [r.run_one(bytes([b]))[0] for b in (6, 13, 6)]
            assert r.input_path == m.path
            assert r.exec_mode == "forkserver"
        finally:
            r.stop()
            shm.cleanup()
            m.close()

        assert rcs == [6, 13, 6]
