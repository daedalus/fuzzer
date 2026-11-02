"""FUZZER_ASAN_STRICT=1: fill whole allocations, pin stack-use-after-return.

Off by default because filling whole allocations is expensive (gcc 13 libasan,
malloc+free: 64 KiB 85 -> 325 us, 1 MiB 91 -> 2753 us). On, it makes an
uninitialized read past the 4 KiB default fill visible: without it the memory
is zero from the OS and ``strlen`` silently stops at 4096.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.cli import ldpreload_wrapper as wrapper
from fuzzer_tool.services.fuzzer import Fuzzer

_FILL = f"max_malloc_fill_size={wrapper.ASAN_STRICT_FILL_MAX}"
_UAR = "detect_stack_use_after_return=1"
_STRICT_KEYS = {"max_malloc_fill_size", "detect_stack_use_after_return"}


class _Exec(Exception):
    """Stands in for execvpe so main() returns control to the test."""


def _wrapper_opts(monkeypatch, tmp_path, strict: str | None, preset: str = "") -> list[str]:
    target = tmp_path / "t.so"
    target.write_bytes(b"")
    monkeypatch.setattr(wrapper.sys, "argv", ["fuzzer-tool", "fuzz", str(target)])
    monkeypatch.setattr(wrapper, "_detect_asan", lambda t: True)
    monkeypatch.setattr(wrapper, "_detect_ubsan", lambda t: False)
    monkeypatch.setattr(wrapper, "_resolve_asan", lambda: None)
    monkeypatch.setattr(wrapper.os, "execvpe", lambda *a: (_ for _ in ()).throw(_Exec()))
    monkeypatch.setenv("ASAN_OPTIONS", preset)
    if strict is None:
        monkeypatch.delenv("FUZZER_ASAN_STRICT", raising=False)
    else:
        monkeypatch.setenv("FUZZER_ASAN_STRICT", strict)
    with pytest.raises(_Exec):
        wrapper.main()
    return wrapper.os.environ["ASAN_OPTIONS"].split(":")


def _keys(opts: list[str]) -> set[str]:
    return {o.split("=")[0] for o in opts}


def test_off_by_default_wrapper(monkeypatch, tmp_path):
    assert not _STRICT_KEYS & _keys(_wrapper_opts(monkeypatch, tmp_path, None))


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_only_exactly_one_enables(monkeypatch, tmp_path, value):
    assert not _STRICT_KEYS & _keys(_wrapper_opts(monkeypatch, tmp_path, value))


def test_on_wrapper(monkeypatch, tmp_path):
    opts = _wrapper_opts(monkeypatch, tmp_path, "1")
    assert _FILL in opts and _UAR in opts


def test_wrapper_keeps_user_values(monkeypatch, tmp_path):
    opts = _wrapper_opts(monkeypatch, tmp_path, "1", preset="max_malloc_fill_size=64")
    assert "max_malloc_fill_size=64" in opts
    assert _FILL not in opts
    assert [o.split("=")[0] for o in opts].count("max_malloc_fill_size") == 1


def test_target_env_off_by_default(monkeypatch):
    monkeypatch.delenv("FUZZER_ASAN_STRICT", raising=False)
    env: dict[str, str] = {}
    Fuzzer._setup_asan_env(env)
    assert not _STRICT_KEYS & _keys(env["ASAN_OPTIONS"].split(":"))


def test_target_env_on(monkeypatch):
    monkeypatch.setenv("FUZZER_ASAN_STRICT", "1")
    env: dict[str, str] = {}
    Fuzzer._setup_asan_env(env)
    opts = env["ASAN_OPTIONS"].split(":")
    assert _FILL in opts and _UAR in opts
    assert "halt_on_error=0" in opts  # recover-mode options unchanged


def test_target_env_keeps_user_value(monkeypatch):
    monkeypatch.setenv("FUZZER_ASAN_STRICT", "1")
    env = {"ASAN_OPTIONS": "detect_stack_use_after_return=0"}
    Fuzzer._setup_asan_env(env)
    opts = env["ASAN_OPTIONS"].split(":")
    assert "detect_stack_use_after_return=0" in opts and _UAR not in opts


_UNINIT = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    char *p = malloc(8192);   /* never initialized, never terminated */
    printf("strlen=%zu\n", strlen(p));
    free(p);
    return 0;
}
"""


@pytest.fixture(scope="module")
def uninit_exe(tmp_path_factory):
    cc = shutil.which("gcc") or shutil.which("clang")
    if not cc:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("strict")
    src, exe = d / "u.c", d / "u"
    src.write_text(_UNINIT)
    r = subprocess.run([cc, "-O0", "-fsanitize=address", "-o", str(exe), str(src)], capture_output=True)
    if r.returncode != 0:
        pytest.skip("compiler lacks ASan")
    return exe


def _run(exe, opts: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ASAN_OPTIONS": f"detect_leaks=0:{opts}"}
    return subprocess.run([str(exe)], capture_output=True, env=env, text=True)


def test_default_fill_hides_uninit_read_past_4k(uninit_exe):
    r = _run(uninit_exe, "max_malloc_fill_size=4096")
    assert "strlen=4096" in r.stdout and "AddressSanitizer" not in r.stderr


def test_strict_fill_reports_it(uninit_exe):
    r = _run(uninit_exe, _FILL)
    assert "heap-buffer-overflow" in r.stderr
