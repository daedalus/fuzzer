"""ASAN hands freed memory back to the OS once a second.

Default is never: ~870 MB freed by the Katz ICFG build on ffmpeg stayed
resident (1509 -> 642 MB with release on). 1000 ms costs no measurable
exec/s on ffmpeg_read_asan.so (95-97 vs 96-97); 0 ms costs ~3%.
"""

import pytest

from fuzzer_tool.cli import ldpreload_wrapper as wrapper
from fuzzer_tool.services.fuzzer import Fuzzer

_OPT = "allocator_release_to_os_interval_ms"
_USER = f"{_OPT}=0"


class _Exec(Exception):
    """Stands in for execvpe so main() returns control to the test."""


def _wrapper_opts(monkeypatch, tmp_path, preset: str | None) -> list[str]:
    target = tmp_path / "t.so"
    target.write_bytes(b"")
    monkeypatch.setattr(wrapper.sys, "argv", ["fuzzer-tool", "fuzz", str(target)])
    monkeypatch.setattr(wrapper, "_detect_asan", lambda t: True)
    monkeypatch.setattr(wrapper, "_detect_ubsan", lambda t: False)
    monkeypatch.setattr(wrapper, "_resolve_asan", lambda: None)
    monkeypatch.setattr(wrapper.os, "execvpe", lambda *a: (_ for _ in ()).throw(_Exec()))
    # setenv first so teardown restores the original state even when absent:
    # delenv(raising=False) on a missing key records nothing to undo, and
    # main() then leaks halt_on_error=0 into every later ASAN test.
    monkeypatch.setenv("ASAN_OPTIONS", preset or "")
    if preset is None:
        monkeypatch.delenv("ASAN_OPTIONS")

    with pytest.raises(_Exec):
        wrapper.main()
    return wrapper.os.environ["ASAN_OPTIONS"].split(":")


def _keys(opts: list[str]) -> list[str]:
    return [o.split("=")[0] for o in opts]


def test_wrapper_sets_release(monkeypatch, tmp_path):
    """Falsification: in-process runs get the release interval."""
    opts = _wrapper_opts(monkeypatch, tmp_path, None)

    assert f"{_OPT}=1000" in opts


def test_wrapper_keeps_user_value(monkeypatch, tmp_path):
    """Adversarial: a user value wins and is not duplicated."""
    opts = _wrapper_opts(monkeypatch, tmp_path, _USER)

    assert _USER in opts
    assert _keys(opts).count(_OPT) == 1


def test_target_env_sets_release():
    env: dict[str, str] = {}

    Fuzzer._setup_asan_env(env)

    assert f"{_OPT}=1000" in env["ASAN_OPTIONS"].split(":")


def test_target_env_keeps_user_value():
    env = {"ASAN_OPTIONS": _USER}

    Fuzzer._setup_asan_env(env)

    opts = env["ASAN_OPTIONS"].split(":")
    assert _USER in opts
    assert _keys(opts).count(_OPT) == 1


def test_wrapper_msan_defaults(monkeypatch, tmp_path):
    target = tmp_path / "t.bin"
    target.write_bytes(b"")
    monkeypatch.setattr(wrapper.sys, "argv", ["fuzzer-tool", "fuzz", str(target)])
    monkeypatch.setattr(wrapper, "_detect_asan", lambda t: False)
    monkeypatch.setattr(wrapper, "_detect_ubsan", lambda t: False)
    monkeypatch.setattr(wrapper, "_detect_msan", lambda t: True)
    monkeypatch.setattr(wrapper.os, "execvpe", lambda *a: (_ for _ in ()).throw(_Exec()))
    monkeypatch.setenv("MSAN_OPTIONS", "symbolize=1")
    with pytest.raises(_Exec):
        wrapper.main()
    opts = wrapper.os.environ["MSAN_OPTIONS"].split(":")
    assert "exit_code=86" in opts
    assert "symbolize=1" in opts and "symbolize=0" not in opts  # user key wins


def test_setup_msan_env_and_odr():
    env: dict = {}
    Fuzzer._setup_msan_env(env)
    assert env["MSAN_OPTIONS"].split(":") == ["exit_code=86", "symbolize=0"]
    Fuzzer._setup_asan_env(env)
    assert "detect_odr_violation=0" in env["ASAN_OPTIONS"].split(":")
