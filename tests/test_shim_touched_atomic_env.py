"""Atomic vs plain touched-bit OR, chosen per run by $__AFL_TOUCHED_ATOMIC.

The shim reads the variable once at attach; ``--touched-atomic`` /
``--no-touched-atomic`` export it. Unset keeps the build default (atomic, or
plain under ``-D__AFL_TOUCHED_ATOMIC=0``).
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.cli import commands
from fuzzer_tool.core.shim_health import TOUCHED_ATOMIC_ENV, export_touched_atomic
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env
from tests.test_shm_touched_bitmap import _bits, _live_slots

# Prints the mode the shim chose, then maps argv ids raw.
_DRIVER = """
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    printf("%d\\n", __afl_touched_atomic);
    for (int i = 1; i < argc; i++)
        __afl_map_id_raw((uint32_t)strtoul(argv[i], NULL, 10));
    return 0;
}
"""
SIZE = 1000


@pytest.fixture(scope="module")
def default_target(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("ta_default"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0")


@pytest.fixture(scope="module")
def plain_target(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("ta_plain"),
        _DRIVER,
        "-D__AFL_CTX_SENSITIVE=0",
        "-D__AFL_TOUCHED_ATOMIC=0",
    )


def _run(exe, env_value, ids=()):
    cov = ShmCoverage(size=SIZE, touched_bitmap=True)
    try:
        extra = {} if env_value is None else {TOUCHED_ATOMIC_ENV: env_value}
        env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries), **extra)
        r = subprocess.run(
            [str(exe), *map(str, ids)], env=env, capture_output=True, text=True, timeout=30
        )
        assert r.returncode == 0, r.stderr
        return int(r.stdout.split()[0]), r.stderr, _bits(cov.touched_snapshot()), _live_slots(cov)
    finally:
        cov.cleanup()


@requires_clang
class TestShimReadsEnv:
    @pytest.mark.parametrize(("env_value", "expected"), [(None, 1), ("1", 1), ("0", 0)])
    def test_default_build(self, default_target, env_value, expected):
        mode, err, _, _ = _run(default_target, env_value)
        assert mode == expected
        assert err == ""

    @pytest.mark.parametrize(("env_value", "expected"), [(None, 0), ("0", 0), ("1", 1)])
    def test_env_overrides_a_plain_build(self, plain_target, env_value, expected):
        assert _run(plain_target, env_value)[0] == expected

    @pytest.mark.parametrize("bad", ["banana", "2", "01", "-1"])
    def test_typo_keeps_default_and_says_so(self, default_target, bad):
        mode, err, _, _ = _run(default_target, bad)
        assert mode == 1
        assert "__AFL_TOUCHED_ATOMIC" in err
        assert "ignored" in err

    def test_empty_value_is_unset(self, default_target):
        mode, err, _, _ = _run(default_target, "")
        assert (mode, err) == (1, "")

    def test_both_modes_set_the_same_bits(self, default_target):
        ids = [1, 2, 999, 1001, 5000, 0xFFFFFFFE]
        a = _run(default_target, "1", ids)
        p = _run(default_target, "0", ids)
        assert a[2] == p[2] == a[3] == p[3]
        assert len(a[2]) == len(ids)


# ── CLI / wiring ───────────────────────────────────────────────────────


def _fuzz_kwargs(monkeypatch, tmp_path, extra):
    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    argv = ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    commands.main()
    return captured


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], None), (["--touched-atomic"], True), (["--no-touched-atomic"], False)],
)
def test_cli_reaches_fuzzer(monkeypatch, tmp_path, flags, expected):
    assert _fuzz_kwargs(monkeypatch, tmp_path, flags)["touched_atomic"] is expected


def test_hail_mary_leaves_it_unset(monkeypatch, tmp_path):
    assert _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])["touched_atomic"] is None


def test_last_flag_wins(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--no-touched-atomic", "--touched-atomic"])
    assert kw["touched_atomic"] is True


def test_export_sets_env_only_when_asked(monkeypatch):
    # setenv-then-delenv leaves teardown restoring "absent": export_* writes
    # os.environ directly, and a bare delenv would restore whatever it last set.
    monkeypatch.setenv(TOUCHED_ATOMIC_ENV, "x")
    monkeypatch.delenv(TOUCHED_ATOMIC_ENV)
    export_touched_atomic(None)
    assert TOUCHED_ATOMIC_ENV not in os.environ
    export_touched_atomic(False)
    assert os.environ[TOUCHED_ATOMIC_ENV] == "0"
    export_touched_atomic(True)
    assert os.environ[TOUCHED_ATOMIC_ENV] == "1"


def test_unset_does_not_clear_an_inherited_value(monkeypatch):
    """None means "no opinion": a value the user exported by hand survives."""
    monkeypatch.setenv(TOUCHED_ATOMIC_ENV, "0")
    export_touched_atomic(None)
    assert os.environ[TOUCHED_ATOMIC_ENV] == "0"


def test_fuzzer_exports_it():
    assert "export_touched_atomic(touched_atomic)" in inspect.getsource(Fuzzer.__init__)
