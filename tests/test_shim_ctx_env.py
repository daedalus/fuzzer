"""Call-stack context on or off per run, chosen by $__AFL_CTX_SENSITIVE.

The shim reads the variable once at attach; ``--ctx-sensitive`` /
``--no-ctx-sensitive`` export it. Off makes ids context-free without a
rebuild. It can only switch a context build off: a ``-D__AFL_CTX_SENSITIVE=0``
build has no walk to switch on. The Python side reads the same variable
(``effective_ctx_bits``) wherever it reads the build's marker symbol.
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
from fuzzer_tool.core import elf
from fuzzer_tool.core.elf import CTX_SENSITIVE_ENV, effective_ctx_bits
from fuzzer_tool.core.shim_health import export_ctx_sensitive
from fuzzer_tool.services.corpus_manager import check_coverage_contract
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

# One edge, reached from two call sites. With context on the two arrive as
# different ids; context-free they are the same edge.
_DRIVER = """
#include <stdio.h>
__attribute__((noinline)) void hit(void) {
    static uint32_t g = 7;
    __sanitizer_cov_trace_pc_guard(&g);
}
__attribute__((noinline)) void site_a(void) { hit(); __asm__ volatile(""); }
__attribute__((noinline)) void site_b(void) { hit(); __asm__ volatile(""); }
int main(void) {
    printf("%d\\n", __afl_ctx_on);
    for (int i = 0; i < 4; i++) { site_a(); site_b(); }
    return 0;
}
"""
SIZE = 1024


@pytest.fixture(scope="module")
def ctx_target(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cx_on"), _DRIVER, "-D__AFL_CTX_SENSITIVE=1")


@pytest.fixture(scope="module")
def free_target(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cx_off"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0")


def _run(exe, env_value):
    cov = ShmCoverage(size=SIZE)
    try:
        extra = {} if env_value is None else {CTX_SENSITIVE_ENV: env_value}
        # Context ids hash return addresses; base-relative mode keeps them
        # comparable between two processes (see FUZZER_KEEP_ASLR in the shim).
        env = _clean_env(
            __AFL_SHM_ID=cov.env_id,
            AFL_MAP_SIZE=str(cov.num_entries),
            FUZZER_KEEP_ASLR="1",
            **extra,
        )
        r = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        return int(r.stdout.split()[0]), r.stderr, set(cov.get_edge_ids())
    finally:
        cov.cleanup()


@requires_clang
class TestShimReadsEnv:
    @pytest.mark.parametrize(("env_value", "mode"), [(None, 1), ("1", 1), ("0", 0)])
    def test_ctx_build(self, ctx_target, env_value, mode):
        got, err, _ = _run(ctx_target, env_value)
        assert (got, err) == (mode, "")

    def test_off_gives_the_same_ids_as_a_context_free_build(self, ctx_target, free_target):
        _, _, off_by_env = _run(ctx_target, "0")
        _, _, off_by_build = _run(free_target, None)
        assert off_by_env == off_by_build
        assert off_by_env

    def test_on_separates_the_two_call_sites(self, ctx_target):
        _, _, on = _run(ctx_target, None)
        _, _, off = _run(ctx_target, "0")
        assert len(on) > len(off)
        assert _run(ctx_target, "1")[2] == on

    @pytest.mark.parametrize("env_value", [None, "0", "1"])
    def test_context_free_build_cannot_be_switched_on(self, free_target, ctx_target, env_value):
        got, _, ids = _run(free_target, env_value)
        assert got == 0
        assert ids == _run(ctx_target, "0")[2]

    @pytest.mark.parametrize("bad", ["banana", "2", "01", "-1", "off"])
    def test_typo_keeps_default_and_says_so(self, ctx_target, bad):
        got, err, _ = _run(ctx_target, bad)
        assert got == 1
        assert "__AFL_CTX_SENSITIVE" in err
        assert "ignored" in err

    def test_empty_value_is_unset(self, ctx_target):
        assert _run(ctx_target, "")[:2] == (1, "")


# ── effective_ctx_bits ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("marker", "env_value", "expected"),
    [
        (8, None, 8),
        (8, "1", 8),
        (8, "0", 0),
        (8, "banana", 8),  # the shim ignores it, so the planner must too
        (0, "0", 0),
        (0, "1", 0),
        (None, "0", None),  # unknown stays unknown: no marker is not evidence
        (None, None, None),
    ],
)
def test_effective_ctx_bits(monkeypatch, marker, env_value, expected):
    monkeypatch.setattr(elf, "detect_ctx_bits", lambda _t: marker)
    if env_value is None:
        monkeypatch.delenv(CTX_SENSITIVE_ENV, raising=False)
    else:
        monkeypatch.setenv(CTX_SENSITIVE_ENV, env_value)
    assert effective_ctx_bits("/x") == expected


def test_map_sizing_follows_the_switch(monkeypatch):
    monkeypatch.setattr(elf, "detect_ctx_bits", lambda _t: 8)
    monkeypatch.setattr(elf, "parse_sancov_guard_count", lambda _t: 5000)
    monkeypatch.delenv(CTX_SENSITIVE_ENV, raising=False)
    wide = elf.estimate_map_size_detail("/x")
    monkeypatch.setenv(CTX_SENSITIVE_ENV, "0")
    narrow = elf.estimate_map_size_detail("/x")
    assert wide.ctx_bits == 8
    assert narrow.ctx_bits == 0
    assert narrow.entries < wide.entries


# ── export / CLI / contract ────────────────────────────────────────────


def test_export_sets_env_only_when_asked(monkeypatch):
    monkeypatch.delenv(CTX_SENSITIVE_ENV, raising=False)
    export_ctx_sensitive(None)
    assert CTX_SENSITIVE_ENV not in os.environ
    export_ctx_sensitive(False)
    assert os.environ[CTX_SENSITIVE_ENV] == "0"
    export_ctx_sensitive(True)
    assert os.environ[CTX_SENSITIVE_ENV] == "1"
    monkeypatch.delenv(CTX_SENSITIVE_ENV)


def test_unset_does_not_clear_an_inherited_value(monkeypatch):
    monkeypatch.setenv(CTX_SENSITIVE_ENV, "0")
    export_ctx_sensitive(None)
    assert os.environ[CTX_SENSITIVE_ENV] == "0"


def test_fuzzer_exports_it():
    assert "export_ctx_sensitive(ctx_sensitive)" in inspect.getsource(Fuzzer.__init__)


def _fuzz_kwargs(monkeypatch, tmp_path, extra):
    exe = tmp_path / "t"
    exe.write_bytes(b"\x7fELF")
    exe.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    monkeypatch.setattr(
        sys, "argv", ["fuzzer-tool", "fuzz", str(exe), "-d", str(tmp_path / "c"), *extra]
    )
    commands.main()
    return captured


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], None), (["--ctx-sensitive"], True), (["--no-ctx-sensitive"], False)],
)
def test_cli_reaches_fuzzer(monkeypatch, tmp_path, flags, expected):
    assert _fuzz_kwargs(monkeypatch, tmp_path, flags)["ctx_sensitive"] is expected


def test_hail_mary_leaves_it_unset(monkeypatch, tmp_path):
    assert _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])["ctx_sensitive"] is None


def test_last_flag_wins(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--no-ctx-sensitive", "--ctx-sensitive"])
    assert kw["ctx_sensitive"] is True


class TestContract:
    def test_mismatch_refuses_resume(self):
        with pytest.raises(RuntimeError, match="ctx_bits"):
            check_coverage_contract({"ctx_bits": 8}, {"ctx_bits": 0})

    def test_match_resumes(self):
        check_coverage_contract({"ctx_bits": 8}, {"ctx_bits": 8})
        check_coverage_contract({"ctx_bits": 0}, {"ctx_bits": 0})

    def test_state_without_the_key_resumes_freely(self):
        check_coverage_contract(
            {"ngram_k": 2, "node_channel": False},
            {"ngram_k": 2, "node_channel": False, "ctx_bits": 8},
        )

    def test_unknown_on_either_side_is_not_a_mismatch(self):
        check_coverage_contract({"ctx_bits": None}, {"ctx_bits": 8})
        check_coverage_contract({"ctx_bits": 8}, {"ctx_bits": None})
