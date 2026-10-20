"""`minimize` replays through the in-process runners, like `fuzz`.

`minimize -c` could only spawn the target, so a fuzz-harness `.so` (no
`main`, entry `LLVMFuzzerTestOneInput`/`fuzz_*`) never produced an edge and
the blackout guard refused to prune. `--inprocess`, `--inprocess-direct` and
`--inprocess-func` pick the same backends `fuzz` does:

  --inprocess                     subprocess loader (crash-isolated)
  --inprocess-direct              direct_lite ctypes call (what fuzz picks
                                  for a .so with that flag alone)
  --inprocess --inprocess-direct  direct ctypes call
  neither                         spawn (unchanged)

The shim binds `__afl_area` at dlopen, and one path dlopens once per process,
so every run here builds its own `.so`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.cli.commands import _minimize_exec_mode
from fuzzer_tool.services.minimize import ExecMode, minimize_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Distinct first bytes fire distinct guard sets: a={1,10,11}, b={1,20,21},
# c={1,10,11,30} (a is dominated by c), 'x' aborts after marking edge 99.
TARGET_SRC = r"""
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
void __sanitizer_cov_trace_pc_guard(uint32_t *guard);
static void fire(uint32_t g) { __sanitizer_cov_trace_pc_guard(&g); }
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    fire(1);
    if (!len) return 0;
    if (buf[0] == 'a')      { fire(10); fire(11); }
    else if (buf[0] == 'b') { fire(20); fire(21); }
    else if (buf[0] == 'c') { fire(10); fire(11); fire(30); }
    else if (buf[0] == 'x') { fire(99); abort(); }
    return 0;
}
"""

SEEDS = {
    "a.bin": b"a",
    "a_dup.bin": b"aaaa",
    "b.bin": b"b",
    "c.bin": b"c",
}
EXPECTED_KEPT = {"b.bin", "c.bin"}

IN_PROCESS_MODES = [ExecMode.LOADER, ExecMode.DIRECT_LITE, ExecMode.DIRECT]


def _compiler() -> str | None:
    for cc in ("clang", "gcc", "cc"):
        if shutil.which(cc):
            return cc
    return None


@pytest.fixture
def build_so(tmp_path):
    """Factory: a fresh shim-linked target `.so` per call."""
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler available")
    count = 0

    def build() -> str:
        nonlocal count
        count += 1
        d = tmp_path / f"so{count}"
        d.mkdir()
        src = d / "t.c"
        src.write_text(TARGET_SRC)
        so = d / "t.so"
        proc = subprocess.run(
            [cc, "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            pytest.skip(f"target failed to build: {proc.stderr[-400:]}")
        return str(so)

    return build


def _corpus(tmp_path: Path, name: str, seeds: dict[str, bytes]) -> Path:
    d = tmp_path / name
    d.mkdir()
    for fname, data in seeds.items():
        (d / fname).write_bytes(data)
    return d


def _live(d: Path) -> set[str]:
    return {p.name for p in d.iterdir() if p.is_file()}


class TestFlags:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        """Run the real parser in main(); capture the Namespace cmd_minimize gets."""
        self.seen: list[argparse.Namespace] = []
        monkeypatch.setattr(commands, "cmd_minimize", lambda a: self.seen.append(a) or 0)
        self.monkeypatch = monkeypatch

    def _args(self, *extra: str) -> argparse.Namespace:
        argv = ["fuzzer-tool", "minimize", "./t.so", "-d", "corpus", "-c", *extra]
        self.monkeypatch.setattr(sys, "argv", argv)
        assert commands.main() == 0
        return self.seen[-1]

    def test_defaults_off(self):
        a = self._args()
        assert a.inprocess is False
        assert a.inprocess_direct is False
        assert a.inprocess_func == "LLVMFuzzerTestOneInput"

    def test_flags_parse(self):
        a = self._args("--inprocess", "--inprocess-direct", "--inprocess-func", "fuzz_test")
        assert (a.inprocess, a.inprocess_direct, a.inprocess_func) == (True, True, "fuzz_test")

    @pytest.mark.parametrize(
        ("flags", "target", "want"),
        [
            ((), "t.so", ExecMode.SPAWN),
            (("--inprocess",), "t.so", ExecMode.LOADER),
            (("--inprocess-direct",), "t.so", ExecMode.DIRECT_LITE),
            (("--inprocess", "--inprocess-direct"), "t.so", ExecMode.DIRECT),
            (("--inprocess",), "pkg.mod:fn", ExecMode.LOADER),
        ],
    )
    def test_mode_table(self, flags, target, want):
        assert _minimize_exec_mode(self._args(*flags), target) is want

    def test_executable_falls_back_to_spawn(self):
        # An ET_EXEC cannot be dlopen'd; fuzz drops to exec mode, so does this.
        exe = shutil.which("true")
        assert exe is not None
        args = self._args("--inprocess", "--inprocess-direct")
        assert _minimize_exec_mode(args, exe) is ExecMode.SPAWN


@pytest.mark.parametrize("mode", IN_PROCESS_MODES, ids=lambda m: m.value)
class TestReplay:
    def test_prunes_by_real_coverage(self, tmp_path, build_so, mode):
        """Dominated and duplicate-edge seeds go; the two that cover stay."""
        corpus = _corpus(tmp_path, "c", SEEDS)
        kept, removed = minimize_corpus(
            build_so(),
            str(corpus),
            use_coverage=True,
            exec_mode=mode,
            inprocess_func="fuzz_test",
        )
        assert (kept, removed) == (len(EXPECTED_KEPT), len(SEEDS) - len(EXPECTED_KEPT))
        assert _live(corpus) == EXPECTED_KEPT

    def test_control_two_runs_agree(self, tmp_path, build_so, mode):
        """The oracle against itself: identical corpora, identical result."""
        results = []
        for name in ("one", "two"):
            corpus = _corpus(tmp_path, name, SEEDS)
            minimize_corpus(
                build_so(),
                str(corpus),
                use_coverage=True,
                exec_mode=mode,
                inprocess_func="fuzz_test",
            )
            results.append(_live(corpus))
        assert results[0] == results[1]

    def test_missing_function_never_wipes(self, tmp_path, build_so, mode):
        """A wrong --inprocess-func is a failed measurement, not 'covers nothing'."""
        corpus = _corpus(tmp_path, "c", SEEDS)
        kept, removed = minimize_corpus(
            build_so(),
            str(corpus),
            use_coverage=True,
            exec_mode=mode,
            inprocess_func="no_such_function",
        )
        assert (kept, removed) == (len(SEEDS), 0)
        assert _live(corpus) == set(SEEDS)

    def test_empty_seed_is_replayed(self, tmp_path, build_so, mode):
        """A zero-length input reaches the entry edge only; it is dominated."""
        corpus = _corpus(tmp_path, "c", {**SEEDS, "empty.bin": b""})
        minimize_corpus(
            build_so(),
            str(corpus),
            use_coverage=True,
            exec_mode=mode,
            inprocess_func="fuzz_test",
        )
        assert _live(corpus) == EXPECTED_KEPT


class TestLoaderIsolation:
    def test_crashing_seed_does_not_kill_minimize(self, tmp_path, build_so):
        """The loader keeps the abort() in the child; its unique edge keeps the seed."""
        corpus = _corpus(tmp_path, "c", {**SEEDS, "crash.bin": b"x"})
        kept, _ = minimize_corpus(
            build_so(),
            str(corpus),
            use_coverage=True,
            exec_mode=ExecMode.LOADER,
            inprocess_func="fuzz_test",
        )
        assert kept == len(EXPECTED_KEPT) + 1
        assert _live(corpus) == EXPECTED_KEPT | {"crash.bin"}
