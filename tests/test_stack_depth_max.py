"""--stack-depth-max: admit an input whose stack depth beats the campaign max.

honggfuzz instrumentCheckStackDepth + fuzz_perfFeedback. The shim already
publishes the per-run depth at SHM offset 0 (tests/test_shim_stack_depth.py);
this checks the novelty rule, the FuzzRound gate and the Fuzzer wiring.
"""

import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.shim_health import STACK_DEPTH_ENV
from fuzzer_tool.core.stack_depth import StackMax

SHIM = "src/fuzzer_tool/adapters/afl_shim.c"

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")


# ── StackMax ──────────────────────────────────────────────────────────


def test_first_sighting_is_baseline_not_novelty():
    sm = StackMax()
    assert not sm.observe(4096)
    assert sm.hits == 0


def test_deeper_reports():
    sm = StackMax()
    sm.observe(4096)
    assert sm.observe(4097)
    assert sm.hits == 1
    assert sm.top == 4097


def test_equal_or_shallower_never_reports():
    """Falsification: the same depth again, or less, is not novelty."""
    sm = StackMax()
    sm.observe(4096)
    for d in (4096, 4095, 16, 1):
        assert not sm.observe(d)
    assert sm.hits == 0
    assert sm.top == 4096


def test_zero_depth_ignored():
    """Adversarial: builds with no per-edge callback publish 0 forever."""
    sm = StackMax()
    for _ in range(1000):
        assert not sm.observe(0)
    assert sm.hits == 0
    assert sm.top == 0


def test_zero_does_not_become_baseline():
    """A 0 before the first real depth must not make that depth a report."""
    sm = StackMax()
    sm.observe(0)
    assert not sm.observe(4096)


def test_reports_once_per_new_level():
    """Each report admits an input: a ramp reports once per distinct depth."""
    sm = StackMax()
    depths = [256 * k for k in range(1, 11)]
    reports = [sm.observe(d) for d in depths + depths[::-1]]
    assert sum(reports) == len(depths) - 1


# ── FuzzRound gate ────────────────────────────────────────────────────


class _Shm:
    def __init__(self, depth):
        self._depth = depth

    def read_stack_depth(self):
        return self._depth


def _round(stack_max, depth, timeout=False, crash=False, shm=True):
    from fuzzer_tool.services.fuzz_round import FuzzRound

    f = SimpleNamespace(_stack_max=stack_max)
    r = FuzzRound(f, b"seed")
    r._is_timeout = timeout
    r._is_crash = crash
    r._scanned_shm = _Shm(depth) if shm else None
    r._observe_stack()
    return r


def test_deeper_round_admits_and_credits():
    sm = StackMax()
    sm.observe(1024)
    r = _round(sm, 8192)
    assert r._is_stack_max
    assert r._admits()

    r._judge()
    assert r._success


def test_truncated_runs_ignored():
    """Adversarial: a crash or timeout stopped mid-recursion; not a max."""
    sm = StackMax()
    sm.observe(1024)
    assert not _round(sm, 10**6, timeout=True)._is_stack_max
    assert not _round(sm, 10**6, crash=True)._is_stack_max
    assert sm.hits == 0
    assert sm.top == 1024


def test_off_or_no_shm_never_admits():
    assert not _round(None, 10**6)._is_stack_max
    sm = StackMax()
    sm.observe(1024)
    assert not _round(sm, 10**6, shm=False)._is_stack_max


# ── Fuzzer wiring ─────────────────────────────────────────────────────


def _fz(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus = tmp_path / "c"
    corpus.mkdir(exist_ok=True)
    (corpus / "a").write_bytes(b"seed")
    return Fuzzer(
        target="/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "k"),
        quiet_stats=True,
        **kw,
    )


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(STACK_DEPTH_ENV, raising=False)
    f = _fz(tmp_path, use_coverage=True)
    assert f._stack_max is None
    assert os.environ[STACK_DEPTH_ENV] == "0"


def test_flag_arms_and_exports(tmp_path, monkeypatch):
    monkeypatch.delenv(STACK_DEPTH_ENV, raising=False)
    f = _fz(tmp_path, use_coverage=True, stack_depth_max=True)
    assert isinstance(f._stack_max, StackMax)
    assert os.environ[STACK_DEPTH_ENV] == "1"


def test_no_shm_refused_loudly(tmp_path, monkeypatch, capsys):
    """Adversarial: no SHM means no depth; the flag must not claim to run."""
    monkeypatch.delenv(STACK_DEPTH_ENV, raising=False)
    f = _fz(tmp_path, use_coverage=False, stack_depth_max=True)
    assert f._stack_max is None
    assert "--stack-depth-max" in capsys.readouterr().out


# ── End to end ────────────────────────────────────────────────────────

# Coverage is flat whatever the input: one guard at the top (sets the
# shim's stack base), one at the bottom of a recursion as deep as the
# input is long. Only the stack depth varies.
_DRIVER = r"""
#include <stdio.h>
#include <stdint.h>
static uint32_t top = 1, bottom = 2;
__attribute__((noinline)) static int rec(int n) {
    char pad[128];
    pad[0] = (char)n;
    __asm__ volatile("" : : "r"(pad) : "memory");
    if (n <= 0) { __sanitizer_cov_trace_pc_guard(&bottom); return pad[0]; }
    return rec(n - 1) + pad[0];
}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); if (!f) return 0;
    unsigned char buf[64]; size_t n = fread(buf, 1, sizeof buf, f); fclose(f);
    __sanitizer_cov_trace_pc_guard(&top);
    return rec((int)n) == 0x7fffffff;
}
"""


_SEED_LEN = 8


def _e2e(tmp_path, monkeypatch, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    monkeypatch.delenv(STACK_DEPTH_ENV, raising=False)
    tmp_path.mkdir()
    src, exe = tmp_path / "drv.c", tmp_path / "drv"
    src.write_text(_DRIVER)
    r = subprocess.run(
        ["clang", "-O1", "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[:300]

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a").write_bytes(b"x" * _SEED_LEN)
    f = Fuzzer(
        target=str(exe),
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=64,
        timeout=1,
        mutations_per_input=2,
        quiet_stats=True,
        use_coverage=True,
        file_mode=True,
        **kw,
    )
    f.run(iterations=100_000, max_execs=400)
    return f


@needs_clang
def test_deeper_inputs_admitted_end_to_end(tmp_path, monkeypatch):
    """Control first (Hard Rule 46): coverage is flat, so without the flag no
    input longer (deeper) than the seed is kept."""
    off = _e2e(tmp_path / "off", monkeypatch)
    assert max(len(d) for d in off.corpus) == _SEED_LEN

    on = _e2e(tmp_path / "on", monkeypatch, stack_depth_max=True)
    assert on._stack_max.hits > 0
    assert max(len(d) for d in on.corpus) > _SEED_LEN
