"""Per-seed hw counters belong to the seed's own execution.

Two defects:

1. ``FuzzRound._commit_edges`` stored the *child's* counts under the
   *parent's* key, so ``seed_hw_*`` described some mutant, not the seed.
2. The seed scorer read ``_last_perf_deltas`` -- the previous execution,
   unrelated to the picked seed -- and only under ``--honggfuzz``.

The fake counters derive each execution's counts from the input bytes, so
the expected value for any seed is computable independently of the fuzzer.
"""

import shutil
import subprocess
import zlib

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

SHIM = "src/fuzzer_tool/adapters/afl_shim.c"

_DRIVER = r"""
#include <stdio.h>
#include <stdint.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); if (!f) return 0;
    unsigned char buf[64]; size_t n = fread(buf, 1, sizeof buf, f); fclose(f);
    for (size_t i = 0; i < n; i++) {
        uint32_t guard = 1 + (buf[i] >> 2);
        __sanitizer_cov_trace_pc_guard(&guard);
    }
    return 0;
}
"""


# One fixed edge whatever the input: coverage saturates on the first exec.
_FLAT = r"""
#include <stdint.h>
int main(int argc, char **argv) {
    uint32_t guard = 1;
    __sanitizer_cov_trace_pc_guard(&guard);
    return 0;
}
"""


def _instr(data: bytes) -> int:
    """Instruction count the fake reports for *data*: unique per input."""
    return 1000 + zlib.crc32(data) % 1_000_000


class _InputPerf:
    """PerfCounters stand-in: counts are a pure function of the input run."""

    counter_names = ["instructions", "branches"]

    def __init__(self, instr=None):
        self.data = b""
        self._read_count = 0
        self.instr = instr or _instr

    def open_for_pid(self, pid):
        return True

    def read_and_reset(self):
        self._read_count += 1
        n = self.instr(self.data)
        return {"instructions": n, "branches": 1 + n // 2}

    def close(self):
        return None


def _fuzzer(tmp_path, driver=_DRIVER, instr=None) -> Fuzzer:
    src, exe = tmp_path / "drv.c", tmp_path / "drv"
    src.write_text(driver)
    r = subprocess.run(
        ["clang", "-O1", "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[:300]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a").write_bytes(b"hello world")
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
    )
    perf = _InputPerf(instr)
    f._perf_counters = perf
    f._ran = {}
    run = f._runner.run_target

    def tagged(data):
        perf.data = data
        f._ran[f._seed_key(data)] = data
        return run(data)

    f._runner.run_target = tagged
    return f


@pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")
def test_regression_seed_hw_is_own_exec(tmp_path):
    """Falsification: every stored count matches the seed's own bytes."""
    f = _fuzzer(tmp_path)
    f.run(iterations=100_000, max_execs=400)

    stored = f._edge_tracker.seed_hw_instructions
    admitted = [k for k in map(f._seed_key, f.corpus) if k in stored]
    assert len(admitted) >= 2, "admissions must record their counts"

    # Trim and minimize retire seeds, so check against every executed input.
    for key, val in stored.items():
        assert val == _instr(f._ran[key])


@pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")
def test_regression_score_reads_picked_seed(tmp_path):
    """Adversarial: scorer gets the picked seed's counts, without --honggfuzz."""
    f = _fuzzer(tmp_path)
    assert not f.honggfuzz
    seen: list[tuple[int, int]] = []
    score = f._seed_scorer.score
    last: dict = {}

    def spy_score(**kw):
        last.clear()
        last.update(kw)
        return score(**kw)

    fuzz_one = f.fuzz_one

    def spy_fuzz_one(seed):
        if last:
            want = f._edge_tracker.get_seed_hw_instructions(f._seed_key(seed))
            seen.append((last.get("hw_instructions", 0), want))
            last.clear()
        return fuzz_one(seed)

    f._seed_scorer.score = spy_score
    f.fuzz_one = spy_fuzz_one
    f.run(iterations=100_000, max_execs=400)

    measured = [s for s in seen if s[1] > 0]
    assert measured, "no scored pick had recorded counts"
    for got, want in measured:
        assert got == want


@pytest.mark.skipif(shutil.which("clang") is None, reason="no clang")
def test_hw_max_input_is_admitted(tmp_path):
    """--hw-perf novelty end to end: the campaign-max input is in the corpus."""
    from fuzzer_tool.core.hw_signals import HwMax

    # Flat coverage: only the instruction count (input length) can admit.
    f = _fuzzer(tmp_path, driver=_FLAT, instr=lambda d: 1000 + 100 * len(d))
    f._hw_max = HwMax()
    f.run(iterations=100_000, max_execs=400)

    assert f._hw_max.hits > 0
    top = f._hw_max._max["instructions"]
    assert top in f._edge_tracker.seed_hw_instructions.values()
