"""Ground-truth checks for the synthetic coverage target.

`tools/gen_synthetic_target.py` exists because three questions could not be
answered against the real target matrix:

- Nothing but `ffmpeg_read` exceeds the 8192-entry map floor, so anything
  that only bites at high load could be argued about but not measured
  without a heavy vendor build.
- No target has a genuinely coverage-dead byte region. Four campaigns
  (zlib, png twice, jpeg) failed to produce one, two of them for structural
  reasons: compressed data has no padding, and any CRC-covered format rules
  out dead bytes outright since mutating any byte moves the CRC-check edge.
  That left `LiveBitMaskEstimator`'s false-negative rate unmeasurable.
- No target has known-nondeterministic edges, so seed stability calibration
  could only ever be validated against fakes.

The value here is that ground truth is known *by construction*. These tests
assert the construction actually holds, because if the "dead" region turns
out to be live, every measurement built on this target is silently wrong --
and that failure mode is invisible unless something checks it.

Built with gcc and `-DSYNTH_MANUAL_GUARDS`: gcc cannot do
`-fsanitize-coverage=trace-pc-guard`, so the generated blocks call the
guard callback themselves. That is the same shim entry point clang's
instrumentation targets, which is the code path under test either way.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "tools", "gen_synthetic_target.py")
SHIM = os.path.join(ROOT, "src", "fuzzer_tool", "adapters", "afl_shim.c")

_DRIVER = """
#include <stdio.h>
#include <stdlib.h>
extern int fuzz_synthetic(const unsigned char *, size_t);
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    static unsigned char buf[65536];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    fuzz_synthetic(buf, n);
    return 0;
}
"""

pytestmark = pytest.mark.skipif(
    shutil.which("gcc") is None or not os.path.exists(SHIM),
    reason="needs gcc and afl_shim.c",
)

LIVE = (0, 32)
DEAD = (32, 96)


def _build(tmp_path, blocks=400, fanout=32, unstable=0):
    """Generate, compile and link a driver for one target variant."""
    src = tmp_path / f"synth_{blocks}_{unstable}.c"
    r = subprocess.run(
        [
            sys.executable,
            GEN,
            "--blocks",
            str(blocks),
            "--fanout",
            str(fanout),
            "--unstable",
            str(unstable),
            "-o",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    obj = tmp_path / f"synth_{blocks}_{unstable}.o"
    r = subprocess.run(
        ["gcc", "-O1", "-DSYNTH_MANUAL_GUARDS", "-c", str(src), "-o", str(obj)],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-800:]

    drv = tmp_path / "driver.c"
    drv.write_text(_DRIVER)
    exe = tmp_path / f"drive_{blocks}_{unstable}"
    r = subprocess.run(
        [
            "gcc",
            "-O1",
            # The exact dead/live-region count assertions assume one edge
            # per synthetic guard; ctx hashing (default-on) would add
            # caller-derived splits.
            "-D__AFL_CTX_SENSITIVE=0",
            f"-include{SHIM}",
            "-o",
            str(exe),
            str(drv),
            str(obj),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-800:]
    return str(exe)


def _run(exe, data, tmp_path, size=8192):
    cov = ShmCoverage(size=size)
    try:
        inp = tmp_path / "in.bin"
        inp.write_bytes(data)
        env = dict(os.environ, __AFL_SHM_ID=str(cov.shm_id), AFL_MAP_SIZE=str(size))
        subprocess.run([exe, str(inp)], env=env, capture_output=True)
        return cov.get_edge_ids()
    finally:
        cov.cleanup()


def _flip(rnd, base, region):
    d = bytearray(base)
    o = rnd.randrange(*region)
    d[o] ^= 1 << rnd.randrange(8)
    return bytes(d)


@pytest.fixture(scope="module")
def base():
    rnd = random.Random(7)
    return bytes(rnd.randrange(256) for _ in range(256))


class TestDeterministicVariant:
    """`--unstable 0`: the variant any liveness measurement must use."""

    def test_identical_input_gives_identical_coverage(self, tmp_path_factory, base):
        """Without this, nothing else in the file means anything: a target
        that varies run to run makes 'the dead region changed coverage'
        unfalsifiable."""
        tp = tmp_path_factory.mktemp("det")
        exe = _build(tp, unstable=0)
        first = _run(exe, base, tp)
        for _ in range(8):
            assert _run(exe, base, tp) == first

    def test_dead_region_mutations_never_change_coverage(self, tmp_path_factory, base):
        """The whole point of the target. Ground truth: zero."""
        tp = tmp_path_factory.mktemp("dead")
        exe = _build(tp, unstable=0)
        rnd = random.Random(11)
        baseline = _run(exe, base, tp)
        changed = sum(1 for _ in range(40) if _run(exe, _flip(rnd, base, DEAD), tp) != baseline)
        assert changed == 0, f"{changed}/40 dead-region mutations moved coverage"

    def test_live_region_mutations_do_change_coverage(self, tmp_path_factory, base):
        """Control for the test above. A target where *nothing* moves
        coverage would pass the dead-region test trivially."""
        tp = tmp_path_factory.mktemp("live")
        exe = _build(tp, unstable=0)
        rnd = random.Random(11)
        baseline = _run(exe, base, tp)
        changed = sum(1 for _ in range(40) if _run(exe, _flip(rnd, base, LIVE), tp) != baseline)
        assert changed >= 30, f"only {changed}/40 live-region mutations moved coverage"

    def test_no_checksum_over_the_input(self, tmp_path_factory, base):
        """Implied by the dead-region result, asserted separately because it
        is the property every real format failed to provide: under a
        whole-file checksum, *every* byte is live."""
        tp = tmp_path_factory.mktemp("nocsum")
        exe = _build(tp, unstable=0)
        rnd = random.Random(3)
        baseline = _run(exe, base, tp)
        tail_region = (96, 256)
        unchanged = sum(
            1 for _ in range(20) if _run(exe, _flip(rnd, base, tail_region), tp) == baseline
        )
        assert unchanged > 0, "every byte affected coverage — looks checksum-like"


class TestUnstableVariant:
    """`--unstable N`: ASLR-gated blocks, for validating stability work."""

    @staticmethod
    def _observe(exe, base, tp, max_runs=40):
        """Collect runs until instability shows, up to max_runs.

        Deliberately not a fixed 12. Detecting instability is itself
        probabilistic -- that is the whole finding in
        docs/sweeps/synthetic_target_ground_truth_2026-08-19.md, where 3
        runs miss it 6% of the time -- so a fixed small count makes this
        test flaky by construction. It went red exactly that way once
        before this early-exit was added, which is the best evidence the
        finding is real.

        The blocks are gated on ASLR, so an environment without address
        randomisation legitimately produces stable coverage. That is a skip,
        not a failure: it says nothing about the target.
        """
        runs = []
        for _ in range(max_runs):
            runs.append(_run(exe, base, tp))
            if len(runs) >= 4 and set().union(*runs) - set.intersection(*runs):
                return runs
        return runs

    def test_identical_input_gives_varying_coverage(self, tmp_path_factory, base):
        tp = tmp_path_factory.mktemp("unstable")
        exe = _build(tp, unstable=4)
        runs = self._observe(exe, base, tp)
        unstable = set().union(*runs) - set.intersection(*runs)
        if not unstable:
            aslr = "0"
            try:
                with open("/proc/sys/kernel/randomize_va_space") as fh:
                    aslr = fh.read().strip()
            except OSError:
                pass
            pytest.skip(
                f"no ASLR variance observed in {len(runs)} runs (randomize_va_space={aslr})"
            )
        assert unstable

    def test_instability_is_confined_to_the_designated_blocks(self, tmp_path_factory, base):
        """A handful of unstable edges, not wholesale nondeterminism —
        otherwise it is useless as a controlled test case."""
        tp = tmp_path_factory.mktemp("unstable2")
        exe = _build(tp, unstable=4)
        runs = self._observe(exe, base, tp)
        unstable = set().union(*runs) - set.intersection(*runs)
        if not unstable:
            pytest.skip("no ASLR variance observed; nothing to bound")
        stable = set.intersection(*runs)
        assert len(unstable) <= 8, f"{len(unstable)} unstable edges, expected <= 8"
        assert len(stable) > len(unstable) * 4


class TestGuardCountScales:
    def test_block_count_drives_distinct_edges(self, tmp_path_factory, base):
        """Map sizing keys on guard count, so --blocks has to actually move
        it. This is what lets the target reach loads only ffmpeg_read
        could otherwise reach."""
        tp = tmp_path_factory.mktemp("scale")
        small = _build(tp, blocks=200, fanout=16, unstable=0)
        large = _build(tp, blocks=4000, fanout=16, unstable=0)
        rnd = random.Random(5)
        inputs = [bytes(rnd.randrange(256) for _ in range(256)) for _ in range(25)]
        small_ids = set().union(*(_run(small, i, tp) for i in inputs))
        large_ids = set().union(*(_run(large, i, tp) for i in inputs))
        assert len(large_ids) > len(small_ids) * 2, (
            f"200 blocks -> {len(small_ids)} edges, 4000 blocks -> {len(large_ids)}"
        )
