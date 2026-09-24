"""FormatLearner transitions must carry the round's real coverage delta.

``fuzz_one`` samples ``_cov_before_fuzz = len(_global_edge_hits)`` at the top
of the round and hands ``coverage_after = len(_global_edge_hits)`` to
``FormatLearner.record_transition``. Only ``EdgeTracker.record_edges`` grows
that dict, and the format-learner block used to run *before* the
record_edges block in the same round, so the two reads were always equal:
every TimelineEntry carried ``coverage_after == coverage_before`` and the
learner's delta statistics (``_delta_moments``, the z/MAD gate, the backtest
description) never saw a discovery. Measured on png_read with
``--learn-format``: 36 of 36 transitions had delta 0 before the move, 45 of
45 had delta > 0 after it.

This drives a real campaign against a small trace-pc-guard target, because
the defect is an ordering bug between two blocks of ``fuzz_one`` and no unit
of either block can show it. Verified to fail on the pre-fix ordering
(every delta zero) and pass after it.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

SHIM = Path(__file__).resolve().parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

# Branchy enough that a few hundred execs find new edges several times.
_TARGET_SRC = r"""
#include <stddef.h>
__attribute__((visibility("default")))
int fuzz_shm_run(const unsigned char *b, size_t n) {
    int s = 0;
    for (size_t i = 0; i < n && i < 16; i++) {
        switch (b[i] & 7) {
        case 0: s += 1; break; case 1: s ^= 3; break; case 2: s -= 2; break;
        case 3: s *= 3; break; case 4: s += (int)i; break; case 5: s |= 8; break;
        case 6: s &= 5; break; default: s = -s; break;
        }
    }
    return s == 12345;
}
"""


@pytest.fixture(scope="module")
def branchy_so(tmp_path_factory):
    if shutil.which("clang") is None:
        pytest.skip("trace-pc-guard needs clang")
    d = tmp_path_factory.mktemp("fl_delta")
    src = d / "branchy.c"
    src.write_text(_TARGET_SRC)
    out = d / "branchy.so"
    subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize-coverage=trace-pc-guard",
            "-shared",
            "-fPIC",
            "-include",
            str(SHIM),
            "-o",
            str(out),
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    return out


def _run_and_capture(target: Path, tmp_path: Path) -> tuple[list[tuple[int, int]], Fuzzer]:
    seeds = tmp_path / "corpus" / "seeds"
    seeds.mkdir(parents=True)
    (tmp_path / "crashes").mkdir()
    (seeds / "s").write_bytes(b"\x00\x01\x02\x03")
    f = Fuzzer(
        target=str(target),
        corpus_dir=str(tmp_path / "corpus"),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=64,
        inprocess_direct=True,
        use_coverage=True,
        learn_format=True,
        seed=3,
    )
    assert f._format_learner is not None
    seen: list[tuple[int, int]] = []
    original = f._format_learner.record_transition

    def spy(**kw):
        seen.append((kw["coverage_before"], kw["coverage_after"]))
        return original(**kw)

    f._format_learner.record_transition = spy
    f.run(max_execs=300)
    return seen, f


def test_transitions_carry_nonzero_coverage_delta(branchy_so, tmp_path):
    seen, f = _run_and_capture(branchy_so, tmp_path)
    assert seen, "campaign produced no coverage transitions; the target is too flat"
    deltas = [after - before for before, after in seen]
    # Pre-fix every delta was exactly 0.
    assert any(d > 0 for d in deltas), f"all {len(deltas)} deltas were zero: {seen[:8]}"
    # The dict only grows, so a delta can never be negative.
    assert all(d >= 0 for d in deltas)
    # Deltas count globally new edges, so they cannot exceed what was found.
    assert sum(deltas) <= len(f._edge_tracker._global_edge_hits)
