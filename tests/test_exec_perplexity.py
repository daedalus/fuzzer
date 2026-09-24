"""Effective edges in the stall reason (P2-1 of handover_edge_id_axis_2026-09-18).

The handover asked for ``2 ** H`` as a *trend*: the value when coverage last
grew against the value now.  ``EdgeTracker.effective_edges()`` cannot carry
that -- it is cumulative, and in the fuzz loop it only sees inputs admitted
for new coverage -- so ``ExecutionPerplexity`` samples executed inputs per
discovery window.  ``TestWhyNotCumulative`` pins the reason with numbers.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core.scheduler_substrate import ExecutionPerplexity, effective_edges

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"


def _uniform(edges, count=10):
    return {e: count for e in edges}


def _filled(p, counts, n):
    for _ in range(n):
        p.observe(counts)
    return p


class TestWindows:
    def test_nothing_measured_before_a_discovery(self):
        p = _filled(ExecutionPerplexity(stride=1, min_samples=4), _uniform(range(8)), 10)
        assert p.reference is None
        assert p.current == pytest.approx(8.0)
        assert p.reason_suffix() == ""

    def test_discovery_closes_a_measured_window(self):
        p = _filled(ExecutionPerplexity(stride=1, min_samples=4), _uniform(range(8)), 4)
        p.note_new_edge()
        assert p.reference == pytest.approx(8.0)
        assert p.current is None  # the new window is empty
        _filled(p, _uniform(range(2)), 4)
        assert p.current == pytest.approx(2.0)
        assert p.reason_suffix() == " + effective edges 8->2"

    def test_short_window_stays_open(self):
        """Early discoveries arrive every few execs: not a measurement."""
        p = _filled(ExecutionPerplexity(stride=1, min_samples=4), _uniform(range(8)), 3)
        p.note_new_edge()
        assert p.reference is None
        _filled(p, _uniform(range(8)), 1)
        p.note_new_edge()
        assert p.reference == pytest.approx(8.0)

    def test_reference_equals_effective_edges_of_the_window(self):
        p = ExecutionPerplexity(stride=1, min_samples=2)
        a, b = {1: 5, 2: 1, 3: 1}, {1: 3, 4: 7}
        p.observe(a)
        p.observe(b)
        p.note_new_edge()
        assert p.reference == pytest.approx(effective_edges({1: 8, 2: 1, 3: 1, 4: 7}))

    def test_empty_and_zero_counts_are_not_samples(self):
        p = ExecutionPerplexity(stride=1, min_samples=1)
        p.observe({})
        assert p.current is None
        p.observe({1: 0, 2: 3})
        assert p.current == pytest.approx(1.0)

    def test_due_is_one_in_stride(self):
        p = ExecutionPerplexity(stride=4)
        assert [p.due() for _ in range(12)].count(True) == 3

    def test_bad_parameters_rejected(self):
        with pytest.raises(ValueError):
            ExecutionPerplexity(stride=0)
        with pytest.raises(ValueError):
            ExecutionPerplexity(min_samples=0)


class TestWhyNotCumulative:
    def test_collapse_raises_the_cumulative_value(self):
        """Mass piling onto the history's tail flattens the cumulative
        distribution: its 2**H goes UP while the executions collapse."""
        history = {e: 1000 // (e + 1) for e in range(40)}  # heavy head, thin tail
        stall_exec = {e: 10 for e in range(30, 34)}  # 4 tail edges, over and over
        p = ExecutionPerplexity(stride=1, min_samples=4)
        _filled(p, history, 4)
        p.note_new_edge()
        _filled(p, stall_exec, 50)

        cumulative_before = effective_edges(history)
        cumulative_after = effective_edges(
            {e: history[e] + 50 * stall_exec.get(e, 0) for e in history}
        )
        assert cumulative_after > cumulative_before  # the wrong direction
        assert p.current < p.reference / 4  # the window sees the collapse
        assert p.current == pytest.approx(4.0)


def _stalled_fuzzer():
    from tests.test_reseed_on_stall import _stalled_fuzzer as make

    return make()


class TestStallReason:
    def test_reason_carries_the_trend(self, capsys):
        f = _stalled_fuzzer()
        p = ExecutionPerplexity(stride=1, min_samples=2)
        _filled(p, _uniform(range(40)), 2)
        p.note_new_edge()
        _filled(p, _uniform(range(10)), 2)
        f._exec_perplexity = p
        assert f._maybe_trigger_stall_recovery(400) is True
        assert "effective edges 40->10" in capsys.readouterr().out

    def test_no_suffix_without_a_measurement(self, capsys):
        f = _stalled_fuzzer()
        assert f._maybe_trigger_stall_recovery(400) is True
        out = capsys.readouterr().out
        assert "STALL #1" in out
        assert "effective edges" not in out

    def test_fuzzer_owns_one(self):
        assert isinstance(_stalled_fuzzer()._exec_perplexity, ExecutionPerplexity)


_DRIVER = r"""
#include <stdio.h>
#include <stdint.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); if (!f) return 0;
    unsigned char buf[64]; size_t n = fread(buf, 1, sizeof buf, f); fclose(f);
    for (size_t i = 0; i < n; i++) {
        uint32_t reps = 1 + (buf[i] & 7);
        for (uint32_t r = 0; r < reps; r++) {
            uint32_t guard = 1 + (buf[i] >> 2);
            __sanitizer_cov_trace_pc_guard(&guard);
        }
    }
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")
def test_fuzz_loop_samples_executed_inputs(tmp_path):
    """The loop wiring, end to end: without the sampling call in fuzz_one the
    window never fills, whatever the unit tests above say."""
    from fuzzer_tool.services.fuzzer import Fuzzer

    src, exe = tmp_path / "drv.c", tmp_path / "drv"
    src.write_text(_DRIVER)
    r = subprocess.run(
        ["gcc", "-O1", "-include", str(SHIM), "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"driver failed to build: {r.stderr[:300]}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a").write_bytes(b"hello world")
    (corpus / "b").write_bytes(b"\x01\x02\x03")
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
    assert f.shm_cov is not None
    f.run(iterations=100_000, max_execs=600)
    p = f._exec_perplexity
    assert p._seen >= 500
    assert p._samples > 0 or p.reference is not None
