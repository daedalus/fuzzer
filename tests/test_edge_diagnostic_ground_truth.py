"""``edge_diagnostic.py matrix --ground-truth``: the reference edge count.

The tracer (tools/ground_truth_tracer.c) logs (prev, cur, call site) per
coverage event and computes no ids, so it can judge the shim's id function
instead of sharing its defects. These tests pin the Python half: record
parsing, dedup across inputs, tolerance of a truncated tail (fuzzgoat aborts
mid-run on its planted bugs), and the end-to-end count on a real tracer
build when a compiler is available.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from shutil import which

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "edge_diagnostic.py"
TRACER_C = ROOT / "tools" / "ground_truth_tracer.c"


def _load():
    spec = importlib.util.spec_from_file_location("edge_diagnostic", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ED = _load()


def _fake_tracer(tmp_path: Path) -> Path:
    """A stand-in tracer: the input file holds the uint64 words to emit."""
    script = tmp_path / "tracer"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os, sys
            with open(sys.argv[1], "rb") as src, open(os.environ["GT_OUT"], "ab") as out:
                out.write(src.read())
            """
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _words(*vals: int) -> bytes:
    return b"".join(v.to_bytes(8, "little") for v in vals)


def test_counts_distinct_edges_and_triples_across_inputs(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    # a: edge (0,5) from two call sites, then (5,6); b repeats (5,6) and adds (6,7)
    a.write_bytes(_words(0, 5, 100, 0, 5, 200, 5, 6, 100))
    b.write_bytes(_words(5, 6, 100, 6, 7, 100))
    got = ED.ground_truth(_fake_tracer(tmp_path), [a, b], timeout=10.0)
    assert got == {"edges": 3, "triples": 4}


def test_truncated_tail_is_dropped_not_fatal(tmp_path):
    a = tmp_path / "a"
    a.write_bytes(_words(0, 5, 100, 5, 6))  # second record cut short
    got = ED.ground_truth(_fake_tracer(tmp_path), [a], timeout=10.0)
    assert got == {"edges": 1, "triples": 1}


@pytest.mark.skipif(which("clang") is None and which("gcc") is None, reason="needs a C compiler")
def test_real_tracer_sees_sibling_edges(tmp_path):
    """Built for real: two runs taking sibling successors are two edges."""
    cc = which("clang") or which("gcc")
    src = tmp_path / "h.c"
    src.write_text(
        textwrap.dedent(
            """
            #include <stdint.h>
            #include <stdio.h>
            static uint32_t g[16];
            int main(int argc, char **argv) {
                __sanitizer_cov_trace_pc_guard_init(g, g + 16);
                FILE *f = fopen(argv[1], "rb");
                int c = f ? fgetc(f) : 'a';
                __sanitizer_cov_trace_pc_guard(&g[3]);
                __sanitizer_cov_trace_pc_guard(c == 'a' ? &g[9] : &g[10]);
                return 0;
            }
            """
        )
    )
    exe = tmp_path / "gt"
    r = subprocess.run(
        [cc, "-O1", f"-include{TRACER_C}", "-o", str(exe), str(src), "-ldl"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-1500:]
    ia, ib = tmp_path / "ia", tmp_path / "ib"
    ia.write_bytes(b"a")
    ib.write_bytes(b"b")
    got = ED.ground_truth(exe, [ia, ib], timeout=10.0)
    # (0 -> 4) shared, then (4 -> 10) and (4 -> 11): three distinct edges
    assert got["edges"] == 3, got
    assert os.access(exe, os.X_OK)
