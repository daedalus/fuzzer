"""Tests for whole-program ICFG construction (K-Scheduler).

The fixture compiles the same self-contained trace-pc target the SHM
channel tests use, builds a TargetDistance against it, and lifts the
whole program: every decoded function contributes nodes, direct calls
contribute caller→callee-entry edges, and the runtime probe-key table
maps the shim's exact PCs onto those nodes.
"""

import shutil
import subprocess

import pytest

from fuzzer_tool.core.distance import TargetDistance
from fuzzer_tool.core.icfg import build_interprocedural_cfg, probe_key_node_table

SRC = """\
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
static volatile int sink;
__attribute__((noinline)) int target_fn(int a) {
    if (a > 10) { sink = 1; } else { sink = 2; }
    return a + 1;
}
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 1 && data[0] == 'T')
        sink = target_fn(data[1]);
    return 0;
}
int main(int argc, char **argv) {
    unsigned char buf[1024];
    size_t n = 0;
    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 0;
        n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
    }
    LLVMFuzzerTestOneInput(buf, n);
    return 0;
}
"""

SHIM = "src/fuzzer_tool/adapters/afl_shim.c"


@pytest.fixture(scope="module")
def tp_target(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("icfg")
    src = d / "icfg_target.c"
    src.write_text(SRC)
    out = d / "icfg_target"
    r = subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-D__AFL_DISTANCE_MODE",
            "-fsanitize-coverage=trace-pc",
            "-include",
            SHIM,
            "-o",
            str(out),
            str(src),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()
    return str(out)


@pytest.fixture(scope="module")
def tp_icfg(tp_target):
    td = TargetDistance(tp_target, targets=["target_fn"])
    assert td.load()
    icfg = build_interprocedural_cfg(td)
    assert icfg is not None
    return td, icfg


class TestWholeProgramScope:
    def test_nodes_cover_non_target_functions(self, tp_icfg):
        """The point of the whole-program ICFG: not just the target
        function's blocks."""
        _, icfg = tp_icfg
        funcs = set(icfg.node_funcs)
        assert {"main", "LLVMFuzzerTestOneInput", "target_fn"} <= funcs

    def test_interprocedural_edges_point_at_callee_entry(self, tp_icfg):
        """Direct calls contribute caller→callee-entry edges."""
        _, icfg = tp_icfg
        callee_entries = {i for i, f in enumerate(icfg.node_funcs) if f == "target_fn"}
        callers = {i for i, f in enumerate(icfg.node_funcs) if f == "LLVMFuzzerTestOneInput"}
        cross = [
            (s, d)
            for s, d in zip(icfg.src.tolist(), icfg.dst.tolist(), strict=False)
            if s in callers and d in callee_entries
        ]
        assert cross, "no caller→callee edge from LLVMFuzzerTestOneInput to target_fn"


class TestRuntimeKeyTable:
    def test_every_probe_key_maps_to_a_valid_node(self, tp_icfg):
        td, icfg = tp_icfg
        table = probe_key_node_table(td, icfg)
        assert table, "expected probe keys on a trace-pc build"
        for idx in table.values():
            assert 0 <= idx < icfg.n_nodes

    def test_keys_land_inside_their_node(self, tp_icfg):
        td, icfg = tp_icfg
        base = td._base_addr or 0
        table = probe_key_node_table(td, icfg)
        starts = icfg.node_addrs
        for key, idx in table.items():
            site = base + key
            lo = starts[idx]
            hi = starts[idx + 1] if idx + 1 < len(starts) else lo + 0x1000
            assert lo <= site < hi

    def test_some_site_maps_into_the_target_function(self, tp_icfg):
        td, icfg = tp_icfg
        table = probe_key_node_table(td, icfg)
        funcs = {icfg.node_funcs[i] for i in table.values()}
        assert "target_fn" in funcs


class TestDeterminism:
    def test_two_builds_identical(self, tp_target):
        td = TargetDistance(tp_target, targets=["target_fn"])
        assert td.load()
        a = build_interprocedural_cfg(td)
        b = build_interprocedural_cfg(td)
        assert a.node_addrs == b.node_addrs
        assert a.n_edges == b.n_edges
        assert (a.src == b.src).all() and (a.dst == b.dst).all()
