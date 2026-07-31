"""Integration tests for the AFLGo BB-level distance (core/distance.py).

Real binaries are compiled with clang at test time (skipped when clang
is unavailable). Expected values are hand-derived:

  - CG distance: target function 1.0, direct caller 2.0, callees of the
    target that cannot reach it back get the penalty distance.
  - BB level: the block containing the target address is valued 0.0;
    other blocks in the target function are valued >= 1.0 (harmonic mean
    of (1 + CFG distance) is at least 1); addresses outside the target
    function are not valued, and a seed touching only them reports
    ``_NO_VALUE_DISTANCE`` (20.0).
"""

import shutil
import subprocess

import pytest

from fuzzer_tool.core.distance import _NO_VALUE_DISTANCE, TargetDistance

SRC = """\
__attribute__((noinline)) int leaf(int x) {
    return x + 1;
}
__attribute__((noinline)) int target_fn(int a) {
    int r = 0;
    if (a > 10)
        r = leaf(a);
    else
        r = a - 1;
    return r;
}
int main(void) {
    return target_fn(42);
}
"""


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("dist")
    src = d / "dist_target.c"
    src.write_text(SRC)
    exe = d / "dist_target"
    so = d / "dist_target.so"
    r1 = subprocess.run(["clang", "-O0", "-g", "-o", str(exe), str(src)], capture_output=True)
    r2 = subprocess.run(
        ["clang", "-O0", "-g", "-shared", "-fPIC", "-o", str(so), str(src)],
        capture_output=True,
    )
    assert r1.returncode == 0, r1.stderr.decode()
    assert r2.returncode == 0, r2.stderr.decode()
    return {"exe": str(exe), "so": str(so), "src": str(src)}


class TestFunctionTarget:
    def test_load_and_cg_distances(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["target_fn"])
        assert td.load()
        assert td._distances["target_fn"] == 1.0
        assert td._distances["main"] == 2.0  # main -> target_fn
        # leaf cannot reach target_fn → penalty (AFLGo: distance TO target)
        assert td._distances["leaf"] > 5.0

    def test_cfg_built_for_target_function(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["target_fn"])
        assert td.load()
        cfg = td._cfgs.get("target_fn")
        assert cfg is not None
        # -O0 branch (if/else) ⇒ at least two blocks
        assert len(cfg.blocks) >= 2

    def test_entry_block_is_target_block(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["target_fn"])
        assert td.load()
        start = td.functions["target_fn"][0]
        assert td._bb_value.get(start) == 0.0
        assert td.is_target(start)

    def test_non_entry_blocks_valued_positive(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["target_fn"])
        assert td.load()
        cfg = td._cfgs["target_fn"]
        start = td.functions["target_fn"][0]
        others = [bs for bs in cfg.blocks if bs != start]
        assert others  # the function has a branch
        for bs in others:
            # harmonic mean of (1 + d) over target blocks >= 1
            assert td._bb_value.get(bs, -1.0) >= 1.0, hex(bs)

    def test_seed_distance_valued_blocks_only(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["target_fn"])
        assert td.load()
        start = td.functions["target_fn"][0]
        # trace hitting only the target block → distance 0
        assert td.seed_distance({(0, start)}) == 0.0
        # trace hitting a non-target block of the target function → > 0
        cfg = td._cfgs["target_fn"]
        other = next(bs for bs in cfg.blocks if bs != start)
        dist = td.seed_distance({(0, other)})
        assert dist > 0.0
        # trace hitting only non-target-function code → no valued blocks
        main_start = td.functions["main"][0]
        assert td.seed_distance({(0, main_start)}) == _NO_VALUE_DISTANCE

    def test_is_target_other_function_false(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["target_fn"])
        assert td.load()
        leaf_start = td.functions["leaf"][0]
        assert not td.is_target(leaf_start)


class TestFileLineTarget:
    def test_resolves_to_function_internal_address(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["dist_target.c:6"])
        assert td.load()
        # line 6 ("if (a > 10)") is inside target_fn
        target_fn_start, target_fn_end = td.functions["target_fn"]
        assert len(td.target_addrs) >= 1
        for addr in td.target_addrs:
            assert target_fn_start <= addr < target_fn_end

    def test_line_target_block_valued_zero(self, binaries):
        td = TargetDistance(binaries["exe"], targets=["dist_target.c:6"])
        assert td.load()
        cfg = td._cfgs["target_fn"]
        assert cfg is not None
        # the block containing the line address is a target block
        line_addr = min(td.target_addrs)
        blk = cfg.block_containing(line_addr)
        assert blk is not None
        assert td._bb_value.get(blk.start) == 0.0
        assert td.is_target(line_addr)


class TestPieSharedObject:
    """.so targets have vaddr != file offset — the code-slice translation
    must make the CG and CFG analysis work."""

    def test_cg_and_cfg_correct_in_so(self, binaries):
        td = TargetDistance(binaries["so"], targets=["target_fn"])
        assert td.load()
        assert td._distances["target_fn"] == 1.0
        cfg = td._cfgs.get("target_fn")
        assert cfg is not None
        start = td.functions["target_fn"][0]
        assert cfg.block_containing(start) is not None
        assert td._bb_value.get(start) == 0.0

    def test_seed_distance_in_so(self, binaries):
        td = TargetDistance(binaries["so"], targets=["target_fn"])
        assert td.load()
        start = td.functions["target_fn"][0]
        assert td.seed_distance({(0, start)}) == 0.0
