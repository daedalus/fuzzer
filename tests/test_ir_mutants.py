"""core/ir_mutants: mutation sites in textual LLVM IR (docs/action_plan_mutant_kill_metric.md P1)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.ir_mutants import MutOp, Site, apply_site, find_sites, sample_sites
from tests.support.scripted_rng import ScriptedRng

# One function per pattern, plus lines that must never be sites.
_LL = """\
; ModuleID = 'fixture.c'
@.str = private unnamed_addr constant [12 x i8] c"icmp slt %x\\00", align 1

define dso_local i32 @f(i32 noundef %n, ptr noundef %p) #0 {
entry:
  %n.addr = alloca i32, align 4
  %p.addr = alloca ptr, align 8
  store i32 %n, ptr %n.addr, align 4, !tbaa !5
  store ptr %p, ptr %p.addr, align 8, !tbaa !5
  %0 = load i32, ptr %n.addr, align 4
  %cmp = icmp slt i32 %0, 4
  br i1 %cmp, label %if.then, label %if.else, !llvm.loop !7

if.then:                                          ; preds = %entry
  %eq = icmp eq i32 %0, 9
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %n.addr, align 4
  call void @g(ptr noundef %p)
  call void @llvm.lifetime.start.p0(i64 4, ptr %n.addr)
  br label %if.else

if.else:                                          ; preds = %if.then, %entry
  %buf = call noalias ptr @malloc(i64 noundef %sz) #3
  %c2 = call noalias ptr @calloc(i64 noundef 1, i64 noundef 8) #3
  store i32 1, ptr %cleanup.dest.slot, align 4
  ; icmp ult i32 %0, 1 in a comment
  ret i32 0
}

!7 = !{!"icmp slt"}
"""


def _line(text: str, needle: str) -> int:
    """Index of the one line containing *needle* (derived, never a literal index)."""
    hits = [i for i, ln in enumerate(text.split("\n")) if needle in ln]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def _changed(a: str, b: str) -> list[tuple[str, str]]:
    """Pairs of differing lines; requires equal line counts."""
    la, lb = a.split("\n"), b.split("\n")
    assert len(la) == len(lb)
    return [(x, y) for x, y in zip(la, lb, strict=True) if x != y]


class TestFindSites:
    def test_each_pattern_found(self) -> None:
        got = set(find_sites(_LL))
        want = {
            Site(_line(_LL, "%cmp = icmp slt"), MutOp.ICMP_BOUNDARY),
            Site(_line(_LL, "%cmp = icmp slt"), MutOp.ICMP_SIGN),
            Site(_line(_LL, "%eq = icmp eq"), MutOp.ICMP_EQ_NE),
            Site(_line(_LL, "br i1 %cmp"), MutOp.BR_NEGATE),
            Site(_line(_LL, "%inc = add nsw"), MutOp.ARITH_SWAP),
            Site(_line(_LL, "store i32 %inc"), MutOp.STORE_DELETE),
            Site(_line(_LL, "call void @g("), MutOp.CALL_DELETE),
            Site(_line(_LL, "@malloc("), MutOp.ALLOC_SHRINK),
            Site(_line(_LL, "@calloc("), MutOp.ALLOC_SHRINK),
        }
        assert got == want

    def test_sites_sorted_by_line(self) -> None:
        sites = find_sites(_LL)
        assert sites == sorted(sites, key=lambda s: (s.line, s.op.value))

    def test_adversarial_lines_are_not_sites(self) -> None:
        lines = {s.line for s in find_sites(_LL)}
        never = [
            'c"icmp slt',  # string constant
            '!{!"icmp slt"}',  # metadata
            "; icmp ult",  # comment
            "store i32 %n, ptr %n.addr",  # argument spill
            "store ptr %p, ptr %p.addr",  # argument spill
            "%cleanup.dest.slot",  # compiler bookkeeping
            "br label %if.else",  # unconditional branch
            "@llvm.lifetime.start",  # intrinsic
        ]
        for needle in never:
            assert _line(_LL, needle) not in lines, needle

    def test_empty_module(self) -> None:
        assert find_sites("") == []

    def test_crlf_same_sites_as_lf(self) -> None:
        assert find_sites(_LL.replace("\n", "\r\n")) == find_sites(_LL)


class TestApplySite:
    @pytest.mark.parametrize(
        ("needle", "op", "old", "new"),
        [
            ("%cmp = icmp slt", MutOp.ICMP_BOUNDARY, "icmp slt", "icmp sle"),
            ("%cmp = icmp slt", MutOp.ICMP_SIGN, "icmp slt", "icmp ult"),
            ("%eq = icmp eq", MutOp.ICMP_EQ_NE, "icmp eq", "icmp ne"),
            ("%inc = add nsw", MutOp.ARITH_SWAP, "= add nsw", "= sub nsw"),
        ],
    )
    def test_one_token_swapped(self, needle: str, op: MutOp, old: str, new: str) -> None:
        i = _line(_LL, needle)
        orig = _LL.split("\n")[i]
        assert _changed(_LL, apply_site(_LL, Site(i, op))) == [(orig, orig.replace(old, new, 1))]

    def test_br_negate_swaps_targets_keeps_metadata(self) -> None:
        i = _line(_LL, "br i1 %cmp")
        ((_, new),) = _changed(_LL, apply_site(_LL, Site(i, MutOp.BR_NEGATE)))
        assert new == "  br i1 %cmp, label %if.else, label %if.then, !llvm.loop !7"

    def test_store_delete_removes_line_only(self) -> None:
        i = _line(_LL, "store i32 %inc")
        out = apply_site(_LL, Site(i, MutOp.STORE_DELETE)).split("\n")
        assert out == _LL.split("\n")[:i] + _LL.split("\n")[i + 1 :]

    def test_alloc_shrink_inserts_sub_and_rewires_size(self) -> None:
        i = _line(_LL, "@malloc(")
        out = apply_site(_LL, Site(i, MutOp.ALLOC_SHRINK)).split("\n")
        assert out[:i] == _LL.split("\n")[:i]
        sub, call = out[i], out[i + 1]
        name = sub.split(" = ")[0].strip()
        assert sub == f"  {name} = sub i64 %sz, 16"
        assert call == f"  %buf = call noalias ptr @malloc(i64 noundef {name}) #3"
        assert name not in _LL

    def test_calloc_shrinks_element_size(self) -> None:
        i = _line(_LL, "@calloc(")
        out = apply_site(_LL, Site(i, MutOp.ALLOC_SHRINK)).split("\n")
        name = out[i].split(" = ")[0].strip()
        assert (
            out[i + 1] == f"  %c2 = call noalias ptr @calloc(i64 noundef 1, i64 noundef {name}) #3"
        )

    def test_crlf_preserved(self) -> None:
        crlf = _LL.replace("\n", "\r\n")
        site = Site(_line(_LL, "%eq = icmp eq"), MutOp.ICMP_EQ_NE)
        assert apply_site(crlf, site) == apply_site(_LL, site).replace("\n", "\r\n")

    def test_stale_site_raises(self) -> None:
        with pytest.raises(ValueError, match="not a"):
            apply_site(_LL, Site(_line(_LL, "ret i32 0"), MutOp.ICMP_EQ_NE))


class TestSampleSites:
    def test_cap_at_or_above_size_keeps_all(self) -> None:
        sites = find_sites(_LL)
        assert sample_sites(sites, len(sites), ScriptedRng()) == sites

    def test_scripted_subset(self) -> None:
        # ScriptedRng.shuffle reverses: the kept subset is the last *cap*
        # sites, returned in line order.
        sites = find_sites(_LL)
        cap = len(sites) // 2
        assert sample_sites(sites, cap, ScriptedRng()) == sites[-cap:]

    def test_negative_cap_raises(self) -> None:
        with pytest.raises(ValueError):
            sample_sites(find_sites(_LL), -1, ScriptedRng())


_TOY_C = """\
#include <stdlib.h>
#include <string.h>
int run(const unsigned char *b, size_t n) {
    char *p = malloc(4);
    if (!p) return 0;
    if (n < 4) p[n] = 1; else p[0] = 2;
    int r = 0;
    for (size_t i = 0; i < n; i++) r += b[i] == 'x' ? 1 : -1;
    free(p);
    return r > 2;
}
"""


@pytest.mark.skipif(
    not (shutil.which("clang") and shutil.which("llvm-as")), reason="needs clang + llvm-as"
)
def test_every_site_on_clang_ir_verifies(tmp_path) -> None:
    """Falsification on real IR: each mutant is valid LLVM (llvm-as accepts it)."""
    src = tmp_path / "toy.c"
    src.write_text(_TOY_C)
    ll = tmp_path / "toy.ll"
    subprocess.run(
        [
            "clang",
            "-O2",
            "-Xclang",
            "-disable-llvm-passes",
            "-emit-llvm",
            "-S",
            str(src),
            "-o",
            str(ll),
        ],
        check=True,
    )
    text = ll.read_text()
    sites = find_sites(text)
    assert {s.op for s in sites} >= {
        MutOp.ICMP_BOUNDARY,
        MutOp.BR_NEGATE,
        MutOp.STORE_DELETE,
        MutOp.ALLOC_SHRINK,
    }

    for site in sites:
        mut = tmp_path / "m.ll"
        mut.write_text(apply_site(text, site))
        res = subprocess.run(
            ["llvm-as", str(mut), "-o", "/dev/null"], capture_output=True, text=True
        )
        assert res.returncode == 0, (site, res.stderr)
