"""tools/frontier.py: branch conditions a corpus takes one way only."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import requires_clang

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import frontier  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STUB = REPO / "tools" / "lib" / "frontier_stub.c"

# Branch at line 9 (c == 'a') and line 11 (c == 'b', nested under 'a').
_TARGET = r"""
#include <stdio.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    int c = fgetc(f), d = fgetc(f);
    fclose(f);
    int r = 0;
    (void)argc;
    if (c == 'a') {
        r = 1;
        if (d == 'b') r = 2;
    }
    return r;
}
"""
LINE_A, LINE_B = 9, 11


def test_frontier_of_keeps_one_sided():
    br = {
        ("f.c", 1, 1): (3, 0),
        ("f.c", 2, 1): (1, 1),
        ("f.c", 3, 1): (0, 0),
        ("f.c", 4, 1): (0, 2),
    }
    assert frontier.frontier_of(br) == {
        ("f.c", 1, 1): frontier.Side.FALSE,
        ("f.c", 4, 1): frontier.Side.TRUE,
    }


def test_classify_source_hints():
    assert frontier.classify("if (!(p = json_alloc(state, n, 0)))") is frontier.Kind.ALLOC
    assert (
        frontier.classify("if (state.used_memory > state.settings.max_memory)")
        is frontier.Kind.LIMIT
    )
    assert (
        frontier.classify("if (state.settings.settings & json_enable_comments)")
        is frontier.Kind.SETTINGS
    )
    assert frontier.classify("if (*(++ptr) != 'r')") is frontier.Kind.OPEN


@pytest.fixture
def cov_bin(tmp_path):
    if not (shutil.which("llvm-cov") and shutil.which("llvm-profdata")):
        pytest.skip("llvm-cov / llvm-profdata not installed")
    src = tmp_path / "t.c"
    src.write_text(_TARGET)
    out = tmp_path / "t_cov"
    cmd = [
        "clang",
        "-O0",
        "-g",
        "-fprofile-instr-generate",
        "-fcoverage-mapping",
        "-o",
        str(out),
        str(src),
        str(STUB),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def _corpus(tmp_path, *inputs: bytes) -> list[Path]:
    paths = []
    for i, data in enumerate(inputs):
        p = tmp_path / f"in{i}"
        p.write_bytes(data)
        paths.append(p)
    return paths


@requires_clang
def test_frontier_on_built_target(cov_bin, tmp_path):
    # 'ax' takes a's true side; 'zz' its false side; b's true side never.
    cov = frontier.Coverage(cov_bin, tmp_path / "work")
    runs = cov.run(_corpus(tmp_path, b"ax", b"zz"))
    front = frontier.frontier_of(cov.merged(runs))
    lines = {k[1]: side for k, side in front.items() if k[0] == "t.c"}
    assert lines == {LINE_B: frontier.Side.TRUE}


@requires_clang
def test_falsification_closing_the_gap_empties_frontier(cov_bin, tmp_path):
    cov = frontier.Coverage(cov_bin, tmp_path / "work")
    runs = cov.run(_corpus(tmp_path, b"ax", b"zz", b"ab"))
    assert not {k for k in frontier.frontier_of(cov.merged(runs)) if k[0] == "t.c"}


@requires_clang
def test_adversarial_crashing_input_keeps_profile(tmp_path):
    # The stub dumps the profile on SIGSEGV; the crashing run still counts.
    if not (shutil.which("llvm-cov") and shutil.which("llvm-profdata")):
        pytest.skip("llvm-cov / llvm-profdata not installed")
    src = tmp_path / "c.c"
    src.write_text(
        '#include <stdio.h>\nint main(int c, char **v) {\n FILE *f = fopen(v[1], "rb");\n'
        " int x = fgetc(f);\n fclose(f);\n (void)c;\n if (x == 'k') { volatile int *p = 0; *p = 1; }\n"
        " return 0;\n}\n"
    )
    out = tmp_path / "c_cov"
    cmd = [
        "clang",
        "-O0",
        "-g",
        "-fprofile-instr-generate",
        "-fcoverage-mapping",
        "-o",
        str(out),
        str(src),
        str(STUB),
    ]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    cov = frontier.Coverage(out, tmp_path / "work")
    runs = cov.run(_corpus(tmp_path, b"k"))
    assert any(t for (f, line, _), (t, _f) in runs[0].branches.items() if f == "c.c" and line == 7)


def test_empty_corpus(tmp_path):
    cov = frontier.Coverage(tmp_path / "missing", tmp_path / "work")
    assert cov.run([]) == []
    assert cov.merged([]) == {}
