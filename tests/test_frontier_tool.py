"""tools/frontier.py: branch conditions a corpus takes one way only."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import requires_clang

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import frontier  # noqa: E402

from fuzzer_tool.adapters import frontier_cov  # noqa: E402

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


def _ffmpeg_cov() -> Path | None:
    root = Path(os.environ.get("FUZZ_BUILD_ROOT", Path.home() / "fuzzing" / "builds"))
    found = sorted(root.glob("ffmpeg_read_*_cov"))
    return found[-1] if found else None


@pytest.mark.skipif(
    _ffmpeg_cov() is None, reason="no ffmpeg_read_<ver>_cov (build_targets.sh --frontier)"
)
def test_ffmpeg_cov_counts_library_branches(tmp_path):
    # The archives carry coverage too: a WAV input must count branches in
    # libavformat's wav demuxer, not only in the wrapper.
    wav = tmp_path / "in.wav"
    pcm = bytes(64)
    wav.write_bytes(
        b"RIFF" + (36 + len(pcm)).to_bytes(4, "little") + b"WAVEfmt "
        + (16).to_bytes(4, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
        + (8000).to_bytes(4, "little") + (16000).to_bytes(4, "little")
        + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
        + b"data" + len(pcm).to_bytes(4, "little") + pcm
    )  # fmt: skip
    runs = frontier.Coverage(_ffmpeg_cov(), tmp_path / "work").run([wav])
    files = {f for f, _, _ in runs[0].branches}
    assert "ffmpeg_read.c" in files
    assert "wavdec.c" in files
    assert frontier.STUB_NAME not in files


@requires_clang
def test_merged_profile_matches_per_input_sum(cov_bin, tmp_path, monkeypatch):
    # Two paths to corpus counts: one merged profile vs summed per-input exports.
    # MERGE_BATCH=1 forces a fold per input (the batch boundary on every step).
    monkeypatch.setattr(frontier_cov, "MERGE_BATCH", 1)
    inputs = _corpus(tmp_path, b"ax", b"zz", b"ab")
    cov = frontier.Coverage(cov_bin, tmp_path / "work")
    summed = frontier.Coverage.merged(cov.run(inputs))
    assert frontier.Coverage.merged(cov.run(inputs)) == summed  # control: reference vs itself
    assert cov.profile(inputs) == summed


def test_sample_prefers_fuzzed_seeds(tmp_path):
    from fuzzer_tool.core.hashing import key16

    files = _corpus(tmp_path, *(bytes([i]) * 4 for i in range(10)))
    fuzzed = files[7]
    meta = {key16(fuzzed.read_bytes()): {"fuzz_count": 3, "child_count": 1}}
    picked = frontier._sample(files, meta, 2)
    assert picked[0] == fuzzed and len(set(picked)) == 2
    assert frontier._sample(files, meta, 50) == files  # adversarial: n above corpus size
    assert frontier.Coverage(tmp_path / "x", tmp_path / "w").profile([]) == {}
