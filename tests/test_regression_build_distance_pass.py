"""build_targets.sh --distance: one failing target must not abort the pass.

The script runs under ``set -e``. A target whose system library is absent
(jpeg_read without jpeglib.h) failed its clang line and aborted the whole
build, so every later target (and fuzzgoat_read_dist.so, the Katz trace-pc
fuzzgoat build) was never attempted. The function is extracted from the
script and run against a stub clang, so the test exercises the script's
own text.
"""

import re
import subprocess
from pathlib import Path

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"

# Stub clang: fail on any source named in $FAIL_ON, else touch the -o file.
_STUB = r"""
clang() {
    local out="" a
    for a in "$@"; do
        for f in $FAIL_ON; do [[ "$a" == *"$f"* ]] && return 1; done
    done
    while [ $# -gt 0 ]; do [ "$1" = "-o" ] && out="$2"; shift; done
    [ -n "$out" ] && : > "$out"
}
ok() { echo "OK $*"; }
warn() { echo "WARN $*"; }
"""


def _function(name: str) -> str:
    text = BUILD_SCRIPT.read_text()
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.M | re.S)
    assert m, f"{name} not found in build_targets.sh"
    return m.group(0)


def _run(tmp_path: Path, fail_on: str) -> subprocess.CompletedProcess:
    src = tmp_path / "src"
    out = tmp_path / "out"
    vendor = tmp_path / "vendor"
    for d in (src, out, vendor / "fuzzgoat"):
        d.mkdir(parents=True)
    for name in ("png_read", "jpeg_read", "test_target", "fuzzgoat_read"):
        (src / f"{name}.c").write_text("")
    (vendor / "fuzzgoat" / "fuzzgoat.c").write_text("")

    env = (
        f'set -e\nWITH_DISTANCE=1; WITH_CMPLOG=0; HAS_FUZZGOAT=1; ASAN_CFLAGS=""\n'
        f'TARGETS="{out}"; TARGETS_SRC="{src}"; VENDOR="{vendor}"; SHIM=/dev/null\n'
        f'BUILD_LOG="{tmp_path}/build.log"; FAIL_ON="{fail_on}"\n'
    )
    sh = env + _STUB + _function("build_distance_so_targets") + "build_distance_so_targets\n"
    return subprocess.run(["bash", "-c", sh], capture_output=True, text=True)


def test_regression_missing_lib_does_not_abort_distance_pass(tmp_path: Path):
    res = _run(tmp_path, "jpeg_read.c")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARN failed: jpeg_read_dist.so (nosan)" in res.stdout
    assert (tmp_path / "out" / "test_target_dist.so").exists()
    assert (tmp_path / "out" / "fuzzgoat_read_dist.so").exists()


def test_control_all_build(tmp_path: Path):
    res = _run(tmp_path, "")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARN" not in res.stdout
    assert (tmp_path / "out" / "fuzzgoat_read_dist.so").exists()


def test_adversarial_every_compile_fails(tmp_path: Path):
    res = _run(tmp_path, ".c")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK" not in res.stdout
    assert "WARN failed: fuzzgoat_read_dist.so (nosan)" in res.stdout
