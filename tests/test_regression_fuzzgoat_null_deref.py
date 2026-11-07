"""compile_fuzzgoat_object keeps fuzzgoat's planted NULL dereference.

clang -O2 treats ``*NULL`` as UB and deletes the load: the ``"A"`` payload
(oneByteString) exited 0, so calibrations had 3 findable planted bugs,
not 4. The function is extracted and sourced with the real compiler.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"
VENDOR = Path(os.environ.get("FUZZ_VENDOR_ROOT", Path.home() / "fuzzing" / "vendoring"))
FUZZGOAT = VENDOR / "fuzzgoat"
SIGSEGV = 11

pytestmark = pytest.mark.skipif(
    not (FUZZGOAT / "fuzzgoat.c").exists() or not shutil.which("clang"),
    reason="needs vendored fuzzgoat (tools/vendor_fuzzgoat.sh) and clang",
)


def _extract() -> str:
    text = BUILD_SCRIPT.read_text()
    start = text.index("compile_fuzzgoat_object() {")
    end = text.index("\n}\n", start) + 3
    flags = next(ln for ln in text.splitlines() if ln.startswith("FUZZGOAT_CFLAGS="))
    return flags + "\n" + text[start:end]


@pytest.fixture
def fuzzgoat_bin(tmp_path):
    """Upstream main.c linked against the object build_targets.sh compiles."""
    suffix = f"_pytest_{tmp_path.name}"
    obj = Path(f"/tmp/fuzzgoat{suffix}.o")  # path fixed by the function
    script = (
        f'VENDOR="{VENDOR}"; DEFAULT_CC=clang; DEBUG_CFLAGS=""; BUILD_LOG="{tmp_path}/log"\n'
        'cov_flag_for_cc() { echo ""; }\n'
        + _extract()
        + f'\ncompile_fuzzgoat_object "{suffix}" "" clang ""\n'
    )
    try:
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and obj.exists(), (tmp_path / "log").read_text()[-400:]

        exe = tmp_path / "fuzzgoat"
        link = [
            "clang",
            "-O2",
            f"-I{FUZZGOAT}",
            "-o",
            str(exe),
            str(FUZZGOAT / "main.c"),
            str(obj),
            "-lm",
        ]
        subprocess.run(link, check=True, capture_output=True, timeout=120)
        yield exe
    finally:
        obj.unlink(missing_ok=True)


def _rc(exe: Path, tmp_path: Path, payload: bytes) -> int:
    inp = tmp_path / "in"
    inp.write_bytes(payload)
    return subprocess.run([str(exe), str(inp)], capture_output=True, timeout=30).returncode


def test_regression_one_byte_string_segfaults(fuzzgoat_bin, tmp_path):
    assert _rc(fuzzgoat_bin, tmp_path, b'"A"') == -SIGSEGV


def test_valid_json_still_clean(fuzzgoat_bin, tmp_path):
    """Adversarial: the flag must not turn bug-free inputs into crashes."""
    assert _rc(fuzzgoat_bin, tmp_path, b'["AB", 1, true]') == 0
