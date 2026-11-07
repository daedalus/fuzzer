"""verify_afl in tools/build_targets.sh checks every built target.

It walked a fixed list of legacy names (png_read, fuzzgoat_read, ...) and
skipped missing files silently; the build writes *_noasan / *_asan.so /
*_ubsan.so, so 8 of ~60 artifacts were checked. The function is extracted
and sourced over a fake build root.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(
    not shutil.which("clang") or not shutil.which("nm"), reason="needs clang + nm"
)


def _extract() -> str:
    text = BUILD_SCRIPT.read_text()
    start = text.index("verify_afl() {")
    return text[start : text.index("\n}\n", start) + 3]


def _elf(path: Path, src: str) -> None:
    c = path.with_suffix(".c.tmp")
    c.write_text(src)
    subprocess.run(["clang", "-x", "c", "-o", str(path), str(c)], check=True, capture_output=True)
    c.unlink()


def _verify(root: Path) -> list[str]:
    script = 'ok() { echo "OK: $1"; }\nwarn() { echo "WARN: $1"; }\n'
    script += f'TARGETS="{root}"\n' + _extract() + "\nverify_afl\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_regression_verify_covers_current_names(tmp_path):
    _elf(tmp_path / "fuzzgoat_read_noasan", "int __afl_area_ptr; int main(void){return 0;}")
    _elf(tmp_path / "png_read_ubsan", "int main(void){return 0;}")
    (tmp_path / "build.log").write_text("not a target\n")
    (tmp_path / "png_read_noasan.profile_cache").write_bytes(b"\x80\x04N.")

    out = _verify(tmp_path)
    assert "OK: 1 targets with AFL symbols" in out
    assert "WARN: png_read_ubsan: no AFL symbols" in out
    assert not any("build.log" in ln or "profile_cache" in ln for ln in out)


def test_verify_empty_root(tmp_path):
    """Adversarial: nothing built is zero checked, not an error."""
    assert "OK: 0 targets with AFL symbols" in _verify(tmp_path)
