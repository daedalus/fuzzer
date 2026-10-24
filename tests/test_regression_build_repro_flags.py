"""Path-stable builds in tools/build_targets.sh.

Every compile passes `-g`, so DWARF and `__FILE__` carried the absolute
repo / vendor / build-root paths: the same source built under two roots
gave different bytes. DEBUG_CFLAGS remaps those prefixes to fixed
placeholders. `ls`/`sort` order also followed the caller's locale, so the
script pins LC_ALL=C.

DEBUG_CFLAGS is extracted from the script and evaluated, so the compile
tests exercise the script's own definition rather than a copy.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(not BUILD_SCRIPT.exists(), reason="build_targets.sh not present")

needs_clang = pytest.mark.skipif(shutil.which("clang") is None, reason="clang not available")

SRC = "int f(void) { return __LINE__; }\nconst char *name = __FILE__;\n"

# A compile line: a compiler word followed (anywhere) by an -O level.
_COMPILE_RE = re.compile(r"(\$cc|\$CC|\$cxx|\$target_cc|\bclang)\b.*\s-O[0-3s$]|CFLAGS=\"-O")


@pytest.fixture(scope="module")
def script_text() -> str:
    return BUILD_SCRIPT.read_text()


def _debug_cflags(text: str, repo: Path, vendor: Path, targets: Path) -> list[str]:
    """Evaluate the script's DEBUG_CFLAGS line under the given roots."""
    line = next(ln for ln in text.splitlines() if ln.startswith("DEBUG_CFLAGS="))
    sh = f'REPO_ROOT="{repo}"; VENDOR="{vendor}"; TARGETS="{targets}"\n{line}\necho "$DEBUG_CFLAGS"'
    out = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout
    return out.split()


def _compile(src: Path, cwd: Path, flags: list[str]) -> bytes:
    """Compile src with an absolute path, as the script does for vendor trees."""
    obj = cwd / "out.o"
    subprocess.run(["clang", "-O2", *flags, "-c", str(src), "-o", str(obj)], cwd=cwd, check=True)
    data = obj.read_bytes()
    obj.unlink()
    return data


def _layout(root: Path, nested: bool) -> tuple[Path, Path, Path, Path]:
    """Create repo/vendor/build roots under root; return them and the source."""
    repo = root / "repo"
    vendor = repo / "vendor" if nested else root / "vendoring"
    targets = root / "builds"
    lib = vendor / "lib"
    lib.mkdir(parents=True)
    repo.mkdir(exist_ok=True)
    targets.mkdir()
    src = lib / "a.c"
    src.write_text(SRC)
    return repo, vendor, targets, src


def _build(tmp_path: Path, name: str, text: str, mapped: bool, nested: bool = False) -> bytes:
    """Build the same source under a fresh root, with or without DEBUG_CFLAGS."""
    root = tmp_path / name
    repo, vendor, targets, src = _layout(root, nested)
    flags = _debug_cflags(text, repo, vendor, targets) if mapped else ["-g"]
    data = _compile(src, repo, flags)
    assert str(root).encode() not in data or not mapped, f"{root} leaked into the object"
    return data


def test_regression_lc_all_pinned_before_listing(script_text: str):
    """ls/sort order follows LC_COLLATE; pin it before the first listing."""
    m = re.search(r"^export LC_ALL=C$", script_text, re.M)
    assert m, "build_targets.sh must export LC_ALL=C"
    first_ls = re.search(r"\$\(ls |\| sort", script_text)
    assert first_ls and m.start() < first_ls.start(), "LC_ALL=C exported after a listing"


def test_regression_every_debug_compile_uses_debug_cflags(script_text: str):
    """A bare -g re-introduces absolute paths into that one target."""
    offenders = [
        (n, ln.strip())
        for n, ln in enumerate(script_text.splitlines(), 1)
        if not ln.lstrip().startswith("#")
        and _COMPILE_RE.search(ln)
        and re.search(r'(^|[\s"])-g([\s"]|$)', ln)
    ]
    assert not offenders, f"bare -g instead of $DEBUG_CFLAGS: {offenders}"


def test_regression_ffmpeg_gets_debug_cflags(script_text: str):
    """FFmpeg adds -g itself; the map rides in COV_FLAGS (--extra-cflags),
    set before cfg_stamp so a flag change reconfigures."""
    hook = script_text.find('COV_FLAGS="$COV_FLAGS $DEBUG_CFLAGS"')
    assert hook != -1, "FFmpeg COV_FLAGS lacks $DEBUG_CFLAGS"
    assert hook < script_text.find("cfg_stamp=$(printf")
    assert '--extra-cflags="$COV_FLAGS"' in script_text


@needs_clang
def test_control_unmapped_roots_differ(tmp_path: Path, script_text: str):
    """Control: without the map, two roots must differ, else the oracle is blind."""
    a = _build(tmp_path, "a", script_text, mapped=False)
    b = _build(tmp_path, "bb", script_text, mapped=False)
    assert a != b


@needs_clang
def test_control_mapped_same_root_matches(tmp_path: Path, script_text: str):
    """Control: one root built twice must match (compiler is deterministic)."""
    root = tmp_path / "same"
    repo, vendor, targets, src = _layout(root, nested=False)
    flags = _debug_cflags(script_text, repo, vendor, targets)
    assert _compile(src, repo, flags) == _compile(src, repo, flags)


@needs_clang
def test_regression_mapped_roots_build_identical(tmp_path: Path, script_text: str):
    """Same source under two roots gives identical bytes once mapped."""
    a = _build(tmp_path, "a", script_text, mapped=True)
    b = _build(tmp_path, "bb", script_text, mapped=True)
    assert a == b


@needs_clang
def test_adversarial_nested_vendor_maps_to_vendor(tmp_path: Path, script_text: str):
    """--in-tree-vendor nests vendor/ inside the repo; the more specific
    vendor map must win, else sources land under /fuzz/src/vendor."""
    a = _build(tmp_path, "a", script_text, mapped=True, nested=True)
    b = _build(tmp_path, "bb", script_text, mapped=True, nested=True)
    assert a == b
    assert b"/fuzz/vendor/lib/a.c" in a
    assert b"/fuzz/src/vendor" not in a
