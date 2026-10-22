"""Regression: the linked FFmpeg build dropped reachable code.

build_targets.sh configured with --disable-parsers --disable-bsfs, left
zlib/bzlib/lzma to host autodetect, and kept asm on when nasm existed --
each one removes instrumented code from the edge map. The shared flag set
lives in tools/lib/ffmpeg_config.sh.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
LIB = ROOT / "tools" / "lib" / "ffmpeg_config.sh"
BUILD = ROOT / "tools" / "build_targets.sh"
VENDOR = ROOT / "tools" / "vendor_ffmpeg.sh"

DEP_FLAGS = ("--enable-zlib", "--enable-bzlib", "--enable-lzma", "--enable-libxml2")


def _flags(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Source the lib and print ffmpeg_feature_flags."""
    return subprocess.run(
        ["bash", "-c", f'. "{LIB}" && ffmpeg_feature_flags'],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=30,
    )


def _fake_tool(tmp_path: Path, name: str) -> Path:
    """A tool that always fails: every probe through it reports 'absent'."""
    tool = tmp_path / name
    tool.write_text("#!/bin/sh\nexit 1\n")
    tool.chmod(0o755)
    return tool


def test_regression_parsers_bsfs_not_disabled():
    """The sancov configure line must keep parsers and bsfs."""
    text = BUILD.read_text()

    assert "--disable-parsers" not in text
    assert "--disable-bsfs" not in text


def test_regression_shared_flags_used_and_stamped():
    """Both scripts source the lib; the stamp covers the feature flags."""
    build = BUILD.read_text()

    assert "tools/lib/ffmpeg_config.sh" in build
    assert "lib/ffmpeg_config.sh" in VENDOR.read_text()
    assert '"$FFMPEG_FEATURE_FLAGS"' in build


def test_asm_forced_off():
    """Asm is uninstrumented: always --disable-asm, nasm or not."""
    r = subprocess.run(
        ["bash", "-c", f'. "{LIB}" && echo "$FFMPEG_ASM_FLAG"'],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert r.stdout.strip() == "--disable-asm"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang required")
def test_present_deps_enabled(tmp_path):
    """Falsification: every dep the host can link is enabled explicitly."""
    have_xml = subprocess.run(["pkg-config", "--exists", "libxml-2.0"]).returncode == 0
    r = _flags({})
    flags = r.stdout.split()

    assert r.returncode == 0
    assert "--disable-autodetect" in flags
    for dep, lib in (("zlib", "-lz"), ("bzlib", "-lbz2"), ("lzma", "-llzma")):
        links = (
            subprocess.run(
                ["clang", "-x", "c", "-", lib, "-o", os.devnull],
                input="int main(void){return 0;}",
                text=True,
                capture_output=True,
            ).returncode
            == 0
        )
        assert (f"--enable-{dep}" in flags) == links
    assert ("--enable-libxml2" in flags) == have_xml


def test_absent_deps_dropped(tmp_path):
    """Adversarial: unlinkable deps are dropped with a warning, never enabled.

    configure aborts on an explicit --enable-X it cannot satisfy, so an
    absent dep must cost that feature, not the whole build.
    """
    cc = _fake_tool(tmp_path, "cc")
    _fake_tool(tmp_path, "pkg-config")
    r = _flags({"FFMPEG_PROBE_CC": str(cc), "PATH": f"{tmp_path}:{os.environ['PATH']}"})
    flags = r.stdout.split()

    assert r.returncode == 0
    assert flags == ["--disable-autodetect"]
    for dep in DEP_FLAGS:
        assert dep.removeprefix("--enable-") in r.stderr


# ── Versioned trees (ffmpeg-<ver>, vendor_ffmpeg.sh --top=N) ───────────

CONFIGURE_STUB = """#!/bin/sh
echo "$@" >> "$CONF_LOG"
mkdir -p ffbuild && : > ffbuild/config.mak
printf 'all:\\n\\tmkdir -p libavformat libavcodec libavutil libswresample\\n' > Makefile
printf '\\tfor l in libavformat libavcodec libavutil libswresample; do : > $$l/$$l.a; done\\n' >> Makefile
"""


def _sancov(
    tmp_path: Path, features: str, configure: str = CONFIGURE_STUB, pre: str = ""
) -> list[str]:
    """Run build_vendored_ffmpeg_sancov on a stub ffmpeg-9.0.2 tree; return configure calls.

    *pre* is shell run just before the call (e.g. sourcing the shared lib).
    """
    vendor = tmp_path / "vendor" / "ffmpeg-9.0.2"
    vendor.mkdir(parents=True, exist_ok=True)
    conf = vendor / "configure"
    conf.write_text(configure)
    conf.chmod(0o755)
    log = tmp_path / "configure.log"

    # cow_copy_tree stages the tree; without it every stub build failed at configure.
    fns = [
        re.search(rf"^{name}\(\) \{{.*?^\}}", BUILD.read_text(), re.M | re.S)
        for name in ("cow_copy_tree", "build_vendored_ffmpeg_sancov")
    ]
    assert all(fns)
    script = (
        "warn() { :; }; ok() { :; }; warn_failed() { echo FAILED; }; log_section() { :; }\n"
        + "\n".join(f.group(0) for f in fns)
        + "\n"
        f'VENDOR="{tmp_path}/vendor"; FUZZ_BUILD_ROOT="{tmp_path}/build"; REPO_ROOT="{ROOT}"\n'
        f'BUILD_LOG="{tmp_path}/build.log"; WITH_FFMPEG_SANCOV=1; FORCE_REBUILD=0; USE_CCACHE=0\n'
        f'ASAN_CFLAGS="-fsanitize=address"; FFMPEG_ASM_FLAG=--disable-asm\n'
        f'FFMPEG_FEATURE_FLAGS="{features}"\n'
        f"{pre}\n"
        'build_vendored_ffmpeg_sancov "" ffmpeg-9.0.2\n'
    )
    r = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "CONF_LOG": str(log)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAILED" not in r.stdout
    return log.read_text().splitlines() if log.exists() else []


needs_build_tools = pytest.mark.skipif(
    not (shutil.which("clang") and shutil.which("rsync") and shutil.which("make")),
    reason="clang, rsync and make required",
)


@needs_build_tools
def test_versioned_tree_gets_shared_flags(tmp_path):
    """Falsification: a versioned tree's configure sees asm off and the feature flags."""
    calls = _sancov(tmp_path, "--disable-autodetect --enable-zlib")
    args = calls[0].split()

    assert len(calls) == 1
    assert "--disable-asm" in args
    assert "--disable-autodetect" in args
    assert "--enable-zlib" in args
    assert "--disable-parsers" not in args
    assert "--disable-bsfs" not in args


@needs_build_tools
def test_versioned_tree_reconfigures_on_flag_change(tmp_path):
    """Adversarial: same flags reuse the tree; a new dep forces a reconfigure."""
    _sancov(tmp_path, "--disable-autodetect")
    same = _sancov(tmp_path, "--disable-autodetect")
    changed = _sancov(tmp_path, "--disable-autodetect --enable-libxml2")

    assert len(same) == 1
    assert len(changed) == 2
    assert "--enable-libxml2" in changed[-1].split()


# ── Frame pointers (FFmpeg configure:6537 vs the shim's caller-ctx walk) ──

# FFmpeg's own flag handling, reduced to the two lines that decide it:
#   configure    appends -fomit-frame-pointer after --extra-cflags unless
#                CFLAGS already carry -fsanitize= (coverage-only trees don't)
#   common.mak   CFLAGS += $(ECFLAGS), the only thing compiled after that
# Every compile logs its effective CFLAGS to $FLAGS_LOG.
CONFIGURE_FP_STUB = """#!/bin/sh
echo "$@" >> "$CONF_LOG"
for a; do case "$a" in --extra-cflags=*) X="${a#--extra-cflags=}";; esac; done
FP=-fomit-frame-pointer
case "$X" in *-fsanitize=*) FP=;; esac
mkdir -p ffbuild && echo "CFLAGS= $X $FP" > ffbuild/config.mak
L="libavformat/libavformat.a libavcodec/libavcodec.a libavutil/libavutil.a libswresample/libswresample.a"
printf 'include ffbuild/config.mak\\nCFLAGS += $(ECFLAGS)\\nall: %s\\n' "$L" > Makefile
printf '%%.a:\\n\\tmkdir -p $(dir $@) && : > $@\\n\\techo "$(CFLAGS)" >> "$(FLAGS_LOG)"\\n' >> Makefile
printf 'clean:\\n\\trm -f %s\\n' "$L" >> Makefile
"""

FP_FLAGS = ("-fomit-frame-pointer", "-fno-omit-frame-pointer")


def _last_fp(cflags: str) -> str:
    """The frame-pointer flag clang obeys: the last one on the line."""
    return [f for f in cflags.split() if f in FP_FLAGS][-1]


def _sancov_fp(tmp_path: Path, pre: str) -> tuple[list[str], list[str]]:
    """Stub-FFmpeg sancov build; return (configure calls, effective CFLAGS per compile)."""
    log = tmp_path / "cflags.log"
    os.environ["FLAGS_LOG"] = str(log)
    try:
        calls = _sancov(tmp_path, "--disable-autodetect", CONFIGURE_FP_STUB, pre)
    finally:
        del os.environ["FLAGS_LOG"]
    return calls, log.read_text().splitlines() if log.exists() else []


@needs_build_tools
def test_regression_ffmpeg_frame_pointer_kept(tmp_path):
    """Falsification: every coverage-only compile ends on -fno-omit-frame-pointer.

    configure appended -fomit-frame-pointer last, so 0 of 2,221 libavformat
    functions had a frame and the shim's caller-ctx walk bailed on 96% of edges.
    """
    _, cflags = _sancov_fp(tmp_path, f'. "{LIB}"')

    assert cflags
    assert all(_last_fp(c) == "-fno-omit-frame-pointer" for c in cflags)


@needs_build_tools
def test_regression_ffmpeg_fp_stale_tree_rebuilds(tmp_path):
    """Adversarial: a tree stamped before the fix must reconfigure, not stay 'up to date'."""
    calls, _ = _sancov_fp(tmp_path, f'. "{LIB}"; FFMPEG_MAKE_FLAGS=""')
    calls, cflags = _sancov_fp(tmp_path, f'. "{LIB}"')

    assert len(calls) == 2
    assert _last_fp(cflags[-1]) == "-fno-omit-frame-pointer"


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang required")
@pytest.mark.skipif(shutil.which("objdump") is None, reason="objdump required")
def test_ffmpeg_make_flags_give_clang_a_frame(tmp_path):
    """Adversarial: clang, fed configure's order plus the lib's make flags, emits a frame."""
    r = subprocess.run(
        ["bash", "-c", f'. "{LIB}" && echo "${{FFMPEG_MAKE_FLAGS#ECFLAGS=}}"'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    src = tmp_path / "f.c"
    src.write_text("int g(int);\nint f(int x) { return g(x) + 1; }\n")
    obj = tmp_path / "f.o"
    cflags = ["-O3", "-fomit-frame-pointer", *r.stdout.split()]
    subprocess.run(["clang", *cflags, "-c", str(src), "-o", str(obj)], check=True, timeout=60)
    asm = subprocess.run(["objdump", "-d", str(obj)], capture_output=True, text=True).stdout

    assert "push   %rbp" in asm


def test_regression_vendor_script_uses_make_flags():
    """vendor_ffmpeg.sh builds the same archives; its make must carry the flags too."""
    text = VENDOR.read_text()
    make = re.search(r'make -j"\$NPROC"[^\n]*', text)

    assert make
    assert "$FFMPEG_MAKE_FLAGS" in make.group(0)
