"""GStreamer must be built by build_targets.sh, with the same defaults as FFmpeg."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "tools" / "build_targets.sh"
TARGET = ROOT / "targets" / "gstreamer_read.c"
VENDOR = ROOT / "tools" / "vendor_gstreamer.sh"


def src() -> str:
    return BUILD.read_text()


def gstreamer_build_fn() -> str:
    text = src()
    start = text.index("build_vendored_gstreamer() {")
    return text[start : text.index("\n}\n", start)]


def test_standalone_build_script_removed():
    assert not (ROOT / "tools" / "build_gstreamer.sh").exists()


def test_gstreamer_is_part_of_the_main_build():
    text = src()
    assert "tools/vendor_gstreamer.sh" in text
    assert "build_vendored_gstreamer()" in text
    assert "build_gstreamer_versions()" in text
    assert 'build_gstreamer_versions "_asan" "$ASAN_CFLAGS $GSTREAMER_COV_FLAGS"' in text
    assert 'build_gstreamer_versions "" "$GSTREAMER_COV_FLAGS"' in text


def test_gstreamer_uses_ccache_and_coverage_by_default():
    text = src()
    assert (
        'GSTREAMER_COV_FLAGS="-fsanitize-coverage=trace-pc-guard -fsanitize-coverage=trace-cmp"'
        in text
    )
    assert 'cc="ccache clang"' in text
    assert 'env CC="$cc" CXX="$cxx" meson setup' in text
    assert "-Dc_args=$COV_FLAGS" in text
    assert "-Dgstreamer:gst_parse=false" in text
    fn = gstreamer_build_fn()
    assert "sancov_stub.c" in fn
    assert "-lsancov_stub" in fn
    assert ".compiler" in fn
    assert "need_fresh" in fn
    assert "--no-ccache" in text


def test_gstreamer_targets_are_verified():
    text = src()
    assert text.count('"$TARGETS"/gstreamer/*.so') >= 2
    assert '"$TARGETS"/gstreamer/*' in text


def test_references_point_at_main_build():
    assert "tools/build_targets.sh" in VENDOR.read_text()
    assert "tools/build_targets.sh" in TARGET.read_text()
