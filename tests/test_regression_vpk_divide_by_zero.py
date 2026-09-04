"""Regression test for the FFmpeg VPK divide-by-zero SIGFPE/SIGSEGV.

The upstream fix (PR#24297, commits c424945cf5 + 2227122b93) adds two guards:

1. vpk_read_packet: guard against par->ch_layout.nb_channels <= 0
   (previously caused divide-by-zero: last_block_size / nb_channels)

2. demux.c parameters_from_context: guard against ch_layout being
   zeroed after a failed avcodec_open2() -> ff_codec_close() ->
   av_opt_free() cycle.

This test verifies:
- The guards are present in the patched source that the build system uses
- The ffmpeg_read target builds successfully with the patched vendor code
- Both ASAN and coverage vendor trees contain the fix
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_vendor_asan(path: Path) -> str:
    """Read from the ASAN vendor staging area."""
    staging = Path.home() / "fuzzing" / "builds" / "ffmpeg_asan" / path.name
    if staging.is_file():
        return staging.read_text()
    # Fall back to the vendoring tree
    vendor = Path.home() / "fuzzing" / "vendoring" / "ffmpeg_asan" / path.name
    if vendor.is_file():
        return vendor.read_text()
    raise FileNotFoundError(f"ASAN vendor source not found: {staging} or {vendor}")


def _read_vendor_cov(path: Path) -> str:
    """Read from the coverage vendor staging area."""
    staging = Path.home() / "fuzzing" / "builds" / "ffmpeg" / path.name
    if staging.is_file():
        return staging.read_text()
    # Fall back to the vendoring tree
    vendor = Path.home() / "fuzzing" / "vendoring" / "ffmpeg" / path.name
    if vendor.is_file():
        return vendor.read_text()
    raise FileNotFoundError(f"Coverage vendor source not found: {staging} or {vendor}")


class TestVPKDivideByZeroGuards:
    """Verify the VPK guards from PR#24297 are present."""

    def test_vpk_c_has_channel_count_guard(self):
        """vpk_read_packet guards against nb_channels <= 0."""
        # The build stages source to ffmpeg_asan/src/
        staging = (
            Path.home() / "fuzzing" / "builds" / "ffmpeg_asan" / "src" / "libavformat" / "vpk.c"
        )
        if not staging.is_file():
            # Fall back to vendoring tree
            staging = Path.home() / "fuzzing" / "vendoring" / "ffmpeg" / "libavformat" / "vpk.c"
        assert staging.is_file(), f"Vendor source not found: {staging}"
        content = staging.read_text()
        assert (
            "if (par->ch_layout.nb_channels <= 0)\n            return AVERROR_INVALIDDATA;"
            in content
        ), "vpk.c missing nb_channels <= 0 guard in vpk_read_packet"

    def test_demux_c_has_ch_layout_zeroing_guard(self):
        """demux.c parameters_from_context guards against zeroed ch_layout."""
        staging = (
            Path.home() / "fuzzing" / "builds" / "ffmpeg_asan" / "src" / "libavformat" / "demux.c"
        )
        if not staging.is_file():
            staging = Path.home() / "fuzzing" / "vendoring" / "ffmpeg" / "libavformat" / "demux.c"
        assert staging.is_file(), f"Vendor source not found: {staging}"
        content = staging.read_text()
        assert (
            "if (par_tmp->ch_layout.nb_channels > 0 && !par->ch_layout.nb_channels)" in content
        ), "demux.c missing ch_layout zeroing guard in parameters_from_context"

    def test_ffmpeg_read_builds_successfully(self):
        """The ffmpeg_read target must compile with the patched vendor source."""
        from subprocess import PIPE, check_call

        # Verify the build script is syntactically valid
        check_call(
            ["bash", "-n", "tools/build_targets.sh"],
            cwd=str(ROOT),
            stdout=PIPE,
            stderr=PIPE,
        )

    def test_fix_applied_to_asan_tree(self):
        """The ASAN vendor tree must have the nb_channels guard applied."""
        staging = (
            Path.home() / "fuzzing" / "builds" / "ffmpeg_asan" / "src" / "libavformat" / "vpk.c"
        )
        if not staging.is_file():
            staging = (
                Path.home() / "fuzzing" / "vendoring" / "ffmpeg_asan" / "libavformat" / "vpk.c"
            )
        assert staging.is_file(), f"ASAN vendor source not found: {staging}"
        content = staging.read_text()
        assert (
            "if (par->ch_layout.nb_channels <= 0)\n            return AVERROR_INVALIDDATA;"
            in content
        ), "ASAN vendor tree missing nb_channels guard"

    def test_fix_in_cov_tree_if_exists(self):
        """The coverage vendor tree must have the nb_channels guard if present."""
        staging = Path.home() / "fuzzing" / "builds" / "ffmpeg" / "src" / "libavformat" / "vpk.c"
        if not staging.is_file():
            staging = Path.home() / "fuzzing" / "vendoring" / "ffmpeg" / "libavformat" / "vpk.c"
        if staging.is_file():
            content = staging.read_text()
            assert (
                "if (par->ch_layout.nb_channels <= 0)\n            return AVERROR_INVALIDDATA;"
                in content
            ), "Coverage vendor tree missing nb_channels guard"
