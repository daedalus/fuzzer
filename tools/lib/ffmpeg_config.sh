# Shared FFmpeg configure flags, sourced by tools/vendor_ffmpeg.sh and
# tools/build_targets.sh. Chosen for edge coverage.
#
#   --disable-asm         sancov cannot instrument nasm or inline asm. With
#                         asm on, SIMD replaces instrumented C DSP code, and
#                         inline asm (cabac, mathops) hides loads from ASAN.
#   --disable-autodetect  coverage must not depend on host packages;
#                         the deps below are enabled explicitly instead.
#   zlib/bzlib/lzma       png/apng/exr/flashsv/zmbv/tscc decoders, tiff
#                         deflate, mov cmov, mkv compressed tracks, cws swf.
#   libxml2               dash and imf demuxers.

FFMPEG_ASM_FLAG="--disable-asm"

# feature:header:lib for deps probed by a link test, as configure does.
_FFMPEG_LINK_DEPS="zlib:zlib.h:-lz bzlib:bzlib.h:-lbz2 lzma:lzma.h:-llzma"

# True when <header> compiles and <lib> links.
_ffmpeg_links() {
    # Unquoted: the probe compiler may be "ccache clang".
    # shellcheck disable=SC2086
    printf '#include <%s>\nint main(void){return 0;}\n' "$1" \
        | ${FFMPEG_PROBE_CC:-clang} -x c - "$2" -o /dev/null 2>/dev/null
}

# Print configure flags for every available dep. An absent one is dropped
# with a warning on stderr: configure aborts on an --enable-X it cannot
# satisfy, so a missing dev package costs one feature, not the build.
ffmpeg_feature_flags() {
    local flags="--disable-autodetect" dep feature header lib

    for dep in $_FFMPEG_LINK_DEPS; do
        IFS=: read -r feature header lib <<<"$dep"
        if _ffmpeg_links "$header" "$lib"; then
            flags="$flags --enable-$feature"
            continue
        fi
        echo "note: $feature not found ($header, $lib) -- FFmpeg built without it" >&2
    done

    # configure finds libxml2 through pkg-config.
    if pkg-config --exists libxml-2.0 2>/dev/null; then
        flags="$flags --enable-libxml2"
    else
        echo "note: libxml2 not found (pkg-config libxml-2.0) -- no dash/imf demuxers" >&2
    fi

    echo "$flags"
}
