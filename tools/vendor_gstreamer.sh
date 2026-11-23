#!/bin/bash
# Vendor the GStreamer monorepo (core + gst-plugins-base/good) for fuzz targets.
#
# Usage:
#   tools/vendor_gstreamer.sh                    # default version
#   tools/vendor_gstreamer.sh --version=1.28.9   # specific release tag
#
# Source: github.com/GStreamer/gstreamer (mirror; gitlab.freedesktop.org is
# not reachable from the sandbox).  Shallow clone of the release tag into
# $FUZZ_VENDOR_ROOT/gstreamer-<version>.
# Build it with: tools/build_gstreamer.sh

set -e
: "${FUZZ_VENDOR_ROOT:=$HOME/fuzzing/vendoring}"
GST_VERSION="${GST_VERSION:-1.28.9}"
for arg in "$@"; do
    case "$arg" in --version=*) GST_VERSION="${arg#*=}" ;; esac
done
DIR="$FUZZ_VENDOR_ROOT/gstreamer-$GST_VERSION"
mkdir -p "$FUZZ_VENDOR_ROOT"
if [ -f "$DIR/meson.build" ]; then
    echo "GStreamer $GST_VERSION already vendored at $DIR"
else
    rm -rf "$DIR"
    git clone --depth 1 --branch "$GST_VERSION" \
        https://github.com/GStreamer/gstreamer.git "$DIR"
fi
for f in meson.build subprojects/gstreamer/meson.build \
         subprojects/gst-plugins-base/meson.build \
         subprojects/gst-plugins-good/meson.build; do
    [ -f "$DIR/$f" ] || { echo "ERROR: missing $DIR/$f"; exit 1; }
done
echo "=== GStreamer $GST_VERSION vendored: $DIR ==="
