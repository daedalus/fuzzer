#!/bin/bash
# Fetch a nightly rustc toolchain, strictly opt-in, for
# tools/build_rust_target.sh's optional real-edge-coverage mode.
#
# READ THIS BEFORE RUNNING IT.
#
# The official channel for nightly Rust (rustup.rs / static.rust-lang.org)
# is not reachable from every environment this project runs in (blocked
# by network policy in the sandbox this was written in, for instance).
# There is no official nightly rustc distributed any other way -- rust-
# lang/rust's own GitHub releases carry only auto-generated source
# archives, never compiled binaries.
#
# This script instead fetches a prebuilt nightly toolchain published by
# a16z/rust (https://github.com/a16z/rust) -- a fork maintained for their
# unrelated Jolt zkVM project, whose releases happen to include a full
# x86_64-unknown-linux-gnu host toolchain built from close-to-upstream
# rustc nightly sources. It is NOT an official Rust distribution:
#   - no independently-published checksum exists to verify against --
#     the SHA256 pinned below is one this project computed itself, once,
#     from the specific release tag named below, over a single TLS
#     connection to GitHub. It catches a corrupted or substituted download
#     on a LATER run; it does not independently prove the original
#     upload was untampered.
#   - it is built by a third party for their own purposes, not audited
#     by this project or by Rust's own release process.
# Treat it accordingly: fine for building a disposable local fuzz target
# in a sandbox, not something to wire into a trusted supply chain without
# your own review of that repo.
#
# Usage:
#   tools/fetch_rust_nightly.sh [install-dir]     # default: ~/.cache/rust-nightly-jolt
#
# tools/build_rust_target.sh picks this up automatically via
# RUSTC_NIGHTLY, or auto-detects the default install dir if that env var
# isn't set. Nothing else in this project requires this script to have
# been run.

set -euo pipefail

# Pinned to one specific release tag, not "latest" -- a moving target
# defeats the point of pinning a hash to verify against on later runs.
RELEASE_TAG="nightly-04c2cbcc03a19bdb4bd60ed8b4688068c6a2befa"
ASSET="rust-toolchain-nightly-x86_64-unknown-linux-gnu.tar.gz"
URL="https://github.com/a16z/rust/releases/download/${RELEASE_TAG}/${ASSET}"
# Computed by this project from the one download performed while writing
# this script (see the provenance note above) -- not sourced from an
# independent, official checksum file, because a16z's releases don't
# publish one.
EXPECTED_SHA256="a5bdf4c47e82c67a3440109ff71ab0222736e7a02f4424513b908b8d38d3bd0a"

INSTALL_DIR="${1:-$HOME/.cache/rust-nightly-jolt}"

warn() { printf '\033[0;33mWARN\033[0m: %s\n' "$*" >&2; }
ok()   { printf '\033[0;32mOK\033[0m: %s\n' "$1"; }
fail() { printf '\033[0;31mFAIL\033[0m: %s\n' "$1" >&2; exit 1; }

if [ -x "$INSTALL_DIR/bin/rustc" ]; then
    ok "already installed at $INSTALL_DIR — remove it first to re-fetch"
    exit 0
fi

warn "fetching an UNOFFICIAL third-party nightly rustc build (a16z/rust)." \
     "See this script's header comment for what that does and doesn't mean" \
     "for trust. Ctrl-C now if that's not acceptable for your use case."

mkdir -p "$INSTALL_DIR"
TMP_TAR="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$TMP_TAR"' EXIT

echo "Downloading $URL ..."
curl -sL -o "$TMP_TAR" "$URL"

ACTUAL_SHA256="$(sha256sum "$TMP_TAR" | cut -d' ' -f1)"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    fail "sha256 mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256 — refusing to extract. Either the release asset changed or the download was corrupted/tampered; do not proceed without investigating which."
fi
ok "sha256 matches pinned value"

TMP_EXTRACT="$(mktemp -d)"
tar xzf "$TMP_TAR" -C "$TMP_EXTRACT"
STAGE2="$TMP_EXTRACT/rust/build/host/stage2"
[ -x "$STAGE2/bin/rustc" ] || fail "expected $STAGE2/bin/rustc after extraction, not found — archive layout changed?"

mkdir -p "$INSTALL_DIR/bin"
cp -a "$STAGE2/bin/." "$INSTALL_DIR/bin/"
cp -a "$STAGE2/lib" "$INSTALL_DIR/lib"
rm -rf "$TMP_EXTRACT"

ok "installed to $INSTALL_DIR"
"$INSTALL_DIR/bin/rustc" --version --verbose
echo
echo "Note: this toolchain does NOT bundle the rust-src component, so"
echo "-Z build-std (needed for ASAN coverage of the Rust side itself,"
echo "not just edge coverage) is not available with it. See"
echo "docs/handover/handover_rust_target_2026-09-15.md for what that"
echo "does and doesn't affect."
echo
echo "Set RUSTC_NIGHTLY=$INSTALL_DIR/bin/rustc (or just run"
echo "tools/build_rust_target.sh, which auto-detects this path) to use it."
