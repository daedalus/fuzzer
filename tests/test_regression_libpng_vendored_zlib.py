"""Regression: vendored libpng must configure against the vendored zlib headers.

libpng's configure generates ``pnglibconf.h`` with the preprocessor, which
reads ``CPPFLAGS``, not ``CFLAGS``. Without ``CPPFLAGS=-I../zlib`` it records
the system zlib's ``ZLIB_VERNUM`` (e.g. 0x1300) and ``pngpriv.h`` then fails
against the vendored 1.3.1 headers (0x1310): ``ZLIB_VERNUM != PNG_ZLIB_VERNUM``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="build_targets.sh not present")

# `cd` into the libpng tree, then everything up to its ./configure call.
_LIBPNG_CONFIGURE = re.compile(
    r'cd "\$(?:VENDOR_LIBPNG_DIR|VENDOR/libpng)".*?\./configure', re.DOTALL
)


def _configure_blocks(text: str) -> list[str]:
    return _LIBPNG_CONFIGURE.findall(text)


def test_regression_libpng_configure_gets_vendored_zlib_cppflags():
    """Every libpng configure call passes the vendored zlib include via CPPFLAGS."""
    blocks = _configure_blocks(SCRIPT.read_text())
    assert blocks, "no libpng configure call found; the pattern is stale"

    missing = [b for b in blocks if not re.search(r'CPPFLAGS="[^"]*-I\.\./zlib', b)]
    assert not missing, f"libpng configure without CPPFLAGS=-I../zlib:\n{missing}"


def test_cflags_only_include_is_rejected():
    """Falsification: -I../zlib in CFLAGS alone (the original bug) must not pass."""
    buggy = (
        '(cd "$VENDOR_LIBPNG_DIR" && \\\n'
        '    CC=clang CFLAGS="-O2 -I../zlib" LDFLAGS="-L../zlib" \\\n'
        "    ./configure --quiet)"
    )
    blocks = _configure_blocks(buggy)
    assert len(blocks) == 1
    assert not re.search(r'CPPFLAGS="[^"]*-I\.\./zlib', blocks[0])
