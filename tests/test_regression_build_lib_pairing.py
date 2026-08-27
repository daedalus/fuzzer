"""Every target compiled with vendored headers must link the vendored library.

`inflateInit2` and friends are macros that bake the *header's* version
string into a runtime check against the *library*. Compiling against
vendored `zlib.h` while linking system `libz` therefore produces a target
that builds cleanly, runs, and silently refuses to decompress anything --
which reads as a fuzzer regression, not a build bug.

This asserts the pairing structurally so the three copies of the vendored-
library detection in tools/build_targets.sh cannot drift apart again
without a test failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "build_targets.sh"

# Target basename -> (include variable, library variable) it is built with.
PAIRED = {
    "png_read": ("PNG_INC", "PNG_LIBS"),
    "zlib_read": ("ZLIB_INC", "ZLIB_LIBS"),
    "gzip_read": ("ZLIB_INC", "GZIP_LIBS"),
    # sqlite3.h is parsed under $SQLITE_DEFINES (carried in $SQLITE_INC) and
    # must link the object compiled with that same define list: a header and
    # a library that disagree about SQLITE_* build options is the same class
    # of silent mismatch as the zlib version check above.
    "sqlite_read": ("SQLITE_INC", "SQLITE_LIBS"),
}


@pytest.fixture(scope="module")
def script() -> str:
    return BUILD_SCRIPT.read_text()


def _build_lines(script: str, target: str) -> list[str]:
    """Every build_target / build_so_target invocation for *target*."""
    return [
        line.strip()
        for line in script.splitlines()
        if re.search(rf"build_(so_)?target\s+\"\$TARGETS/{target}\.c\"", line)
    ]


class TestIncludeLinkPairing:
    def test_targets_are_actually_built(self, script):
        for target in PAIRED:
            assert _build_lines(script, target), f"no build line found for {target}"

    @pytest.mark.parametrize("target", sorted(PAIRED))
    def test_include_flag_implies_matching_library_var(self, target, script):
        """A build line using $X_INC must use the library var paired with it."""
        inc_var, lib_var = PAIRED[target]
        for line in _build_lines(script, target):
            if f"${inc_var}" not in line:
                continue
            assert f"${lib_var}" in line, (
                f"{target} is compiled with ${inc_var} but does not link ${lib_var}: {line}"
            )

    def test_gzip_never_hardcodes_the_system_library(self, script):
        """gzip_read must go through GZIP_LIBS, not a literal -lz."""
        for line in _build_lines(script, "gzip_read"):
            assert "-lz" not in line, f"gzip_read links -lz directly: {line}"


class TestVendoredBranchesSetEveryConsumer:
    """Each `if [ -f "$VENDOR_ZLIB_A" ]` branch must set *both* zlib consumers.

    ZLIB_INC is shared by zlib_read and gzip_read, so a branch that flips
    the include path has to flip both library variables or it desynchronizes
    one of them.
    """

    def _vendored_zlib_branches(self, script: str) -> list[str]:
        # Matches the branch by what it tests for -- a vendored libz.a --
        # not by a specific variable name, so a rename does not silently
        # turn this suite into a no-op.
        blocks, current = [], None
        for line in script.splitlines():
            if re.match(r'\s*if \[ -f "\$\{?\w*[Zz][Ll][Ii][Bb]\w*\}?" \]; then\s*$', line):
                current = []
                continue
            if current is not None:
                if re.match(r"\s*fi\s*$", line):
                    blocks.append("\n".join(current))
                    current = None
                else:
                    current.append(line)
        return blocks

    def test_branches_exist(self, script):
        assert self._vendored_zlib_branches(script)

    def test_every_branch_sets_zlib_and_gzip_libs(self, script):
        for i, block in enumerate(self._vendored_zlib_branches(script)):
            if "ZLIB_INC=" not in block:
                continue
            assert "ZLIB_LIBS=" in block, (
                f"vendored-zlib branch {i} sets ZLIB_INC but not ZLIB_LIBS"
            )
            assert "GZIP_LIBS=" in block, (
                f"vendored-zlib branch {i} sets ZLIB_INC but not GZIP_LIBS"
            )

    def test_zlib_and_gzip_get_the_same_library(self, script):
        """The two must resolve to the same archive, not merely both be set."""
        for i, block in enumerate(self._vendored_zlib_branches(script)):
            if "ZLIB_INC=" not in block:
                continue
            zlib = re.search(r'ZLIB_LIBS="([^"]*)"', block)
            gzip = re.search(r'GZIP_LIBS="([^"]*)"', block)
            assert zlib and gzip
            assert zlib.group(1) == gzip.group(1), (
                f"branch {i}: zlib_read links {zlib.group(1)!r} "
                f"but gzip_read links {gzip.group(1)!r} from the same headers"
            )
