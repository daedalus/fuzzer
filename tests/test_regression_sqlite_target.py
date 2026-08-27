"""The SQLite target must stay byte-compatible with the SQLite sniffer.

``targets/sqlite_read.c`` reads the fuzz input as a database image when it
carries the SQLite magic, and as SQL text otherwise. That dispatch has to
use the *same* conditions as the ``sqlite_chunk_mutate`` availability
predicate in ``core/operator_registry.py``, because the predicate is what
decides whether the structure-aware mutator fires on a corpus file.

The failure this guards against is silent and total: give the wrapper a
mode-selector prefix byte the way ``lz4_read.c`` has one, and the magic
moves to offset 1, the predicate never matches, every database in the
corpus gets mutated as an unstructured byte string, and
``dictionaries/sqlite.dict`` tokens land at offsets that mean nothing. The
campaign still runs, still reports edges, and quietly does flat-byte
fuzzing on a structured format.

The rest of the file asserts the build wiring, which follows the lz4 and
secp256k1 pattern (library TU compiled separately, without the shim).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "sqlite_read.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"
VENDOR_SCRIPT = ROOT / "tools" / "vendor_sqlite.sh"
REGISTRY = ROOT / "src" / "fuzzer_tool" / "core" / "operator_registry.py"

MAGIC = b"SQLite format 3\x00"
HEADER_SIZE = 100


@pytest.fixture(scope="module")
def target_src() -> str:
    return TARGET.read_text()


@pytest.fixture(scope="module")
def build_src() -> str:
    return BUILD_SCRIPT.read_text()


class TestSnifferAgreement:
    def test_target_exists(self):
        assert TARGET.is_file(), "targets/sqlite_read.c is missing"

    def test_registry_predicate_is_magic_and_header_size(self):
        """The predicate this target mirrors, pinned so a change to either
        side breaks here rather than silently downgrading the mutator."""
        from fuzzer_tool.core.operator_registry import _FORMAT_SNIFFERS

        sniff = _FORMAT_SNIFFERS.get("sqlite_chunk_mutate")
        assert sniff is not None, "sqlite_chunk_mutate has no format sniffer"
        assert sniff(MAGIC + b"\x00" * (HEADER_SIZE - len(MAGIC)))
        assert not sniff(MAGIC + b"\x00" * 10), "short input must not sniff as sqlite"
        assert not sniff(b"\x00" + MAGIC + b"\x00" * HEADER_SIZE), (
            "a one-byte prefix must not sniff as sqlite — which is exactly why "
            "the target cannot use a mode-selector byte"
        )

    def test_dispatch_uses_the_same_header_size(self, target_src):
        assert f"#define SQLITE_FUZZ_HDR_SIZE {HEADER_SIZE}u" in target_src
        assert "size >= SQLITE_FUZZ_HDR_SIZE" in target_src

    def test_dispatch_compares_the_magic_at_offset_zero(self, target_src):
        """memcmp against buf, not buf+1 or buf[0] as a mode selector."""
        assert '#define SQLITE_FUZZ_MAGIC "SQLite format 3"' in target_src
        assert "memcmp(buf, SQLITE_FUZZ_MAGIC, sizeof(SQLITE_FUZZ_MAGIC)) == 0" in target_src

    def test_entry_point_exported(self, target_src):
        assert "int fuzz_shm_run(const unsigned char *buf, size_t size)" in target_src


class TestSandboxing:
    """direct_lite runs the target in-process, so an input that hangs or
    OOMs takes the whole campaign with it, not one exec."""

    @pytest.mark.parametrize(
        "guard",
        [
            "sqlite3_hard_heap_limit64",  # cannot OOM the fuzzer
            "sqlite3_progress_handler",  # cannot spin forever
            "sqlite3_set_authorizer",  # cannot ATTACH/PRAGMA out of process
            "SQLITE_DBCONFIG_DEFENSIVE",
            "SQLITE_DBCONFIG_TRUSTED_SCHEMA",  # documented untrusted-DB mitigation
            "SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION",
        ],
    )
    def test_guard_present(self, target_src, guard):
        assert guard in target_src, f"{guard} missing — target is not safely bounded"

    def test_memory_database_only(self, target_src):
        assert '":memory:"' in target_src
        assert "sqlite3_open_v2" in target_src


class TestBuildWiring:
    def test_vendor_script_exists_and_is_executable(self):
        assert VENDOR_SCRIPT.is_file(), "tools/vendor_sqlite.sh is missing"
        assert VENDOR_SCRIPT.stat().st_mode & 0o111, "vendor_sqlite.sh is not executable"
        text = VENDOR_SCRIPT.read_text()
        assert "vendor/sqlite" in text or 'SQLITE_DIR="$VENDOR_DIR/sqlite"' in text

    def test_build_script_defines_paths(self, build_src):
        assert 'SQLITE="${SQLITE_DIR:-vendor/sqlite}"' in build_src
        assert "SQLITE_DEFINES=" in build_src

    def test_amalgamation_compiled_without_the_shim(self, build_src):
        """Hard Rule 8: `-include $SHIM` applies to every .c on a command
        line, so the library TU must be compiled in its own pass."""
        m = re.search(r"compile_sqlite_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
        assert m, "compile_sqlite_objects not found"
        body = m.group(1)
        assert "-include" not in body, "sqlite3.c must not be compiled with the shim"
        assert '-c "$SQLITE/sqlite3.c"' in body

    def test_wrapper_and_library_share_one_define_list(self, build_src):
        """sqlite3.h parsed under different SQLITE_* defines than the object
        it links against is a silent ABI mismatch, so both sides use
        $SQLITE_DEFINES rather than two hand-kept copies."""
        m = re.search(r"compile_sqlite_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
        assert "$SQLITE_DEFINES" in m.group(1)
        assert 'SQLITE_INC="-I$SQLITE $SQLITE_DEFINES"' in build_src

    def test_target_is_built_and_verified(self, build_src):
        assert 'build_so_target "$TARGETS/sqlite_read.c"' in build_src
        assert '"$TARGETS"/sqlite_read.so' in build_src, (
            "sqlite_read.so missing from the AFL-symbol verify pass"
        )
        assert '"sqlite_read"' in build_src, "sqlite_read missing from the feature matrix"

    def test_gcc_fallback_does_not_lose_the_target(self, build_src):
        """trace-pc-guard is clang-only; gcc must still get a (shallower)
        target rather than a failed compile and a skipped build."""
        m = re.search(r"compile_sqlite_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
        body = m.group(1)
        assert "*clang*" in body, "coverage flag must be gated on the compiler"
