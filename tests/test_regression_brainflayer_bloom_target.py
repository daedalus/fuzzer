"""targets/brainflayer_read.c fuzzes brainflayer's sliced bloom filter
(bloom.c): trailer validation (bloom_init), slice extraction (bloom_slice),
set/check, and the scalar vs AVX2 batch lookup (bloom_chk_n).

It is a differential target, so the oracle is load-bearing: a broken
slicer or a lane bug in the AVX2 gathers must abort, and the unmodified
library must not (Hard Rule 46 control). The build-and-run tests need the
vendored sources (tools/vendor_brainflayer.sh) and clang; they skip
otherwise. The wiring tests always run.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "brainflayer_read.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"
VENDOR_SCRIPT = ROOT / "tools" / "vendor_brainflayer.sh"
DICT = ROOT / "dictionaries" / "brainflayer.dict"

VENDOR_ROOT = Path(os.environ.get("FUZZ_VENDOR_ROOT", Path.home() / "fuzzing" / "vendoring"))
BRAINFLAYER = Path(os.environ.get("BRAINFLAYER_DIR", VENDOR_ROOT / "brainflayer"))

MAGIC = b"BLFSLICE"
VERSION = 1
REC_SZ = 20
SIGABRT_RC = -6

# One slice-offset line in bloom_init; sabotage flips bit 0 of every offset
# (stays < 160, so no out-of-bounds word index).
SLICE_OFFSET = "o = i * 160 / k;"
SLICE_OFFSET_BROKEN = "o = (i * 160 / k) ^ 1;"

# Driver: stdin -> fuzz_shm_run. Stub edge map, the shim is not under test.
DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
void __afl_map_edge(unsigned int loc) { (void)loc; }
int fuzz_shm_run(const unsigned char *, size_t);
int main(void) {
    static unsigned char buf[1 << 16];
    size_t n = fread(buf, 1, sizeof(buf), stdin);
    unsigned char *exact = malloc(n ? n : 1);
    for (size_t i = 0; i < n; i++) { exact[i] = buf[i]; }
    int rc = fuzz_shm_run(exact, n);
    free(exact);
    return rc;
}
"""


def trailer(log2bits: int, k: int, magic: bytes = MAGIC, version: int = VERSION) -> bytes:
    """16-byte trailer, laid out as bloom_trailer() writes it."""
    return magic + bytes([version, log2bits, k]) + bytes(5)


def records(n: int) -> bytes:
    """n distinct hash160 records with every bit position varied."""
    out = bytearray()
    for i in range(n):
        seed = (i + 1) * 0x9E3779B1
        out += b"".join(struct.pack("<I", (seed * (j + 3)) & 0xFFFFFFFF) for j in range(5))
    return bytes(out)


# ── wiring ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def src() -> str:
    return TARGET.read_text()


@pytest.fixture(scope="module")
def build_src() -> str:
    return BUILD_SCRIPT.read_text()


class TestSource:
    def test_entry_point_exported(self, src):
        assert "int fuzz_shm_run(const unsigned char *buf, size_t size)" in src

    def test_library_validator_gates_trailer(self, src):
        assert "bloom_init(" in src, "trailer must go through the library validator"
        assert "BLOOM_MAGIC" in src and "BLOOM_VERSION" in src

    def test_independent_slice_reference(self, src):
        assert "bloom_slice(" in src
        assert "bf_ref_slice(" in src, "needs an independent bit-extraction oracle"

    def test_scalar_and_simd_batch_compared(self, src):
        assert "bloom_chk_n(" in src
        assert '__builtin_cpu_supports("avx2")' in src

    def test_work_is_bounded(self, src):
        """direct_lite runs in-process: record count and mapping are capped."""
        assert "#define BF_MAX_RECS" in src
        assert "MAP_NORESERVE" in src and "munmap(" in src

    def test_mismatches_abort(self, src):
        assert src.count("abort()") >= 3


class TestWiring:
    def test_vendor_verifies_bloom_sources(self):
        vendor_src = VENDOR_SCRIPT.read_text()
        assert "bloom.c" in vendor_src and "bloom.h" in vendor_src

    def test_objects_compiled_without_the_shim(self, build_src):
        m = re.search(r"compile_brainflayer_bloom_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
        assert m, "compile_brainflayer_bloom_objects not found"
        assert "-include" not in m.group(1)
        assert '"$BRAINFLAYER/bloom.c"' in m.group(1)

    def test_target_built_verified_and_listed(self, build_src):
        assert 'build_so_target "${TARGETS_SRC:-$TARGETS}/brainflayer_read.c"' in build_src
        assert '"$TARGETS"/brainflayer_read.so' in build_src
        assert '"brainflayer_read"' in build_src

    def test_dictionary_has_magic(self):
        assert '"BLFSLICE"' in DICT.read_text()


# ── build and run ─────────────────────────────────────────────────

needs_vendor = pytest.mark.skipif(
    not (BRAINFLAYER / "bloom.c").is_file() or shutil.which("clang") is None,
    reason="brainflayer not vendored (tools/vendor_brainflayer.sh) or clang missing",
)


def _build(out: Path, bloom_c: Path, san: list[str]) -> Path:
    driver = out / "driver.c"
    driver.write_text(DRIVER)
    exe = out / "brainflayer_read"
    cmd = [
        "clang", "-O1", "-g", *san, "-I", str(BRAINFLAYER),
        str(TARGET), str(bloom_c), str(BRAINFLAYER / "mmapf.c"), str(driver),
        "-lm", "-o", str(exe),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    return exe


def _asan_flags(out: Path) -> list[str]:
    """ASAN when clang's runtime (libclang-rt-<ver>-dev) links, else none:
    the oracle aborts are what is under test, ASAN is extra."""
    probe = out / "probe.c"
    probe.write_text("int main(void) { return 0; }\n")
    cmd = ["clang", "-fsanitize=address", str(probe), "-o", str(out / "probe")]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0
    return ["-fsanitize=address"] if ok else []


def _run(exe: Path, data: bytes, simd: str = "1") -> int:
    env = {**os.environ, "BRAINFLAYER_SIMD": simd}
    proc = subprocess.run([str(exe)], input=data, capture_output=True, env=env, timeout=60)
    return proc.returncode


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """(stock, sabotaged) executables; tmp_path_factory cleans up."""
    stock_dir = tmp_path_factory.mktemp("bf_stock")
    broken_dir = tmp_path_factory.mktemp("bf_broken")
    san = _asan_flags(stock_dir)
    stock = _build(stock_dir, BRAINFLAYER / "bloom.c", san)

    bloom_src = (BRAINFLAYER / "bloom.c").read_text()
    assert SLICE_OFFSET in bloom_src, "sabotage anchor moved in bloom.c"
    broken_c = broken_dir / "bloom.c"
    broken_c.write_text(bloom_src.replace(SLICE_OFFSET, SLICE_OFFSET_BROKEN))
    return stock, _build(broken_dir, broken_c, san)


# log2bits/k covering: min size, the r + log2bits > 64 (lshift) path, k = 1,
# k = max, and the legacy filter (log2bits = 0).
VALID = [(20, 1), (20, 7), (33, 17), (40, 160), (40, 3), (0, 0)]


@needs_vendor
class TestOracle:
    @pytest.mark.parametrize(("log2bits", "k"), VALID)
    def test_control_stock_library_agrees(self, builds, log2bits, k):
        """A/A control: unmodified bloom.c passes every check."""
        stock, _ = builds
        data = trailer(log2bits, k) + records(9)
        assert _run(stock, data) == 0
        assert _run(stock, data, simd="0") == 0

    @pytest.mark.parametrize(("log2bits", "k"), [(20, 7), (33, 17), (40, 160)])
    def test_falsification_broken_slicer_aborts(self, builds, log2bits, k):
        """bloom_init offsets shifted by one bit: set/chk still agree with
        each other, only the independent reference can catch it."""
        _, broken = builds
        assert _run(broken, trailer(log2bits, k) + records(9)) == SIGABRT_RC

    @pytest.mark.parametrize(
        "head",
        [
            trailer(19, 7),  # below BLOOM_LOG2_MIN
            trailer(41, 7),  # above BLOOM_LOG2_MAX
            trailer(20, 161),  # above BLOOM_SLICES_MAX
            trailer(20, 0),  # k = 0
            trailer(20, 7, magic=b"BLFSLICF"),
            trailer(20, 7, version=2),
            trailer(255, 255),
        ],
    )
    def test_adversarial_rejected_trailers_are_clean(self, builds, head):
        """Rejected trailers never reach the filter, even on the broken build."""
        stock, broken = builds
        data = head + records(9)
        assert _run(stock, data) == 0
        assert _run(broken, data) == 0

    @pytest.mark.parametrize("data", [b"", MAGIC, trailer(20, 7)[:15], trailer(20, 7)])
    def test_adversarial_short_inputs(self, builds, data):
        """Empty, truncated trailer, trailer without records."""
        stock, _ = builds
        assert _run(stock, data) == 0

    def test_adversarial_record_cap(self, builds):
        """Far more records than BF_MAX_RECS plus a ragged tail: bounded, clean."""
        stock, _ = builds
        data = trailer(40, 160) + records(2000) + b"\x01\x02\x03"
        assert _run(stock, data) == 0
