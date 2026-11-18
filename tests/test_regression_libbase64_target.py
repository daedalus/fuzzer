"""libbase64 differential target: source bounds, build wiring, and oracle.

``targets/libbase64_read.c`` runs one input through every SIMD codec this CPU
supports and aborts on the first disagreement with the plain codec.

Two things are easy to get silently wrong:

* the build wiring. libbase64's ``lib/config.h`` is *generated* (upstream's
  Makefile does it), each SIMD codec needs its own ``-m`` flag, and
  ``codec_choose.c`` references only the codecs that config.h enables. So the
  objects are compiled in their own pass, without the shim, against one
  generated config.h per build suffix (lz4/secp256k1 discipline, Hard Rule 9).
* the oracle. A differential that cannot fire on a broken codec is not a
  differential. ``TestOracle`` builds the target against a clean tree
  (control), then against a tree with one byte of the AVX2 encoder's
  translate LUT changed, and requires SIGABRT. The control must pass first
  (Hard Rule 46).

The behaviour tests need clang and the vendored tree
(``tools/vendor_libbase64.sh``); they skip otherwise.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "libbase64_read.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"
VENDOR_SCRIPT = ROOT / "tools" / "vendor_libbase64.sh"
VENDOR_ROOT = Path(os.environ.get("FUZZ_VENDOR_ROOT", Path.home() / "fuzzing" / "vendoring"))
VENDOR_TREE = VENDOR_ROOT / "libbase64"

SIGABRT_RC = -6

# Inputs the wrapper must survive without aborting: empty, mode byte only,
# stream modes missing their chunk seed, and a chunk seed that overflows the
# remaining payload (b64_next_chunk must clamp to `left`).
EDGE_INPUTS = {
    "empty": b"",
    "mode_decode": b"\x00",
    "mode_encode": b"\x01",
    "stream_decode_no_seed": b"\x02",
    "stream_encode_no_seed": b"\x03",
    "stream_huge_seed": b"\x02\xff" + b"QUJD" * 4,
}

# One zero byte per 6-bit group lands in the [0..25] range of the AVX2
# encode LUT, so shifting its first entry (65 -> 66) corrupts every output
# byte of a zero payload. Any input with a zero first byte triggers it.
FAULT_INPUT = b"\x01" + b"\x00" * 64
LUT_HEAD = "\t\t65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,"


@pytest.fixture(scope="module")
def build_src() -> str:
    return BUILD_SCRIPT.read_text()


@pytest.fixture(scope="module")
def compile_body(build_src) -> str:
    m = re.search(r"compile_libbase64_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
    assert m, "compile_libbase64_objects not found in tools/build_targets.sh"
    return m.group(1)


class TestSource:
    def test_target_exists(self):
        assert TARGET.is_file(), "targets/libbase64_read.c is missing"

    def test_entry_points_exported(self):
        src = TARGET.read_text()
        assert "int fuzz_shm_run(const unsigned char *buf, size_t size)" in src
        assert "int fuzz_libbase64(const unsigned char *buf, size_t size)" in src

    def test_every_codec_is_pitted_against_plain(self):
        """A codec missing from the table is a codec the campaign never sees."""
        src = TARGET.read_text()
        for flag in (
            "BASE64_FORCE_PLAIN",
            "BASE64_FORCE_SSSE3",
            "BASE64_FORCE_SSE41",
            "BASE64_FORCE_SSE42",
            "BASE64_FORCE_AVX",
            "BASE64_FORCE_AVX2",
            "BASE64_FORCE_AVX512",
        ):
            assert flag in src, f"{flag} missing — that codec is never exercised"

    def test_cpu_gate_before_forcing_a_codec(self):
        """Forcing a codec the CPU lacks is SIGILL, not a mismatch."""
        src = TARGET.read_text()
        assert "__builtin_cpu_supports" in src
        assert "b64_cpu_has" in src

    def test_input_is_bounded(self):
        """direct_lite runs in-process: an adversarial length stalls the run."""
        src = TARGET.read_text()
        assert "#define B64_FUZZ_MAX_IN" in src
        assert "plen > B64_FUZZ_MAX_IN" in src


class TestBuildWiring:
    def test_vendor_script_exists_and_is_executable(self):
        assert VENDOR_SCRIPT.is_file(), "tools/vendor_libbase64.sh is missing"
        assert VENDOR_SCRIPT.stat().st_mode & 0o111, "vendor_libbase64.sh is not executable"
        assert 'LIBBASE64_DIR="$VENDOR_DIR/libbase64"' in VENDOR_SCRIPT.read_text()

    def test_build_script_defines_path(self, build_src):
        assert 'LIBBASE64="${LIBBASE64_DIR:-$VENDOR/libbase64}"' in build_src

    def test_objects_compiled_without_the_shim(self, compile_body):
        """Hard Rule 9: `-include $SHIM` applies to every .c on a command
        line, so the library TUs must be compiled in their own pass."""
        assert "-include" not in compile_body, "library objects must not get the shim"

    def test_one_object_per_codec(self, build_src, compile_body):
        """Each SIMD TU needs its own -m flag, so the codec table in
        build_targets.sh must name every one and the compile pass must use
        that table. A codec missing from the table is a codec the target never
        sees, with no symptom beyond a narrower differential."""
        m = re.search(r'^LIBBASE64_CODECS="([^"]+)"', build_src, re.M)
        assert m, "LIBBASE64_CODECS table not found"
        for arch in ("ssse3", "sse41", "sse42", "avx", "avx2", "avx512"):
            assert f"{arch}:" in m.group(1), f"{arch} codec missing from LIBBASE64_CODECS"
        assert "lib/arch/$name/codec.c" in compile_body, "codec objects not built from the table"
        assert "$mflags" in compile_body, "per-codec -m flags not applied"

    def test_config_h_is_generated(self, compile_body):
        """lib/config.h is generated, not shipped: without it every codec.c
        and codec_choose.c fail to compile, or worse, resolves to a stale
        upstream one."""
        assert "config.h" in compile_body
        assert "HAVE_" in compile_body

    def test_gcc_fallback_does_not_lose_coverage(self, compile_body):
        m = re.search(r"cov_flag_for_cc \"\$cc\"", compile_body)
        assert m, "coverage flag must be gated on the compiler"
        assert "-fsanitize-coverage=trace-pc-guard" not in compile_body, (
            "clang-only guard flag hardcoded; gcc must get trace-pc via cov_flag_for_cc"
        )

    def test_target_is_built_and_in_the_matrix(self, build_src):
        assert 'build_so_target "${TARGETS_SRC:-$TARGETS}/libbase64_read.c"' in build_src
        assert '"libbase64_read"' in build_src, "libbase64_read missing from the feature matrix"

    def test_vendor_script_documented_in_header(self, build_src):
        assert "tools/vendor_libbase64.sh ->" in build_src


def _cpu_has(feature: str) -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return feature in flags


def _compile(tree: Path, out: Path, cfg: Path) -> Path:
    """Build the target plus a file harness against one SIMD codec (AVX2).

    config.h mirrors what build_targets.sh generates: every codec macro
    defined, only AVX2 enabled. All codec objects are still linked, because
    codec_choose_forced() names all seven unconditionally; the disabled ones
    compile to upstream's stubs.
    """
    cfg.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (cfg / "config.h").write_text(
        "#define HAVE_SSSE3 0\n"
        "#define HAVE_SSE41 0\n"
        "#define HAVE_SSE42 0\n"
        "#define HAVE_AVX 0\n"
        "#define HAVE_AVX2 1\n"
        "#define HAVE_AVX512 0\n"
        "#define HAVE_NEON32 0\n"
        "#define HAVE_NEON64 0\n"
    )

    harness = out / "harness.c"
    harness.write_text(
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <stddef.h>\n"
        "void __afl_map_edge(unsigned int e) { (void)e; }\n"
        "extern int fuzz_shm_run(const unsigned char *buf, size_t size);\n"
        "int main(int argc, char **argv) {\n"
        '    FILE *f = fopen(argv[1], "rb");\n'
        "    if (!f) { return 2; }\n"
        "    fseek(f, 0, SEEK_END);\n"
        "    long n = ftell(f);\n"
        "    fseek(f, 0, SEEK_SET);\n"
        "    unsigned char *buf = malloc((size_t)n + 1);\n"
        "    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) { return 2; }\n"
        "    fclose(f);\n"
        "    int rc = fuzz_shm_run(buf, (size_t)n);\n"
        "    free(buf);\n"
        "    return rc;\n"
        "}\n"
    )

    sources = {
        "lib": (tree / "lib" / "lib.c", []),
        "codec_choose": (tree / "lib" / "codec_choose.c", []),
        "tables": (tree / "lib" / "tables" / "tables.c", []),
        "generic": (tree / "lib" / "arch" / "generic" / "codec.c", []),
        "avx2": (
            tree / "lib" / "arch" / "avx2" / "codec.c",
            # Force the portable C encoder: the x86-64 default is the inline
            # assembly loop in enc_loop_asm.c, and the fault injected below
            # lives in the C translate LUT.
            ["-mavx2", "-DBASE64_AVX2_USE_ASM=0"],
        ),
        "ssse3": (tree / "lib" / "arch" / "ssse3" / "codec.c", []),
        "sse41": (tree / "lib" / "arch" / "sse41" / "codec.c", []),
        "sse42": (tree / "lib" / "arch" / "sse42" / "codec.c", []),
        "avx": (tree / "lib" / "arch" / "avx" / "codec.c", []),
        "avx512": (tree / "lib" / "arch" / "avx512" / "codec.c", []),
        "neon32": (tree / "lib" / "arch" / "neon32" / "codec.c", []),
        "neon64": (tree / "lib" / "arch" / "neon64" / "codec.c", []),
    }
    objs = []
    for name, (src, tree_flags) in sources.items():
        obj = out / f"{name}.o"
        subprocess.run(
            [
                "clang",
                "-O1",
                "-g",
                "-fPIC",
                "-I",
                str(cfg),
                "-I",
                str(tree / "lib"),
                "-I",
                str(tree / "include"),
                *tree_flags,
                "-c",
                str(src),
                "-o",
                str(obj),
            ],
            check=True,
        )
        objs.append(str(obj))

    target_obj = out / "target.o"
    subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-fPIC",
            "-I",
            str(tree / "include"),
            "-c",
            str(TARGET),
            "-o",
            str(target_obj),
        ],
        check=True,
    )
    harness_obj = out / "harness.o"
    subprocess.run(["clang", "-O1", "-g", "-c", str(harness), "-o", str(harness_obj)], check=True)

    binary = out / "run"
    subprocess.run(
        ["clang", "-O1", "-g", "-o", str(binary), str(harness_obj), str(target_obj), *objs],
        check=True,
    )
    return binary


def _run(binary: Path, data: bytes) -> int:
    payload = binary.parent / "input.bin"
    payload.write_bytes(data)
    return subprocess.run([str(binary), str(payload)], capture_output=True).returncode


@pytest.mark.skipif(
    shutil.which("clang") is None or not (VENDOR_TREE / "lib" / "lib.c").is_file(),
    reason="needs clang and tools/vendor_libbase64.sh (FUZZ_VENDOR_ROOT/libbase64)",
)
@pytest.mark.skipif(
    shutil.which("clang") is None or not (VENDOR_TREE / "lib" / "lib.c").is_file(),
    reason="needs clang and tools/vendor_libbase64.sh (FUZZ_VENDOR_ROOT/libbase64)",
)
class TestOracle:
    @pytest.fixture(scope="class")
    def control(self, tmp_path_factory) -> Path:
        return _compile(
            VENDOR_TREE, tmp_path_factory.mktemp("b64control"), tmp_path_factory.mktemp("b64cfg")
        )

    def test_control_passes_on_clean_tree(self, control):
        """Hard Rule 46: a differential that cannot fail on identical inputs
        is not an oracle. The control has to be green before the fault means
        anything."""
        assert _run(control, FAULT_INPUT) == 0, "differential fired on a clean tree"

    @pytest.mark.parametrize("name", sorted(EDGE_INPUTS))
    def test_edge_inputs_do_not_abort(self, control, name):
        assert _run(control, EDGE_INPUTS[name]) == 0, f"{name} aborted the target"

    @pytest.mark.skipif(not _cpu_has("avx2"), reason="the injected fault lives in the AVX2 codec")
    def test_injected_avx2_fault_aborts(self, tmp_path):
        """Falsification: corrupt one LUT entry of the AVX2 encoder and the
        differential must abort. Anything else means the target is not
        comparing codec output."""
        tree = tmp_path / "tree"
        shutil.copytree(VENDOR_TREE, tree)
        translate = tree / "lib" / "arch" / "avx2" / "enc_translate.c"
        text = translate.read_text()
        assert LUT_HEAD in text, "AVX2 translate LUT not found — update the fault injection"
        translate.write_text(text.replace(LUT_HEAD, LUT_HEAD.replace("65, 71", "66, 71", 1), 1))

        binary = _compile(tree, tmp_path / "faulty", tmp_path / "faultycfg")
        assert _run(binary, FAULT_INPUT) == SIGABRT_RC, "corrupted AVX2 encoder was not detected"
