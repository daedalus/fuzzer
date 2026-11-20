"""OpenSSL base64 differential target: AVX2 vs scalar encoder.

``targets/openssl_b64_diff.c`` runs each input through OpenSSL's public
base64 API twice, with the AVX2 capability bit cleared and set, and aborts
on any difference; the decoder round trip is a second oracle. The AVX2
dispatch exists only when libcrypto is built with asm (encode.c gates it on
``OPENSSL_CPUID_OBJ && !OPENSSL_NO_ASM``), so this target links an
asm-enabled build tree, not openssl_read's no-asm one.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "openssl_b64_diff.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"
VENDOR_SCRIPT = ROOT / "tools" / "vendor_openssl.sh"
BUILD_ROOT = Path(os.environ.get("FUZZ_BUILD_ROOT", Path.home() / "fuzzing" / "builds"))
BUILT_SO = BUILD_ROOT / "openssl_b64_diff_noasan.so"
ASM_TREE = BUILD_ROOT / "openssl" / "build_asm_nosan"

# Header bytes: mode, stream chunk size.
HDR = 2


@pytest.fixture(scope="module")
def target_src() -> str:
    return TARGET.read_text()


@pytest.fixture(scope="module")
def build_src() -> str:
    return BUILD_SCRIPT.read_text()


def _compile_body(build_src: str) -> str:
    m = re.search(r"compile_openssl_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
    assert m, "compile_openssl_objects not found"
    return m.group(1)


class TestTarget:
    def test_target_exists(self):
        assert TARGET.is_file(), "targets/openssl_b64_diff.c is missing"

    def test_entry_point_exported(self, target_src):
        assert "int fuzz_shm_run(const unsigned char *buf, size_t size)" in target_src

    def test_toggles_avx2_capability_bit(self, target_src):
        """encode.c dispatches on OPENSSL_ia32cap_P[2] bit 5."""
        assert "OPENSSL_ia32cap_P[2]" in target_src
        assert "1u << 5" in target_src

    def test_avx2_gated_on_cpu_once(self, target_src):
        """Setting the bit on a CPU without AVX2 would SIGILL: decide at init."""
        assert '__builtin_cpu_supports("avx2")' in target_src

    @pytest.mark.parametrize(
        "api", ["EVP_EncodeBlock(", "EVP_EncodeUpdate(", "EVP_DecodeBlock(", "EVP_DecodeUpdate("]
    )
    def test_api_exercised(self, target_src, api):
        assert api in target_src

    def test_regression_mismatch_not_swallowed_by_shim(self, target_src):
        """afl_shim.c rewrites abort() into a logged return (unless
        --abort-is-crash), so a mismatch reported via abort() is lost."""
        code = re.sub(r"/\*.*?\*/", "", target_src, flags=re.S)
        assert "abort()" not in code
        assert "raise(SIGABRT)" in code

    def test_input_capped(self, target_src):
        assert "#define B64_DIFF_MAX_IN" in target_src
        assert "> B64_DIFF_MAX_IN" in target_src


class TestBuildWiring:
    def test_vendor_default_is_4_1(self):
        assert 'OPENSSL_VERSION="${OPENSSL_VERSION:-4.1.0}"' in VENDOR_SCRIPT.read_text()

    def test_vendor_refuses_stale_tree(self):
        """A tree of another version must not be silently reused."""
        text = VENDOR_SCRIPT.read_text()
        assert "VERSION.dat" in text

    def test_stamp_includes_source_version(self, build_src):
        """Re-vendoring a new version must rebuild libcrypto."""
        assert "VERSION.dat" in _compile_body(build_src)

    def test_asm_variant_has_own_tree(self, build_src):
        """asm and no-asm objects must never share an archive."""
        assert "build_asm" in _compile_body(build_src)
        base = re.search(r'OPENSSL_CONFIG_BASE="([^"]*)"', build_src)
        assert base, "OPENSSL_CONFIG_BASE missing"
        assert "no-asm" not in base.group(1), "asm variant would inherit no-asm"

    def test_noasm_tree_unchanged_for_openssl_read(self, build_src):
        assert 'OPENSSL_BDIR="$TARGETS/openssl/build${suffix}"' in build_src

    def test_target_built_and_in_matrix(self, build_src):
        assert 'build_so_target "${TARGETS_SRC:-$TARGETS}/openssl_b64_diff.c"' in build_src
        assert '"openssl_b64_diff"' in build_src


@pytest.mark.skipif(not BUILT_SO.is_file(), reason="openssl_b64_diff_noasan.so not built")
class TestBuiltTargetAdversarial:
    """Boundary lengths around the 96-byte AVX2 block and 48-byte line."""

    _RUNNER = (
        "import ctypes, sys\n"
        "f = ctypes.CDLL(sys.argv[1]).fuzz_shm_run\n"
        "f.argtypes = [ctypes.c_char_p, ctypes.c_size_t]\n"
        "d = sys.stdin.buffer.read()\n"
        "sys.exit(f(d, len(d)))\n"
    )

    @pytest.mark.parametrize("n", [0, 1, 47, 48, 95, 96, 99, 100, 101, 196, 197, 4095])
    @pytest.mark.parametrize("mode", [0, 1, 2, 3])
    def test_boundary_lengths_agree(self, n, mode):
        chunk = 7
        data = bytes((mode, chunk)) + bytes((i * 37) & 0xFF for i in range(n))
        proc = subprocess.run(
            [sys.executable, "-I", "-c", self._RUNNER, str(BUILT_SO)],
            input=data,
            capture_output=True,
            timeout=10,
        )
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")


@pytest.mark.skipif(
    not (ASM_TREE / "libcrypto.a").is_file() or not shutil.which("clang"),
    reason="asm libcrypto not built",
)
def test_falsification_oracle_catches_corruption(tmp_path):
    """Built with B64_DIFF_CORRUPT_SCALAR the scalar output is altered by
    one byte; the oracle must kill the process, or it could never report a
    real bug. Linked like build_target: shim in the wrapper TU only."""
    stub = tmp_path / "stub.c"
    stub.write_text(
        "#include <stddef.h>\n"
        "int fuzz_shm_run(const unsigned char *, size_t);\n"
        "int main(void) { unsigned char b[202] = {0, 0};\n"
        "  for (int i = 2; i < 202; i++) b[i] = (unsigned char)i;\n"
        "  return fuzz_shm_run(b, sizeof b); }\n"
    )
    stub_o = tmp_path / "stub.o"
    subprocess.run(["clang", "-c", str(stub), "-o", str(stub_o)], check=True)

    def build(name: str, *defines: str) -> Path:
        exe = tmp_path / name
        subprocess.run(
            [
                "clang",
                "-O1",
                *defines,
                "-D__AFL_CMPLOG=1",
                "-include",
                str(ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"),
                f"-I{ASM_TREE / 'include'}",
                f"-I{Path.home() / 'fuzzing' / 'vendoring' / 'openssl' / 'include'}",
                "-o",
                str(exe),
                str(TARGET),
                str(stub_o),
                str(ASM_TREE / "libcrypto.a"),
                "-lpthread",
                "-ldl",
            ],
            check=True,
            capture_output=True,
        )
        return exe

    # Control first (Hard Rule 46): the uncorrupted build must pass.
    clean = subprocess.run([str(build("clean"))], capture_output=True, timeout=10)
    assert clean.returncode == 0, clean.stderr.decode(errors="replace")

    bad = subprocess.run(
        [str(build("corrupt", "-DB64_DIFF_CORRUPT_SCALAR"))], capture_output=True, timeout=10
    )
    assert bad.returncode != 0, "corrupted scalar output was not detected"


# ── OpenSSL master: AVX2 decoder (dec_b64_avx2.c, not in 4.1.0) ─────────
MASTER_SO = BUILD_ROOT / "openssl_b64_diff_master_noasan.so"
MASTER_TREE = BUILD_ROOT / "openssl" / "build_master_nosan"
MODE_DECODE = 4  # byte 0 bit 2: payload is base64 text, diff the decoders


class TestMasterWiring:
    def test_decode_mode_bit(self, target_src):
        assert "#define B64_MODE_DECODE 4u" in target_src

    @pytest.mark.parametrize("api", ["EVP_DecodeBlock(", "EVP_DecodeUpdate(", "EVP_DecodeFinal("])
    def test_decoders_diffed(self, target_src, api):
        body = target_src.split("static void diff_decode", 1)
        assert len(body) == 2, "diff_decode missing"
        assert api in body[1].split("\nstatic int fuzz_b64", 1)[0]

    def test_vendor_master_is_git_clone(self):
        text = VENDOR_SCRIPT.read_text()
        assert "openssl-master" in text
        assert "git clone" in text

    def test_master_variant_builds_master_source(self, build_src):
        body = _compile_body(build_src)
        assert "build_master" in body
        assert "$OPENSSL_MASTER" in body
        assert 'OPENSSL_MASTER="${OPENSSL_MASTER_DIR:-$VENDOR/openssl-master}"' in build_src

    def test_stamp_includes_git_commit(self, build_src):
        """master's VERSION.dat stays x.y.0-dev across commits."""
        assert "rev-parse HEAD" in _compile_body(build_src)

    def test_master_target_built_and_in_matrix(self, build_src):
        assert '"$TARGETS/openssl_b64_diff_master${out_suffix}.so"' in build_src
        assert '"openssl_b64_diff_master"' in build_src


_DECODE_INPUTS = {
    "valid_64": b"QUJD" * 16,
    "valid_96_pad": b"QUJD" * 23 + b"QQ==",
    "ws_inside": (b"QUJD" * 8 + b" \n\t") * 3,
    "bad_char_at_63": b"QUJD" * 15 + b"QUJ!",
    "pad_mid": b"QUJD" * 8 + b"QQ==" + b"QUJD" * 8,
    "srp_chars": b"./09" * 20,
    "trailing_ws": b"QUJD" * 16 + b"\r\n  ",
    "len_mod_4_1": b"QUJD" * 16 + b"Q",
}


@pytest.mark.skipif(not MASTER_SO.is_file(), reason="openssl_b64_diff_master_noasan.so not built")
class TestMasterBuiltAdversarial:
    """Hostile base64 text around the 64-byte AVX2 decode threshold."""

    @pytest.mark.parametrize("name", sorted(_DECODE_INPUTS))
    @pytest.mark.parametrize("chunk", [0, 1, 63, 64])
    def test_decoders_agree(self, name, chunk):
        data = bytes((MODE_DECODE, chunk)) + _DECODE_INPUTS[name]
        proc = subprocess.run(
            [sys.executable, "-I", "-c", TestBuiltTargetAdversarial._RUNNER, str(MASTER_SO)],
            input=data,
            capture_output=True,
            timeout=10,
        )
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")


@pytest.mark.skipif(
    not (MASTER_TREE / "libcrypto.a").is_file() or not shutil.which("clang"),
    reason="master libcrypto not built",
)
def test_falsification_decode_oracle_catches_corruption(tmp_path):
    """Decode mode, 128 chars (AVX2 decoder engages at >= 64): a corrupted
    scalar output byte must kill the process; the clean build must not."""
    if platform.machine() not in ("x86_64", "AMD64"):
        pytest.skip("AVX2 path is x86-64 only")
    stub = tmp_path / "stub.c"
    stub.write_text(
        "#include <stddef.h>\n#include <string.h>\n"
        "int fuzz_shm_run(const unsigned char *, size_t);\n"
        "int main(void) { unsigned char b[130] = {4, 0};\n"
        "  memset(b + 2, 'Q', 128);\n"
        "  return fuzz_shm_run(b, sizeof b); }\n"
    )
    stub_o = tmp_path / "stub.o"
    subprocess.run(["clang", "-c", str(stub), "-o", str(stub_o)], check=True)
    master_inc = Path.home() / "fuzzing" / "vendoring" / "openssl-master" / "include"

    def build(name: str, *defines: str) -> Path:
        exe = tmp_path / name
        cmd = [
            "clang",
            "-O1",
            *defines,
            "-D__AFL_CMPLOG=1",
            "-include",
            str(ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"),
        ]
        cmd += [
            f"-I{MASTER_TREE / 'include'}",
            f"-I{master_inc}",
            "-o",
            str(exe),
            str(TARGET),
            str(stub_o),
        ]
        cmd += [str(MASTER_TREE / "libcrypto.a"), "-lpthread", "-ldl"]
        subprocess.run(cmd, check=True, capture_output=True)
        return exe

    # Control first (Hard Rule 46).
    clean = subprocess.run([str(build("clean"))], capture_output=True, timeout=10)
    assert clean.returncode == 0, clean.stderr.decode(errors="replace")

    bad = subprocess.run(
        [str(build("corrupt", "-DB64_DIFF_CORRUPT_SCALAR"))], capture_output=True, timeout=10
    )
    assert bad.returncode != 0, "corrupted scalar decode output was not detected"
