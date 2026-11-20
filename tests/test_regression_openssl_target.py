"""OpenSSL target: sniffer agreement and build wiring.

``targets/openssl_read.c`` feeds the raw input to OpenSSL's DER decoders
(X.509 cert/CRL/CSR, PKCS#7, keys). No mode-selector byte: the DER
mutators fire only when ``_sniff_der`` sees a SEQUENCE/SET at offset 0
(same trap as ``tests/test_regression_sqlite_target.py``).

Build wiring follows the sqlite/secp256k1 pattern: libcrypto built
separately (out-of-tree, never under the vendor root), shim only in the
wrapper TU.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "targets" / "openssl_read.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"
VENDOR_SCRIPT = ROOT / "tools" / "vendor_openssl.sh"

# Minimal DER: SEQUENCE { INTEGER 0 }.
DER_SEQ = b"\x30\x03\x02\x01\x00"


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


class TestSnifferAgreement:
    def test_target_exists(self):
        assert TARGET.is_file(), "targets/openssl_read.c is missing"

    def test_der_sniffer_needs_offset_zero(self):
        """A prefix byte hides the DER structure from the mutators."""
        from fuzzer_tool.core.operator_registry import _sniff_der

        assert _sniff_der(DER_SEQ)
        assert not _sniff_der(b"\x00" + DER_SEQ)

    def test_input_passed_unshifted(self, target_src):
        """Decoders read from buf itself, never buf + 1."""
        assert "buf + 1" not in target_src
        assert "buf[0]" not in target_src

    def test_entry_point_exported(self, target_src):
        assert "int fuzz_shm_run(const unsigned char *buf, size_t size)" in target_src

    @pytest.mark.parametrize("decoder", ["d2i_X509", "d2i_X509_CRL", "d2i_X509_REQ", "d2i_PKCS7"])
    def test_decoder_reached(self, target_src, decoder):
        assert f"{decoder}(" in target_src


class TestBounded:
    """In-process: a leak or unbounded input takes the campaign down."""

    def test_input_capped(self, target_src):
        assert "#define OPENSSL_FUZZ_MAX_IN" in target_src
        assert "size > OPENSSL_FUZZ_MAX_IN" in target_src

    @pytest.mark.parametrize(
        "free_fn", ["X509_free", "X509_CRL_free", "X509_REQ_free", "PKCS7_free", "EVP_PKEY_free"]
    )
    def test_objects_freed(self, target_src, free_fn):
        assert f"{free_fn}(" in target_src

    def test_error_queue_cleared(self, target_src):
        """The per-thread error queue grows per failed decode otherwise."""
        assert "ERR_clear_error()" in target_src


class TestBuildWiring:
    def test_vendor_script_exists_and_is_executable(self):
        assert VENDOR_SCRIPT.is_file(), "tools/vendor_openssl.sh is missing"
        assert VENDOR_SCRIPT.stat().st_mode & 0o111, "vendor_openssl.sh is not executable"
        assert 'OPENSSL_DIR="$VENDOR_DIR/openssl"' in VENDOR_SCRIPT.read_text()

    def test_build_script_defines_paths(self, build_src):
        assert 'OPENSSL="${OPENSSL_DIR:-$VENDOR/openssl}"' in build_src

    def test_library_built_without_the_shim(self, build_src):
        assert "-include" not in _compile_body(build_src)

    def test_library_built_out_of_tree(self, build_src):
        """Vendor root stays read-only; build tree lives under the build root."""
        body = _compile_body(build_src)
        assert "$TARGETS/openssl/build" in body
        assert '"$src/Configure"' in body
        assert 'local src="$OPENSSL"' in body, "default variant must build the release tree"

    def test_library_coverage_gated_on_compiler(self, build_src):
        body = _compile_body(build_src)
        assert 'cov_flag_for_cc "$cc"' in body
        assert "trace-pc-guard" not in body

    def test_library_is_pic(self, build_src):
        """libcrypto.a is linked into a .so."""
        assert "-fPIC" in _compile_body(build_src)

    def test_target_is_built_and_in_matrix(self, build_src):
        assert 'build_so_target "${TARGETS_SRC:-$TARGETS}/openssl_read.c"' in build_src
        assert '"openssl_read"' in build_src, "openssl_read missing from the feature matrix"


BUILT_SO = (
    Path(os.environ.get("FUZZ_BUILD_ROOT", Path.home() / "fuzzing" / "builds"))
    / "openssl_read_noasan.so"
)

# Hostile DER: deep indefinite nesting, 4 GiB long-form length, truncated
# long form, cap + 1 bytes of a SEQUENCE header.
_MAX_IN = 16 * 1024
HOSTILE = {
    "deep_indefinite": b"\x30\x80" * 4000,
    "huge_length": b"\x30\x84\xff\xff\xff\xff" + b"\x02\x01\x00",
    "truncated_long_form": b"\x30\x82\x01",
    "over_cap": b"\x30\x82\xff\xff" + b"\x00" * (_MAX_IN + 1 - 4),
}

_RUNNER = """
import ctypes, sys
f = ctypes.CDLL(sys.argv[1]).fuzz_shm_run
f.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
d = sys.stdin.buffer.read()
sys.exit(f(d, len(d)))
"""


@pytest.mark.skipif(not BUILT_SO.is_file(), reason="openssl_read_noasan.so not built")
class TestBuiltTargetAdversarial:
    """Malformed DER must return 0, never fault: a fault here is a bug."""

    @pytest.mark.parametrize("name", sorted(HOSTILE))
    def test_hostile_input_returns_zero(self, name):
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER, str(BUILT_SO)],
            input=HOSTILE[name],
            capture_output=True,
            timeout=10,
        )
        assert proc.returncode == 0, proc.stderr.decode(errors="replace")
