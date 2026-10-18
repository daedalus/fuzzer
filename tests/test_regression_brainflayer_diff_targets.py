"""The brainflayer differential targets must keep their oracle and their
build wiring intact.

``targets/ec_pubkey_diff.c`` and ``targets/hash_simd_diff.c`` are *differential*
targets: each runs two independent implementations of the same computation and
``abort()``s when they disagree. That makes two things load-bearing in a way a
crash-only target does not have:

1. The oracle must actually compare. If the gate that skips invalid EC keys
   ever swallowed the valid ones too, or a hash comparison were dropped, the
   target would run clean forever and silently test nothing — the same
   failure mode the sqlite target guards against, here for a differential
   instead of a sniffer.
2. The library translation units (brainflayer's sources and the reference
   libsecp256k1) must be compiled WITHOUT ``-include afl_shim.c`` (Hard Rule
   8), or the link fails with multiple-definition errors on the shim symbols.

CI does not vendor or build these (they need a network clone and OpenSSL), so
this is a source/wiring regression test in the shape of
``test_regression_sqlite_target.py`` — it pins the contract so a change to the
harness or the build script breaks here rather than silently.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EC_TARGET = ROOT / "targets" / "ec_pubkey_diff.c"
HASH_TARGET = ROOT / "targets" / "hash_simd_diff.c"
BUILD_SCRIPT = ROOT / "tools" / "build_targets.sh"
VENDOR_SCRIPT = ROOT / "tools" / "vendor_brainflayer.sh"


@pytest.fixture(scope="module")
def ec_src() -> str:
    return EC_TARGET.read_text()


@pytest.fixture(scope="module")
def hash_src() -> str:
    return HASH_TARGET.read_text()


@pytest.fixture(scope="module")
def build_src() -> str:
    return BUILD_SCRIPT.read_text()


@pytest.fixture(scope="module")
def vendor_src() -> str:
    return VENDOR_SCRIPT.read_text()


class TestTargetsExist:
    def test_ec_target_exists(self):
        assert EC_TARGET.is_file(), "targets/ec_pubkey_diff.c is missing"

    def test_hash_target_exists(self):
        assert HASH_TARGET.is_file(), "targets/hash_simd_diff.c is missing"

    def test_entry_points_exported(self, ec_src, hash_src):
        sig = "int fuzz_shm_run(const unsigned char *buf, size_t size)"
        assert sig in ec_src
        assert sig in hash_src


class TestEcOracle:
    """ec_pubkey_diff: brainflayer's precomp fast path vs the reference
    secp256k1_ec_pubkey_create, compared for every valid key."""

    def test_both_implementations_are_called(self, ec_src):
        assert "secp256k1_ec_pubkey_create(" in ec_src, "reference side missing"
        assert "secp256k1_ec_pubkey_create_precomp(" in ec_src, "fast path missing"

    def test_gate_skips_only_invalid_keys(self, ec_src):
        """A key the reference rejects (0 or >= n) is the gate and returns 0;
        everything else must fall through to the comparison. Pinning the gate
        shape stops it from being widened into 'skip everything'."""
        assert "secp256k1_ec_pubkey_create(ec_diff_ctx, &pk, seckey) != 1" in ec_src
        # the reject branch returns, it does not abort
        gate = ec_src.split("!= 1")[1].split("}")[0]
        assert "return 0" in gate and "abort" not in gate

    def test_mismatch_aborts_on_full_65_byte_compare(self, ec_src):
        assert "memcmp(ref, got, 65) != 0" in ec_src
        # the disagreement branch must abort (reported as a crash)
        assert re.search(r"memcmp\(ref, got, 65\) != 0\)\s*\{[^}]*abort\(\)", ec_src, re.S)

    def test_serialized_as_uncompressed(self, ec_src):
        assert "SECP256K1_EC_UNCOMPRESSED" in ec_src
        assert "ref_len != 65" in ec_src


class TestHashOracle:
    """hash_simd_diff: brainflayer SIMD vs scalar/OpenSSL reference, per record."""

    def test_sha256_both_paths_called(self, hash_src):
        # scalar references
        assert "sha256_upub(" in hash_src
        assert "sha256_cpub(" in hash_src
        # SIMD under test
        assert "sha256_upub_n(" in hash_src
        assert "sha256_cpub_n(" in hash_src

    def test_ripemd160_both_paths_called(self, hash_src):
        assert "ripemd160_256(" in hash_src       # scalar reference
        assert "ripemd160_256_n(" in hash_src     # SIMD under test

    def test_each_family_aborts_on_mismatch(self, hash_src):
        assert "memcmp(ref, simd[i], 32) != 0) { __afl_map_edge(0x9301); abort(); }" in hash_src
        assert "memcmp(ref, simd[i], 32) != 0) { __afl_map_edge(0x9302); abort(); }" in hash_src
        assert "memcmp(ref, simd[i], 20) != 0) { __afl_map_edge(0x9401); abort(); }" in hash_src

    def test_record_counts_are_bounded(self, hash_src):
        """direct_lite runs in-process, so an adversarial length must not be
        allowed to allocate or spin without bound."""
        assert "#define HASH_DIFF_MAX_RECS" in hash_src
        assert "n > HASH_DIFF_MAX_RECS" in hash_src

    def test_record_sizes_match_the_hashers(self, hash_src):
        assert "size / 65u" in hash_src   # 65-byte uncompressed pubkey records
        assert "size / 32u" in hash_src   # 32-byte ripemd records


class TestBuildWiring:
    def test_vendor_script_exists_and_is_executable(self, vendor_src):
        assert VENDOR_SCRIPT.is_file(), "tools/vendor_brainflayer.sh is missing"
        assert VENDOR_SCRIPT.stat().st_mode & stat.S_IXUSR, "vendor script not executable"
        assert "brainflayer" in vendor_src

    def test_vendor_pins_are_present(self, vendor_src):
        """The targets were validated against these exact commits; a silent
        pin bump is a review-worthy change, not a transparent one."""
        assert "BRAINFLAYER_REF" in vendor_src
        assert "acb7f97eb82dfbbdb797354e1550b910055b4422" in vendor_src, (
            "libsecp256k1 pin changed — ec_pubkey_fast.c needs this vintage"
        )
        # the config the two TUs must agree on
        assert "USE_FIELD_10X26" in vendor_src

    def test_build_script_defines_paths(self, build_src):
        assert 'BRAINFLAYER="${BRAINFLAYER_DIR:-$VENDOR/brainflayer}"' in build_src
        assert 'BRAINFLAYER_SECP256K1="$BRAINFLAYER/secp256k1"' in build_src

    @pytest.mark.parametrize(
        "fn", ["compile_brainflayer_ec_objects", "compile_brainflayer_hash_objects"]
    )
    def test_library_objects_compiled_without_the_shim(self, build_src, fn):
        """Hard Rule 8: `-include $SHIM` applies to every .c on a command
        line, so each library TU must be compiled in its own pass."""
        m = re.search(rf"{fn}\(\)\s*\{{(.*?)\n\}}", build_src, re.S)
        assert m, f"{fn} not found"
        assert "-include" not in m.group(1), f"{fn} must not compile with the shim"

    def test_ec_objects_use_config_and_matching_includes(self, build_src):
        m = re.search(r"compile_brainflayer_ec_objects\(\)\s*\{(.*?)\n\}", build_src, re.S)
        body = m.group(1)
        assert "-DHAVE_CONFIG_H=1" in body, "reference lib needs the vendored config"
        assert "-Wno-unused-function" in body, "ec_pubkey_fast.c has unused helpers"
        assert '-I"$BRAINFLAYER"' in body

    def test_hash_target_links_openssl(self, build_src):
        assert "hash_simd_diff" in build_src
        assert "-lcrypto" in build_src, "OpenSSL scalar SHA-256 reference needs libcrypto"

    def test_targets_are_built(self, build_src):
        assert 'build_so_target "${TARGETS_SRC:-$TARGETS}/ec_pubkey_diff.c"' in build_src
        assert 'build_so_target "${TARGETS_SRC:-$TARGETS}/hash_simd_diff.c"' in build_src

    def test_targets_in_verify_pass(self, build_src):
        assert '"$TARGETS"/ec_pubkey_diff.so' in build_src
        assert '"$TARGETS"/hash_simd_diff.so' in build_src

    def test_targets_in_feature_matrix(self, build_src):
        assert '"ec_pubkey_diff"' in build_src
        assert '"hash_simd_diff"' in build_src
