"""Crypto-constant sniffer: algorithm tables in a target's data sections.

Fixtures are tiny clang-built ELFs whose ``const`` arrays hold the
published constants (FIPS 180-4, RFC 1321/7539/7693, Keccak reference).
The expected sets are literals, not derived from the module under test.
"""

import shutil
import subprocess
from unittest.mock import patch

import pytest

from fuzzer_tool.core.crypto_consts import CryptoAlgo, resolve_hlx, sniff_elf, sniff_target

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang required")

A = CryptoAlgo

# C initialisers, one array per signature. *_BE arrays hold byte-swapped
# words, i.e. the table as a big-endian target would lay it out.
SHA256_K = "const uint32_t k256[8] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 1, 2, 3, 4};"
SHA256_K_2 = "const uint32_t k256[2] = {0x428a2f98, 0x71374491};"
SHA256_IV = "const uint32_t iv256[4] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a};"
SHA512_IV = (
    "const uint64_t iv512[4] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,"
    " 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL};"
)
BLAKE2_SIGMA = (
    "const uint8_t sigma[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
    "14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3};"
)
TEA_BE = "const uint32_t delta[1] = {0xb979379e};"
CHACHA = 'const char sigma_s[17] = "expand 32-byte k";'
KECCAK = (
    "const uint64_t rc[4] = {0x0000000000000001ULL, 0x0000000000008082ULL,"
    " 0x800000000000808aULL, 0x8000000080008000ULL};"
)
AES = (
    "const uint8_t sbox[16] = {0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,"
    "0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76};"
)
CRC32 = "const uint32_t crc[4] = {0x00000000, 0x77073096, 0xee0e612c, 0x990951ba};"
MD5_IV = "const uint32_t md5iv[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};"
SHA1_IV = "const uint32_t sha1iv[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};"
SHA1_IV_BE = (
    "const uint32_t sha1be[5] = {0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210, 0xf0e1d2c3};"
)


def _build(tmp_path, *decls, name="t"):
    """Compile a probe whose ``main`` touches every array so none is dropped."""
    names = [d.split("[")[0].split()[-1] for d in decls]
    uses = " + ".join(f"(int)((const unsigned char *){n})[argc]" for n in names) or "0"
    src = tmp_path / f"{name}.c"
    src.write_text(
        "#include <stdint.h>\n"
        + "\n".join(f"__attribute__((used)) {d}" for d in decls)
        + f"\nint main(int argc, char **argv) {{ (void)argv; return {uses}; }}\n"
    )
    out = tmp_path / name
    subprocess.run(["clang", "-O0", "-o", str(out), str(src)], check=True)
    return str(out)


def test_detects_every_family(tmp_path):
    target = _build(tmp_path, SHA256_K, TEA_BE, CHACHA, KECCAK, AES, CRC32, MD5_IV)
    assert sniff_target(target) == {A.SHA256, A.TEA, A.CHACHA, A.KECCAK, A.AES, A.CRC32, A.MD5}


def test_sha512_iv_with_sigma_is_blake2b(tmp_path):
    assert sniff_target(_build(tmp_path, SHA512_IV, BLAKE2_SIGMA)) == {A.BLAKE2B}
    assert sniff_target(_build(tmp_path, SHA512_IV, name="u")) == {A.SHA512}


def test_sha256_iv_with_sigma_is_blake2s(tmp_path):
    assert sniff_target(_build(tmp_path, SHA256_IV, BLAKE2_SIGMA)) == {A.BLAKE2S}
    assert sniff_target(_build(tmp_path, SHA256_IV, name="u")) == {A.SHA256}


def test_sha1_iv_is_not_md5(tmp_path):
    # SHA-1's IV starts with MD5's four words; only the five-word run counts.
    assert sniff_target(_build(tmp_path, SHA1_IV)) == {A.SHA1}
    assert sniff_target(_build(tmp_path, SHA1_IV, MD5_IV, name="u")) == {A.SHA1, A.MD5}


def test_big_endian_table(tmp_path):
    assert sniff_target(_build(tmp_path, SHA1_IV_BE)) == {A.SHA1}


def test_falsify_partial_table_is_not_a_hit(tmp_path):
    """Two of SHA-256's K words is a chance-level match, not a table."""
    assert sniff_target(_build(tmp_path, SHA256_K_2)) == set()


def test_falsify_plain_binary_is_empty(tmp_path):
    assert sniff_target(_build(tmp_path)) == set()


@pytest.mark.parametrize(
    "mangle",
    [
        lambda b: b[:40],  # truncated header
        lambda b: b[:40] + (2**63).to_bytes(8, "little") + b[48:],  # e_shoff past EOF
        lambda b: b[:60] + b"\xff\xff" + b[62:],  # e_shnum huge
        lambda b: b"MZ" + b[2:],  # not ELF
    ],
)
def test_adversarial_corrupt_headers(tmp_path, mangle):
    with open(_build(tmp_path, SHA256_K), "rb") as f:
        elf = f.read()
    assert sniff_elf(mangle(elf)) == frozenset()


def test_adversarial_table_straddling_section_end(tmp_path):
    """Table bytes split across the file's end are not a match."""
    with open(_build(tmp_path, SHA256_K), "rb") as f:
        elf = f.read()
    k = (0x428A2F98).to_bytes(4, "little")
    cut = elf.find(k) + 6
    assert sniff_elf(elf[:cut]) == frozenset()


def test_missing_target_is_empty(tmp_path):
    assert sniff_target(str(tmp_path / "absent")) == frozenset()


# ── --hlx-algo auto ─────────────────────────────────────────────────────


def test_resolve_hlx_prefers_strongest():
    assert resolve_hlx("auto", {A.MD5, A.SHA1}) == "sha1"
    assert resolve_hlx("auto", {A.SHA256, A.MD5}) == "sha256"
    assert resolve_hlx("auto", {A.CRC32}) is None
    assert resolve_hlx("md5", {A.SHA256}) == "md5"  # explicit choice wins
    assert resolve_hlx(None, {A.SHA256}) is None  # off stays off


def test_fuzzer_arms_hash_len_ext_from_target(tmp_path, capsys):
    from fuzzer_tool.core.operator_registry import REGISTRY
    from fuzzer_tool.services.fuzzer import Fuzzer

    target = _build(tmp_path, SHA256_K, CRC32)
    with patch("os.access", return_value=True):
        f = Fuzzer(
            target=target,
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=4096,
            timeout=1,
            hlx_algo="auto",
            cmplog=False,  # no FIFO left in ~/.cache
        )
    assert f.crypto_algos == {A.SHA256, A.CRC32}
    assert f.hlx_algo == "sha256"
    assert "hash_len_ext" in REGISTRY.available(f, b"x" * 64)
    assert capsys.readouterr().out.count("Crypto constants:") == 1


def test_fuzzer_auto_without_hash_stays_off(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    target = _build(tmp_path, CRC32)
    with patch("os.access", return_value=True):
        f = Fuzzer(
            target=target,
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=4096,
            timeout=1,
            hlx_algo="auto",
            cmplog=False,  # no FIFO left in ~/.cache
        )
    assert f.hlx_algo is None
