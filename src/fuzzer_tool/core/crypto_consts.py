"""Crypto-constant sniffer: which hash/cipher tables a target carries.

Scans the target's data sections (``elf.word_section_spans``: rodata, data,
relro) for published algorithm constants, little- and big-endian::

    .rodata: ... 98 2f 8a 42 91 44 37 71 cf fb c0 b5 a5 db b5 e9 ...
                 ^ SHA-256 K[0..3], LE                -> CryptoAlgo.SHA256

Tables match on several consecutive words (>= 96 bits), so a chance hit in
random data is ~2**-96 per offset. Single-word or short signatures (TEA
delta, AES S-box head) are weaker; see ``_SIGS``. Shared IVs are split by a
second table:

    SHA-256 IV + BLAKE2 sigma -> BLAKE2S     SHA-256 IV alone -> SHA256
    SHA-512 IV + BLAKE2 sigma -> BLAKE2B     SHA-512 IV alone -> SHA512
    MD5 IV not followed by SHA-1's 5th word  -> MD5

``resolve_hlx`` turns ``--hlx-algo auto`` into the detected Merkle-Damgard
hash, which arms ``hash_len_ext``.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable
from typing import NamedTuple

from fuzzer_tool.core.elf import word_section_spans

log = logging.getLogger(__name__)


class CryptoAlgo(enum.Enum):
    TEA = "tea"  # TEA/XTEA delta
    CHACHA = "chacha"  # Salsa20/ChaCha sigma/tau
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"
    SHA512 = "sha512"
    KECCAK = "keccak"  # SHA-3 / SHAKE
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    CRC32 = "crc32"
    AES = "aes"


class _Mark(enum.Enum):
    """Raw signature hits, before IV disambiguation."""

    TEA_DELTA = enum.auto()
    CHACHA_SIGMA = enum.auto()
    CHACHA_TAU = enum.auto()
    MD5_T = enum.auto()
    MD5_IV = enum.auto()
    SHA1_IV = enum.auto()
    SHA256_K = enum.auto()
    SHA256_IV = enum.auto()
    SHA512_K = enum.auto()
    SHA512_IV = enum.auto()
    KECCAK_RC = enum.auto()
    BLAKE2_SIGMA = enum.auto()
    CRC32_TABLE = enum.auto()
    AES_SBOX = enum.auto()


_U32 = 4
_U64 = 8

# Published constants (spec values). Tables are truncated to their head.
TEA_DELTA = 0x9E3779B9  # floor(2**32 / golden ratio)
CHACHA_SIGMA = b"expand 32-byte k"  # RFC 7539
CHACHA_TAU = b"expand 16-byte k"  # Salsa20 16-byte key
MD5_T_HEAD = (0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE)  # RFC 1321
MD5_IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476)  # RFC 1321
SHA1_IV = MD5_IV + (0xC3D2E1F0,)  # FIPS 180-4 5.3.1
SHA256_K_HEAD = (0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5)  # FIPS 180-4 4.2.2
SHA256_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A)  # = BLAKE2s IV[0..3]
SHA512_K_HEAD = (
    0x428A2F98D728AE22,
    0x7137449123EF65CD,
    0xB5C0FBCFEC4D3B2F,
    0xE9B5DBA58189DBBC,
)  # FIPS 180-4 4.2.3
SHA512_IV = (
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
)  # FIPS 180-4 5.3.5, = BLAKE2b IV[0..3]
KECCAK_RC_HEAD = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
)  # Keccak-f[1600] iota
BLAKE2_SIGMA_HEAD = bytes(range(16)) + bytes(
    (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3)
)  # RFC 7693 sigma rows 0-1, uint8
CRC32_TABLE_HEAD = (0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA)  # reflected 0xEDB88320
AES_SBOX_HEAD = bytes.fromhex("637c777bf26b6fc53001672bfed7ab76")  # FIPS 197 row 0


def _words(values: tuple[int, ...], width: int) -> tuple[bytes, ...]:
    """LE and BE byte layouts of a word table (deduped)."""
    le = b"".join(v.to_bytes(width, "little") for v in values)
    be = b"".join(v.to_bytes(width, "big") for v in values)
    return tuple(dict.fromkeys((le, be)))


def _swap32(raw: bytes) -> bytes:
    """A byte string read as BE u32 words (ChaCha sigma on a BE host)."""
    return b"".join(raw[i : i + _U32][::-1] for i in range(0, len(raw), _U32))


# Mark -> needles; any needle hit sets the mark.
_SIGS: dict[_Mark, tuple[bytes, ...]] = {
    _Mark.TEA_DELTA: _words((TEA_DELTA,), _U32),  # 32 bits: weakest signature
    _Mark.CHACHA_SIGMA: (CHACHA_SIGMA, _swap32(CHACHA_SIGMA)),
    _Mark.CHACHA_TAU: (CHACHA_TAU, _swap32(CHACHA_TAU)),
    _Mark.MD5_T: _words(MD5_T_HEAD, _U32),
    _Mark.MD5_IV: _words(MD5_IV, _U32),
    _Mark.SHA1_IV: _words(SHA1_IV, _U32),
    _Mark.SHA256_K: _words(SHA256_K_HEAD, _U32),
    _Mark.SHA256_IV: _words(SHA256_IV, _U32),
    _Mark.SHA512_K: _words(SHA512_K_HEAD, _U64),
    _Mark.SHA512_IV: _words(SHA512_IV, _U64),
    _Mark.KECCAK_RC: _words(KECCAK_RC_HEAD, _U64),
    _Mark.BLAKE2_SIGMA: (BLAKE2_SIGMA_HEAD,),
    _Mark.CRC32_TABLE: _words(CRC32_TABLE_HEAD, _U32),
    _Mark.AES_SBOX: (AES_SBOX_HEAD,),
}

# Marks that name one algorithm outright.
_DIRECT = {
    _Mark.TEA_DELTA: CryptoAlgo.TEA,
    _Mark.CHACHA_SIGMA: CryptoAlgo.CHACHA,
    _Mark.CHACHA_TAU: CryptoAlgo.CHACHA,
    _Mark.MD5_T: CryptoAlgo.MD5,
    _Mark.SHA1_IV: CryptoAlgo.SHA1,
    _Mark.SHA256_K: CryptoAlgo.SHA256,
    _Mark.SHA512_K: CryptoAlgo.SHA512,
    _Mark.KECCAK_RC: CryptoAlgo.KECCAK,
    _Mark.CRC32_TABLE: CryptoAlgo.CRC32,
    _Mark.AES_SBOX: CryptoAlgo.AES,
}


class _SharedIv(NamedTuple):
    iv: _Mark
    blake: CryptoAlgo  # IV plus BLAKE2 sigma
    sha: CryptoAlgo  # IV alone


_SHARED_IVS = (
    _SharedIv(_Mark.SHA256_IV, CryptoAlgo.BLAKE2S, CryptoAlgo.SHA256),
    _SharedIv(_Mark.SHA512_IV, CryptoAlgo.BLAKE2B, CryptoAlgo.SHA512),
)

HLX_AUTO = "auto"
"""``--hlx-algo auto``: pick the length-extension hash from the target."""

# Strongest first: a binary with SHA-256 and MD5 most likely MACs with SHA-256.
_HLX_ORDER = (
    (CryptoAlgo.SHA256, "sha256"),
    (CryptoAlgo.SHA1, "sha1"),
    (CryptoAlgo.MD5, "md5"),
)


def _count(elf: bytes, spans: list[tuple[int, int]], needle: bytes) -> int:
    return sum(elf.count(needle, lo, hi) for lo, hi in spans)


def _marks(elf: bytes, spans: list[tuple[int, int]]) -> set[_Mark]:
    """Every signature with a needle inside some span (``bytes.find``, no copies)."""
    found = set()
    for mark, needles in _SIGS.items():
        if any(elf.find(nd, lo, hi) != -1 for nd in needles for lo, hi in spans):
            found.add(mark)

    # MD5's IV is SHA-1's first four words: count only the runs SHA-1 lacks.
    if _Mark.MD5_IV in found:
        md5 = sum(_count(elf, spans, nd) for nd in _SIGS[_Mark.MD5_IV])
        sha1 = sum(_count(elf, spans, nd) for nd in _SIGS[_Mark.SHA1_IV])
        if md5 <= sha1:
            found.discard(_Mark.MD5_IV)
    return found


def _classify(marks: set[_Mark]) -> frozenset[CryptoAlgo]:
    algos = {_DIRECT[m] for m in marks if m in _DIRECT}
    if _Mark.MD5_IV in marks:
        algos.add(CryptoAlgo.MD5)

    blake = _Mark.BLAKE2_SIGMA in marks
    for shared in _SHARED_IVS:
        if shared.iv not in marks:
            continue
        algos.add(shared.blake if blake else shared.sha)
    return frozenset(algos)


def sniff_elf(elf: bytes) -> frozenset[CryptoAlgo]:
    """Algorithms whose constants appear in the ELF image's data sections.

    A corrupt or non-ELF image yields an empty set (the section walk declines).
    """
    spans = word_section_spans(elf)
    if not spans:
        return frozenset()
    return _classify(_marks(elf, spans))


def sniff_target(target: str) -> frozenset[CryptoAlgo]:
    """``sniff_elf`` on a binary on disk; empty if unreadable."""
    try:
        with open(target, "rb") as f:
            elf = f.read()
    except OSError as exc:
        log.debug("crypto sniff: cannot read %s: %s", target, exc)
        return frozenset()
    return sniff_elf(elf)


def resolve_hlx(choice: str | None, algos: Iterable[CryptoAlgo]) -> str | None:
    """Concrete ``--hlx-algo``: ``auto`` -> strongest detected MD hash, else unchanged.

    Example: ``resolve_hlx("auto", {SHA1, MD5}) == "sha1"``; none found -> None.
    """
    if choice != HLX_AUTO:
        return choice
    present = set(algos)
    for algo, name in _HLX_ORDER:
        if algo in present:
            return name
    return None


def describe(algos: Iterable[CryptoAlgo]) -> str:
    """Stable, comma-joined names for the startup line, e.g. ``crc32, sha256``."""
    return ", ".join(sorted(a.value for a in algos))
