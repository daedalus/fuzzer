"""core/hashing.py: in-house XXH64 (persisted keys) and hash64 (in-memory keys).

Vectors come from the reference xxhash library, generated once; the
package is not a dependency.
"""

import pytest

from fuzzer_tool.core.hashing import hash64, hash64_seeded, key16, xxh64

_U64 = (1 << 64) - 1
_SEED = 0x9E3779B1
_LENS = [0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, 1000]


def _blob(n: int) -> bytes:
    return bytes((i * 7 + 3) % 256 for i in range(n))


_XXH64 = dict(
    zip(
        _LENS,
        [
            0xEF46DB3751D8E999,
            0x1F25C8D0BC1F4BB6,
            0x31D2363F52E564C9,
            0x9BB64B7D66EE9FDA,
            0x9A7B149959CE60D8,
            0xDAB99D95C6F90092,
            0x1B47CB8243CC8E32,
            0x434850232B787BE2,
            0xA2AA5F33CC4A6119,
            0x23C3C17EF790FD97,
            0x50A7CFC7BA588784,
            0x5E3E54B431C7493C,
            0x0EB64B3EF6EEB01F,
            0xA383B724B2BD12F1,
            0xA61F8D4C170FE531,
            0x5F235FA033F1A3FB,
        ],
        strict=True,
    )
)
_XXH64_SEEDED = dict(
    zip(
        _LENS,
        [
            0xAC75FDA2929B17EF,
            0x79CF94F5ACE37FE1,
            0x96D2A5E583DDD738,
            0x4A862FBB4B73776E,
            0xDDC5CF71D37A6368,
            0xC6E75E2CBD8E07EE,
            0x8120A1EDA8AC77DA,
            0x17F387CE30885DE5,
            0x381822F33376D6B6,
            0x9ABD315470AC276A,
            0x72CDD3262F467933,
            0x634DA7D265979E61,
            0xBCB3847D70525180,
            0x97F9D8F996719021,
            0x86F549EDE5AC87E2,
            0x83080310EE83CC20,
        ],
        strict=True,
    )
)


@pytest.mark.parametrize("n", _LENS)
def test_xxh64_vectors(n):
    """Every stripe/tail boundary: <4, 4-7, 8-31, 32+ bytes."""
    assert xxh64(_blob(n)) == _XXH64[n]


@pytest.mark.parametrize("n", _LENS)
def test_xxh64_seeded_vectors(n):
    assert xxh64(_blob(n), _SEED) == _XXH64_SEEDED[n]


def test_xxh64_text_vector():
    assert xxh64(b"Nobody inspects the spammish repetition") == 0xFBCEA83C8A378BF1


def test_xxh64_accepts_buffers():
    data = _blob(100)
    want = xxh64(data)
    assert xxh64(bytearray(data)) == want
    assert xxh64(memoryview(data)) == want


def test_key16_is_xxh64_hex_prefix():
    """Persisted corpus names and seed_meta keys must not change."""
    assert key16(b"") == "ef46db3751d8e999"
    assert key16(_blob(33)) == "50a7cfc7ba588784"


def test_key16_zero_padded():
    """0x0EB6... must keep its leading zero: 16 chars, not 15."""
    key = key16(_blob(64))

    assert key == "0eb64b3ef6eeb01f"
    assert len(key) == 16


def test_hash64_range_and_determinism():
    data = _blob(500)

    assert hash64(data) == hash64(data)
    assert 0 <= hash64(data) <= _U64
    assert hash64(b"") == hash64(b"")


def test_hash64_stable_across_processes():
    """Persisted pickles key on it: pinned values, no per-process salt."""
    assert hash64(b"abc") == 0x024D0127352441C2
    assert hash64(b"") == 0x1_0000_0000


def test_hash64_accepts_buffers():
    data = _blob(100)

    assert hash64(bytearray(data)) == hash64(data)
    assert hash64(memoryview(data)) == hash64(data)


def test_hash64_seed_changes_digest():
    data = _blob(64)

    assert hash64_seeded(data, 1) != hash64_seeded(data, 2)
    assert hash64_seeded(data, 1) == hash64_seeded(data, 1)


def test_hash64_seed_is_not_concatenation():
    """Data still matters under a seed; a negative seed wraps to 64 bits."""
    assert hash64_seeded(b"ab", 1) != hash64_seeded(b"a", 1)
    assert hash64_seeded(b"a", -1) == hash64_seeded(b"a", _U64)
    assert hash64_seeded(b"a", -1) != hash64_seeded(b"a", 1)


def test_hash64_every_single_bit_flip_distinct():
    """Adversarial: a 512-byte buffer and all 4096 one-bit mutants collide nowhere.

    The per-op "did the buffer change" check hangs on this property.
    """
    base = bytearray(_blob(512))
    seen = {hash64(base)}
    for byte in range(len(base)):
        for bit in range(8):
            base[byte] ^= 1 << bit
            seen.add(hash64(base))
            base[byte] ^= 1 << bit

    assert len(seen) == 1 + 512 * 8


def test_hash64_length_extension_distinct():
    """Zero padding changes the digest (trailing NULs are not ignored)."""
    assert len({hash64(b"\x00" * n) for n in range(64)}) == 64


def test_xxh64_rejects_str():
    with pytest.raises(TypeError):
        xxh64("abc")  # type: ignore[arg-type]


def test_hash64_len_separates_equal_digests(monkeypatch):
    """Equal 32-bit-weak digests, different lengths -> different keys."""
    import fuzzer_tool.core.hashing as h

    monkeypatch.setattr(h, "hash64", lambda data: 7)
    assert h.hash64_len(b"a" * 10) != h.hash64_len(b"b" * 11)


def test_hash64_len_same_input_same_key():
    from fuzzer_tool.core.hashing import hash64_len

    assert hash64_len(_blob(100)) == hash64_len(bytes(_blob(100)))


def test_hash64_len_empty_and_memoryview():
    from fuzzer_tool.core.hashing import hash64_len

    assert hash64_len(b"") == hash64_len(memoryview(b""))
    assert hash64_len(b"") != hash64_len(b"\0")


def test_length_keyed_table_no_merge_on_collision(monkeypatch):
    """A colliding digest must not hand a seed another length's bins."""
    import fuzzer_tool.core.hashing as h
    from fuzzer_tool.core.schedulers._bin_rates import BinRates
    from tests.test_pos_bin_rates import ScriptedRng

    monkeypatch.setattr(h, "hash64", lambda data: 7)
    rates = BinRates(ScriptedRng(), 1.0, 1.0)
    rates.credit(b"a" * 100, [3], 1.0)

    assert rates.width(b"b" * 90_000) is None
    assert rates.counts(b"b" * 90_000) is None


def _key_ref(data) -> int:
    """hash64_len derived from zlib directly, independent of the memo."""
    from zlib import adler32, crc32

    return crc32(data) | (adler32(data) << 32) | (len(data) << 64)


def test_regression_hash64_len_memo_reuses_same_object(monkeypatch):
    """Schedulers key the same parent object back to back: digest it once."""
    import fuzzer_tool.core.hashing as h

    calls = []
    real = h.hash64
    monkeypatch.setattr(h, "hash64", lambda d: calls.append(1) or real(d))
    parent = bytes(_blob(300))

    keys = [h.hash64_len(parent) for _ in range(4)]

    assert keys == [_key_ref(parent)] * 4
    assert len(calls) == 1


def test_regression_hash64_len_memo_misses_on_other_object():
    """Falsification: a new object, even one equal in length, is rehashed."""
    from fuzzer_tool.core.hashing import hash64_len

    a, b = bytes(_blob(64)), bytes(reversed(_blob(64)))
    hash64_len(a)

    assert hash64_len(b) == _key_ref(b) != _key_ref(a)
    assert hash64_len(bytes(a)) == _key_ref(a)


@pytest.mark.parametrize("view", [bytearray, lambda b: memoryview(bytearray(b))])
def test_regression_hash64_len_memo_skips_mutable(view):
    """Adversarial: a buffer edited in place between calls keeps its identity,
    so a memo keyed on identity would return the stale key."""
    from fuzzer_tool.core.hashing import hash64_len

    buf = view(_blob(32))
    before = hash64_len(buf)
    buf[0] ^= 0xFF

    assert hash64_len(buf) == _key_ref(buf) != before


def test_regression_hash64_len_memo_holds_one_object():
    """Bounded: the memo releases a parent once another one is keyed."""
    import sys

    from fuzzer_tool.core.hashing import hash64_len

    big = bytes(1 << 16)
    base = sys.getrefcount(big)
    hash64_len(big)
    hash64_len(bytes(_blob(8)))

    assert sys.getrefcount(big) == base
