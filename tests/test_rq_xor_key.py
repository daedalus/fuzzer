"""Redqueen I2S through an unknown XOR key: ``XorKeyEncoder`` (core/rq_encodings.py).

    input  x = "MAGI" ^ K  --deobfuscate-->  "MAGI"  ==  "WXYZ"   (cmplog operands)
    K cancels in x[i] ^ x[i+p], so search lag_p("MAGI") in lag_p(input),
    read K = chunk ^ "MAGI", write "WXYZ" ^ K.

Expected bytes come from ``_xor``/``_oracle`` below, not from the module.
"""

import pytest

from fuzzer_tool.core.rq_encodings import XorKeyStage, generate_mutations, set_xor_key_stage

_STR_A, _STR_B = b"MAGIC_OK", b"OPENSESA"
_PAD = b"\x10\x32\x54\x76\x98\xba\xdc\xfe\x01\x23"


@pytest.fixture(autouse=True)
def _armed():
    """Arm for the test, disarm after: the stage is module state."""
    set_xor_key_stage(XorKeyStage.ON)
    yield
    set_xor_key_stage(XorKeyStage.OFF)


def _xor(data: bytes, key: bytes, phase: int = 0) -> bytes:
    """Reference XOR with a repeating key starting at key[phase]."""
    p = len(key)
    return bytes(b ^ key[(i + phase) % p] for i, b in enumerate(data))


def _apply(data: bytes, mutation) -> bytes:
    offsets, repls, _enc = mutation
    buf = bytearray(data)
    for off, chunk in zip(offsets, repls, strict=True):
        buf[off : off + len(chunk)] = chunk
    return bytes(buf)


def _xor_hits(op_a, op_b, size, kind, data, **kw):
    muts = generate_mutations(op_a, op_b, size, kind, data, **kw)
    return [m for m in muts if m[2].name().startswith("xor_key")]


def _oracle(op_a: bytes, data: bytes, max_p: int, min_inv: int) -> set[int]:
    """Brute force: offsets where chunk ^ op_a is a non-zero periodic key."""
    n = len(op_a)
    found = set()
    for off in range(len(data) - n + 1):
        key = bytes(c ^ a for c, a in zip(data[off : off + n], op_a, strict=True))
        if not any(key):
            continue
        for p in range(1, max_p + 1):
            if n - p < min_inv:
                break
            if all(op_a[i] == op_a[i + p] for i in range(n - p)):
                continue  # operand itself p-periodic: rejected by design
            if all(key[i] == key[i + p] for i in range(n - p)):
                found.add(off)
                break
    return found


@pytest.mark.parametrize("key", [b"\x5a", b"\x13\x37", b"\xde\xad\xbe\xef"])
def test_solves_xored_string_compare(key):
    # Falsification: planted at an odd offset so the key phase is rotated.
    off = 3
    data = _PAD[:off] + _xor(_STR_A, key, off) + _PAD
    hits = _xor_hits(_STR_A, _STR_B, 64, "STR", data)

    want = _PAD[:off] + _xor(_STR_B, key, off) + _PAD
    assert want in [_apply(data, m) for m in hits]


def test_solves_reversed_integer_compare():
    # Target reads a big-endian u32, XORs it, compares as a LE integer.
    a, b, key = 0x41424344, 0x0BADF00D, b"\xa5"
    data = b"\x00\x01" + _xor(a.to_bytes(4, "big"), key) + b"\x02"
    hits = _xor_hits(a.to_bytes(4, "little"), b.to_bytes(4, "little"), 32, "CMP", data)

    assert _xor(b.to_bytes(4, "big"), key) in [m[1][0] for m in hits]


def test_longer_replacement_extends_key_periodically():
    key = b"\x01\x02"
    data = _xor(b"ABCDE", key) + bytes(8)
    hits = _xor_hits(b"ABCDE", b"ABCDEFG", 64, "STR", data)

    assert _xor(b"ABCDEFG", key) in [m[1][0] for m in hits]


def test_matches_brute_force_oracle():
    # Several planted keys plus filler; offsets must match brute force exactly.
    op_a = b"\x11\x22\x33\x44\x55"
    data = (
        b"\x00\x09\x07"
        + _xor(op_a, b"\x80")
        + b"\x61\x62\x63\x64"
        + _xor(op_a, b"\x01\x02", 1)
        + b"\xff\xfe"
        + op_a
        + b"\x10\x20\x30"
    )
    # Control: the oracle must see the two keyed copies and skip the plain one.
    expected = _oracle(op_a, data, max_p=8, min_inv=3)
    assert {3, 12} <= expected
    assert data.find(op_a) not in expected

    hits = _xor_hits(op_a, b"\x99\x88\x77\x66\x55", 512, "STR", data)
    assert {m[0][0] for m in hits} == expected


def test_regression_plain_match_left_to_plain_encoder():
    # Adversarial: key 0 is the plain I2S case; no xor_key duplicate.
    data = b"\x00" + _STR_A + b"\x00"
    assert _xor_hits(_STR_A, _STR_B, 64, "STR", data) == []


def test_periodic_operand_rejected():
    # Adversarial: "AAAA" ^ K matches every byte run; must not flood.
    data = bytes(range(64)) + b"\x07" * 32
    assert _xor_hits(b"AAAA", b"BBBB", 32, "CMP", data) == []


def test_short_operand_skipped():
    # 2-byte operand: 1 invariant byte matches 1/256 offsets; too weak.
    data = bytes(range(256))
    assert _xor_hits(b"\x01\x02", b"\x03\x04", 16, "CMP", data) == []


def test_hash_like_pair_skipped():
    data = _xor(_STR_A, b"\x5a") + _PAD
    hits = _xor_hits(_STR_A, _STR_B, 64, "STR", data, is_hash=lambda _a, _b: True)
    assert hits == []


def test_regression_input_cache_not_stale():
    # Adversarial: same length, different content; lag cache keyed by identity.
    key = b"\x5a"
    hit = bytes(_xor(_STR_A, key) + _PAD)
    miss = bytes(len(hit))
    assert _xor_hits(_STR_A, _STR_B, 64, "STR", hit)
    assert _xor_hits(_STR_A, _STR_B, 64, "STR", miss) == []


def test_disarmed_no_hit():
    data = _xor(_STR_A, b"\x5a") + _PAD
    assert _xor_hits(_STR_A, _STR_B, 64, "STR", data)

    set_xor_key_stage(XorKeyStage.OFF)
    assert _xor_hits(_STR_A, _STR_B, 64, "STR", data) == []


@pytest.mark.parametrize("flag", [False, True])
def test_fuzzer_flag_sets_stage(tmp_path, flag):
    from fuzzer_tool.core import rq_encodings
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "corpus", tmp_path / "crashes"
    corpus.mkdir()
    crashes.mkdir()
    # Start from the opposite state: construction must override it.
    set_xor_key_stage(XorKeyStage.OFF if flag else XorKeyStage.ON)
    Fuzzer(target="/bin/true", corpus_dir=str(corpus), crashes_dir=str(crashes), rq_xor_key=flag)
    assert (rq_encodings._xor_stage is XorKeyStage.ON) is flag


def test_cli_flag_reaches_fuzzer():
    from fuzzer_tool.cli import commands

    assert "rq_xor_key" in commands._HAIL_MARY_FLAGS
