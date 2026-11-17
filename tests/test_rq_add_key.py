"""Redqueen I2S through an unknown additive key: ``AddKeyEncoder`` (core/rq_encodings.py).

    input  x = "MAGI" + K (mod 256 per byte)  --decode-->  "MAGI" == "WXYZ"
    K cancels in x[i+p] - x[i], so search that lag of "MAGI" in the input's,
    read K = chunk - "MAGI", write "WXYZ" + K.   (AFL++ rtn "arith" transform)

Expected bytes come from ``_add``/``_oracle`` below, not from the module.
"""

import pytest

from fuzzer_tool.core.rq_encodings import AddKeyStage, generate_mutations, set_add_key_stage

_STR_A, _STR_B = b"MAGIC_OK", b"OPENSESA"
_PAD = b"\x10\x32\x54\x76\x98\xba\xdc\xfe\x01\x23"


@pytest.fixture(autouse=True)
def _armed():
    """Arm for the test, disarm after: the stage is module state."""
    set_add_key_stage(AddKeyStage.ON)
    yield
    set_add_key_stage(AddKeyStage.OFF)


def _add(data: bytes, key: bytes, phase: int = 0) -> bytes:
    """Reference: add a repeating key mod 256, starting at key[phase]."""
    p = len(key)
    return bytes((b + key[(i + phase) % p]) & 0xFF for i, b in enumerate(data))


def _apply(data: bytes, mutation) -> bytes:
    offsets, repls, _enc = mutation
    buf = bytearray(data)
    for off, chunk in zip(offsets, repls, strict=True):
        buf[off : off + len(chunk)] = chunk
    return bytes(buf)


def _add_hits(op_a, op_b, size, kind, data):
    muts = generate_mutations(op_a, op_b, size, kind, data)
    return [m for m in muts if m[2].name().startswith("add_key")]


def _oracle(op_a: bytes, data: bytes, max_p: int, min_inv: int) -> set[int]:
    """Brute force: offsets where chunk - op_a is a non-zero periodic key."""
    n = len(op_a)
    found = set()
    for off in range(len(data) - n + 1):
        key = bytes((c - a) & 0xFF for c, a in zip(data[off : off + n], op_a, strict=True))
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


@pytest.mark.parametrize("key", [b"\x0d", b"\x13\x37", b"\xde\xad\xbe\xef"])
def test_solves_shifted_string_compare(key):
    # Falsification: planted at an odd offset so the key phase is rotated.
    off = 3
    data = _PAD[:off] + _add(_STR_A, key, off) + _PAD
    hits = _add_hits(_STR_A, _STR_B, 64, "STR", data)

    want = _PAD[:off] + _add(_STR_B, key, off) + _PAD
    assert want in [_apply(data, m) for m in hits]


def test_rot13_like_digits():
    """`c - '0'` style decode: input digits are operand bytes + 0x30."""
    op_a, op_b = bytes([1, 2, 3, 4]), bytes([9, 8, 7, 6])
    data = b"x=" + _add(op_a, b"\x30") + b";"
    hits = _add_hits(op_a, op_b, 32, "CMP", data)
    assert _add(op_b, b"\x30") in [m[1][0] for m in hits]


def test_matches_brute_force_oracle():
    op_a = b"\x11\x22\x33\x44\x55"
    data = (
        b"\x00\x09\x07"
        + _add(op_a, b"\x80")
        + b"\x61\x62\x63\x64"
        + _add(op_a, b"\x01\x02", 1)
        + b"\xff\xfe"
        + op_a
        + b"\x10\x20\x30"
    )
    # Control: the oracle sees both keyed copies and skips the plain one.
    expected = _oracle(op_a, data, max_p=8, min_inv=3)
    assert {3, 12} <= expected
    assert data.find(op_a) not in expected

    hits = _add_hits(op_a, b"\x99\x88\x77\x66\x55", 512, "STR", data)
    assert {m[0][0] for m in hits} == expected


def test_wraps_mod_256():
    """Adversarial: key and replacement both wrap past 0xFF."""
    op_a, op_b, key = b"\xf0\xf1\xf2\xf3", b"\xfe\xfd\xfc\xfb", b"\x20"
    data = b"\x00" + _add(op_a, key) + b"\x00"
    hits = _add_hits(op_a, op_b, 32, "CMP", data)
    assert _add(op_b, key) in [m[1][0] for m in hits]


def test_periodic_operand_rejected():
    """Adversarial: a constant operand would match every constant run."""
    data = b"\x05" * 32
    assert _add_hits(b"AAAA", b"BBBB", 32, "CMP", data) == []


def test_disarmed_no_hit():
    set_add_key_stage(AddKeyStage.OFF)
    data = _PAD[:3] + _add(_STR_A, b"\x13\x37", 3) + _PAD
    assert _add_hits(_STR_A, _STR_B, 64, "STR", data) == []


def test_cli_flag(monkeypatch):
    from tests.test_lightweight_mode import _cli

    args, _ = _cli(monkeypatch, "--rq-add-key")
    assert args.rq_add_key is True


@pytest.mark.parametrize("flag", [False, True])
def test_fuzzer_flag_sets_stage(tmp_path, flag):
    from fuzzer_tool.core import rq_encodings
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "corpus", tmp_path / "crashes"
    corpus.mkdir()
    crashes.mkdir()
    # Start from the opposite state: construction must override it.
    set_add_key_stage(AddKeyStage.OFF if flag else AddKeyStage.ON)
    Fuzzer(target="/bin/true", corpus_dir=str(corpus), crashes_dir=str(crashes), rq_add_key=flag)
    armed = rq_encodings._add_stage_fn is rq_encodings._add_key_mutations
    assert armed is flag
