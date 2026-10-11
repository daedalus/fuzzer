"""Tests for core/byte_hash_mitm.py and ByteHashEncoder (FNV-1, djb2, sdbm, OAAT).

Reference hashes live here, written independently of the module under test.
"""

import pytest

from fuzzer_tool.core.byte_hash_mitm import SPECS, ByteHashSpec, preimages
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.rq_encodings import (
    BUILTIN_ENCODERS,
    ByteHashEncoder,
    Fnv1aEncoder,
    _fnv1a,
    _fnv1a_invert_all,
    generate_mutations,
)

M = 0xFFFFFFFF
SEED = 7


def _fnv1(d: bytes) -> int:
    h = 2166136261
    for b in d:
        h = (h * 16777619) & M
        h ^= b
    return h


def _djb2(d: bytes) -> int:
    h = 5381
    for b in d:
        h = (h * 33 + b) & M
    return h


def _sdbm(d: bytes) -> int:
    h = 0
    for b in d:
        h = (b + (h << 6) + (h << 16) - h) & M
    return h


def _oaat(d: bytes) -> int:
    h = 0
    for b in d:
        h = (h + b) & M
        h = (h + (h << 10)) & M
        h ^= h >> 6
    h = (h + (h << 3)) & M
    h ^= h >> 11
    return (h + (h << 15)) & M


REF = {"fnv1": _fnv1, "djb2": _djb2, "sdbm": _sdbm, "oaat": _oaat}
EDGE = [b"\x00" * 4, b"\xff" * 4, b"crsh"]


def _le(h: int) -> bytes:
    return h.to_bytes(4, "little")


@pytest.mark.parametrize("name", REF)
class TestPreimages:
    def test_roundtrip_contains_source_and_all_hash_to_target(self, name):
        rng = RandPool(seed=SEED)
        for x in EDGE + [rng.randbytes(4) for _ in range(20)]:
            pres = preimages(SPECS[name], REF[name](x))
            assert x in pres
            assert all(REF[name](p) == REF[name](x) for p in pres)

    def test_falsification_wrong_family_does_not_invert(self, name):
        # A hash value of another family must not yield preimages of this one.
        other = next(n for n in REF if n != name)
        x = b"crsh"
        pres = preimages(SPECS[name], REF[other](x))
        assert all(REF[name](p) == REF[other](x) for p in pres)
        assert x not in pres or REF[name](x) == REF[other](x)


def test_matches_existing_fnv1a_inverter():
    # Spec-driven engine reproduces the shipped FNV-1a inverter byte for byte.
    inv = pow(16777619, -1, 1 << 32)
    spec = ByteHashSpec(
        "fnv1a_ref",
        2166136261,
        lambda h, c: ((h ^ c) * 16777619) & 0xFFFFFFFF,
        lambda h, c: ((h * inv) & 0xFFFFFFFF) ^ c,
        lambda h: h,
    )
    rng = RandPool(seed=SEED)
    for x in EDGE + [rng.randbytes(4) for _ in range(10)]:
        t = _fnv1a(x)
        assert sorted(preimages(spec, t)) == sorted(_fnv1a_invert_all(t))


class TestEncoder:
    @pytest.mark.parametrize("name", REF)
    def test_solves_hashed_compare(self, name):
        garbage, want = b"xxxx", b"crsh"
        data = b"AA" + garbage + b"BB"
        muts = generate_mutations(_le(REF[name](garbage)), _le(REF[name](want)), 32, "CMP", data)
        hits = [m for m in muts if m[2].name() == f"{name}_p"]
        assert any(m[0] == (2,) and m[1] == (want,) for m in hits)

    def test_registered_without_duplicating_fnv1a(self):
        names = [e.name() for e in BUILTIN_ENCODERS]
        assert len(names) == len(set(names))
        assert sum(isinstance(e, Fnv1aEncoder) for e in BUILTIN_ENCODERS) == 2
        assert sum(isinstance(e, ByteHashEncoder) for e in BUILTIN_ENCODERS) == 2 * len(SPECS)

    def test_adversarial_filters(self):
        enc = ByteHashEncoder(SPECS["djb2"])
        big = (1 << 28).to_bytes(4, "little")
        assert not enc.is_applicable(32, "CMP", (12).to_bytes(4, "little"), (13).to_bytes(4, "little"))
        assert not enc.is_applicable(32, "CMP", (12).to_bytes(4, "little"), big)
        assert not enc.is_applicable(64, "CMP", b"\x01" * 8, b"\x02" * 8)
        assert not enc.is_applicable(32, "STR", big, big)
        assert enc.is_applicable(32, "CMP", big, big)

    def test_reversed_field(self):
        enc = ByteHashEncoder(SPECS["sdbm"], reverse=True)
        assert enc.encode(_le(_sdbm(b"crsh"))).count(b"hsrc") == 1
