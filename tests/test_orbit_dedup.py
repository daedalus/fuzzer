"""Orbit dedup (algebra handover G1): canonical forms, parser, _is_repeat hook."""

import itertools

import pytest

from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.orbit_dedup import canonical, parse_orbit_spec
from fuzzer_tool.services.fuzzer import Fuzzer

SWAP3 = parse_orbit_spec("swap@2:2x3")
ENDIAN4 = parse_orbit_spec("endian@0:4")
HEAD = b"HD"


class _Stub:
    """Only what _is_repeat touches."""

    def __init__(self, specs):
        self._dedup_execs = True
        self._exec_bloom = BloomFilter(capacity=1000, error_rate=1e-3)
        self.cuckoo_seed_filter = None
        self._cuckoo_recovered: set[str] = set()
        self._orbit_specs = specs
        self._orbit_folded = 0

    _is_repeat = Fuzzer._is_repeat


def test_swap_orbit_has_single_minimum():
    chunks = [b"cc", b"aa", b"bb"]
    forms = {canonical(HEAD + b"".join(p), SWAP3) for p in itertools.permutations(chunks)}

    assert forms == {HEAD + b"aabbcc"}


def test_endian_orbit_is_symmetric():
    fwd = b"\x01\x02\x03\x04rest"

    assert canonical(fwd, ENDIAN4) == canonical(b"\x04\x03\x02\x01rest", ENDIAN4)
    assert canonical(fwd, ENDIAN4) == fwd


def test_canonical_idempotent():
    data = HEAD + b"zzyyxx"

    assert canonical(canonical(data, SWAP3), SWAP3) == canonical(data, SWAP3)


def test_falsification_distinct_multisets_stay_distinct():
    a = HEAD + b"aabbcc"
    b = HEAD + b"aabbcd"

    assert canonical(a, SWAP3) != canonical(b, SWAP3)


def test_falsification_bytes_outside_span_not_folded():
    a = b"XY" + b"bbaacc"
    b = b"XZ" + b"bbaacc"

    assert canonical(a, SWAP3) != canonical(b, SWAP3)


def test_adversarial_short_input_unchanged():
    assert canonical(b"\x01\x02", ENDIAN4) == b"\x01\x02"
    assert canonical(HEAD + b"bbaa", SWAP3) == HEAD + b"bbaa"


def test_adversarial_empty_input():
    assert canonical(b"", SWAP3 + ENDIAN4) == b""


def test_combined_items_fold_independently():
    specs = parse_orbit_spec("endian@0:2,swap@2:1x2")
    x = b"\x02\x01" + b"ba"
    y = b"\x01\x02" + b"ab"

    assert canonical(x, specs) == canonical(y, specs) == b"\x01\x02ab"


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "swap@0:2",  # no count
        "swap@0:2x1",  # count < 2
        "swap@0:0x2",  # size 0
        "swap@0:1x4",  # 24 > MAX_ORBIT
        "endian@0:1",  # too small
        "endian@0:9",  # too large
        "endian@0:4x2",  # count on endian
        "swap@0:2x2,endian@3:2",  # overlap
        "swap@0:1x3,swap@3:1x3,endian@6:2",  # 6*6*2 > MAX_ORBIT
        "rot@0:2",
    ],
)
def test_adversarial_bad_specs_rejected(spec):
    with pytest.raises(ValueError):
        parse_orbit_spec(spec)


def test_max_orbit_boundary():
    at_cap = ",".join(f"endian@{2 * i}:2" for i in range(4))  # 2**4 == MAX_ORBIT

    assert len(parse_orbit_spec(at_cap)) == 4
    with pytest.raises(ValueError):
        parse_orbit_spec(at_cap + ",endian@8:2")


def test_is_repeat_folds_permutation():
    f = _Stub(SWAP3)
    parent = HEAD + b"xxyyzz"
    first = HEAD + b"bbaacc"
    second = HEAD + b"ccaabb"

    assert f._is_repeat(first, parent) is False
    assert f._is_repeat(second, parent) is True
    assert f._orbit_folded == 1


def test_is_repeat_parent_orbit_is_repeat():
    f = _Stub(SWAP3)
    parent = HEAD + b"aabbcc"

    assert f._is_repeat(HEAD + b"ccbbaa", parent) is True


def test_falsification_off_keeps_exact_match():
    f = _Stub([])
    parent = HEAD + b"xxyyzz"

    assert f._is_repeat(HEAD + b"bbaacc", parent) is False
    assert f._is_repeat(HEAD + b"ccaabb", parent) is False
    assert f._orbit_folded == 0
