"""Orbit dedup ``rot@OFF:SIZExCOUNT``: ring buffers equal under cyclic shift fold to one."""

import pytest

from fuzzer_tool.core.orbit_dedup import MAX_ORBIT, OrbitKind, canonical, parse_orbit_spec

HEAD = b"RG"


def _booth(seq: list) -> int:
    """Booth's least-rotation index: independent oracle for the shipped rule."""
    ss = seq + seq
    fail = [-1] * len(ss)
    k = 0
    for j in range(1, len(ss)):
        i = fail[j - k - 1]
        while i != -1 and ss[j] != ss[k + i + 1]:
            if ss[j] < ss[k + i + 1]:
                k = j - i - 1
            i = fail[i]
        if i == -1 and ss[j] != ss[k + i + 1]:
            if ss[j] < ss[k + i + 1]:
                k = j
            fail[j - k] = -1
        else:
            fail[j - k] = i + 1
    return k


def _chunks(span: bytes, size: int) -> list[bytes]:
    return [span[i : i + size] for i in range(0, len(span), size)]


def _rotate(chunks: list[bytes], r: int) -> bytes:
    return b"".join(chunks[r:] + chunks[:r])


def test_parse_rot_item():
    (spec,) = parse_orbit_spec("rot@2:4x3")
    assert (spec.kind, spec.offset, spec.size, spec.count) == (OrbitKind.ROT, 2, 4, 3)
    assert spec.orbit_size == 3


def test_every_rotation_has_one_canonical_form():
    chunks = [b"cc", b"aa", b"bb", b"aa"]
    specs = parse_orbit_spec("rot@2:2x4")
    forms = {canonical(HEAD + _rotate(chunks, r), specs) for r in range(len(chunks))}

    k = _booth(chunks)
    assert forms == {HEAD + _rotate(chunks, k)}


def test_matches_booth_on_periodic_ring():
    # Periodic rings hit Booth's failure-function branches.
    chunks = _chunks(b"abab" * 3 + b"aa", 1)[:16]
    specs = parse_orbit_spec(f"rot@0:1x{len(chunks)}")
    assert canonical(b"".join(chunks), specs) == _rotate(chunks, _booth(chunks))


def test_control_booth_against_itself():
    # Hard Rule 46: the oracle must agree with a second run of itself.
    chunks = [b"\x03", b"\x01", b"\x02", b"\x01"]
    assert _rotate(chunks, _booth(chunks)) == _rotate(chunks, _booth(list(chunks)))


def test_falsification_permutation_is_not_folded():
    # A non-rotation reordering stays distinct: rot is not swap.
    specs = parse_orbit_spec("rot@0:1x3")
    assert canonical(b"abc", specs) != canonical(b"acb", specs)


def test_adversarial_short_input_unchanged():
    specs = parse_orbit_spec("rot@4:4x4")
    data = b"tooshort"
    assert canonical(data, specs) is data


def test_adversarial_orbit_cap_and_count():
    with pytest.raises(ValueError):
        parse_orbit_spec(f"rot@0:1x{MAX_ORBIT + 1}")
    with pytest.raises(ValueError):
        parse_orbit_spec("rot@0:4x1")
    with pytest.raises(ValueError):
        parse_orbit_spec("rot@0:4")


def test_adversarial_rot_overlap_refused():
    with pytest.raises(ValueError):
        parse_orbit_spec("rot@0:2x4,endian@6:2")
