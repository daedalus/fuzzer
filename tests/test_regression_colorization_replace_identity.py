"""Drawing until the byte differs is a shifted draw, done once.

Uniform on {0..255} minus one excluded value ``x`` is exactly

    (x + 1 + u) mod 256,   u uniform on {0..254}

so the ``while c == data[i]: c = random.randint(0, 255)`` retry loop is a
single draw with an offset. The tests below pin the map itself rather than
sampling it: for a fixed original byte the offsets 0..254 must be a
bijection onto the 255 values that are not the original, which is the whole
correctness claim.
"""

import random

import pytest

from fuzzer_tool.core.colorization import _diverse_copy

ALL_BYTES = bytes(range(256))


@pytest.mark.parametrize("original", [0x00, 0x41, 0x7F, 0xFE, 0xFF])
def test_offsets_are_a_bijection_onto_the_allowed_values(original):
    """Offsets 0..254 hit every value except the original, exactly once."""
    produced = {(original + 1 + u) % 256 for u in range(255)}

    assert original not in produced
    assert produced == set(range(256)) - {original}


def test_never_returns_the_original_byte(monkeypatch):
    """Every offset the draw can yield moves the byte."""
    for forced in range(255):
        monkeypatch.setattr(random, "randbytes", lambda n, v=forced: bytes([v]) * n)
        out = _diverse_copy(ALL_BYTES)

        assert all(o != d for o, d in zip(out, ALL_BYTES, strict=True))


def test_offset_matches_the_closed_form(monkeypatch):
    """Exact output for a pinned offset stream, not a sampled property."""
    offsets = bytes([0, 1, 2, 3, 100, 254])
    data = bytes([0x00, 0x41, 0xFF, 0x7F, 0x10, 0x80])
    monkeypatch.setattr(random, "randbytes", lambda _n: offsets)

    assert _diverse_copy(data) == bytearray(
        (d + 1 + u) % 256 for d, u in zip(data, offsets, strict=True)
    )


def test_offset_255_is_redrawn(monkeypatch):
    """Adversarial: 255 is the one offset that maps a byte onto itself."""
    # Slot 0 draws 255 a second time and must be retried again; slots 1 and
    # 2 are then redrawn once each.
    draws = [b"\xff\xff\xff", b"\xff", b"\x07", b"\x01", b"\x02"]
    monkeypatch.setattr(random, "randbytes", lambda _n: draws.pop(0))
    data = b"\x41\x41\x41"

    out = _diverse_copy(data)

    assert not draws, "the redraw loop must consume every scripted draw"
    assert out == bytearray([0x49, 0x43, 0x44])
    assert all(b != 0x41 for b in out)


def test_redraw_only_touches_the_rejected_slot(monkeypatch):
    """Falsification: a redraw must not disturb offsets already accepted."""
    draws = [bytes([10, 255, 20]), b"\x30"]
    monkeypatch.setattr(random, "randbytes", lambda _n: draws.pop(0))
    data = bytes([0x00, 0x00, 0x00])

    out = _diverse_copy(data)

    assert out == bytearray([11, 0x31, 21])


def test_empty_input():
    assert _diverse_copy(b"") == bytearray()


def test_seeded_runs_are_reproducible():
    """Entropy still comes from `random`, so --seed keeps determining it."""
    random.seed(4242)
    first = _diverse_copy(ALL_BYTES * 4)
    random.seed(4242)

    assert _diverse_copy(ALL_BYTES * 4) == first


def test_length_is_preserved():
    random.seed(1)
    for n in (1, 2, 255, 256, 257, 4096):
        assert len(_diverse_copy(bytes(n))) == n
