"""Drawing until the byte differs is a shifted draw, done once.

Uniform on {0..255} minus one excluded value ``x`` is exactly

    (x + 1 + u) mod 256,   u uniform on {0..254}

so the ``while c == data[i]: c = random.randint(0, 255)`` retry loop is a
single draw with an offset. The tests below pin the map itself rather than
sampling it: for a fixed original byte the offsets 0..254 must be a
bijection onto the 255 values that are not the original, which is the whole
correctness claim.

The draws used to come from the global ``random`` module and were scripted
by monkeypatching ``random.randbytes``. ``_diverse_copy`` takes its pool as
an argument now, so they are scripted by injection instead -- which is also
the tripwire the monkeypatch could not be: a fallback to the global stream
would have kept passing under the patch and now raises StopIteration.
"""

import pytest

from fuzzer_tool.core.colorization import _diverse_copy
from fuzzer_tool.core.rand_pool import RandPool

from .support.scripted_rng import ScriptedRng

ALL_BYTES = bytes(range(256))


@pytest.mark.parametrize("original", [0x00, 0x41, 0x7F, 0xFE, 0xFF])
def test_offsets_are_a_bijection_onto_the_allowed_values(original):
    """Offsets 0..254 hit every value except the original, exactly once."""
    produced = {(original + 1 + u) % 256 for u in range(255)}

    assert original not in produced
    assert produced == set(range(256)) - {original}


def test_never_returns_the_original_byte():
    """Every offset the draw can yield moves the byte."""
    for forced in range(255):
        # 255 is the only rejected offset, so a uniform blob below it is
        # consumed in one call and no redraw is scripted.
        rng = ScriptedRng(randbytes=[bytes([forced]) * len(ALL_BYTES)])
        out = _diverse_copy(ALL_BYTES, rng)

        assert all(o != d for o, d in zip(out, ALL_BYTES, strict=True))


def test_offset_matches_the_closed_form():
    """Exact output for a pinned offset stream, not a sampled property."""
    offsets = bytes([0, 1, 2, 3, 100, 254])
    data = bytes([0x00, 0x41, 0xFF, 0x7F, 0x10, 0x80])

    out = _diverse_copy(data, ScriptedRng(randbytes=[offsets]))

    assert out == bytearray((d + 1 + u) % 256 for d, u in zip(data, offsets, strict=True))


def test_offset_255_is_redrawn():
    """Adversarial: 255 is the one offset that maps a byte onto itself."""
    # Slot 0 draws 255 a second time and must be retried again; slots 1 and
    # 2 are then redrawn once each.
    rng = ScriptedRng(randbytes=[b"\xff\xff\xff", b"\xff", b"\x07", b"\x01", b"\x02"])
    data = b"\x41\x41\x41"

    out = _diverse_copy(data, rng)

    with pytest.raises(StopIteration):
        rng.randbytes(1)  # the redraw loop must consume every scripted draw
    assert out == bytearray([0x49, 0x43, 0x44])
    assert all(b != 0x41 for b in out)


def test_redraw_only_touches_the_rejected_slot():
    """Falsification: a redraw must not disturb offsets already accepted."""
    rng = ScriptedRng(randbytes=[bytes([10, 255, 20]), b"\x30"])
    data = bytes([0x00, 0x00, 0x00])

    out = _diverse_copy(data, rng)

    assert out == bytearray([11, 0x31, 21])


def test_empty_input():
    assert _diverse_copy(b"", RandPool(seed=1)) == bytearray()


def test_seeded_runs_are_reproducible():
    """Entropy comes from the injected pool, so --seed keeps determining it."""
    first = _diverse_copy(ALL_BYTES * 4, RandPool(seed=4242))

    assert _diverse_copy(ALL_BYTES * 4, RandPool(seed=4242)) == first


def test_a_different_seed_gives_a_different_result():
    """Falsification for the test above: equality must come from the seed,
    not from the function ignoring its pool."""
    assert _diverse_copy(ALL_BYTES * 4, RandPool(seed=1)) != _diverse_copy(
        ALL_BYTES * 4, RandPool(seed=2)
    )


def test_length_is_preserved():
    rng = RandPool(seed=1)
    for n in (1, 2, 255, 256, 257, 4096):
        assert len(_diverse_copy(bytes(n), rng)) == n
