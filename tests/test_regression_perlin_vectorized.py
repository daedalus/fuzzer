"""PerlinNoiseMutator evaluated fBm noise byte by byte in Python.

~10 ms per KB (4 octaves): 8.8 s per 3k --hail-mary execs. The field is now
evaluated over the whole buffer in numpy with the same float operations in
the same order and round-half-even, so the output is byte-identical.
"""

import math

import pytest

from fuzzer_tool.core.mutations.perlin_noise import PerlinNoise1D, PerlinNoiseMutator
from fuzzer_tool.core.rand_pool import RandPool


def _old_mutate(m: PerlinNoiseMutator, data: bytes, rng) -> bytes | None:
    """The pre-change per-byte loop, verbatim (max_len aside)."""
    n = len(data)
    if n < 8:
        return None
    noise = m._noise_for(data, rng)
    out = bytearray(data)
    phase = rng.randint(0, 1 << 20)
    for i in range(n):
        x = (i + phase) / m.scale
        v = noise.octaves(x, n_octaves=m.octaves) if m.octaves > 1 else noise(x)
        delta = int(round(v * m.strength))
        if delta:
            out[i] = (out[i] + delta) & 0xFF
    return bytes(out)


@pytest.mark.parametrize(
    ("octaves", "scale", "length"), [(4, 32.0, 1000), (1, 7.5, 333), (3, 1.0, 64)]
)
def test_regression_perlin_vectorized(octaves, scale, length):
    data = RandPool(octaves).randbytes(length)
    new = PerlinNoiseMutator(scale=scale, octaves=octaves)
    old = PerlinNoiseMutator(scale=scale, octaves=octaves)
    for seed in range(5):
        assert new.mutate(data, RandPool(seed)) == _old_mutate(old, data, RandPool(seed))


def test_old_matches_itself():
    """Control (Hard Rule 46)."""
    data = RandPool(9).randbytes(200)
    m = PerlinNoiseMutator()
    assert _old_mutate(m, data, RandPool(1)) == _old_mutate(m, data, RandPool(1))


def test_short_input_declines():
    """Adversarial: under 8 bytes there is no noise cycle to apply."""
    assert PerlinNoiseMutator().mutate(b"1234567", RandPool(0)) is None


def test_field_values_match_scalar():
    """Falsification: the vector field equals the scalar one point by point."""
    noise = PerlinNoise1D(seed=3)
    xs = [i / 9.0 for i in range(-40, 400)]
    vec = noise.octaves_array(xs, n_octaves=4)
    assert [float(v) for v in vec] == [noise.octaves(x, n_octaves=4) for x in xs]
    assert math.isfinite(sum(vec))
