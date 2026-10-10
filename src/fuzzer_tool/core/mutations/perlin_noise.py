"""Perlin (gradient) noise coherent-perturbation mutation operator.

Ordinary byte-level mutation (bit/byte flips, arithmetic, havoc) produces
*uncorrelated* noise: each touched byte is independent of its neighbours.
For structured pixel/sample buffers (PNG IDAT after filter-decode, raw BMP
pixels, PCM/ADTS sample arrays) that kind of noise is usually the wrong
shape -- real images and audio have *smoothly varying* local structure, and
decoders' interesting code paths (DCT/quantization, subband synthesis,
predictive filters) are exercised more by inputs that preserve that local
smoothness while still being anomalous than by inputs that are locally
flat garbage.

Perlin noise (Ken Perlin, 1982; see
https://en.wikipedia.org/wiki/Perlin_noise) is gradient noise: a grid of
pseudo-random unit gradient vectors, each candidate point scored by the dot
product of the surrounding cell's four gradients with their offset vectors,
interpolated with a C1-continuous ease curve (smootherstep here). Output is
continuous and band-limited by a lattice frequency parameter -- exactly the
"coherent, tunable-scale randomness" byte/bit mutators don't give you.

This module is a *generic*, format-agnostic coherent-perturbation operator:
it treats the buffer as a 1D signal and adds scaled noise deltas to byte
values. It deliberately does not parse PNG/BMP/WAV structure -- pairing it
with format awareness (decoding IDAT, locating the PCM data chunk) is a
follow-up, tracked as a TODO below, not this iteration's scope.

Algorithm reference: https://en.wikipedia.org/wiki/Perlin_noise
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np

from fuzzer_tool.core.mutator_interface import Availability, MutationContext, MutatorBase


def _smootherstep(t: float) -> float:
    """Ken Perlin's improved ease curve: 6t^5 - 15t^4 + 10t^3.

    Zero first *and* second derivative at t=0 and t=1, unlike the classic
    cubic smoothstep -- avoids second-derivative discontinuities at cell
    boundaries (visible as faint grid lines in image use; here it just
    means the noise field has no artificial "seams" at lattice points).
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


# Module-level, keyed on (seed, node) rather than a bound method keyed on
# (self, node): an ``lru_cache`` on an instance method holds a strong
# reference to ``self`` in every cache entry, which keeps every
# ``PerlinNoise1D`` that ever populated an entry alive for as long as that
# entry survives eviction (ruff B019 -- the same leak already fixed for
# fractal_voronoi.py's ``_site``/``_root``). Since almost every mutation
# call derives a fresh, effectively-unique seed (content hash XORed with an
# rng draw -- see ``PerlinNoiseMutator._noise_for``), a method-level cache
# here meant up to 8192 otherwise-dead ``PerlinNoise1D`` instances pinned
# alive at once, continuously replaced as new seeds arrived: steady,
# unbounded-looking RSS growth over a campaign even though the *cache*
# itself was size-capped. Keying on the plain ``(seed, node)`` tuple keeps
# the same cache hit rate (repeated lattice points within one noise field)
# without retaining the instance.
@lru_cache(maxsize=8192)
def _gradient(seed: int, node: int) -> float:
    """Random unit gradient (scalar in [-1, 1]) at integer lattice point *node*."""
    h = hashlib.sha256(f"{seed}:{node}".encode()).digest()
    # 2 bytes -> [0, 65535] -> [-1, 1)
    raw = (h[0] << 8) | h[1]
    return (raw / 32768.0) - 1.0


class PerlinNoise1D:
    """Deterministic 1D gradient noise, seeded and cacheable.

    Gradients are derived from a SHA-256 hash of ``(seed, node)`` rather
    than Perlin's original permutation-table hash -- avoids a 256-entry
    lookup table and its wraparound-at-256 correlation artifacts, at the
    cost of being slower per node. Nodes are cached (module-level, keyed on
    ``(seed, node)``; see ``_gradient`` above) since a mutation pass
    revisits the same integer lattice points across every fractional
    offset in a cell.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _gradient(self, node: int) -> float:
        return _gradient(self.seed, node)

    def __call__(self, x: float) -> float:
        """Noise value at *x*, approximately in [-1, 1]."""
        node0 = math.floor(x)
        node1 = node0 + 1
        frac = x - node0

        # Dot product in 1D degenerates to gradient * (offset distance).
        d0 = self._gradient(node0) * frac
        d1 = self._gradient(node1) * (frac - 1.0)

        t = _smootherstep(frac)
        return _lerp(d0, d1, t)

    def field_array(self, xs: np.ndarray) -> np.ndarray:
        """``__call__`` over an array: same float operations, same order."""
        node0 = np.floor(xs)
        frac = xs - node0
        nodes = node0.astype(np.int64)
        # Few distinct lattice nodes per buffer: hash each once.
        uniq, inv = np.unique(nodes, return_inverse=True)
        g0 = np.fromiter((self._gradient(int(v)) for v in uniq), float, len(uniq))[inv]
        g1 = np.fromiter((self._gradient(int(v) + 1) for v in uniq), float, len(uniq))[inv]
        d0 = g0 * frac
        d1 = g1 * (frac - 1.0)
        t = frac * frac * frac * (frac * (frac * 6 - 15) + 10)
        return d0 + t * (d1 - d0)

    def octaves_array(self, xs, n_octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
        """``octaves`` over an array of points (bit-identical per point)."""
        xs = np.asarray(xs, dtype=np.float64)
        total = np.zeros_like(xs)
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0
        for _ in range(max(1, n_octaves)):
            total = total + self.field_array(xs * frequency) * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / max_amplitude if max_amplitude else np.zeros_like(xs)

    def octaves(self, x: float, n_octaves: int = 4, persistence: float = 0.5) -> float:
        """Fractal Brownian motion: sum of *n_octaves* doubling-frequency layers.

        Each octave halves in amplitude (``persistence``) while doubling in
        frequency, the standard fBm construction -- broad low-frequency
        drift plus finer high-frequency detail, normalized so the result
        stays roughly in [-1, 1] regardless of ``n_octaves``.
        """
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0
        for _ in range(max(1, n_octaves)):
            total += self(x * frequency) * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / max_amplitude if max_amplitude else 0.0


class PerlinNoiseMutator(MutatorBase):
    """Coherent-noise byte perturbation operator.

    Adds scaled 1D Perlin/fBm noise deltas across the buffer instead of
    independent per-byte randomness. ``scale`` controls the lattice
    frequency (smaller = smoother, longer-wavelength drift; larger =
    higher-frequency, more locally jittery); ``strength`` caps the max
    per-byte delta; ``octaves`` layers multiple frequencies (fBm) for
    natural-looking multi-scale texture.

    Args:
        scale: Wavelength divisor -- buffer offset is divided by this
            before being fed to the noise field. Default 32 (one full
            noise cycle per ~32-64 bytes).
        strength: Maximum absolute byte delta applied (0-255). Default 24.
        octaves: fBm layer count. 1 disables fBm (pure single-frequency
            noise).
        seed: Noise-field seed. ``None`` derives one from the input data's
            hash, so the same input always gets the same noise field
            (determinism across runs / across replay), while different
            inputs get different fields (not degenerate mutation of every
            seed the same way).
    """

    availability = Availability.INPUT  # is_available reads only the input

    name = "perlin_noise"
    category = "structural"

    def __init__(
        self,
        scale: float = 32.0,
        strength: int = 24,
        octaves: int = 4,
        seed: int | None = None,
    ) -> None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        if not (0 < strength <= 255):
            raise ValueError("strength must be in (0, 255]")
        if octaves < 1:
            raise ValueError("octaves must be >= 1")
        self.scale = scale
        self.strength = strength
        self.octaves = octaves
        self._fixed_seed = seed
        self._noise_cache: dict[int, PerlinNoise1D] = {}

    def _noise_for(self, data: bytes, rng) -> PerlinNoise1D:
        if self._fixed_seed is not None:
            seed = self._fixed_seed
        else:
            # Derive from input content + one rng draw, so replaying the
            # same input+rng-state reproduces the same field (determinism
            # for triage/replay), while distinct inputs or rng draws diverge.
            digest = hashlib.sha256(data).digest()
            content_seed = int.from_bytes(digest[:4], "big")
            seed = content_seed ^ rng.randint(0, 0xFFFFFFFF)
        noise = self._noise_cache.get(seed)
        if noise is None:
            noise = PerlinNoise1D(seed=seed)
            if len(self._noise_cache) >= 64:
                self._noise_cache.clear()
            self._noise_cache[seed] = noise
        return noise

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        """Always available -- stdlib only, no external dependency."""
        return True

    def mutate(
        self,
        data: bytes,
        rng,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx,
    ) -> bytes | None:
        """Apply coherent noise perturbation across the whole buffer.

        Declines (returns None) on inputs too short to carry a meaningful
        noise cycle. Returns a new ``bytes`` object; never mutates ``data``
        in place.
        """
        n = len(data)
        if n < 8:
            return None

        noise = self._noise_for(data, rng)
        octaves = self.octaves
        scale = self.scale
        strength = self.strength

        # Random phase offset so repeated calls on the same input don't
        # always perturb the same bytes the same way even with a fixed
        # noise field -- mirrors havoc's non-determinism within a run
        # while keeping the underlying field replayable.
        phase = rng.randint(0, 1 << 20)

        # Whole buffer at once; np.rint is round-half-even like round().
        # One octave reduces to noise(x) exactly (v * 1.0 / 1.0).
        xs = (np.arange(n) + phase) / scale
        delta = np.rint(noise.octaves_array(xs, n_octaves=octaves) * strength).astype(np.int64)
        shifted = (np.frombuffer(bytes(data), dtype=np.uint8).astype(np.int64) + delta) & 0xFF
        result = shifted.astype(np.uint8).tobytes()
        if max_len and len(result) > max_len:
            result = result[:max_len]
        return result


# ------------------------------------------------------------------
# Self-registration on module import
# ------------------------------------------------------------------


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    m = PerlinNoiseMutator()
    if m.name not in REGISTRY.names():
        REGISTRY.register_mutator(m)


_register()

# TODO (follow-up iteration): format-aware variants that decode the pixel
# or PCM buffer first (PNG IDAT after zlib-inflate + per-scanline filter
# reversal, raw BMP pixel array, WAV/ADTS sample array) and apply 2D noise
# over (row, col) or 1D noise over the sample index, then re-encode/
# re-filter -- rather than perturbing the compressed/container bytes
# directly, which mostly just breaks checksums before reaching the
# interesting decode paths. Needs the existing png.py/bmp.py chunk-location
# helpers as a base.
