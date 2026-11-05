"""Context inflation in map sizing was a guess 8-9x above measurement.

``ctx_inflation_factor(8)`` returned sqrt(2**8) = 16. Measured distinct
edges, ctx on vs off, same build and corpus:

    fuzzgoat_read_asan.so   628 -> 1,118     (1.78x, 50k execs)
    ffmpeg_read 9.0.2      16,417 -> 27,602  (1.68x, 32k-exec corpus)

So fuzzgoat (278 guards) was sized at 32,768 entries for 1,118 edges:
peak load 1.66%. See docs/handover/handover_shm_displacement_2026-11-05.md P2.
"""

from fuzzer_tool.core.elf import (
    MAP_SIZE_DEFAULT,
    MAP_SIZE_MAX,
    TARGET_LOAD_FACTOR,
    _size_from_blocks,
    ctx_inflation_factor,
)

# (ctx off, ctx on) distinct edges per target.
MEASURED = {"fuzzgoat": (628, 1_118), "ffmpeg": (16_417, 27_602)}

FUZZGOAT_GUARDS = 278
FUZZGOAT_EDGES = MEASURED["fuzzgoat"][1]
FFMPEG_GUARDS = 214_532
CTX_BITS = 8


def test_regression_ctx_inflation_matches_measurement():
    """Covers every measured ratio, and stays within 2x of the largest."""
    worst = max(on / off for off, on in MEASURED.values())
    got = ctx_inflation_factor(CTX_BITS)

    assert worst <= got <= 2 * worst


def test_regression_fuzzgoat_sized_at_default():
    entries = _size_from_blocks(FUZZGOAT_GUARDS, CTX_BITS)

    assert entries == MAP_SIZE_DEFAULT
    assert FUZZGOAT_EDGES / entries <= TARGET_LOAD_FACTOR


def test_ffmpeg_still_at_cap():
    """Falsification: the big target must not shrink below its cap."""
    assert _size_from_blocks(FFMPEG_GUARDS, CTX_BITS) == MAP_SIZE_MAX


def test_inflation_never_exceeds_ctx_ceiling():
    """Adversarial: 1 ctx bit gives at most 2 variants per edge."""
    for bits in range(1, 33):
        assert 1.0 <= ctx_inflation_factor(bits) <= 2.0**bits
