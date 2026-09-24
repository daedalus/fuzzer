"""Deterministic stage rotates its start offset by fuzz_count.

Per-pass quotas (P0-1) keep all four passes alive under the cap, but each
pass still covers only a byte *prefix*: the tail of a long seed never gets
deterministic treatment. Rotating the start by ``fuzz_count * span`` makes
successive runs tile the seed.
"""

from __future__ import annotations

import math
from pathlib import Path

from fuzzer_tool.services.operators import _det_start, _deterministic_mutation_stream

PER_BYTE = 33
TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


def _seed(length: int) -> bytes:
    # Disjoint from INTERESTING_UNSIGNED_8 for the first 0xBE bytes; any
    # repeated value still yields a mutant differing at exactly one index.
    return bytes((0x41 + i) & 0xFF for i in range(length))


def _diff(data: bytes, mutant: bytes) -> int:
    (idx,) = [i for i, (a, b) in enumerate(zip(data, mutant, strict=True)) if a != b]
    return idx


def _bitflip_positions(data: bytes, cap: int, start: int) -> list[int]:
    stream = _deterministic_mutation_stream(data, cap, start=start)
    n_bit = cap * 8 // PER_BYTE  # proportional bitflip quota (floor)
    return sorted({_diff(data, next(stream)) for _ in range(n_bit)})


def test_start_zero_is_the_old_schedule():
    data = _seed(40)
    cap = PER_BYTE * 10
    assert list(_deterministic_mutation_stream(data, cap)) == list(
        _deterministic_mutation_stream(data, cap, start=0)
    )


def test_rotated_window_starts_at_offset():
    data = _seed(100)
    cap = PER_BYTE * 10  # 10 bytes per pass
    assert _bitflip_positions(data, cap, start=50) == list(range(50, 60))


def test_rotation_wraps_around():
    """Adversarial: a window crossing the end continues at byte 0."""
    data = _seed(100)
    cap = PER_BYTE * 10
    assert _bitflip_positions(data, cap, start=95) == [0, 1, 2, 3, 4, 95, 96, 97, 98, 99]


def test_start_reduced_modulo_length():
    data = _seed(100)
    cap = PER_BYTE * 10
    assert _bitflip_positions(data, cap, start=150) == list(range(50, 60))


def test_untruncated_schedule_is_a_permutation():
    """Falsification: rotation must not add or drop mutants."""
    data = _seed(20)
    full = sorted(_deterministic_mutation_stream(data))
    assert sorted(_deterministic_mutation_stream(data, start=7)) == full


# ---------------------------------------------------------------------------
# _det_start
# ---------------------------------------------------------------------------


def test_det_start_zero_without_truncation():
    assert _det_start(100, fuzz_count=5, max_mutations=PER_BYTE * 100) == 0


def test_det_start_zero_for_first_run():
    assert _det_start(10_000, fuzz_count=0, max_mutations=PER_BYTE * 10) == 0


def test_det_start_empty_seed():
    assert _det_start(0, fuzz_count=3, max_mutations=1) == 0


def test_det_start_tiles_the_seed():
    """Successive fuzz_counts cover every byte: the tail is reached."""
    length, cap = 1000, PER_BYTE * 64
    span = cap // PER_BYTE
    covered: set[int] = set()
    for fc in range(math.ceil(length / span)):
        s = _det_start(length, fuzz_count=fc, max_mutations=cap)
        covered.update((s + k) % length for k in range(span))
    assert covered == set(range(length))


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def test_engine_rotates_by_seed_fuzz_count(tmp_path):
    from fuzzer_tool.core.skipdet import MAX_DET_MUTATIONS
    from fuzzer_tool.services.fuzzer import Fuzzer

    data = _seed(4000)  # 33 * 4000 > MAX_DET_MUTATIONS: truncated
    corpus = tmp_path / "corpus"
    (corpus / "seeds").mkdir(parents=True)
    (tmp_path / "crashes").mkdir()
    (corpus / "seeds" / "seed1").write_bytes(data)
    f = Fuzzer(
        target=TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(tmp_path / "crashes"),
        max_len=4096,
        deterministic=True,
    )
    seed_key = f._seed_key(data)
    f._favored = {seed_key}
    f._edge_tracker.seed_edges[seed_key] = {1, 2, 3}
    f.seed_meta[data]["fuzz_count"] = 1

    mutant = f._operators.maybe_deterministic_mutation(data)
    expected = _det_start(len(data), fuzz_count=1, max_mutations=MAX_DET_MUTATIONS)
    assert expected > 0
    assert _diff(data, mutant) == expected
