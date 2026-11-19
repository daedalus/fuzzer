"""Shared constants, failure type and seed inputs."""

from __future__ import annotations

import functools
import pickle
import sys
from pathlib import Path

# A call slower than this is a finding (pathological input -> hang risk).
SLOW_SECONDS = 5.0

# Rule 17/18: crashes live under ~, never in the source tree.
CRASH_DIR = Path.home() / "fuzzing" / "selffuzz-crashes"

ELF_MAGIC = b"\x7fELF"
_ELF64_LE_HEADER = 64
_ELF_CANDIDATES = ("/bin/true", "/usr/bin/true", sys.executable)
_LEARNER_SEED_ROWS = 24

# (label, detail)
Failure = tuple[str, str]


def err(exc: BaseException) -> str:
    """One-line ``Type: message`` for a failure report."""
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def elf_seeds() -> list[bytes]:
    """Smallest real ELF on the box plus a bare 64-bit LE header.

    Small keeps per-case mutation cheap; the bare header reaches the
    section-table bounds checks with nothing behind them.
    """
    seeds = []
    for cand in _ELF_CANDIDATES:
        path = Path(cand).resolve()
        if not path.is_file():
            continue

        data = path.read_bytes()
        if data.startswith(ELF_MAGIC):
            seeds.append(data)

    seeds.sort(key=len)
    bare = bytearray(_ELF64_LE_HEADER)
    bare[:7] = ELF_MAGIC + b"\x02\x01\x01"  # class 64, little-endian, version 1
    return [*seeds[:1], bytes(bare)]


def text_seeds() -> list[bytes]:
    """AFL dictionary lines: quoted, escaped, named, commented, malformed."""
    lines = [
        b'"IHDR"',
        b'kw1="\\x89PNG"',
        b'kw2@5="\\\\\\""',
        b"# comment",
        b'"unterminated',
        b'"\\xZZ"',
        b"",
    ]
    return [b"\n".join(lines)]


def state_seed() -> bytes:
    """A plausible persisted-state pickle (shape of ``StateStore`` sections)."""
    return pickle.dumps({"version": 1, **section_seeds(), "stats": {"execs": 10}})


@functools.cache
def section_seeds() -> dict[str, dict]:
    """Real ``to_dict``/``get_state`` output per restorable section.

    Built from the live classes so the seeds track their current schema.
    """
    from fuzzer_tool.core.analyzers.analyzer_format_learner import FormatLearner
    from fuzzer_tool.core.edge_tracker import EdgeTracker
    from fuzzer_tool.core.markov import MarkovEnsemble
    from fuzzer_tool.core.rand_pool import RandPool

    markov = MarkovEnsemble(rng=RandPool(1))
    markov.train(b"\x89PNG\r\n\x1a\nIHDR" * 8)

    tracker = EdgeTracker(map_size=1024)

    learner = FormatLearner(rng=RandPool(1))
    for i in range(_LEARNER_SEED_ROWS):
        learner.record_transition(
            b"ABCD" + bytes([i]) * 8, "bit_flip", i % 8, 1, 10, 10 + i % 2, {i}, set()
        )

    return {
        "markov": markov.to_dict(),
        "edge_tracker": tracker.to_dict(),
        "learner_state": learner.get_state(),
    }
