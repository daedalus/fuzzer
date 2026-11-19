"""Fixed pathological inputs per suite (deterministic, no RNG)."""

from __future__ import annotations

import pickle
import struct
from typing import TYPE_CHECKING

from . import corpus, targets

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

# ELF64 header field offsets (System V gABI).
E_SHOFF = 0x28
E_SHENTSIZE = 0x3A
E_SHNUM = 0x3C
E_SHSTRNDX = 0x3E
TRUNCATE_POINTS = 8
HUGE_LINE = 1 << 20


def _run(cases: list[tuple[str, str, bytes]]) -> list[corpus.Failure]:
    """Run ``(target, case name, data)`` triples; collect failures."""
    fails = []
    for target, name, data in cases:
        detail = targets.run_target(target, data)
        if detail is not None:
            fails.append((f"{target}:{name}", detail))
    return fails


def _patched(data: bytes, offset: int, fmt: str, value: int) -> bytes:
    """*data* with one header field overwritten."""
    buf = bytearray(data)
    struct.pack_into(fmt, buf, offset, value)
    return bytes(buf)


def _elf_cases() -> list[tuple[str, bytes]]:
    """Header-field extremes and truncations of real and bare ELFs."""
    real, bare = corpus.elf_seeds()[0], corpus.elf_seeds()[-1]
    cases = [("empty", b""), ("magic_only", corpus.ELF_MAGIC), ("bare_header", bare)]

    for name, base in (("real", real), ("bare", bare)):
        cases += [
            (f"{name}_shoff_huge", _patched(base, E_SHOFF, "<Q", (1 << 63) - 1)),
            (f"{name}_shoff_end", _patched(base, E_SHOFF, "<Q", len(base))),
            (f"{name}_shnum_max", _patched(base, E_SHNUM, "<H", 0xFFFF)),
            (f"{name}_shentsize_0", _patched(base, E_SHENTSIZE, "<H", 0)),
            (f"{name}_shstrndx_oob", _patched(base, E_SHSTRNDX, "<H", 0xFFFE)),
        ]

    step = max(1, len(real) // TRUNCATE_POINTS)
    cases += [(f"real_trunc_{n}", real[:n]) for n in range(0, len(real), step)]
    return cases


def fuzz_elf(_workdir: Path, _opts: argparse.Namespace) -> list[corpus.Failure]:
    """ELF parsers (bytes and path entry points) plus AFL dictionaries."""
    cases = []
    for name, data in _elf_cases():
        cases += [("elf_bytes", name, data), ("elf_path", name, data)]

    lines = [b'"' + b"\\x" * 4 + b'"', b'"' + b"A" * HUGE_LINE + b'"', b"\x00\xff\n\r\n"]
    cases += [("dict", f"text_{i}", t) for i, t in enumerate(corpus.text_seeds() + lines)]
    return _run(cases)


def fuzz_state(_workdir: Path, _opts: argparse.Namespace) -> list[corpus.Failure]:
    """Corrupt state files and wrong-shaped persisted sections."""
    cases = [
        ("state", "empty", b""),
        ("state", "seed", corpus.state_seed()),
        ("state", "not_dict", pickle.dumps([1, 2, 3])),
        ("state", "truncated", corpus.state_seed()[:-7]),
    ]

    wrong = {"empty": {}, "nulls": None, "junk_keys": {"zz": 1, "": []}}
    for section, seed in corpus.section_seeds().items():
        cases.append((section, "seed", pickle.dumps(seed)))
        for name, value in wrong.items():
            cases.append((section, name, pickle.dumps(dict.fromkeys(seed, value))))
    return _run(cases)


def fuzz_learner(_workdir: Path, _opts: argparse.Namespace) -> list[corpus.Failure]:
    """Transition streams at the type-valid extremes."""

    def rows(offset, width, before=10, after=11) -> bytes:
        return pickle.dumps([(b"ABCD", "bit_flip", offset, width, before, after, {1}, set())] * 600)

    cases = [
        ("learner", "empty", pickle.dumps([])),
        ("learner", "none_offsets", rows(None, 1)),
        ("learner", "negative_offset", rows(-1, 1)),
        ("learner", "huge_offset", rows(1 << 40, 1)),
        ("learner", "zero_width", rows(0, 0)),
        ("learner", "huge_width", rows(0, 1 << 20)),
        ("learner", "negative_coverage", rows(0, 1, -5, -9)),
    ]
    return _run(cases)


def fuzz_ops(_workdir: Path, _opts: argparse.Namespace) -> list[corpus.Failure]:
    """Every armed operator on degenerate buffers."""
    inputs = {
        "empty": b"",
        "one": b"\x00",
        "ff_maxlen": b"\xff" * targets.MOCK_MAX_LEN,
        "all_bytes": bytes(range(256)),
        "png_header": b"\x89PNG\r\n\x1a\n" + bytes(25),
        "elf_header": corpus.elf_seeds()[-1],
    }
    cases = []
    for name, data in inputs.items():
        cases += [(targets.OP_PREFIX + op, name, data) for op in targets.op_names(data)]
    return _run(cases)
