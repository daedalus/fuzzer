"""Every persistent in-tree target takes its input through the shim's macros.

``__AFL_FUZZ_TEST_CASE_BUF`` / ``_LEN`` poison ``[len, cap)`` under ASAN on
every iteration (``afl_shim.c``, ``__afl_fuzz_len``), so a one-byte overread
of the fixed 1 MiB buffer is reported; ``test_shm_testcase.py`` proves that
dynamically on a harness. This file guards the other half statically: a
target that opened its own buffer inside the loop (``read``/``fread`` into a
fixed array) would silently lose that guarantee.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TARGETS = Path(__file__).resolve().parents[1] / "targets"
LOOP = re.compile(r"\b__AFL_LOOP\s*\(")
BUF = re.compile(r"=\s*__AFL_FUZZ_TEST_CASE_BUF\b")
LEN = re.compile(r"=\s*__AFL_FUZZ_TEST_CASE_LEN\b")
# Raw input reads that would bypass the macros inside the persistent loop.
RAW_READ = re.compile(r"\b(?:fread|read|pread|getline|fgets)\s*\(")


def _persistent_targets() -> list[Path]:
    return sorted(p for p in TARGETS.glob("*.c") if LOOP.search(p.read_text(errors="replace")))


def _loop_body(src: str) -> str:
    """Text from the ``__AFL_LOOP`` call to its matching closing brace."""
    m = LOOP.search(src)
    start = src.index("{", m.end())
    depth = 0
    for i in range(start, len(src)):
        depth += (src[i] == "{") - (src[i] == "}")
        if depth == 0:
            return src[start : i + 1]
    raise AssertionError("unbalanced braces after __AFL_LOOP")


def test_inventory_is_not_empty():
    # Guards the glob: a moved targets/ dir must not turn this file into a no-op.
    assert len(_persistent_targets()) >= 10


@pytest.mark.parametrize("path", _persistent_targets(), ids=lambda p: p.stem)
def test_persistent_target_uses_testcase_macros(path):
    src = path.read_text(errors="replace")
    assert BUF.search(src), f"{path.name}: input buffer must come from __AFL_FUZZ_TEST_CASE_BUF"
    body = _loop_body(src)
    assert LEN.search(body), f"{path.name}: length must be read per iteration from __AFL_FUZZ_TEST_CASE_LEN"


@pytest.mark.parametrize("path", _persistent_targets(), ids=lambda p: p.stem)
def test_persistent_loop_has_no_raw_input_read(path):
    body = _loop_body(path.read_text(errors="replace"))
    assert not RAW_READ.search(body), f"{path.name}: raw read inside __AFL_LOOP bypasses ASAN poisoning"
