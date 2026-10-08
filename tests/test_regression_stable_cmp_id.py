"""Regression: ``_stable_cmp_id`` must be stable across processes.

The no-PC fallback used builtin ``hash()``, which ``PYTHONHASHSEED`` salts
per process, so ids changed across runs/resumes and ``-j N`` workers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.core.weizz_tags import _stable_cmp_id

_ID_MAX = 0xFFFF
_SUBPROC_TIMEOUT_S = 10
_SRC = str(Path(__file__).resolve().parent.parent / "src")

_PROBE = (
    "from fuzzer_tool.core.weizz_tags import _stable_cmp_id\n"
    "print(_stable_cmp_id(b'IHDR', b'\\x00\\x01'), _stable_cmp_id(b'ab', b'c'))\n"
)


def _ids_with_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": _SRC}
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=_SUBPROC_TIMEOUT_S,
        check=True,
    )
    return proc.stdout.strip()


def test_regression_cmp_id_pc_path_exact() -> None:
    """PC path folds high bits: (pc ^ pc >> 16) & 0xFFFF, zero -> 1."""
    assert _stable_cmp_id(b"a", b"b", 0x12345678) == 0x5678 ^ 0x1234
    assert _stable_cmp_id(b"a", b"b", 0x10001) == 1


def test_regression_cmp_id_fallback_contract() -> None:
    """Fallback stays in [1, 0xFFFF] and frames operands (no concat alias)."""
    cid = _stable_cmp_id(b"ab", b"c")

    assert 1 <= cid <= _ID_MAX
    assert cid != _stable_cmp_id(b"a", b"bc")


@pytest.mark.timeout(30)
def test_regression_cmp_id_hashseed_stable() -> None:
    """Control: same seed twice agrees; then differing seeds must agree."""
    assert _ids_with_seed("1") == _ids_with_seed("1")
    assert _ids_with_seed("1") == _ids_with_seed("2")
