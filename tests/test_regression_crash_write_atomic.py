"""Regression: save_crash wrote its five files one by one.

An error (or interrupt) between writes left a partial set: a ``.bin`` with no
``.json``, which the report counts by, or sidecars with no input.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.adapters.filesystem import save_crash
from fuzzer_tool.core.crash_metadata import CrashMetadata

ASAN = "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x1\n    #0 0x401 in f\n"
EXTS = {".bin", ".txt", ".json", ".sh", ".hex"}


def test_regression_failed_write_leaves_nothing(tmp_path, monkeypatch) -> None:
    def _boom(self, data, target):
        raise OSError("disk full")

    monkeypatch.setattr(CrashMetadata, "format_reproducer", _boom)

    with pytest.raises(OSError):
        save_crash(b"x", 1, ASAN, tmp_path, set(), {})

    assert list(tmp_path.iterdir()) == []


def test_saved_crash_has_full_set(tmp_path) -> None:
    # Falsification: a normal save writes all five, no temp files.
    base = save_crash(b"x", 1, ASAN, tmp_path, set(), {})

    assert {p.suffix for p in tmp_path.iterdir()} == EXTS
    assert all(p.name.startswith(base) for p in tmp_path.iterdir())
