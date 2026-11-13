"""Regression: a smaller trigger evicted the saved crash before its own write.

``save_crash`` unlinked the old cluster files, then published the new ones. A
failed publish (ENOSPC) left the signature known with no reproducer on disk.
"""

import errno

import pytest

from fuzzer_tool.adapters import filesystem as fs

_SEGV_RC = -11
_BIG = b"A" * 100
_SMALL = b"A" * 10


def _bins(crashes_dir):
    return sorted(p.read_bytes() for p in crashes_dir.glob("*.bin"))


@pytest.fixture
def saved(tmp_path):
    state = {"hashes": set(), "sigs": {}, "mins": {}}
    assert fs.save_crash(
        _BIG, _SEGV_RC, "", tmp_path, state["hashes"], state["sigs"], crash_min_sizes=state["mins"]
    )
    return tmp_path, state


def _save_small(crashes_dir, state):
    return fs.save_crash(
        _SMALL,
        _SEGV_RC,
        "",
        crashes_dir,
        state["hashes"],
        state["sigs"],
        crash_min_sizes=state["mins"],
    )


def test_regression_failed_replace_keeps_old_crash(saved, monkeypatch):
    crashes_dir, state = saved

    def enospc(*_a, **_k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fs, "_publish_crash", enospc)
    with pytest.raises(OSError):
        _save_small(crashes_dir, state)

    assert _bins(crashes_dir) == [_BIG]
    assert state["mins"] == {next(iter(state["mins"])): len(_BIG)}


def test_smaller_trigger_still_replaces(saved):
    """Falsification: a successful replace leaves only the smaller trigger."""
    crashes_dir, state = saved
    assert _save_small(crashes_dir, state)

    assert _bins(crashes_dir) == [_SMALL]


def test_replace_within_same_second_keeps_new(saved, monkeypatch):
    """Adversarial: same timestamp -> same base name; eviction must spare it."""
    crashes_dir, state = saved
    old_base = next(crashes_dir.glob("*.bin")).stem
    frozen = int(old_base.split("_")[1])
    monkeypatch.setattr(fs.time, "time", lambda: frozen)

    assert _save_small(crashes_dir, state) == old_base
    assert _bins(crashes_dir) == [_SMALL]


def test_regression_failed_first_save_is_retried(tmp_path, monkeypatch):
    """A failed write marked the input and signature known: the crash was never saved."""
    hashes, sigs, mins = set(), {}, {}
    real = fs._publish_crash

    def enospc(*_a, **_k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fs, "_publish_crash", enospc)
    with pytest.raises(OSError):
        fs.save_crash(_BIG, _SEGV_RC, "", tmp_path, hashes, sigs, crash_min_sizes=mins)

    assert (hashes, sigs, mins) == (set(), {}, {})

    monkeypatch.setattr(fs, "_publish_crash", real)
    assert fs.save_crash(_BIG, _SEGV_RC, "", tmp_path, hashes, sigs, crash_min_sizes=mins)
    assert _bins(tmp_path) == [_BIG]
