"""Regression: --save-smaller replaced the wrong crash, and prune popped the wrong key.

Replacement ran only for a NOVEL signature, so a smaller trigger of a known
bug was dropped while a different bug sharing the stack hash was evicted.
``crash_min_sizes`` was keyed by stack hash; ``_prune_crash_data`` popped it
by signature, a no-op.
"""

from __future__ import annotations

from types import SimpleNamespace

import fuzzer_tool.services.fuzzer as fuzzer_mod
from fuzzer_tool.adapters.filesystem import save_crash
from fuzzer_tool.services.fuzzer import Fuzzer


def _report(error_type: str) -> str:
    return (
        f"==1==ERROR: AddressSanitizer: {error_type} on address 0x1\n"
        "    #0 0x401111 in parse_chunk\n"
        "    #1 0x402222 in main\n"
    )


_HBO = _report("heap-buffer-overflow")
_UAF = _report("heap-use-after-free")


class _Crashes:
    """save_crash state for one campaign, --save-smaller on."""

    def __init__(self, crashes_dir) -> None:
        self.dir = crashes_dir
        self.hashes: set[str] = set()
        self.sigs: dict[str, int] = {}
        self.min_sizes: dict[str, int] = {}

    def save(self, data: bytes, stderr: str):
        return save_crash(
            data, 1, stderr, self.dir, self.hashes, self.sigs, crash_min_sizes=self.min_sizes
        )

    def inputs(self) -> list[bytes]:
        return sorted(p.read_bytes() for p in self.dir.glob("*.bin"))


def test_regression_smaller_same_signature_replaces(tmp_path) -> None:
    c = _Crashes(tmp_path)
    c.save(b"L" * 64, _HBO)

    assert c.save(b"S" * 8, _HBO)
    assert c.inputs() == [b"S" * 8]
    assert len(list(tmp_path.iterdir())) == len(list(tmp_path.glob("crash_*")))
    assert c.sigs == {next(iter(c.sigs)): 2}


def test_regression_other_signature_same_stack_kept(tmp_path) -> None:
    # Adversarial: same PCs, different bug; smaller, but not a replacement.
    c = _Crashes(tmp_path)
    c.save(b"L" * 64, _HBO)
    c.save(b"S" * 8, _UAF)

    assert c.inputs() == sorted([b"L" * 64, b"S" * 8])


def test_larger_same_signature_not_replaced(tmp_path) -> None:
    # Falsification: a bigger trigger keeps the smaller one on disk.
    c = _Crashes(tmp_path)
    c.save(b"S" * 8, _HBO)

    assert not c.save(b"L" * 64, _HBO)
    assert c.inputs() == [b"S" * 8]


def test_regression_prune_drops_min_size_by_signature(tmp_path, monkeypatch) -> None:
    c = _Crashes(tmp_path)
    c.save(b"a" * 8, _HBO)
    c.save(b"b" * 8, _UAF)
    frequent = next(sig for sig in c.sigs if "overflow" in sig)
    c.sigs[frequent] = 5

    fake = SimpleNamespace(
        crash_sigs=c.sigs,
        crash_frames={},
        crash_min_sizes=c.min_sizes,
        _crash_replays={},
        _crash_files={},
        crash_hashes=set(),
    )
    monkeypatch.setattr(fuzzer_mod, "MAX_CRASH_SIGS", 1)
    Fuzzer._prune_crash_data(fake)  # type: ignore[arg-type]

    assert set(fake.crash_sigs) == {frequent}
    assert set(fake.crash_min_sizes) == {frequent}


# ── Prune policy ────────────────────────────────────────────────────────


def _fake(sigs: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        crash_sigs=sigs,
        crash_frames={},
        crash_min_sizes={},
        _crash_replays={},
        _crash_files={},
        crash_hashes=set(),
    )


def test_regression_prune_keeps_newest_on_ties(monkeypatch) -> None:
    # Equal counts: the newest (most likely to be hit again) survive.
    fake = _fake({f"sig{i}": 1 for i in range(8)})
    monkeypatch.setattr(fuzzer_mod, "MAX_CRASH_SIGS", 4)
    Fuzzer._prune_crash_data(fake)  # type: ignore[arg-type]

    assert list(fake.crash_sigs) == ["sig5", "sig6", "sig7"]


def test_prune_keeps_frequent_first(monkeypatch) -> None:
    # Falsification: frequency still outranks age.
    fake = _fake({"old_hot": 9, **{f"sig{i}": 1 for i in range(7)}})
    monkeypatch.setattr(fuzzer_mod, "MAX_CRASH_SIGS", 4)
    Fuzzer._prune_crash_data(fake)  # type: ignore[arg-type]

    assert list(fake.crash_sigs) == ["old_hot", "sig5", "sig6"]


def test_regression_pruned_signature_not_resaved(tmp_path) -> None:
    # Forgotten in memory, still on disk: a re-hit is a repeat, not a new bug.
    c = _Crashes(tmp_path)
    c.save(b"first", _HBO)
    c.sigs.clear()

    assert not c.save(b"second", _HBO)
    assert c.inputs() == [b"first"]
    assert c.sigs == {next(iter(c.sigs)): 1}


def test_regression_crash_hashes_bounded(monkeypatch) -> None:
    # One entry per crashing exec, repeats included: it grew without bound.
    fake = _fake({"sig": 1})
    fake.crash_hashes = {f"{i:016x}" for i in range(9)}
    monkeypatch.setattr(fuzzer_mod, "MAX_CRASH_HASHES", 8)
    Fuzzer._prune_crash_data(fake)  # type: ignore[arg-type]

    assert len(fake.crash_hashes) <= 8


def test_crash_hashes_under_bound_kept(monkeypatch) -> None:
    # Falsification: below the bound nothing is forgotten.
    fake = _fake({"sig": 1})
    fake.crash_hashes = {f"{i:016x}" for i in range(8)}
    monkeypatch.setattr(fuzzer_mod, "MAX_CRASH_HASHES", 8)
    Fuzzer._prune_crash_data(fake)  # type: ignore[arg-type]

    assert len(fake.crash_hashes) == 8


def test_forgotten_input_hash_not_resaved(tmp_path) -> None:
    # Edge: a repeat input after the hashes were dropped is still a repeat.
    c = _Crashes(tmp_path)
    c.save(b"same", _HBO)
    c.hashes.clear()

    assert not c.save(b"same", _HBO)
    assert c.inputs() == [b"same"]
