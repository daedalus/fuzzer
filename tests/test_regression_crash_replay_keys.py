"""Regression tests for finding #22 — crash replays keyed by the wrong thing.

``_crash_replays`` is documented as ``sig -> replay return codes`` and every
other consumer keys it that way (``_prune_crash_data`` pops by signature, the
reproducibility report labels the key as a signature). The scheduler built the
key with ``crash_sigs.get(crash_name, crash_name)`` — a FILENAME looked up in a
signature-keyed dict — so the lookup always missed and the filename became the
key.

``run_crash_replays`` then had to find its way back to an input from that key,
and did it with ``f.stem.startswith(sig[:12])``. Crash files are named
``crash_<unix_ts>_<cluster>_<san>_<err>``, so twelve characters cover
``crash_`` plus six digits of a ten-digit timestamp: every crash inside the
same 10**4-second window compares equal and the first one in directory order
wins. That is the defect these tests pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzer_tool.services.stats_reporter import run_crash_replays


def _seed_key(data: bytes) -> str:
    from fuzzer_tool.adapters.filesystem import hash_data

    return hash_data(data)


class _Recorder:
    """Stands in for run_target_stdin, recording what was actually replayed."""

    def __init__(self):
        self.seen: list[bytes] = []

    def __call__(self, target, data, timeout, *a, **kw):
        self.seen.append(data)
        return 0, "", 1234


@pytest.fixture
def patched_runner(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("fuzzer_tool.adapters.process.run_target_stdin", rec)
    return rec


def _write_crash(crashes_dir: Path, ts: int, tag: str, payload: bytes) -> str:
    """Write a crash file in save_crash()'s naming scheme; return the base name."""
    base = f"crash_{ts}_cluster{tag}_sig_signal11"
    (crashes_dir / f"{base}.bin").write_bytes(payload)
    (crashes_dir / f"{base}.txt").write_text("sidecar\n")
    return base


# Two timestamps that share their first six digits, i.e. inside the ~2.7h
# window over which the old sig[:12] prefix could not tell crashes apart.
TS_A = 1756800000
TS_B = 1756809999


class TestReplaysTheRightInput:
    def test_recorded_file_is_used(self, tmp_path, patched_runner):
        crashes = tmp_path / "crashes"
        crashes.mkdir()
        base_a = _write_crash(crashes, TS_A, "a", b"INPUT-A")
        base_b = _write_crash(crashes, TS_B, "b", b"INPUT-B")

        replays: dict[str, list[int]] = {"SIG_B": []}
        run_crash_replays(
            crashes,
            "/bin/true",
            1.0,
            replays,
            replay_n=1,
            seed_key_fn=_seed_key,
            budget_ms=10_000,
            crash_files={"SIG_A": base_a, "SIG_B": base_b},
        )
        assert patched_runner.seen == [b"INPUT-B"]

    def test_neighbouring_timestamp_is_not_confused(self, tmp_path, patched_runner):
        """The exact shape of the old collision: A is written first, B is asked for.

        Under `f.stem.startswith(sig[:12])` any crash in the same ~2.7h window
        matched, and directory order decided the winner.
        """
        crashes = tmp_path / "crashes"
        crashes.mkdir()
        base_a = _write_crash(crashes, TS_A, "a", b"WRONG-INPUT-A")
        base_b = _write_crash(crashes, TS_B, "b", b"RIGHT-INPUT-B")
        assert base_a[:12] == base_b[:12], "test fixture no longer reproduces the collision"

        replays: dict[str, list[int]] = {"SIG_B": []}
        run_crash_replays(
            crashes,
            "/bin/true",
            1.0,
            replays,
            replay_n=1,
            seed_key_fn=_seed_key,
            budget_ms=10_000,
            crash_files={"SIG_B": base_b},
        )
        assert patched_runner.seen == [b"RIGHT-INPUT-B"]

    def test_content_hash_fallback_still_works(self, tmp_path, patched_runner):
        """No recorded file (state from an older run): fall back on identity."""
        crashes = tmp_path / "crashes"
        crashes.mkdir()
        _write_crash(crashes, TS_A, "a", b"OTHER")
        _write_crash(crashes, TS_B, "b", b"TARGET")

        key = _seed_key(b"TARGET")
        replays: dict[str, list[int]] = {key: []}
        run_crash_replays(
            crashes,
            "/bin/true",
            1.0,
            replays,
            replay_n=1,
            seed_key_fn=_seed_key,
            budget_ms=10_000,
            crash_files=None,
        )
        assert patched_runner.seen == [b"TARGET"]

    def test_unknown_signature_records_missing_not_a_neighbour(self, tmp_path, patched_runner):
        """A signature with no file must score -3, not replay someone else's input."""
        crashes = tmp_path / "crashes"
        crashes.mkdir()
        _write_crash(crashes, TS_A, "a", b"SOMEBODY-ELSE")

        replays: dict[str, list[int]] = {"crash_1756800000_unknown": []}
        run_crash_replays(
            crashes,
            "/bin/true",
            1.0,
            replays,
            replay_n=1,
            seed_key_fn=_seed_key,
            budget_ms=10_000,
            crash_files={},
        )
        assert patched_runner.seen == []
        assert replays["crash_1756800000_unknown"] == [-3]

    def test_stale_recorded_name_falls_back(self, tmp_path, patched_runner):
        """A recorded name whose file is gone must not replay a neighbour."""
        crashes = tmp_path / "crashes"
        crashes.mkdir()
        _write_crash(crashes, TS_A, "a", b"NEIGHBOUR")

        replays: dict[str, list[int]] = {"SIG_GONE": []}
        run_crash_replays(
            crashes,
            "/bin/true",
            1.0,
            replays,
            replay_n=1,
            seed_key_fn=_seed_key,
            budget_ms=10_000,
            crash_files={"SIG_GONE": f"crash_{TS_B}_clusterb_sig_signal11"},
        )
        assert patched_runner.seen == []
        assert replays["SIG_GONE"] == [-3]


class TestSchedulerKeySpace:
    """The key the scheduler writes must be the key every consumer reads."""

    def test_replay_key_is_a_signature_not_a_filename(self):
        import inspect

        from fuzzer_tool.services import fuzzer as fuzzer_mod

        src = inspect.getsource(fuzzer_mod.Fuzzer.fuzz_one)
        # Comments in the fix quote the old expression, so scan code only.
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert "crash_sigs.get(crash_name" not in code, (
            "the replay key is being re-derived by looking a filename up in a "
            "signature-keyed dict; use _last_crash_signature"
        )
        assert "_last_crash_signature" in code

    def test_save_crash_publishes_the_counted_signature(self):
        import inspect

        from fuzzer_tool.services.corpus_manager import CorpusManager

        src = inspect.getsource(CorpusManager.save_crash)
        assert "_last_crash_signature" in src
        assert "_crash_files" in src

    def test_prune_evicts_the_file_map_too(self):
        import inspect

        from fuzzer_tool.services import fuzzer as fuzzer_mod

        src = inspect.getsource(fuzzer_mod.Fuzzer._prune_crash_data)
        assert "_crash_files.pop" in src, (
            "_crash_files is per-signature and must be evicted with its siblings"
        )
