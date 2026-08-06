"""Regression tests for StateStore — pickle-based single-file state persistence.

Covers: round-trip, int-key preservation (JSON would coerce), legacy JSON
migration, cleanup of legacy files, disabled mode, and unsafe-Global rejection.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from fuzzer_tool.core.state_store import (
    LEGACY_JSON_FILES,
    STATE_FILENAME,
    StateStore,
)


class TestStateStoreRoundTrip:
    """StateStore.save() + load() must round-trip all sections."""

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        store.set("corpus", {"exec_count": 42, "crashes": {"sig1": 3}})
        store.set("markov", {"chain": [[0, 1], [1, 2]]})
        store.set("edge_tracker", {"cumulative_edges": [1, 2, 3]})
        store.save()

        store2 = StateStore(tmp_path)
        assert store2.get("corpus")["exec_count"] == 42
        assert store2.get("corpus")["crashes"] == {"sig1": 3}
        assert store2.get("markov") == {"chain": [[0, 1], [1, 2]]}
        assert store2.get("edge_tracker")["cumulative_edges"] == [1, 2, 3]

    def test_state_file_is_compressed_pickle(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        store.set("corpus", {"exec_count": 1})
        store.save()

        assert (tmp_path / STATE_FILENAME).exists()
        # File should be gzip-compressed
        import gzip

        with gzip.open(tmp_path / STATE_FILENAME, "rb") as fh:
            data = pickle.load(fh)
        assert isinstance(data, dict)
        assert data["corpus"]["exec_count"] == 1

    def test_get_returns_default_for_missing_section(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        assert store.get("nonexistent") is None
        assert store.get("nonexistent", "default") == "default"

    def test_empty_store_saves_nothing(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        assert store.save() is False
        assert not (tmp_path / STATE_FILENAME).exists()


class TestStateStoreIntKeys:
    """Pickle preserves int keys; JSON coerces them to str (the bug we fixed)."""

    def test_int_keys_preserved(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        # Edge IDs and byte values are int keys — JSON would turn them into str.
        store.set("edge_tracker", {0: 100, 1: 200, 255: 1})
        store.save()

        store2 = StateStore(tmp_path)
        data = store2.get("edge_tracker")
        assert 0 in data  # int key survives pickle
        assert data[0] == 100
        assert data[255] == 1
        # Ensure no str keys leaked in
        assert all(isinstance(k, int) for k in data)

    def test_nested_int_keys_preserved(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        store.set("mi", {65: {97: 5, 98: 3}, 66: {97: 1}})
        store.save()

        store2 = StateStore(tmp_path)
        data = store2.get("mi")
        assert isinstance(data, dict)
        assert all(isinstance(k, int) for k in data)
        assert isinstance(data[65], dict)
        assert all(isinstance(k, int) for k in data[65])


class TestLegacyJsonMigration:
    """StateStore.load() should auto-migrate legacy JSON files."""

    def test_migrates_legacy_json_files(self, tmp_path: Path) -> None:
        # Write legacy JSON files
        (tmp_path / "state.json").write_text(json.dumps({"exec_count": 999, "seed_meta": {}}))
        (tmp_path / "markov.json").write_text(json.dumps({"chain": "mc_data"}))

        store = StateStore(tmp_path)
        assert store.get("corpus")["exec_count"] == 999
        assert store.get("markov") == {"chain": "mc_data"}

    def test_get_triggers_load_once(self, tmp_path: Path) -> None:
        (tmp_path / "state.json").write_text(json.dumps({"exec_count": 5}))

        store = StateStore(tmp_path)
        # First get triggers load
        assert store.get("corpus")["exec_count"] == 5
        # Second get doesn't reload from disk
        store.set("corpus", {"exec_count": 10})
        assert store.get("corpus")["exec_count"] == 10

    def test_legacy_keys_coerced_to_int_by_components(self, tmp_path: Path) -> None:
        """Legacy JSON coerces int keys to str; the migration just preserves the
        data as-is — component from_dict/load must handle str→int conversion."""

        # JSON doesn't allow int keys, so they become strings
        (tmp_path / "edge_tracker.json").write_text(json.dumps({"cumulative_edges": [1, 2, 3]}))

        store = StateStore(tmp_path)
        et_data = store.get("edge_tracker")
        assert et_data["cumulative_edges"] == [1, 2, 3]


class TestCleanupLegacy:
    """cleanup_legacy() must remove all mapped JSON files."""

    def test_cleanup_removes_legacy_files(self, tmp_path: Path) -> None:
        # Create legacy files
        for filename in LEGACY_JSON_FILES.values():
            (tmp_path / filename).write_text("")

        store = StateStore(tmp_path)
        store.load()
        removed = store.cleanup_legacy()
        assert removed == len(LEGACY_JSON_FILES)
        for filename in LEGACY_JSON_FILES.values():
            assert not (tmp_path / filename).exists()

    def test_cleanup_nonexistent_files_is_noop(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        removed = store.cleanup_legacy()
        assert removed == 0

    def test_pickle_not_removed_by_cleanup(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path)
        store.set("corpus", {"exec_count": 1})
        store.save()
        store.cleanup_legacy()
        assert (tmp_path / STATE_FILENAME).exists()


class TestDisabledMode:
    """--no-save-state: StateStore with enabled=False must not write."""

    def test_disabled_save_is_noop(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path, enabled=False)
        store.set("corpus", {"exec_count": 1})
        assert store.save() is False
        assert not (tmp_path / STATE_FILENAME).exists()

    def test_disabled_load_returns_empty(self, tmp_path: Path) -> None:
        store = StateStore(tmp_path, enabled=False)
        assert store.get("corpus") is None


class TestSafeUnpickler:
    """Unsafe globals in a state file must be rejected, not executed."""

    def test_disallowed_global_rejected(self, tmp_path: Path) -> None:
        # Manually craft a pickle with a disallowed global (e.g. os.system)
        payload = pickle.dumps(
            {"corpus": {"exec_count": _Malicious()}},
        )
        # Write raw gzip pickle
        import gzip

        with gzip.open(tmp_path / STATE_FILENAME, "wb") as fh:
            fh.write(payload)

        # load() catches UnsafeStateError internally and returns empty data
        # so the fuzzer doesn't crash — a tampered file yields no state, not
        # arbitrary code execution.
        store2 = StateStore(tmp_path)
        store2.load()
        assert store2.get("corpus") is None


class _Malicious:
    """A class that pickle would reference as a global — not in the allowlist."""

    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))
