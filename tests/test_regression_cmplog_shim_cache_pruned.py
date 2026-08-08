"""Regression: the digest-keyed shim cache grew without bound.

Keying the cached ``.so`` on ``sha256(cmplog_shim.c)[:16]`` is what forces a
recompile after every edit -- but nothing reclaimed the superseded digests,
and the pre-digest fixed-name ``fuzz_cmplog_shim.so`` was never removed
either. Every edit to the shim left another artifact behind in
``~/.cache/fuzzer_cmplog/`` for the life of the machine.

Pruning runs only after the current object is confirmed on disk, so a failed
compile cannot empty the cache.
"""

import os

from fuzzer_tool.core.cmplog import _LEGACY_SHIM_NAME, _prune_stale_shims


def _touch(d, name):
    path = os.path.join(str(d), name)
    with open(path, "wb") as fh:
        fh.write(b"\x7fELF")
    return path


class TestPruneStaleShims:
    def test_superseded_digests_are_removed(self, tmp_path):
        keep = "fuzz_cmplog_shim.0123456789abcdef.so"
        _touch(tmp_path, keep)
        for digest in ("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc"):
            _touch(tmp_path, f"fuzz_cmplog_shim.{digest}.so")

        assert _prune_stale_shims(str(tmp_path), keep) == 3
        assert sorted(os.listdir(tmp_path)) == [keep]

    def test_legacy_fixed_name_is_removed(self, tmp_path):
        keep = "fuzz_cmplog_shim.0123456789abcdef.so"
        _touch(tmp_path, keep)
        _touch(tmp_path, _LEGACY_SHIM_NAME)

        assert _prune_stale_shims(str(tmp_path), keep) == 1
        assert not os.path.exists(os.path.join(tmp_path, _LEGACY_SHIM_NAME))

    def test_current_shim_survives(self, tmp_path):
        keep = "fuzz_cmplog_shim.0123456789abcdef.so"
        path = _touch(tmp_path, keep)
        _prune_stale_shims(str(tmp_path), keep)
        assert os.path.exists(path)

    def test_unrelated_artifacts_are_left_alone(self, tmp_path):
        """Runtime logs and other shims share the directory."""
        keep = "fuzz_cmplog_shim.0123456789abcdef.so"
        _touch(tmp_path, keep)
        others = ["fuzz_cmplog_abc123def456.cmplog", "perf_shim.so", "notes.txt"]
        for name in others:
            _touch(tmp_path, name)

        assert _prune_stale_shims(str(tmp_path), keep) == 0
        assert sorted(os.listdir(tmp_path)) == sorted([keep, *others])

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert _prune_stale_shims(str(tmp_path / "nope"), "anything.so") == 0


class TestStartPrunes:
    def test_start_leaves_exactly_one_shim(self, tmp_path, monkeypatch):
        """End to end: a real start() collapses the cache to the live object."""
        from fuzzer_tool.core import cmplog as C

        monkeypatch.setattr(C, "_CMPLOG_DIR", str(tmp_path))
        _touch(tmp_path, _LEGACY_SHIM_NAME)
        _touch(tmp_path, "fuzz_cmplog_shim.deadbeefdeadbeef.so")

        collector = C.CmplogCollector()
        if not collector.start():  # no compiler available
            return

        objects = [n for n in os.listdir(tmp_path) if n.endswith(".so")]
        assert objects == [os.path.basename(collector._shim_path)]
