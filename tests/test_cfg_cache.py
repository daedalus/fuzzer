"""Tests for core/cfg_cache.py — on-disk CFG decode cache."""

import gzip
import logging
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.core import cfg_cache
from fuzzer_tool.core import distance as distance_mod
from fuzzer_tool.core.distance import TargetDistance
from fuzzer_tool.core.elf import build_id

needs_cc = pytest.mark.skipif(
    not (shutil.which("gcc") or shutil.which("clang")), reason="no C compiler"
)

_SRC = """

__attribute__((visibility(\"default\")))
int helper_add(int a, int b) { return a + b; }

__attribute__((visibility(\"default\")))
int entry_calc(int x) {
    if (x > 10) {
        return helper_add(x, 1);
    }
    return helper_add(x, 2) ^ 0x5a;
}

int main(void) { return entry_calc(3); }
"""


def _cc():
    return shutil.which("gcc") or shutil.which("clang")


@pytest.fixture(autouse=True)
def _cache_sandbox(tmp_path, monkeypatch):
    """Redirect the cache dir before anything can memoize the real one."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setattr(cfg_cache, "_cache_dir_memo", None)


@pytest.fixture(scope="module")
def built_exe(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfgcache")
    src = d / "t.c"
    src.write_text(_SRC)
    cc = _cc()
    out = {}
    for key, extra in (("exe", []), ("nobid", ["-Wl,--build-id=none"])):
        exe = d / f"t_{key}"
        r = subprocess.run(
            [cc, "-O0", "-g", *extra, "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        out[key] = str(exe)
    return out


def _build(exe):
    return TargetDistance(exe, ["entry_calc", "helper_add"])


class TestCacheRoundTrip:
    @needs_cc
    def test_second_run_hits_cache(self, built_exe, monkeypatch):
        """Falsification: the second load must serve CFGs from disk."""
        calls = {"n": 0}
        orig = distance_mod.build_function_cfg

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(distance_mod, "build_function_cfg", counting)

        td = _build(built_exe["exe"])
        assert td.load(), "first load failed"
        assert calls["n"] > 0, "fixture decoded nothing; test cannot prove caching"
        after_first, first_cfgs = calls["n"], td._cfgs

        td2 = _build(built_exe["exe"])
        assert td2.load(), "second load failed"
        assert calls["n"] == after_first, (
            f"second run decoded {calls['n'] - after_first} function(s) "
            "instead of hitting the cache"
        )
        assert td2._cfgs == first_cfgs

    @needs_cc
    def test_artifact_lands_under_xdg(self, built_exe):
        assert _build(built_exe["exe"]).load()
        cache_root = Path(os.environ["XDG_CACHE_HOME"]) / "fuzzer_cfgcache"
        files = list(cache_root.glob("*.pkl.gz"))
        assert len(files) == 1


class TestInvalidation:
    @needs_cc
    def test_invalidation_on_byte_change(self, built_exe, monkeypatch):
        """Falsification: any byte change (no build-id → sha256 fallback)
        must invalidate. Uses the --build-id=none binary because a live
        NT_GNU_BUILD_ID deliberately wins over content hashing."""
        calls = {"n": 0}
        orig = distance_mod.build_function_cfg

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(distance_mod, "build_function_cfg", counting)

        assert _build(built_exe["nobid"]).load()
        after_first = calls["n"]

        blob = bytearray(Path(built_exe["nobid"]).read_bytes())
        blob[-1] ^= 0xFF
        Path(built_exe["nobid"]).write_bytes(blob)

        assert _build(built_exe["nobid"]).load()
        assert calls["n"] > after_first, "changed binary was served stale CFGs"

    @needs_cc
    def test_decoder_version_bump_invalidates(self, built_exe, monkeypatch):
        assert _build(built_exe["exe"]).load()
        monkeypatch.setattr(cfg_cache, "SCHEMA_VERSION", cfg_cache.SCHEMA_VERSION + 1)
        calls = {"n": 0}
        orig = distance_mod.build_function_cfg

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(distance_mod, "build_function_cfg", counting)
        assert _build(built_exe["exe"]).load()
        assert calls["n"] > 0, "schema bump did not force a fresh decode"


class TestAdversarial:
    @needs_cc
    def test_corrupt_file_recomputes(self, built_exe, caplog):
        assert _build(built_exe["exe"]).load()
        ident = cfg_cache.identity(built_exe["exe"])
        artifact = Path(cfg_cache._cache_dir()) / f"{ident}.pkl.gz"
        good = artifact.read_bytes()

        # Truncated in mid-payload...
        artifact.write_bytes(good[: len(good) // 3])
        with caplog.at_level(logging.WARNING, logger="fuzzer_tool.core.cfg_cache"):
            assert cfg_cache.load(ident) is None
        assert any("unreadable" in r.message for r in caplog.records)

        # ...and outright garbage.
        artifact.write_bytes(os.urandom(4096))
        assert cfg_cache.load(ident) is None

        # The pipeline still produces correct CFGs end to end.
        td = _build(built_exe["exe"])
        assert td.load()
        assert td._cfgs, "recompute after corruption produced no CFGs"

    @needs_cc
    def test_safe_unpickler_rejects_foreign_global(self, built_exe):
        """A crafted pickle referencing os.system must be refused, not run."""

        class Exploit:
            def __reduce__(self):
                return (os.system, ("echo pwned",))

        ident = cfg_cache.identity(built_exe["exe"])
        payload = {
            "schema": cfg_cache.SCHEMA_VERSION,
            "decoder_fp": cfg_cache.decoder_fingerprint(),
            "cfgs": {"evil": Exploit()},
        }
        artifact = Path(cfg_cache._cache_dir()) / f"{ident}.pkl.gz"
        with open(artifact, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as fh:
            pickle.dump(payload, fh)
        assert cfg_cache.load(ident) is None


class TestParallelEquivalence:
    @needs_cc
    def test_parallel_equals_serial(self, built_exe, monkeypatch):
        """Falsification: pool and serial decoders must agree exactly."""
        # Two target functions: below the func threshold unless lowered.
        monkeypatch.setattr(cfg_cache, "PARALLEL_MIN_FUNCS", 2)
        monkeypatch.setattr(cfg_cache, "PARALLEL_MIN_BYTES", 0)
        td_par = _build(built_exe["exe"])
        assert td_par.load()

        monkeypatch.setattr(cfg_cache, "PARALLEL_MIN_FUNCS", 10**9)
        monkeypatch.setattr(cfg_cache, "PARALLEL_MIN_BYTES", 10**12)
        td_ser = _build(built_exe["exe"])
        assert td_ser.load()

        assert td_par._cfgs == td_ser._cfgs
        assert len(td_par._cfgs) == 2


class TestIdentity:
    def test_build_id_system_binary(self):
        if not Path("/bin/true").exists():
            pytest.skip("no /bin/true")
        bid = build_id("/bin/true")
        assert bid is not None
        assert 8 <= len(bid) <= 32

    def test_non_elf_is_none(self):
        assert build_id("/etc/hostname") is None

    def test_missing_file_identity_none(self, tmp_path):
        assert cfg_cache.identity(str(tmp_path / "nope")) is None

    @needs_cc
    def test_nobid_falls_back_to_content_hash(self, built_exe):
        i_bid = cfg_cache.identity(built_exe["exe"])
        i_nobid = cfg_cache.identity(built_exe["nobid"])
        assert i_bid and i_nobid and i_bid != i_nobid


class TestToggles:
    def test_env_disable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FUZZER_DISABLE_CFG_CACHE", "1")
        assert cfg_cache.env_enabled() is False
        assert cfg_cache.load("deadbeef") is None
        assert cfg_cache.store("deadbeef", {}) is None

    def test_env_values(self, monkeypatch):
        for v in ("1", "true", "yes"):
            monkeypatch.setenv("FUZZER_DISABLE_CFG_CACHE", v)
            assert cfg_cache.env_enabled() is False
        monkeypatch.setenv("FUZZER_DISABLE_CFG_CACHE", "")
        assert cfg_cache.env_enabled() is True
        monkeypatch.delenv("FUZZER_DISABLE_CFG_CACHE")
        assert cfg_cache.env_enabled() is True
