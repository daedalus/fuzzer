"""Regression tests for ``core.debruijn_cache`` and its wiring into
``core.mutations.structured.de_bruijn_bytes`` / ``de_bruijn_bits``.

Covers handover 10f
(``docs/handover/handover_combinatorics_permutations_2026-09-02.md``):
the de Bruijn construction is a pure function of ``(k, n)`` and was
previously re-derived by every process in a parallel fuzzing campaign.
These tests exercise the disk cache directly (module-level, isolated
per test via a fresh ``XDG_CACHE_HOME``) and the subprocess-level
integration proving two independent Python processes actually share
one computed artifact.
"""

import importlib
import subprocess
import sys

import pytest


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point XDG_CACHE_HOME at a scratch dir and reload the cache module
    so its memoized ``_cache_dir_memo`` doesn't leak across tests."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FUZZER_DISABLE_DEBRUIJN_CACHE", raising=False)
    from fuzzer_tool.core import debruijn_cache

    importlib.reload(debruijn_cache)
    return debruijn_cache, tmp_path


def test_store_then_load_round_trips(isolated_cache):
    cache, _ = isolated_cache
    cache.store("bytes", "k4_n4", "fp123", b"hello world")
    assert cache.load("bytes", "k4_n4", "fp123") == b"hello world"


def test_load_miss_returns_none(isolated_cache):
    cache, _ = isolated_cache
    assert cache.load("bytes", "k4_n4", "fp123") is None


def test_different_fingerprint_is_a_separate_key(isolated_cache):
    # An algorithm-source edit changes the fingerprint; the old artifact
    # must not be returned under the new one.
    cache, _ = isolated_cache
    cache.store("bytes", "k4_n4", "fp_old", b"old sequence")
    assert cache.load("bytes", "k4_n4", "fp_new") is None
    assert cache.load("bytes", "k4_n4", "fp_old") == b"old sequence"


def test_different_kind_is_a_separate_key(isolated_cache):
    # de_bruijn_bytes and de_bruijn_bits must not collide on the same key.
    cache, _ = isolated_cache
    cache.store("bytes", "n8", "fp1", b"byte sequence")
    cache.store("bits", "n8", "fp1", b"bit sequence")
    assert cache.load("bytes", "n8", "fp1") == b"byte sequence"
    assert cache.load("bits", "n8", "fp1") == b"bit sequence"


def test_env_disable_skips_both_load_and_store(isolated_cache, monkeypatch):
    cache, tmp_path = isolated_cache
    monkeypatch.setenv("FUZZER_DISABLE_DEBRUIJN_CACHE", "1")
    cache.store("bytes", "k4_n4", "fp123", b"should not be written")
    assert cache.load("bytes", "k4_n4", "fp123") is None
    assert list(tmp_path.rglob("*.bin")) == []


def test_store_is_atomic_no_tmp_files_left_behind(isolated_cache):
    cache, tmp_path = isolated_cache
    cache.store("bytes", "k4_n4", "fp123", b"payload")
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []


def test_empty_data_is_not_stored(isolated_cache):
    cache, tmp_path = isolated_cache
    cache.store("bytes", "k4_n4", "fp123", b"")
    assert list(tmp_path.rglob("*.bin")) == []


def test_fingerprint_changes_when_source_changes():
    from fuzzer_tool.core import debruijn_cache

    def fn_v1():
        return 1

    def fn_v2():
        return 2

    assert debruijn_cache.fingerprint(fn_v1) != debruijn_cache.fingerprint(fn_v2)


def test_fingerprint_is_stable_for_same_source():
    from fuzzer_tool.core import debruijn_cache
    from fuzzer_tool.core.mutations.structured import _de_bruijn_symbols

    assert debruijn_cache.fingerprint(_de_bruijn_symbols) == debruijn_cache.fingerprint(
        _de_bruijn_symbols
    )


# ── Integration: two real subprocesses actually share one artifact ──────


def test_two_subprocesses_produce_identical_sequences_via_shared_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FUZZER_DISABLE_DEBRUIJN_CACHE", raising=False)
    script = (
        "from fuzzer_tool.core.mutations.structured import de_bruijn_bytes, de_bruijn_bits;"
        "b1 = de_bruijn_bytes(4, 4); b2 = de_bruijn_bits(8);"
        "print(len(b1), len(b2), b1.hex(), b2.hex())"
    )
    results = []
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(r.stdout)
    assert results[0] == results[1]
    assert list(tmp_path.rglob("*.bin")), "expected the cache dir to hold artifacts after run 1"


def test_disabling_cache_env_var_still_produces_correct_output(tmp_path, monkeypatch):
    # The mutator must keep working correctly with the cache fully
    # disabled -- disk caching is an optimization, never a correctness
    # dependency.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("FUZZER_DISABLE_DEBRUIJN_CACHE", "1")
    script = (
        "from fuzzer_tool.core.mutations.structured import de_bruijn_bytes, de_bruijn_bits;"
        "b1 = de_bruijn_bytes(4, 4); b2 = de_bruijn_bits(8);"
        "print(len(b1), len(b2))"
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r.stdout.strip() == "256 32"
    assert list(tmp_path.rglob("*.bin")) == []
