"""environ_copy() matches os.environ.copy() without re-decoding every spawn.

os.environ.copy() decodes every variable (~84us at 160 vars); the spawn
path paid it per exec. The cache must still see every os.environ change.
"""

import os

import pytest

from fuzzer_tool.adapters import process

_KEY = "FUZZER_TEST_ENVIRON_COPY"


@pytest.fixture
def decodes(monkeypatch) -> dict:
    """Fresh cache plus a count of full decodes."""
    monkeypatch.setattr(process, "_ENV_CACHE", process._EnvCache())
    calls = {"n": 0}
    real = process._decode_environ

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(process, "_decode_environ", counting)
    return calls


def test_matches_os_environ_copy(decodes):
    """Falsification: the copy equals the stdlib's own."""
    assert process.environ_copy() == os.environ.copy()


def test_unchanged_environ_decodes_once(decodes):
    """Falsification: repeated spawns with no env change reuse the decode."""
    for _ in range(5):
        process.environ_copy()
    assert decodes["n"] == 1


@pytest.mark.parametrize("value", ["a", "b"])
def test_sees_new_and_changed_vars(decodes, monkeypatch, value):
    """Adversarial: set, then change the value under the same key."""
    process.environ_copy()
    monkeypatch.setenv(_KEY, "first")
    assert process.environ_copy()[_KEY] == "first"

    monkeypatch.setenv(_KEY, value)
    assert process.environ_copy()[_KEY] == value
    assert decodes["n"] == 3


def test_sees_removed_var(decodes, monkeypatch):
    """Adversarial: a deleted variable must not survive in the cache."""
    monkeypatch.setenv(_KEY, "x")
    assert _KEY in process.environ_copy()
    monkeypatch.delenv(_KEY)
    assert _KEY not in process.environ_copy()


def test_caller_mutation_does_not_poison(decodes):
    """Adversarial: the spawn path writes __AFL_SHM_ID into its copy."""
    first = process.environ_copy()
    first[_KEY] = "leak"
    first.pop("PATH", None)

    second = process.environ_copy()
    assert _KEY not in second
    assert second == os.environ.copy()
