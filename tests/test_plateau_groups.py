"""--plateau-groups: union seeds whose mutants exit to each other's exact path.

soksolv fess_moves.go:341-415 (reversible-group exit dedup).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.plateau import RETAIN_FRAC, PlateauGroups
from tests import test_commands_extended

H_A, H_B, H_C = 0xA, 0xB, 0xC


def _groups(max_keys: int = 64) -> PlateauGroups:
    return PlateauGroups(max_keys=max_keys)


# ── union-find ──────────────────────────────────────────────────────


def test_unknown_key_is_its_own_group():
    """Falsification: nothing observed, nothing merged."""
    g = _groups()
    assert g.group("a") == "a"
    assert g.group("b") == "b"
    assert len(g) == 0


def test_boring_exit_to_other_seed_unions():
    g = _groups()
    hashes = {"a": H_A, "b": H_B}
    assert g.observe("a", H_B, hashes)
    assert g.group("a") == g.group("b")


def test_exit_to_own_path_or_unknown_path_is_no_union():
    """Falsification: a mutant replaying its parent, or a new path, merges nothing."""
    g = _groups()
    hashes = {"a": H_A, "b": H_B}
    assert not g.observe("a", H_A, hashes)
    assert not g.observe("a", H_C, hashes)
    assert g.group("a") != g.group("b")
    assert len(g) == 0


def test_union_is_transitive_and_by_size():
    """b joins the larger group {a, c}; its root is that group's."""
    g = _groups()
    hashes = {"a": H_A, "b": H_B, "c": H_C}
    g.observe("a", H_C, hashes)
    root = g.group("a")
    g.observe("b", H_A, hashes)
    assert g.group("b") == g.group("c") == root
    assert g.size(root) == 3


def test_stale_index_entry_is_revalidated():
    """Adversarial: a seed whose path changed is not matched on its old hash."""
    g = _groups()
    hashes = {"a": H_A, "b": H_B}
    assert not g.observe("a", H_C, hashes)  # builds the index
    hashes["b"] = H_C  # same size: the index is stale, not rebuilt by size
    assert not g.observe("a", H_B, hashes)
    assert g.observe("a", H_C, hashes)
    assert g.group("a") == g.group("b")


def test_path_compression_flattens_chain():
    g = _groups()
    keys = [f"k{i}" for i in range(8)]
    hashes = {k: i + 1 for i, k in enumerate(keys)}
    for i in range(len(keys) - 1):
        g.observe(keys[i + 1], i + 1, hashes)
    root = g.group(keys[-1])
    assert all(g.group(k) == root for k in keys)
    assert all(g._parent[k] == root for k in keys)


def test_bounded_and_retains_live_keys():
    """Adversarial: a full table drops dead keys; live groups survive the rebuild."""
    n = 8
    g = _groups(max_keys=n)
    hashes = {f"k{i}": i + 1 for i in range(n)}
    for i in range(0, n, 2):
        g.observe(f"k{i}", i + 2, hashes)  # pair k_i with k_{i+1}
    assert len(g) == n

    live = {k: v for k, v in hashes.items() if k not in ("k2", "k3", "k4")}
    assert len(live) <= n * RETAIN_FRAC
    live["new_a"], live["new_b"] = 100, 101
    assert g.observe("new_a", 101, live)

    assert len(g) <= n
    assert "k2" not in g._parent and "k3" not in g._parent
    assert g.group("k0") == g.group("k1")
    assert g.group("k6") == g.group("k7")
    assert g.group("new_a") == g.group("new_b")
    assert g.group("k5") == "k5"  # partner k4 died: singleton now


def test_full_table_of_live_keys_refuses_new():
    """Adversarial: no dead keys to drop, so a new union is skipped, not grown."""
    n = 4
    g = _groups(max_keys=n)
    hashes = {f"k{i}": i + 1 for i in range(n)} | {"x": 50, "y": 51}
    g.observe("k0", 2, hashes)
    g.observe("k2", 4, hashes)
    assert len(g) == n
    assert not g.observe("x", 51, hashes)
    assert len(g) == n
    assert g.group("x") == "x"


def test_state_roundtrip():
    g = _groups()
    hashes = {"a": H_A, "b": H_B}
    g.observe("a", H_B, hashes)
    h = PlateauGroups.from_dict(g.to_dict(), max_keys=64)
    assert h.group("a") == h.group("b")
    assert PlateauGroups.from_dict({"bad": 1}, max_keys=64).group("a") == "a"


# ── wiring ──────────────────────────────────────────────────────────


@pytest.fixture
def fuzzer():
    from fuzzer_tool.services.fuzzer import Fuzzer

    with tempfile.TemporaryDirectory(prefix="plateau_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
                max_len=256,
                timeout=1,
                plateau_groups=True,
            )
        yield f


def test_fuzzer_builds_groups_only_when_asked(fuzzer):
    from fuzzer_tool.services.fuzzer import Fuzzer

    assert isinstance(fuzzer._plateau, PlateauGroups)
    with tempfile.TemporaryDirectory(prefix="plateau_off_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            off = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
            )
        assert off._plateau is None


def _boring_round(f, parent: bytes, path_hash: int, timeout: bool = False):
    from fuzzer_tool.services.fuzz_round import FuzzRound

    f.shm_cov = MagicMock()
    f.shm_cov.read_path_hash.return_value = path_hash
    r = FuzzRound(f, parent)
    r._mutated = parent + b"!"
    r._is_timeout = timeout
    return r._on_boring()


def test_on_boring_unions_parent_with_matching_seed(fuzzer):
    a, b = b"seed-a", b"seed-b"
    ka, kb = fuzzer._seed_key(a), fuzzer._seed_key(b)
    fuzzer._edge_tracker.seed_path_hash.update({ka: H_A, kb: H_B})

    assert _boring_round(fuzzer, a, H_B) is False
    assert fuzzer._plateau.group(ka) == fuzzer._plateau.group(kb)


def test_on_boring_skips_timeout_and_zero_hash(fuzzer):
    """Falsification: a truncated run or a shim without the hash merges nothing."""
    a, b = b"seed-a", b"seed-b"
    ka, kb = fuzzer._seed_key(a), fuzzer._seed_key(b)
    fuzzer._edge_tracker.seed_path_hash.update({ka: H_A, kb: H_B})

    _boring_round(fuzzer, a, H_B, timeout=True)
    _boring_round(fuzzer, a, 0)
    assert fuzzer._plateau.group(ka) != fuzzer._plateau.group(kb)


def test_lottery_group_signal_reads_plateau(fuzzer):
    fuzzer._plateau.observe("ka", H_B, {"ka": H_A, "kb": H_B})
    signals = fuzzer._seed_picker._seed_signals({"ka": b"a", "kb": b"b", "kc": b"c"})
    assert signals["group"]("ka") == signals["group"]("kb")
    assert signals["group"]("kc") == "kc"


def test_state_persists(fuzzer):
    fuzzer._plateau.observe("ka", H_B, {"ka": H_A, "kb": H_B})
    fuzzer._save_learned()
    fuzzer._plateau = PlateauGroups()
    fuzzer.resume = True
    fuzzer._load_learned()
    assert fuzzer._plateau.group("ka") == fuzzer._plateau.group("kb")


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.plateau_groups = flag
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["plateau_groups"] is flag
