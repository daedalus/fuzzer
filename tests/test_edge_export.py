"""`edge-dump`: replay each corpus seed, pickle {seed_id: sorted uint32 edge ids}."""

import pickle
import sys

import numpy as np

from fuzzer_tool.adapters import process
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.cli import commands
from fuzzer_tool.services import edge_export


class _Shm:
    """ShmCoverage stand-in: one edge per distinct byte of the last input."""

    current: bytes = b""

    def __init__(self):
        self.env_id = "0"
        self.num_entries = 16

    def reset_edge_map(self):
        pass

    def edge_id_array(self):
        return np.array(sorted(set(type(self).current)), dtype=np.uint32) + 1

    def cleanup(self):
        pass


def _expected(data: bytes) -> list[int]:
    return [b + 1 for b in sorted(set(data))]


def _setup(tmp_path, monkeypatch, seeds: dict[str, bytes], blind: bool = False):
    """Corpus of *seeds* under seeds/; fake replay; returns (corpus, out)."""
    root = tmp_path / "corpus" / "seeds"
    root.mkdir(parents=True)
    for name, data in seeds.items():
        (root / name).write_bytes(data)

    def fake_run(target, data, timeout, env=None, **_):
        _Shm.current = b"" if blind else data
        return 0, "", 0

    monkeypatch.setattr(edge_export, "ShmCoverage", _Shm)
    monkeypatch.setattr(process, "run_target_stdin", fake_run)
    return root.parent, tmp_path / "edges.pkl"


def _load(out):
    with open(out, "rb") as f:
        return pickle.load(f)


def test_exports_edges_per_seed(tmp_path, monkeypatch):
    seeds = {"a": b"abc", "b": b"zz", "c": b"\x00\xff"}
    corpus, out = _setup(tmp_path, monkeypatch, seeds)

    n = edge_export.export_edges("t", str(corpus), str(out))

    got = _load(out)
    assert n == len(seeds)
    assert set(got) == {hash_data(d) for d in seeds.values()}
    for data in seeds.values():
        arr = got[hash_data(data)]
        assert arr.dtype == np.uint32
        assert arr.tolist() == _expected(data)


def test_twins_not_collapsed(tmp_path, monkeypatch):
    """Falsification: same edges, different bytes -> two entries (cmin would collapse)."""
    seeds = {"a": b"ab", "b": b"ba"}
    corpus, out = _setup(tmp_path, monkeypatch, seeds)

    edge_export.export_edges("t", str(corpus), str(out))

    got = _load(out)
    assert len(got) == len(seeds)
    assert got[hash_data(b"ab")].tolist() == got[hash_data(b"ba")].tolist()


def test_blackout_writes_nothing(tmp_path, monkeypatch, capsys):
    """Adversarial: no edges at all is a failed measurement, not an empty export."""
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"}, blind=True)

    assert edge_export.export_edges("t", str(corpus), str(out)) == 0
    assert not out.exists()
    assert "No edges" in capsys.readouterr().err


def test_empty_seed_and_pruned_excluded(tmp_path, monkeypatch):
    """Adversarial: empty input exports an empty array; pruned/ is not live corpus."""
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab", "e": b""})
    pruned = corpus / "seeds" / "pruned"
    pruned.mkdir()
    (pruned / "p").write_bytes(b"pp")

    edge_export.export_edges("t", str(corpus), str(out))

    got = _load(out)
    assert hash_data(b"pp") not in got
    assert got[hash_data(b"")].size == 0
    assert got[hash_data(b"ab")].tolist() == _expected(b"ab")


def test_cli_wired(tmp_path, monkeypatch):
    """`fuzzer-tool edge-dump` reaches export_edges and writes the pickle."""
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"})
    monkeypatch.setattr(commands, "_validate_target", lambda t: None)
    monkeypatch.setattr(
        sys, "argv", ["fuzzer-tool", "edge-dump", "t", "-d", str(corpus), "-o", str(out)]
    )

    assert commands.main() == 0
    assert _load(out)[hash_data(b"ab")].tolist() == _expected(b"ab")


def test_running_stats(tmp_path, monkeypatch, capsys):
    """Progress line matches minimize's: rate, ETA, elapsed, unique edges, rss."""
    seeds = {"a": b"ab", "b": b"bc"}
    corpus, out = _setup(tmp_path, monkeypatch, seeds)

    edge_export.export_edges("t", str(corpus), str(out))

    line = capsys.readouterr().out.split("\r")[-1]
    union = set(_expected(b"ab")) | set(_expected(b"bc"))
    assert f"Replayed {len(seeds)}/{len(seeds)}" in line
    assert f"edges: {len(union)}" in line
    for field in ("seeds/s", "ETA:", "elapsed:", "rss:"):
        assert field in line
