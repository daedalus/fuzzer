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

    ctor_calls: list = []

    def __init__(self, num_entries=16, **kwargs):
        type(self).ctor_calls.append((num_entries, kwargs))
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
    """Every pickled batch in *out*, merged."""
    merged: dict = {}
    with open(out, "rb") as f:
        while True:
            try:
                merged.update(pickle.load(f))
            except EOFError:
                return merged


def test_exports_edges_per_seed(tmp_path, monkeypatch):
    seeds = {"a": b"abc", "b": b"zz", "c": b"\x00\xff"}
    corpus, out = _setup(tmp_path, monkeypatch, seeds)

    n = edge_export.export_edges("t", str(corpus), [str(out)])

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

    edge_export.export_edges("t", str(corpus), [str(out)])

    got = _load(out)
    assert len(got) == len(seeds)
    assert got[hash_data(b"ab")].tolist() == got[hash_data(b"ba")].tolist()


def test_blackout_writes_nothing(tmp_path, monkeypatch, capsys):
    """Adversarial: no edges at all is a failed measurement, not an empty export."""
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab"}, blind=True)

    assert edge_export.export_edges("t", str(corpus), [str(out)]) == 0
    assert not out.exists()
    assert "No edges" in capsys.readouterr().err


def test_empty_seed_and_pruned_excluded(tmp_path, monkeypatch):
    """Adversarial: empty input exports an empty array; pruned/ is not live corpus."""
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"ab", "e": b""})
    pruned = corpus / "seeds" / "pruned"
    pruned.mkdir()
    (pruned / "p").write_bytes(b"pp")

    edge_export.export_edges("t", str(corpus), [str(out)])

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

    edge_export.export_edges("t", str(corpus), [str(out)])

    line = capsys.readouterr().out.split("\r")[-1]
    union = set(_expected(b"ab")) | set(_expected(b"bc"))
    assert f"Replayed {len(seeds)}/{len(seeds)}" in line
    assert f"edges: {len(union)}" in line
    for field in ("seeds/s", "ETA:", "elapsed:", "rss:"):
        assert field in line


def test_reports_avg_and_max_edges_per_seed(tmp_path, monkeypatch, capsys):
    """Summary line: mean and peak edges per replayed seed, from the ids written."""
    seeds = {"a": b"abc", "b": b"zz", "c": b"\x01\x02\x03\x04"}
    corpus, out = _setup(tmp_path, monkeypatch, seeds)

    edge_export.export_edges("t", str(corpus), [str(out)])

    counts = [len(_expected(d)) for d in seeds.values()]
    line = capsys.readouterr().out.split("\r")[-1]
    assert f"avg {sum(counts) / len(counts):.1f}" in line
    assert f"max {max(counts)}" in line
    assert f"({len(seeds)} seeds)" in line


def test_avg_counts_edgeless_seeds_and_ignores_skipped(tmp_path, monkeypatch, capsys):
    """Adversarial: an edgeless seed is one replayed seed (avg 2.0, not 4.0); an
    already-exported seed is not replayed, so it must not move the average."""
    corpus, out = _setup(tmp_path, monkeypatch, {"a": b"abc"})
    edge_export.export_edges("t", str(corpus), [str(out)])

    (corpus / "seeds" / "e").write_bytes(b"")
    (corpus / "seeds" / "f").write_bytes(b"wxyz")
    edge_export.export_edges("t", str(corpus), [str(out)])

    line = capsys.readouterr().out.split("\r")[-1]
    assert "avg 2.0" in line
    assert "max 4" in line
    assert "(2 seeds)" in line


def test_table_sized_by_target_and_touched_scan_past_crossover(tmp_path, monkeypatch):
    """Perf regression: a target-sized table makes the per-seed full-table walk
    the dominant cost (edge-dump 1647 -> 864 seeds/s at 524288 entries), so past
    TOUCHED_SCAN_MIN_ENTRIES the replay reads the touched bitmap instead."""
    from fuzzer_tool.adapters.shm import TOUCHED_SCAN_MIN_ENTRIES

    for entries, want in ((8192, False), (TOUCHED_SCAN_MIN_ENTRIES, True), (524288, True)):
        _Shm.ctor_calls.clear()
        corpus, out = _setup(tmp_path / str(entries), monkeypatch, {"a": b"ab"})
        monkeypatch.setattr(edge_export, "estimate_map_size", lambda target, e=entries: e)

        edge_export.export_edges("t", str(corpus), [str(out)])

        assert _Shm.ctor_calls == [(entries, {"touched_scan": want})]
