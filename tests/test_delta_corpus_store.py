"""Delta-compressed corpus store with keyframes (``--delta-corpus``).

soksolv ``reverse/nodestore.go``: store a child as a diff against its
parent, a full keyframe every ``SNAPSHOT_INTERVAL`` hops. Off by default;
``save_to_corpus`` writes full seeds unless handed a ``DeltaStore``.

The hazards that kept it disabled are pinned by ``test_regression_*``:
a live delta whose parent was pruned vanished on load, a corrupt delta
loaded as a seed under the wrong hash, and a delta pruned by ``fuzzer-tool
minimize`` (to ``pruned/deltas/``) could not be rehydrated. Every corpus
reader must see full bytes.
"""

from __future__ import annotations

import inspect
import json
import shutil

from fuzzer_tool.adapters import filesystem as fs
from fuzzer_tool.adapters.filesystem import (
    SNAPSHOT_INTERVAL,
    compute_delta,
    hash_data,
    load_corpus,
    rehydrate_by_hash,
    save_to_corpus,
)


def _child(parent: bytes, i: int) -> bytes:
    """Same-length one-byte mutation of *parent* (always delta-encodable)."""
    b = bytearray(parent)
    b[i % len(b)] = (b[i % len(b)] + 1 + i // len(b)) & 0xFF
    return bytes(b)


def _delta_path(d, data):
    return d / "deltas" / f"delta_{hash_data(data)}.json"


def _write_delta(d, parent, child, diff=None):
    rec = {"parent": hash_data(parent), "diff": diff or compute_delta(parent, child), "v": 1}
    out = _delta_path(d, child)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec))
    return out


def _chain(d, n, store, base=b"A" * 256):
    """Admit a lineage chain of *n* generations through the production writer."""
    seen: set[str] = set()
    seeds = [base]
    save_to_corpus(base, d, seen, delta_store=store)
    for gen in range(1, n + 1):
        child = _child(seeds[-1], gen)
        assert save_to_corpus(
            child, d, seen, parent=seeds[-1], lineage_depth=gen - 1, delta_store=store
        )
        seeds.append(child)
    return seeds


def _hops(d, h, limit=100):
    """Delta hops from *h* to a full entry, read off disk."""
    hops = 0
    while (p := d / "deltas" / f"delta_{h}.json").is_file() and hops < limit:
        h = json.loads(p.read_text())["parent"]
        hops += 1
    return hops


class TestRegressionHazards:
    def test_regression_delta_parent_pruned_load(self, tmp_path):
        """A live delta whose full parent sits in seeds/pruned/ must still load."""
        d, seen = tmp_path / "c", set()
        parent, child = b"P" * 64, b"P" * 63 + b"Q"
        save_to_corpus(parent, d, seen)
        _write_delta(d, parent, child)
        h = hash_data(parent)
        dest = d / "seeds" / "pruned" / h[:2]
        dest.mkdir(parents=True)
        shutil.move(str(d / "seeds" / h[:2] / f"id_{h}"), str(dest / f"id_{h}"))

        corpus, _, _ = load_corpus(d, add_default=False)
        assert corpus == [child]

    def test_regression_corrupt_delta_hash_mismatch(self, tmp_path):
        """A delta rebuilding to other bytes than its name is dropped, not loaded."""
        d, seen = tmp_path / "c", set()
        parent, child = b"P" * 64, b"P" * 63 + b"Q"
        save_to_corpus(parent, d, seen)
        _write_delta(d, parent, child, diff=[[0, ord("Z")]])

        corpus, seen_h, _ = load_corpus(d, add_default=False)
        assert corpus == [parent]
        assert hash_data(child) not in seen_h
        assert rehydrate_by_hash(hash_data(child), d) is None

    def test_regression_minimize_pruned_delta_rehydrates(self, tmp_path):
        """`fuzzer-tool minimize` prunes deltas/x to pruned/deltas/x."""
        d, seen = tmp_path / "c", set()
        parent, child = b"P" * 64, b"P" * 63 + b"Q"
        save_to_corpus(parent, d, seen)
        live = _write_delta(d, parent, child)
        dest = d / "pruned" / "deltas"
        dest.mkdir(parents=True)
        shutil.move(str(live), str(dest / live.name))
        assert rehydrate_by_hash(hash_data(child), d) == child


class TestWriter:
    def test_off_by_default_writes_full(self, tmp_path):
        """Falsification: without a DeltaStore nothing is delta-encoded."""
        d = tmp_path / "c"
        _chain(d, 3, None)
        assert not (d / "deltas").exists()

    def test_delta_written_and_round_trips(self, tmp_path):
        d = tmp_path / "c"
        seeds = _chain(d, 3, fs.DeltaStore(d))
        assert len(list((d / "deltas").glob("delta_*.json"))) == 3
        corpus, _, _ = load_corpus(d, add_default=False)
        assert sorted(corpus) == sorted(seeds)
        for s in seeds:
            assert rehydrate_by_hash(hash_data(s), d) == s

    def test_chain_bounded_by_interval(self, tmp_path):
        d = tmp_path / "c"
        seeds = _chain(d, 3 * SNAPSHOT_INTERVAL, fs.DeltaStore(d))
        hops = [_hops(d, hash_data(s)) for s in seeds]
        assert max(hops) < SNAPSHOT_INTERVAL
        assert max(hops) >= 2  # deltas really chain
        # Keyframes on the depth grid.
        for gen in range(SNAPSHOT_INTERVAL, len(seeds), SNAPSHOT_INTERVAL):
            assert hops[gen] == 0
        corpus, _, _ = load_corpus(d, add_default=False)
        assert sorted(corpus) == sorted(seeds)

    def test_keyframe_when_parent_not_stored(self, tmp_path):
        d, seen = tmp_path / "c", set()
        parent = b"X" * 64
        child = _child(parent, 1)
        save_to_corpus(child, d, seen, parent=parent, lineage_depth=1, delta_store=fs.DeltaStore(d))
        assert not (d / "deltas").exists()
        assert rehydrate_by_hash(hash_data(child), d) == child

    def test_resume_unknown_delta_parent_gets_keyframe(self, tmp_path):
        """A fresh store (resume) cannot see a delta parent's hops: write full."""
        d = tmp_path / "c"
        seeds = _chain(d, 2, fs.DeltaStore(d))
        child = _child(seeds[-1], 99)
        save_to_corpus(
            child, d, set(), parent=seeds[-1], lineage_depth=2, delta_store=fs.DeltaStore(d)
        )
        assert _hops(d, hash_data(child)) == 0

    def test_disk_bytes_smaller(self, tmp_path):
        on, off = tmp_path / "on", tmp_path / "off"
        _chain(on, 40, fs.DeltaStore(on), base=b"B" * 4096)
        _chain(off, 40, None, base=b"B" * 4096)

        def size(p):
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

        assert size(on) * 4 < size(off)


class TestAdversarial:
    def test_parent_retired_then_resume(self, tmp_path):
        from fuzzer_tool.services.corpus_manager import _retire_seed_file

        d = tmp_path / "c"
        seeds = _chain(d, 4, fs.DeltaStore(d))
        for s in seeds[:-1]:
            _retire_seed_file(d, hash_data(s))
        corpus, _, _ = load_corpus(d, add_default=False)
        assert corpus == [seeds[-1]]

    def test_parent_cycle_no_hang(self, tmp_path):
        d = tmp_path / "c"
        a, b = b"A" * 32, b"B" * 32
        (d / "deltas").mkdir(parents=True)
        for x, y in ((a, b), (b, a)):
            rec = {"parent": hash_data(y), "diff": [], "v": 1}
            _delta_path(d, x).write_text(json.dumps(rec))
        assert rehydrate_by_hash(hash_data(a), d) is None
        corpus, _, _ = load_corpus(d, add_default=False)
        assert corpus == []

    def test_truncated_delta_skipped(self, tmp_path):
        d = tmp_path / "c"
        seeds = _chain(d, 2, fs.DeltaStore(d))
        _delta_path(d, seeds[1]).write_text('{"parent": ')
        corpus, _, _ = load_corpus(d, add_default=False)
        # seeds[2] chains through the broken record: lost, but nothing else is.
        assert seeds[0] in corpus
        assert seeds[1] not in corpus

    def test_in_run_prune_keeps_child(self, tmp_path):
        """auto-minimize moving the parent's delta to deltas/pruned keeps the child."""
        d = tmp_path / "c"
        seeds = _chain(d, 3, fs.DeltaStore(d))
        src = _delta_path(d, seeds[1])
        h = hash_data(seeds[1])
        dest = d / "deltas" / "pruned" / h[:2]
        dest.mkdir(parents=True)
        shutil.move(str(src), str(dest / src.name))
        corpus, _, _ = load_corpus(d, add_default=False)
        assert seeds[1] not in corpus
        assert seeds[2] in corpus and seeds[3] in corpus


class TestReaders:
    def _corpus(self, tmp_path):
        d = tmp_path / "c"
        seeds = _chain(d, 3, fs.DeltaStore(d), base=b"tok_alpha tok_beta " * 8)
        return d, seeds

    def test_minimize_by_hash_sees_and_materializes(self, tmp_path):
        from fuzzer_tool.services.minimize import minimize_corpus

        d, seeds = self._corpus(tmp_path)
        out = tmp_path / "out"
        kept, removed = minimize_corpus("/bin/true", str(d), output_dir=str(out))
        assert kept == len(seeds) and removed == 0
        got = sorted(f.read_bytes() for f in out.rglob("id_*"))
        assert got == sorted(seeds)
        assert not list(out.rglob("delta_*"))

    def test_minimize_prune_in_place_keeps_rehydratable(self, tmp_path):
        from fuzzer_tool.services.minimize import minimize_corpus

        d, seeds = self._corpus(tmp_path)
        # Duplicate a delta seed as a full file so by-hash dedup prunes one copy.
        h = hash_data(seeds[2])
        (d / "seeds" / h[:2]).mkdir(parents=True, exist_ok=True)
        (d / "seeds" / h[:2] / f"id_{h}").write_bytes(seeds[2])
        minimize_corpus("/bin/true", str(d))
        for s in seeds:
            assert rehydrate_by_hash(hash_data(s), d) == s
        corpus, _, _ = load_corpus(d, add_default=False)
        assert sorted(set(corpus)) == sorted(seeds)

    def test_edge_export_reader(self, tmp_path):
        from fuzzer_tool.services.edge_export import _CorpusSeeds

        d, seeds = self._corpus(tmp_path)
        assert sorted(_CorpusSeeds(d, 0)) == sorted(seeds)

    def test_root_cause_reader(self, tmp_path):
        from fuzzer_tool.services.root_cause import _load_corpus

        d, seeds = self._corpus(tmp_path)
        assert sorted(data for _, data in _load_corpus(str(d))) == sorted(seeds)

    def test_autotoken_reader(self, tmp_path):
        from fuzzer_tool.services.import_corpus import build_autotoken_dictionary

        d = tmp_path / "c"
        base = b"." * 64
        seen: set[str] = set()
        store = fs.DeltaStore(d)
        save_to_corpus(base, d, seen, delta_store=store)
        child = b"." * 48 + b" zebratok " + b"." * 6
        save_to_corpus(child, d, seen, parent=base, lineage_depth=0, delta_store=store)
        assert _delta_path(d, child).is_file()
        assert b"zebratok" in build_autotoken_dictionary(str(d))

    def test_full_cover_index(self, tmp_path):
        from fuzzer_tool.services.full_cover import load_corpus as fc_load

        d, seeds = self._corpus(tmp_path)
        idx = fc_load(d)
        for s in seeds:
            assert idx.sizes[hash_data(s)] == len(s)


class TestWiring:
    def test_cli_flag(self):
        from fuzzer_tool.cli import commands

        src = inspect.getsource(commands)
        assert '"--delta-corpus"' in src
        assert "delta_corpus=getattr" in src
        assert "delta_corpus" not in commands._HAIL_MARY_FLAGS

    def test_corpus_manager_passes_store(self):
        from fuzzer_tool.services.corpus_manager import CorpusManager

        assert "delta_store=" in inspect.getsource(CorpusManager.save_to_corpus)


def _delta_ref(parent: bytes, child: bytes):
    """Reference: the original per-byte loop."""
    if len(parent) != len(child):
        return None
    diff = [[i, child[i]] for i in range(len(parent)) if parent[i] != child[i]]
    return None if len(diff) > len(child) // 4 else diff


class TestComputeDeltaVectorized:
    CASES = [
        (n, k)
        for n in (1, 10, fs._NP_DELTA_MIN - 1, fs._NP_DELTA_MIN, 1000)
        for k in (0, 1, 3, 400)
    ]

    def _pair(self, n, k):
        parent = bytes(range(256)) * (n // 256 + 1)
        parent = parent[:n]
        child = bytearray(parent)
        for j in range(min(k, n)):
            child[(j * 7) % n] ^= 0x5A
        return parent, bytes(child)

    def test_control_reference_against_itself(self):
        for n, k in self.CASES:
            p, c = self._pair(n, k)
            assert _delta_ref(p, c) == _delta_ref(p, c)

    def test_matches_reference(self):
        for n, k in self.CASES:
            p, c = self._pair(n, k)
            got = compute_delta(p, c)
            assert got == _delta_ref(p, c), (n, k)
            assert got is None or all(type(x) is int for pair in got for x in pair)

    def test_length_mismatch(self):
        assert compute_delta(b"a" * 300, b"a" * 301) is None
