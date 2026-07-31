"""Tests for lineage crash-path replay: rehydrate_by_hash + _lineage_candidate."""

import json
from pathlib import Path

from fuzzer_tool.adapters.filesystem import (
    compute_delta,
    compute_delta_v2,
    hash_data,
    rehydrate_by_hash,
)
from fuzzer_tool.services.tmin import _lineage_candidate


def _write_seed(corpus_dir: Path, data: bytes, subdir: str = "seeds") -> str:
    h = hash_data(data)
    d = corpus_dir / subdir / h[:2]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"id_{h}").write_bytes(data)
    return h


def _write_delta(corpus_dir: Path, parent: bytes, child: bytes, subdir: str = "deltas") -> str:
    h = hash_data(child)
    parent_h = hash_data(parent)
    diff_v1 = compute_delta(parent, child)
    if diff_v1 is not None:
        rec = {"parent": parent_h, "diff": diff_v1, "v": 1}
    else:
        rec = {"parent": parent_h, "diff": compute_delta_v2(parent, child), "v": 2}
    d = corpus_dir / subdir / h[:2]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"delta_{h}.json").write_text(json.dumps(rec))
    return h


class TestRehydrateByHash:
    def test_full_seed(self, tmp_path):
        data = b"hello world" * 4
        h = _write_seed(tmp_path, data)
        assert rehydrate_by_hash(h, tmp_path) == data

    def test_pruned_seed(self, tmp_path):
        data = b"pruned data" * 4
        h = _write_seed(tmp_path, data, subdir="seeds/pruned")
        assert rehydrate_by_hash(h, tmp_path) == data

    def test_irreplaceable_seed(self, tmp_path):
        data = b"irreplaceable" * 4
        h = _write_seed(tmp_path, data, subdir="seeds/irreplaceable")
        assert rehydrate_by_hash(h, tmp_path) == data

    def test_v1_delta_reconstruction(self, tmp_path):
        parent = b"base input bytes for delta" * 2
        child = bytearray(parent)
        child[3] = 0x42
        child[10] = 0xAB
        child = bytes(child)
        _write_seed(tmp_path, parent)
        h = _write_delta(tmp_path, parent, child)
        assert rehydrate_by_hash(h, tmp_path) == child

    def test_v2_delta_reconstruction(self, tmp_path):
        parent = b"base input for v2" * 2
        # Length-changing edit: insert a byte.
        child = parent[:5] + b"XY" + parent[5:]
        _write_seed(tmp_path, parent)
        h = _write_delta(tmp_path, parent, child)
        assert rehydrate_by_hash(h, tmp_path) == child

    def test_pruned_delta_reconstruction(self, tmp_path):
        parent = b"parent bytes here"
        child = parent[:4] + b"Z" + parent[4:]
        _write_seed(tmp_path, parent)
        h = _write_delta(tmp_path, parent, child, subdir="deltas/pruned")
        assert rehydrate_by_hash(h, tmp_path) == child

    def test_missing_hash_returns_none(self, tmp_path):
        assert rehydrate_by_hash("deadbeefdeadbeef", tmp_path) is None

    def test_cyclic_delta_chain_terminates(self, tmp_path):
        # h1 -> parent h2 -> parent h1: cyclic; must return None, not hang.
        h1 = "1111111111111111"
        h2 = "2222222222222222"
        d = tmp_path / "deltas"
        (d / h1[:2]).mkdir(parents=True, exist_ok=True)
        (d / h2[:2]).mkdir(parents=True, exist_ok=True)
        (d / h1[:2] / f"delta_{h1}.json").write_text(
            json.dumps({"parent": h2, "diff": [[0, 0, 1]], "v": 2})
        )
        (d / h2[:2] / f"delta_{h2}.json").write_text(
            json.dumps({"parent": h1, "diff": [[0, 0, 1]], "v": 2})
        )
        assert rehydrate_by_hash(h1, tmp_path) is None

    def test_deep_legal_chain_resolvable(self, tmp_path):
        # save_to_corpus caps real delta chains at SNAPSHOT_INTERVAL (20) hops;
        # a 20-hop chain must fully resolve.
        parent = b"root"
        _write_seed(tmp_path, parent)
        cur = parent
        for _ in range(20):
            child = cur + b"x"
            h = _write_delta(tmp_path, cur, child)
            cur = child
        assert rehydrate_by_hash(h, tmp_path) == cur


class TestLineageCandidate:
    def _make_corpus(self, tmp_path):
        root = b"ROOT"
        mid = b"MIDDLESEED"
        parent = b"PARENTSEED123"
        corpus = tmp_path / "corpus"
        _write_seed(corpus, root)
        _write_seed(corpus, mid)
        _write_seed(corpus, parent)
        # state.json: parent -> mid -> root chain with edge annotations.
        state = {
            "seed_meta": {
                hash_data(parent): {
                    "parent_key": hash_data(mid),
                    "parent_ops": ["bitflip"],
                    "parent_sites": [2],
                },
                hash_data(mid): {
                    "parent_key": hash_data(root),
                    "parent_ops": ["byte_flip"],
                    "parent_sites": [1],
                },
                hash_data(root): {"parent_key": None, "parent_ops": [], "parent_sites": []},
            }
        }
        (corpus / "state.json").write_text(json.dumps(state))
        return corpus, root, mid, parent

    def _write_sidecar(self, crash_path: Path, parent: bytes):
        sidecar = crash_path.with_suffix(".txt")
        sidecar.write_text(
            f"parent_seed:   {hash_data(parent)}\n"
            "mutation_ops:  bitflip, havoc\n"
            "mutation_sites: 2, 5\n"
        )

    def test_returns_root_most_crashing_ancestor(self, tmp_path):
        corpus, root, mid, parent = self._make_corpus(tmp_path)
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        self._write_sidecar(crash, parent)

        # Only the root (smallest) crashes.
        def is_crash(data, sig):
            return "sig" if data == root else None

        assert _lineage_candidate(crash, str(corpus), is_crash, "sig") == root

    def test_returns_first_crashing_from_root_down(self, tmp_path):
        corpus, root, mid, parent = self._make_corpus(tmp_path)
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        self._write_sidecar(crash, parent)

        # Root and mid both crash → root (first, smallest) wins.
        def is_crash(data, sig):
            return "sig" if data in (root, mid) else None

        assert _lineage_candidate(crash, str(corpus), is_crash, "sig") == root

    def test_no_crashing_ancestor_returns_none(self, tmp_path):
        corpus, root, mid, parent = self._make_corpus(tmp_path)
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        self._write_sidecar(crash, parent)
        assert _lineage_candidate(crash, str(corpus), lambda d, s: None, "sig") is None

    def test_missing_sidecar_returns_none(self, tmp_path):
        corpus, root, mid, parent = self._make_corpus(tmp_path)
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        assert _lineage_candidate(crash, str(corpus), lambda d, s: "sig", "sig") is None

    def test_missing_state_json_returns_none(self, tmp_path):
        corpus, root, mid, parent = self._make_corpus(tmp_path)
        (corpus / "state.json").unlink()
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        self._write_sidecar(crash, parent)
        assert _lineage_candidate(crash, str(corpus), lambda d, s: "sig", "sig") is None

    def test_no_corpus_dir_returns_none(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        self._write_sidecar(crash, b"parent")
        assert _lineage_candidate(crash, None, lambda d, s: "sig", "sig") is None

    def test_skips_missing_seed_bytes(self, tmp_path):
        corpus, root, mid, parent = self._make_corpus(tmp_path)
        # Delete the mid seed file → mid can't be rehydrated, but root can.
        (corpus / "seeds" / hash_data(mid)[:2] / f"id_{hash_data(mid)}").unlink()
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        self._write_sidecar(crash, parent)
        assert _lineage_candidate(crash, str(corpus), lambda d, s: "sig", "sig") == root

    def test_cyclic_parent_chain_terminates(self, tmp_path):
        corpus, root, mid, parent = self._make_corpus(tmp_path)
        # Make the chain cyclic: parent -> parent.
        state = {"seed_meta": {hash_data(parent): {"parent_key": hash_data(parent)}}}
        (corpus / "state.json").write_text(json.dumps(state))
        # Delete the seed file so nothing rehydrates — the cycle must
        # terminate (no hang) and yield None, not recurse forever.
        (corpus / "seeds" / hash_data(parent)[:2] / f"id_{hash_data(parent)}").unlink()
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"X" * 100)
        self._write_sidecar(crash, parent)
        assert _lineage_candidate(crash, str(corpus), lambda d, s: "sig", "sig") is None
