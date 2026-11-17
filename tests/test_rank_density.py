"""``rank``: E/bit column, ``--sort density``, and the tracker key fix.

``cmd_rank`` read tracker edges under ``sha256(seed)[:16]``; the tracker keys
seeds by ``corpus_manager.seed_key`` (XXH64), so every tracker column read 0.
"""

from __future__ import annotations

import argparse

import pytest

from fuzzer_tool.cli.commands import cmd_rank
from fuzzer_tool.core.edge_density import edges_per_bit
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.corpus_manager import seed_key

DENSE = b"D" * 4
SPARSE = b"S" * 400
UNTRACKED = b"U" * 40
HITS = {DENSE: {1, 2, 3, 4}, SPARSE: {1, 2, 3, 4, 5, 6}}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    seeds = root / "seeds"
    seeds.mkdir(parents=True)
    for i, s in enumerate((SPARSE, UNTRACKED, DENSE)):
        (seeds / f"s{i}").write_bytes(s)

    et = EdgeTracker()
    for s, hit in HITS.items():
        et.record_edges(seed_key(s), hit)
    store = StateStore(root)
    store.set("edge_tracker", et.to_dict())
    store.save()
    return root


def _rows(capsys, corpus, tmp_path, sort=None) -> dict[str, list[str]]:
    target = tmp_path / "target"
    target.write_bytes(b"\x7fELF")
    target.chmod(0o755)
    args = argparse.Namespace(corpus=str(corpus), top=10, dump=None, target=str(target))
    if sort is not None:
        args.sort = sort
    assert cmd_rank(args) == 0

    out = capsys.readouterr().out
    rows = [ln.split() for ln in out.splitlines() if ln[:4].strip().isdigit()]
    return {seed_key(s): next(r for r in rows if seed_key(s) in r) for s in HITS | {UNTRACKED: 0}}


def test_regression_rank_tracker_key(capsys, corpus, tmp_path):
    """Edges come from the tracker under the corpus key, not 0."""
    rows = _rows(capsys, corpus, tmp_path)
    assert int(rows[seed_key(SPARSE)][2]) == len(HITS[SPARSE])


def test_rank_edges_per_bit_column(capsys, corpus, tmp_path):
    rows = _rows(capsys, corpus, tmp_path)
    for s, hit in HITS.items():
        assert f"{edges_per_bit(len(hit), len(s)):.4f}" in rows[seed_key(s)]


def test_rank_untracked_seed_zero(capsys, corpus, tmp_path):
    """Adversarial: a seed the tracker never saw reads 0 e/bit, no raise."""
    rows = _rows(capsys, corpus, tmp_path)
    assert f"{0.0:.4f}" in rows[seed_key(UNTRACKED)]


def test_rank_sort_density(capsys, corpus, tmp_path):
    """Densest first; the score order puts SPARSE (more edges) above DENSE."""
    by_score = _rows(capsys, corpus, tmp_path)
    by_density = _rows(capsys, corpus, tmp_path, sort="density")

    def pos(rows, s):
        return int(rows[seed_key(s)][0])

    assert pos(by_density, DENSE) < pos(by_density, SPARSE) < pos(by_density, UNTRACKED)
    assert pos(by_score, SPARSE) < pos(by_score, DENSE)
