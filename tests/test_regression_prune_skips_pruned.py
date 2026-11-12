"""Regression: auto-minimize re-walked and re-moved every pruned seed.

``_prune_files`` ran ``rglob("id_*")`` over all of ``seeds/``, including
``seeds/pruned/``, and ``shutil.move``d each pruned file onto its own path on
every minimize: 3,202 pruned vs ~120 live seeds on a fuzzgoat corpus.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.services import corpus_manager
from fuzzer_tool.services.corpus_manager import CorpusManager


def _seed(root, rel: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return str(p)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    seeds = tmp_path / "seeds"
    paths = {
        "kept": _seed(seeds, "aa/id_aakept"),
        "drop": _seed(seeds, "bb/id_bbdrop"),
        "root_drop": _seed(seeds, "id_ccroot"),
        "old": _seed(seeds, "pruned/dd/id_ddold"),
        "readmitted": _seed(seeds, "pruned/ee/id_eeback"),
        "irreplaceable": _seed(seeds, "irreplaceable/ff/id_ffkeep"),
        "crashing": _seed(seeds, "crashing/11/id_11crash"),
        "timeouts": _seed(seeds, "timeouts/22/id_22hang"),
    }
    moves: list[tuple[str, str]] = []
    real_move = corpus_manager.shutil.move

    def spy(src, dst):
        moves.append((src, dst))
        return real_move(src, dst)

    monkeypatch.setattr(corpus_manager.shutil, "move", spy)
    cm = CorpusManager(SimpleNamespace(corpus_dir=tmp_path))
    return cm, seeds, paths, moves


def test_regression_pruned_files_not_rewalked(layout):
    cm, _seeds, paths, moves = layout
    cm._prune_files({"aakept", "eeback"})

    sources = {src for src, _ in moves}
    assert paths["old"] not in sources
    assert paths["readmitted"] not in sources


def test_live_unkept_seeds_still_pruned(layout):
    """Falsification: seeds outside the kept set still move under pruned/."""
    cm, seeds, paths, moves = layout
    cm._prune_files({"aakept", "eeback"})

    assert sorted(src for src, _ in moves) == sorted([paths["drop"], paths["root_drop"]])
    assert (seeds / "pruned/bb/id_bbdrop").is_file()
    assert (seeds / "pruned/cc/id_ccroot").is_file()
    assert (seeds / "aa/id_aakept").is_file()


def test_irreplaceable_never_moved(layout):
    """Adversarial: an empty kept set still leaves irreplaceable/ alone."""
    cm, seeds, paths, moves = layout
    cm._prune_files(set())

    assert paths["irreplaceable"] not in {src for src, _ in moves}
    assert (seeds / "irreplaceable/ff/id_ffkeep").is_file()


@pytest.mark.parametrize("kind", ["crashing", "timeouts"])
def test_regression_crash_seeds_never_pruned(layout, kind):
    """Crash/timeout seeds are not in the corpus, so no kept set names them."""
    cm, _seeds, paths, moves = layout
    cm._prune_files({"aakept"})

    assert paths[kind] not in {src for src, _ in moves}
    assert Path(paths[kind]).is_file()
