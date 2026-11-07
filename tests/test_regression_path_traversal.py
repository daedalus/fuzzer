"""Regression: corpus/crash/download paths must stay inside their roots.

A shared corpus can ship symlinked dirs and crafted hashes; targets print
attacker-shaped sanitizer reports; downloaders read upstream lists. Each test
plants a ``victim`` dir outside the root and asserts it is untouched/unread.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.adapters import filesystem as fs
from fuzzer_tool.adapters.filesystem import hash_data, load_corpus, save_to_corpus

_SECRET = b"secret-config\n"


@pytest.fixture
def victim(tmp_path):
    d = tmp_path / "victim"
    d.mkdir()
    (d / "config.ini").write_bytes(_SECRET)
    return d


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "corpus"
    (d / "seeds").mkdir(parents=True)
    return d


def _names(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


# ── corpus loader ────────────────────────────────────────────────────


def test_regression_loader_skips_symlinked_shard(corpus, victim):
    """seeds/ab -> victim: every victim file was renamed to id_<hash>."""
    (corpus / "seeds" / "ab").symlink_to(victim)
    got, _, _ = load_corpus(corpus, add_default=False)
    assert _names(victim) == ["config.ini"]
    assert _SECRET not in got


def test_loader_skips_symlinked_root(corpus, victim, tmp_path):
    """Adversarial: the canonical root itself (seeds/irreplaceable) is a link."""
    (corpus / "seeds" / "irreplaceable").symlink_to(victim)
    got, _, _ = load_corpus(corpus, add_default=False)
    assert _names(victim) == ["config.ini"]
    assert _SECRET not in got


def test_iter_seeds_skips_symlinked_shard(corpus, victim):
    (corpus / "seeds" / "ab").symlink_to(victim)
    assert _SECRET not in list(fs._iter_seeds(corpus))


def test_loader_still_normalizes_real_shard(corpus):
    """Falsification: a misnamed seed in a real shard is still renamed."""
    shard = corpus / "seeds" / "ab"
    shard.mkdir()
    (shard / "misnamed").write_bytes(b"seed")
    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == [b"seed"]
    assert _names(shard) == [f"id_{hash_data(b'seed')}"]


# ── rehydrate_by_hash ────────────────────────────────────────────────


def _traversal_hash(corpus: Path) -> str:
    """``xx/../..`` climbs out of seeds/ once seeds/xx/id_xx exists."""
    (corpus / "seeds" / "xx" / "id_xx").mkdir(parents=True)
    return "xx/../../../../victim/config.ini"


def test_regression_rehydrate_rejects_traversal(corpus, victim):
    h = _traversal_hash(corpus)
    assert (corpus / "seeds" / h[:2] / f"id_{h}").resolve() == victim / "config.ini"
    assert fs.rehydrate_by_hash(h, corpus) is None


def test_regression_rehydrate_delta_parent_rejected(corpus, victim):
    """Adversarial: a valid delta whose parent field climbs out."""
    import json

    parent = _traversal_hash(corpus)
    deltas = corpus / "deltas"
    deltas.mkdir()
    h = "0123456789abcdef"
    (deltas / f"delta_{h}.json").write_text(json.dumps({"parent": parent, "diff": []}))
    assert fs.rehydrate_by_hash(h, corpus) is None


def test_rehydrate_valid_hash(corpus):
    """Falsification: a real 16-hex hash still rehydrates."""
    save_to_corpus(b"seed", corpus, set())
    assert fs.rehydrate_by_hash(hash_data(b"seed"), corpus) == b"seed"


# ── prune destinations ───────────────────────────────────────────────


def test_regression_retire_refuses_symlinked_pruned(corpus, victim):
    from fuzzer_tool.services.corpus_manager import _retire_seed_file

    save_to_corpus(b"seed", corpus, set())
    (corpus / "seeds" / "pruned").symlink_to(victim)
    h = hash_data(b"seed")
    assert not _retire_seed_file(corpus, h)
    assert _names(victim) == ["config.ini"]
    assert (corpus / "seeds" / h[:2] / f"id_{h}").is_file()


def test_regression_prune_files_refuses_symlinked_pruned(corpus, victim):
    from fuzzer_tool.services.corpus_manager import CorpusManager

    save_to_corpus(b"drop", corpus, set())
    (corpus / "seeds" / "pruned").symlink_to(victim)
    mgr = SimpleNamespace(f=SimpleNamespace(corpus_dir=corpus))
    CorpusManager._prune_files(mgr, set())
    assert _names(victim) == ["config.ini"]


def test_prune_files_symlinked_shard_dest(corpus, victim):
    """Adversarial: pruned/ is real but its shard pruned/<hh> is a link."""
    from fuzzer_tool.services.corpus_manager import CorpusManager

    save_to_corpus(b"drop", corpus, set())
    h = hash_data(b"drop")
    (corpus / "seeds" / "pruned").mkdir()
    (corpus / "seeds" / "pruned" / h[:2]).symlink_to(victim)
    mgr = SimpleNamespace(f=SimpleNamespace(corpus_dir=corpus))
    CorpusManager._prune_files(mgr, set())
    assert _names(victim) == ["config.ini"]


def test_retire_still_moves(corpus):
    """Falsification: a real pruned/ still receives the seed."""
    from fuzzer_tool.services.corpus_manager import _retire_seed_file

    save_to_corpus(b"seed", corpus, set())
    h = hash_data(b"seed")
    assert _retire_seed_file(corpus, h)
    assert (corpus / "seeds" / "pruned" / h[:2] / f"id_{h}").is_file()


def test_regression_minimize_commit_refuses_symlinked_pruned(tmp_path, victim):
    from fuzzer_tool.services import minimize

    corpus = tmp_path / "mc"
    corpus.mkdir()
    seed = corpus / "seed"
    seed.write_bytes(b"a")
    (corpus / "pruned").symlink_to(victim)
    minimize._commit_results([seed], [], None, corpus)
    assert _names(victim) == ["config.ini"]
    assert seed.is_file()


# ── symlinked sidecars / import sources ──────────────────────────────


def test_regression_minimize_skips_symlinked_sidecar(tmp_path, victim):
    from fuzzer_tool.services import minimize

    corpus = tmp_path / "mc"
    corpus.mkdir()
    seed = corpus / "seed"
    seed.write_bytes(b"a")
    (corpus / "seed.txt").symlink_to(victim / "config.ini")
    out = tmp_path / "out"
    minimize._commit_results([seed], [str(seed)], str(out), corpus)
    assert _names(out) == ["seed"]


def test_regression_full_cover_skips_symlinked_sidecar(corpus, victim, tmp_path):
    from fuzzer_tool.services import full_cover

    save_to_corpus(b"seed", corpus, set())
    sid = hash_data(b"seed")
    path = corpus / "seeds" / sid[:2] / f"id_{sid}"
    path.with_suffix(".txt").symlink_to(victim / "config.ini")
    out = tmp_path / "out"
    full_cover._write_cover(full_cover.load_corpus(corpus), corpus, {sid}, out)
    assert _names(out / "seeds" / sid[:2]) == [f"id_{sid}"]


def test_regression_import_skips_symlinked_sources(tmp_path, victim):
    from fuzzer_tool.services import import_corpus

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "real").write_bytes(b"real")
    (lib / "link").symlink_to(victim / "config.ini")
    dest = tmp_path / "dest"
    assert import_corpus.import_from_libfuzzer(str(lib), str(dest)) == 1
    got, _, _ = load_corpus(dest, add_default=False)
    assert got == [b"real"]


def test_regression_afl_import_skips_symlinks(tmp_path, victim):
    """Adversarial: queue file and crash sidecar both symlinked."""
    from fuzzer_tool.services import import_corpus

    afl = tmp_path / "afl"
    (afl / "queue").mkdir(parents=True)
    (afl / "crashes").mkdir()
    (afl / "queue" / "link").symlink_to(victim / "config.ini")
    (afl / "crashes" / "id_0").write_bytes(b"boom")
    (afl / "crashes" / "id_0.txt").symlink_to(victim / "config.ini")
    crashes = tmp_path / "crashes"
    seeds, n_crash = import_corpus.import_from_afl(str(afl), str(tmp_path / "dest"), str(crashes))
    assert (seeds, n_crash) == (0, 1)
    assert [p.suffix for p in crashes.iterdir()] == [".bin"]


# ── crash names from sanitizer output ────────────────────────────────


@pytest.mark.parametrize("raw", ["a/../../b", "/abs/x.c:3:5", "..", "x\x00y"])
def test_regression_safe_component(raw):
    out = fs.safe_component(raw)
    assert "/" not in out and "\x00" not in out and out not in ("", ".", "..")


def test_safe_component_keeps_signature_chars():
    """Falsification: ordinary signature characters survive."""
    sig = "asan:heap-buffer-overflow@foo@x.c:3:5"
    assert fs.safe_component(sig) == sig


def test_regression_crash_name_parts_no_slash():
    report = SimpleNamespace(
        is_valid=lambda: True, error_type="../../etc/x", sanitizer="AddressSanitizer"
    )
    error_short, _ = fs._crash_name_parts(report, 1)
    assert "/" not in error_short


def test_regression_sweep_crash_name_no_slash():
    from fuzzer_tool.cli.commands import _sweep_crash_name

    name = _sweep_crash_name("0123456789abcdef", "ubsan:x@../libavcodec/x.c:3:5")
    assert "/" not in name and name.startswith("crash_0123456789ab_")


# ── downloaders ──────────────────────────────────────────────────────


def _tools():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


@pytest.mark.parametrize("bad", ["../../x", "/abs/dir", "a/b", ".."])
def test_regression_fate_list_rejects_bad_fuzzer(bad):
    mod = pytest.importorskip("tools.extract_ffmpeg_seeds")
    assert mod.parse_fate_list(f"123/456 {bad}\n") == []


def test_fate_list_keeps_valid_fuzzer():
    """Falsification: a plain fuzzer name parses."""
    mod = pytest.importorskip("tools.extract_ffmpeg_seeds")
    got = mod.parse_fate_list("123/456 tools_target_dec_h264_fuzzer\n")
    assert [e["fuzzer"] for e in got] == ["tools_target_dec_h264_fuzzer"]


def test_regression_group_corpus_skips_escaping_symlink(tmp_path, victim):
    mod = pytest.importorskip("tools.group_seed_corpus")
    src = tmp_path / "fate"
    src.mkdir()
    (src / "a.mp4").write_bytes(b"x")
    (src / "inside.mp4").symlink_to(src / "a.mp4")
    (src / "out.mp4").symlink_to(victim / "config.ini")
    assert sorted(p.name for p in mod.parse_corpus(src)) == ["a.mp4", "inside.mp4"]
