"""seed_zip.compact: pruned seeds leave seeds.zip for seeds/pruned/.

Nothing is deleted (corpus rule): a pruned seed is written to
seeds/pruned/<hh>/id_<h> first, then the archive is rewritten without its
data member and tombstone. Superseded tombstones and duplicate members from
re-admission are dropped. Live, protected and foreign members are untouched.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from fuzzer_tool.adapters import seed_zip
from fuzzer_tool.adapters.filesystem import (
    hash_data,
    load_corpus,
    rehydrate_by_hash,
    save_irreplaceable,
    save_to_corpus,
)
from fuzzer_tool.adapters.seed_zip import SeedTree, ZipMode


@pytest.fixture
def zcorpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    store = seed_zip.configure(corpus, ZipMode.ON)
    yield corpus, store
    seed_zip.configure(corpus, ZipMode.OFF)


def _names(corpus: Path) -> list[str]:
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME) as zf:
        return zf.namelist()


def _cold_file(corpus: Path, data: bytes) -> Path:
    h = hash_data(data)
    return corpus / "seeds" / "pruned" / h[:2] / f"id_{h}"


def _closed(corpus: Path) -> None:
    seed_zip.configure(corpus, ZipMode.OFF)


def _live(corpus: Path) -> list[bytes]:
    return sorted(load_corpus(corpus, add_default=False)[0])


def _seeds(n: int) -> list[bytes]:
    return [f"seed-{i}".encode() * 5 for i in range(n)]


# ── behaviour ────────────────────────────────────────────────────────


def test_pruned_move_to_cold_and_rehydrate(zcorpus):
    corpus, store = zcorpus
    keep, drop = _seeds(2)
    for d in (keep, drop):
        save_to_corpus(d, corpus, set())
    store.retire(hash_data(drop))
    _closed(corpus)

    stats = seed_zip.compact(corpus)

    h_drop, h_keep = hash_data(drop), hash_data(keep)
    assert stats.moved == 1
    assert _cold_file(corpus, drop).read_bytes() == drop
    assert _names(corpus) == [f"{h_keep[:2]}/id_{h_keep}"]  # no data, no tombstone
    assert rehydrate_by_hash(h_drop, corpus) == drop
    assert _live(corpus) == [keep]


def test_idempotent(zcorpus):
    corpus, store = zcorpus
    a, b = _seeds(2)
    save_to_corpus(a, corpus, set())
    save_to_corpus(b, corpus, set())
    store.retire(hash_data(a))
    _closed(corpus)

    seed_zip.compact(corpus)
    before = (corpus / seed_zip.ZIP_NAME).read_bytes()
    again = seed_zip.compact(corpus)

    assert (again.moved, again.dropped) == (0, 0)
    assert (corpus / seed_zip.ZIP_NAME).read_bytes() == before


def test_readmitted_seed_stays_live_without_duplicates(zcorpus):
    corpus, store = zcorpus
    (d,) = _seeds(1)
    h = hash_data(d)
    save_to_corpus(d, corpus, set())
    store.retire(h)
    store.put(SeedTree.MAIN, h, d)  # re-admission: same name twice + tombstone
    _closed(corpus)

    stats = seed_zip.compact(corpus)

    assert stats.moved == 0
    assert _names(corpus) == [f"{h[:2]}/id_{h}"]
    assert _live(corpus) == [d]
    assert not _cold_file(corpus, d).exists()


def test_protected_and_foreign_members_kept(zcorpus):
    corpus, store = zcorpus
    (d,) = _seeds(1)
    h = hash_data(d)
    save_irreplaceable(d, corpus, set(), set())
    save_to_corpus(d, corpus, set())
    store.retire(h)
    store.flush()
    _closed(corpus)
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME, "a") as zf:
        zf.writestr("thirdparty/input.bin", b"foreign")

    seed_zip.compact(corpus)

    names = _names(corpus)
    assert f"irreplaceable/{h[:2]}/id_{h}" in names
    assert "thirdparty/input.bin" in names
    assert f"{h[:2]}/id_{h}" not in names
    assert _cold_file(corpus, d).read_bytes() == d


def test_no_zip_is_noop(tmp_path):
    stats = seed_zip.compact(tmp_path)

    assert (stats.moved, stats.dropped) == (0, 0)
    assert not (tmp_path / seed_zip.ZIP_NAME).exists()


def test_shrinks_archive(zcorpus):
    corpus, store = zcorpus
    seeds = [os.urandom(4096) for _ in range(20)]
    for d in seeds:
        save_to_corpus(d, corpus, set())
    for d in seeds[:15]:
        store.retire(hash_data(d))
    _closed(corpus)
    size = (corpus / seed_zip.ZIP_NAME).stat().st_size

    stats = seed_zip.compact(corpus)

    assert stats.moved == 15
    assert (corpus / seed_zip.ZIP_NAME).stat().st_size < size // 2
    assert _live(corpus) == sorted(seeds[15:])


# ── falsification ────────────────────────────────────────────────────


def test_nothing_lost_hot_plus_cold_equals_before(zcorpus):
    corpus, store = zcorpus
    seeds = _seeds(12)
    for d in seeds:
        save_to_corpus(d, corpus, set())
    for d in seeds[::3]:
        store.retire(hash_data(d))
    _closed(corpus)

    seed_zip.compact(corpus)

    for d in seeds[::3]:
        assert rehydrate_by_hash(hash_data(d), corpus) == d
    gone = {hash_data(d) for d in seeds[::3]}
    assert _live(corpus) == sorted(d for d in seeds if hash_data(d) not in gone)
    cold = {p.name for p in (corpus / "seeds" / "pruned").rglob("id_*")}
    assert cold == {f"id_{hash_data(d)}" for d in seeds[::3]}


def test_control_unpruned_archive_is_unchanged_in_content(zcorpus):
    corpus, _ = zcorpus
    seeds = _seeds(5)
    for d in seeds:
        save_to_corpus(d, corpus, set())
    _closed(corpus)

    stats = seed_zip.compact(corpus)

    assert (stats.moved, stats.dropped) == (0, 0)
    assert _live(corpus) == sorted(seeds)
    assert not (corpus / "seeds" / "pruned").exists()


# ── adversarial ──────────────────────────────────────────────────────


def test_refuses_while_store_open(zcorpus):
    corpus, store = zcorpus
    save_to_corpus(b"x" * 9, corpus, set())
    store.flush()

    with pytest.raises(seed_zip.StoreOpenError):
        seed_zip.compact(corpus)


def test_failed_swap_leaves_archive_and_rerun_succeeds(zcorpus, monkeypatch):
    corpus, store = zcorpus
    a, b = _seeds(2)
    save_to_corpus(a, corpus, set())
    save_to_corpus(b, corpus, set())
    store.retire(hash_data(a))
    _closed(corpus)
    path = corpus / seed_zip.ZIP_NAME
    before = path.read_bytes()
    real = os.replace

    def boom(src, dst):
        if Path(dst) == path:
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        seed_zip.compact(corpus)
    monkeypatch.setattr(os, "replace", real)

    assert path.read_bytes() == before  # original intact
    assert not list(corpus.glob("*.tmp"))  # no litter
    assert seed_zip.compact(corpus).moved == 1
    assert rehydrate_by_hash(hash_data(a), corpus) == a


def test_existing_cold_file_not_overwritten(zcorpus):
    corpus, store = zcorpus
    (d,) = _seeds(1)
    save_to_corpus(d, corpus, set())
    store.retire(hash_data(d))
    _closed(corpus)
    cold = _cold_file(corpus, d)
    cold.parent.mkdir(parents=True)
    cold.write_bytes(d)
    mtime = cold.stat().st_mtime_ns

    seed_zip.compact(corpus)

    assert cold.stat().st_mtime_ns == mtime
    assert _names(corpus) == []


def test_torn_archive_is_salvaged_then_compacted(zcorpus):
    corpus, store = zcorpus
    a, b = _seeds(2)
    save_to_corpus(a, corpus, set())
    save_to_corpus(b, corpus, set())
    store.retire(hash_data(a))
    store.flush()
    _closed(corpus)
    path = corpus / seed_zip.ZIP_NAME
    raw = path.read_bytes()
    path.write_bytes(raw[: raw.rfind(b"PK\x01\x02")])  # central directory lost

    stats = seed_zip.compact(corpus)

    assert stats.moved == 1
    assert _live(corpus) == [b]
    assert rehydrate_by_hash(hash_data(a), corpus) == a


def test_cli_compact_seeds(zcorpus, capsys, monkeypatch):
    import sys

    from fuzzer_tool.cli import commands

    corpus, store = zcorpus
    a, b = _seeds(2)
    save_to_corpus(a, corpus, set())
    save_to_corpus(b, corpus, set())
    store.retire(hash_data(a))
    _closed(corpus)

    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "compact-seeds", "-d", str(corpus)])

    assert commands.main() == 0
    assert "moved 1" in capsys.readouterr().out


# ── startup trigger: pruned/live ratio ───────────────────────────────


def _archive(corpus: Path, store, live: int, pruned: int) -> list[bytes]:
    seeds = _seeds(live + pruned)
    for d in seeds:
        save_to_corpus(d, corpus, set())
    for d in seeds[:pruned]:
        store.retire(hash_data(d))
    _closed(corpus)
    return seeds


def test_pruned_ratio(zcorpus):
    corpus, store = zcorpus
    _archive(corpus, store, live=2, pruned=3)

    assert seed_zip.pruned_ratio(corpus) == 1.5


def test_pruned_ratio_no_zip_is_zero(tmp_path):
    assert seed_zip.pruned_ratio(tmp_path) == 0.0


def test_compact_over_fires_above_threshold(zcorpus):
    corpus, store = zcorpus
    _archive(corpus, store, live=2, pruned=3)

    stats = seed_zip.compact_over(corpus, 1.0)

    assert stats.moved == 3
    assert seed_zip.pruned_ratio(corpus) == 0.0


def test_compact_over_holds_below_threshold(zcorpus):
    corpus, store = zcorpus
    _archive(corpus, store, live=2, pruned=3)
    before = (corpus / seed_zip.ZIP_NAME).read_bytes()

    stats = seed_zip.compact_over(corpus, 2.0)

    assert stats.moved == 0
    assert (corpus / seed_zip.ZIP_NAME).read_bytes() == before


def test_compact_over_all_pruned_fires(zcorpus):
    corpus, store = zcorpus
    _archive(corpus, store, live=0, pruned=2)

    assert seed_zip.compact_over(corpus, 100.0).moved == 2


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_compact_over_nonpositive_or_nan_is_off(zcorpus, bad):
    corpus, store = zcorpus
    _archive(corpus, store, live=1, pruned=3)

    assert seed_zip.compact_over(corpus, bad).moved == 0


def _fuzzer(corpus: Path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    return Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus),
        crashes_dir=str(corpus.parent / "crashes"),
        **kw,
    )


def test_fuzzer_compacts_at_startup(tmp_path):
    corpus = tmp_path / "corpus"
    store = seed_zip.configure(corpus, ZipMode.ON)
    seeds = _archive(corpus, store, live=2, pruned=3)
    try:
        f = _fuzzer(corpus, zip_seed_corpus=True, zip_compact_ratio=1.0)

        assert not any(n.startswith(".pruned/") for n in _names(corpus))
        assert rehydrate_by_hash(hash_data(seeds[0]), corpus) == seeds[0]
        assert sorted(f.corpus) == sorted(seeds[3:])
        f.save_to_corpus(b"new after compaction")  # store still writable
        f._save_state()
        with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME) as zf:
            assert b"new after compaction" in [zf.read(n) for n in zf.namelist()]
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_fuzzer_default_never_compacts(tmp_path):
    corpus = tmp_path / "corpus"
    store = seed_zip.configure(corpus, ZipMode.ON)
    _archive(corpus, store, live=1, pruned=3)
    before = (corpus / seed_zip.ZIP_NAME).read_bytes()
    try:
        _fuzzer(corpus, zip_seed_corpus=True)

        assert (corpus / seed_zip.ZIP_NAME).read_bytes() == before
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_fuzzer_survives_stale_open_store(tmp_path):
    corpus = tmp_path / "corpus"
    seed_zip.configure(corpus, ZipMode.ON)  # a previous Fuzzer left it open
    try:
        _fuzzer(corpus, zip_compact_ratio=1.0)  # must not raise StoreOpenError
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def _parsed_ratio(monkeypatch, *argv: str) -> float:
    import sys

    from fuzzer_tool.cli import commands

    captured: dict[str, float] = {}

    def _spy(args):
        captured["v"] = args.zip_compact_ratio
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *argv])
    assert commands.main() == 0
    return captured["v"]


def test_cli_ratio_default_off(monkeypatch):
    assert _parsed_ratio(monkeypatch) == 0.0


def test_cli_ratio_parses(monkeypatch):
    assert _parsed_ratio(monkeypatch, "--zip-compact-ratio", "0.5") == 0.5
