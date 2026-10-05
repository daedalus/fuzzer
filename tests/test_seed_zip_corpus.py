"""--zip-seed-corpus: seeds read from seeds/ AND seeds.zip, written only to
seeds.zip, appended in blocks at deflate level 9.

Covers routing of every seed writer (corpus, irreplaceable, crashing,
timeouts), retire/prune tombstones, rehydration, block flushing, salvage of
an archive whose central directory was lost mid-append, and the CLI gate.
"""

from __future__ import annotations

import zipfile
import zlib
from pathlib import Path

import pytest

from fuzzer_tool.adapters import seed_zip
from fuzzer_tool.adapters.filesystem import (
    hash_data,
    load_corpus,
    rehydrate_by_hash,
    save_crashing_seed,
    save_irreplaceable,
    save_timeout_seed,
    save_to_corpus,
)
from fuzzer_tool.adapters.seed_zip import SeedTree, ZipMode


@pytest.fixture
def zcorpus(tmp_path: Path):
    """Corpus dir with zip mode ON; always switched OFF afterwards."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    store = seed_zip.configure(corpus, ZipMode.ON)
    yield corpus, store
    seed_zip.configure(corpus, ZipMode.OFF)


def _members(corpus: Path) -> list[str]:
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME) as zf:
        return zf.namelist()


def _kill(corpus: Path) -> None:
    """Drop the store as SIGKILL would: fd closed, nothing more written."""
    store = seed_zip._STORES.pop(Path(corpus))
    if store._zf is not None:
        store._zf.fp.close()
        store._zf.fp = None


def _seed_files(corpus: Path) -> list[Path]:
    return [p for p in (corpus / "seeds").rglob("id_*") if p.is_file()]


# ── write routing ────────────────────────────────────────────────────


def test_corpus_seed_lands_only_in_zip(zcorpus):
    corpus, store = zcorpus
    data = b"PNG-ish seed " * 7

    assert save_to_corpus(data, corpus, set())
    store.flush()

    h = hash_data(data)
    assert _members(corpus) == [f"{h[:2]}/id_{h}"]
    assert _seed_files(corpus) == []  # falsification: no loose file


def test_protected_trees_route_to_zip_subtrees(zcorpus):
    corpus, store = zcorpus
    seen: set[str] = set()
    irep: set[str] = set()

    save_irreplaceable(b"irep", corpus, seen, irep)
    save_crashing_seed(b"crash", corpus, seen, irep)
    save_timeout_seed(b"hang", corpus, seen, irep)
    store.flush()

    names = set(_members(corpus))
    for tree, data in (("irreplaceable", b"irep"), ("crashing", b"crash"), ("timeouts", b"hang")):
        h = hash_data(data)
        assert f"{tree}/{h[:2]}/id_{h}" in names
    assert _seed_files(corpus) == []


def test_repeat_crash_is_not_appended_twice(zcorpus):
    corpus, store = zcorpus
    seen: set[str] = set()
    irep: set[str] = set()

    save_crashing_seed(b"crash", corpus, seen, irep)
    save_crashing_seed(b"crash", corpus, seen, irep)  # pending dup
    store.flush()
    save_crashing_seed(b"crash", corpus, seen, irep)  # flushed dup
    store.flush()

    assert len(_members(corpus)) == 1


def test_deltas_stay_in_deltas_dir(zcorpus):
    corpus, store = zcorpus
    parent = bytes(range(64))
    child = bytearray(parent)
    child[3] ^= 0xFF

    save_to_corpus(bytes(child), corpus, set(), parent=parent)
    store.flush()

    assert list((corpus / "deltas").glob("delta_*.json"))
    assert not (corpus / seed_zip.ZIP_NAME).exists()


# ── block mode and compression ───────────────────────────────────────


def test_block_is_held_until_threshold(tmp_path):
    corpus = tmp_path / "c"
    store = seed_zip.configure(corpus, ZipMode.ON, block_seeds=3)
    try:
        save_to_corpus(b"a1", corpus, set())
        save_to_corpus(b"a2", corpus, set())
        assert not (corpus / seed_zip.ZIP_NAME).exists()

        save_to_corpus(b"a3", corpus, set())  # third seed fills the block
        assert len(_members(corpus)) == 3

        save_to_corpus(b"a4", corpus, set())
        assert len(_members(corpus)) == 3
        store.flush()
        assert len(_members(corpus)) == 4
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_default_store_buffers(zcorpus):
    corpus, store = zcorpus
    assert seed_zip.BLOCK_SEEDS > 1
    save_to_corpus(b"one", corpus, set())
    assert not (corpus / seed_zip.ZIP_NAME).exists()  # held for the block


def test_block_grows_with_archive(tmp_path):
    corpus = tmp_path / "c"
    store = seed_zip.configure(corpus, ZipMode.ON, block_seeds=2, growth=8)
    try:
        for i in range(32):  # 32 entries -> block becomes 32 // 8 = 4
            store.put(SeedTree.MAIN, f"{i:016x}", b"x")
        store.flush()
        base = len(_members(corpus))

        for i in range(3):
            store.put(SeedTree.MAIN, f"{100 + i:016x}", b"y")
        assert len(_members(corpus)) == base  # fixed block of 2 would have flushed
        store.put(SeedTree.MAIN, f"{200:016x}", b"y")
        assert len(_members(corpus)) == base + 4
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_growth_zero_keeps_fixed_block(tmp_path):
    corpus = tmp_path / "c"
    store = seed_zip.configure(corpus, ZipMode.ON, block_seeds=2, growth=0)
    try:
        for i in range(32):
            store.put(SeedTree.MAIN, f"{i:016x}", b"x")
        store.flush()
        base = len(_members(corpus))
        store.put(SeedTree.MAIN, f"{100:016x}", b"y")
        store.put(SeedTree.MAIN, f"{101:016x}", b"y")
        assert len(_members(corpus)) == base + 2
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_byte_budget_flushes_large_seed(tmp_path):
    corpus = tmp_path / "c"
    seed_zip.configure(corpus, ZipMode.ON, block_seeds=1000, block_bytes=1024)
    try:
        save_to_corpus(b"x" * 2048, corpus, set())
        assert len(_members(corpus)) == 1
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_members_are_deflate_level_9(zcorpus):
    corpus, store = zcorpus
    data = bytes(i * 7 % 251 for i in range(4096)) + b"tail" * 300
    save_to_corpus(data, corpus, set())
    store.flush()

    # Expected size derived from raw deflate at level 9, not a literal.
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    want = len(co.compress(data) + co.flush())
    co1 = zlib.compressobj(1, zlib.DEFLATED, -15)
    fast = len(co1.compress(data) + co1.flush())
    assert want != fast  # control: payload discriminates the levels

    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME) as zf:
        (info,) = zf.infolist()
    assert info.compress_type == zipfile.ZIP_DEFLATED
    assert info.compress_size == want


def test_archive_valid_between_blocks_handle_open(zcorpus):
    """Every flush leaves a readable archive while the writer stays open."""
    corpus, store = zcorpus
    for i in range(3):
        save_to_corpus(f"blk{i}".encode() * 9, corpus, set())
        store.flush()
        assert len(_members(corpus)) == i + 1
    assert store._zf is not None  # one handle for the whole run


def test_flush_falls_back_to_close(zcorpus, monkeypatch):
    corpus, store = zcorpus
    monkeypatch.setattr(seed_zip, "_checkpoint", lambda zf: False)
    for i in range(2):
        save_to_corpus(f"fb{i}".encode(), corpus, set())
        store.flush()
        assert store._zf is None
    assert len(_members(corpus)) == 2


def test_append_preserves_earlier_blocks(zcorpus):
    corpus, store = zcorpus
    datas = [f"seed-{i}".encode() * 5 for i in range(10)]
    for d in datas[:5]:
        save_to_corpus(d, corpus, set())
    store.flush()
    for d in datas[5:]:
        save_to_corpus(d, corpus, set())
    store.flush()

    got, _, _ = load_corpus(corpus, add_default=False)
    assert sorted(got) == sorted(datas)


# ── read side ────────────────────────────────────────────────────────


def test_load_unions_files_and_zip(zcorpus):
    corpus, store = zcorpus
    h = hash_data(b"from-file")
    (corpus / "seeds" / h[:2]).mkdir(parents=True)
    (corpus / "seeds" / h[:2] / f"id_{h}").write_bytes(b"from-file")

    save_to_corpus(b"from-zip", corpus, set())
    store.flush()

    got, seen, _ = load_corpus(corpus, add_default=False)
    assert sorted(got) == [b"from-file", b"from-zip"]
    assert seen == {h, hash_data(b"from-zip")}


def test_same_seed_in_both_loads_once(zcorpus):
    corpus, store = zcorpus
    h = hash_data(b"dup")
    (corpus / "seeds" / h[:2]).mkdir(parents=True)
    (corpus / "seeds" / h[:2] / f"id_{h}").write_bytes(b"dup")
    store.put(SeedTree.MAIN, h, b"dup")
    store.flush()

    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == [b"dup"]


def test_zip_protected_seeds_marked_irreplaceable(zcorpus):
    corpus, store = zcorpus
    save_crashing_seed(b"crash", corpus, set(), set())
    store.flush()

    _, _, irep = load_corpus(corpus, add_default=False)
    assert irep == {hash_data(b"crash")}


def test_rehydrate_finds_pending_and_flushed(zcorpus):
    corpus, store = zcorpus
    save_to_corpus(b"pending", corpus, set())
    assert rehydrate_by_hash(hash_data(b"pending"), corpus) == b"pending"

    store.flush()
    assert rehydrate_by_hash(hash_data(b"pending"), corpus) == b"pending"


def test_delta_child_of_zip_parent_resolves(zcorpus):
    corpus, store = zcorpus
    parent = bytes(range(64))
    child = bytearray(parent)
    child[3] ^= 0xFF
    save_to_corpus(parent, corpus, set())
    save_to_corpus(bytes(child), corpus, set(), parent=parent)
    store.flush()

    got, _, _ = load_corpus(corpus, add_default=False)
    assert sorted(got) == sorted([parent, bytes(child)])
    assert rehydrate_by_hash(hash_data(bytes(child)), corpus) == bytes(child)


# ── retire / prune ───────────────────────────────────────────────────


def test_retire_hides_from_load_but_rehydrates(zcorpus):
    corpus, store = zcorpus
    save_to_corpus(b"keep", corpus, set())
    save_to_corpus(b"drop", corpus, set())
    store.flush()

    store.retire(hash_data(b"drop"))
    store.flush()

    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == [b"keep"]
    assert rehydrate_by_hash(hash_data(b"drop"), corpus) == b"drop"
    assert list(store.pruned()) == [b"drop"]


def test_retire_does_not_hide_protected_copy(zcorpus):
    corpus, store = zcorpus
    h = hash_data(b"both")
    store.put(SeedTree.MAIN, h, b"both")
    store.put(SeedTree.IRREPLACEABLE, h, b"both")
    store.retire(h)
    store.flush()

    got, _, irep = load_corpus(corpus, add_default=False)
    assert got == [b"both"]
    assert irep == {h}


def test_readmission_after_retire_wins(zcorpus):
    corpus, store = zcorpus
    h = hash_data(b"again")
    store.put(SeedTree.MAIN, h, b"again")
    store.retire(h)
    store.put(SeedTree.MAIN, h, b"again")
    store.flush()

    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == [b"again"]


def test_retire_unknown_hash_is_noop(zcorpus):
    corpus, store = zcorpus
    store.retire("0" * 16)
    store.flush()
    assert not (corpus / seed_zip.ZIP_NAME).exists()


# ── mode gate ────────────────────────────────────────────────────────


def test_mode_off_writes_files_but_still_loads_zip(tmp_path):
    """Without --zip-seed-corpus an existing seeds.zip is read, never written."""
    corpus = tmp_path / "c"
    store = seed_zip.configure(corpus, ZipMode.ON)
    save_to_corpus(b"zipped", corpus, set())
    store.flush()
    assert seed_zip.configure(corpus, ZipMode.OFF) is None
    before = (corpus / seed_zip.ZIP_NAME).read_bytes()

    save_to_corpus(b"loose", corpus, set())
    assert len(_seed_files(corpus)) == 1  # writes still go to files

    got, _, _ = load_corpus(corpus, add_default=False)
    assert sorted(got) == [b"loose", b"zipped"]
    assert seed_zip.lookup(corpus) is None  # read-only: no store registered
    assert (corpus / seed_zip.ZIP_NAME).read_bytes() == before


def test_mode_off_zip_respects_tombstones_and_protection(tmp_path):
    corpus = tmp_path / "c"
    store = seed_zip.configure(corpus, ZipMode.ON)
    save_to_corpus(b"keep" * 8, corpus, set())
    gone = b"gone" * 8
    save_to_corpus(gone, corpus, set())
    store.retire(hash_data(gone))
    store.put(SeedTree.CRASHING, hash_data(b"crash" * 8), b"crash" * 8)
    seed_zip.configure(corpus, ZipMode.OFF)

    got, _, irr = load_corpus(corpus, add_default=False)
    assert sorted(got) == sorted([b"keep" * 8, b"crash" * 8])
    assert hash_data(b"crash" * 8) in irr


def test_configure_off_flushes_pending(tmp_path):
    corpus = tmp_path / "c"
    seed_zip.configure(corpus, ZipMode.ON)
    save_to_corpus(b"late", corpus, set())
    seed_zip.configure(corpus, ZipMode.OFF)
    assert len(_members(corpus)) == 1


def test_lookup_is_per_corpus(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    seed_zip.configure(a, ZipMode.ON)
    try:
        assert seed_zip.lookup(a) is not None
        assert seed_zip.lookup(b) is None
    finally:
        seed_zip.configure(a, ZipMode.OFF)


# ── adversarial archives ─────────────────────────────────────────────


def test_salvage_after_lost_central_directory(zcorpus):
    corpus, store = zcorpus
    datas = [f"s{i}".encode() * 40 for i in range(6)]
    for d in datas:
        save_to_corpus(d, corpus, set())
    store.flush()

    # Simulate a kill mid-append: local entries intact, CD and EOCD gone.
    _kill(corpus)
    path = corpus / seed_zip.ZIP_NAME
    with zipfile.ZipFile(path) as zf:
        cd_start = min(i.header_offset for i in zf.infolist()) + sum(
            30 + len(i.filename.encode()) + len(i.extra) + i.compress_size for i in zf.infolist()
        )
    raw = path.read_bytes()
    path.write_bytes(raw[:cd_start] + raw[cd_start : cd_start + 7])  # torn CD

    store = seed_zip.configure(corpus, ZipMode.ON)

    got, _, _ = load_corpus(corpus, add_default=False)
    assert sorted(got) == sorted(datas)
    assert list(corpus.glob(seed_zip.ZIP_NAME + ".corrupt*"))  # original kept

    save_to_corpus(b"after", corpus, set())  # archive appendable again
    store.flush()
    assert len(_members(corpus)) == len(datas) + 1


def test_salvage_drops_torn_last_member(zcorpus):
    corpus, store = zcorpus
    save_to_corpus(b"whole" * 20, corpus, set())
    save_to_corpus(b"torn" * 50, corpus, set())
    store.flush()

    _kill(corpus)
    path = corpus / seed_zip.ZIP_NAME
    with zipfile.ZipFile(path) as zf:
        last = zf.infolist()[-1]
    raw = path.read_bytes()
    path.write_bytes(raw[: last.header_offset + 40])  # mid-header/data

    seed_zip.configure(corpus, ZipMode.ON)
    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == [b"whole" * 20]


def test_hostile_member_names_ignored(zcorpus):
    """Traversal and absolute names are never corpus data; other names are
    foreign members and load by content (see test_foreign_members_*)."""
    corpus, _ = zcorpus
    seed_zip.configure(corpus, ZipMode.OFF)
    good = hash_data(b"ok")
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME, "w") as zf:
        zf.writestr("../../etc/id_" + "a" * 16, b"evil1")
        zf.writestr("/abs/id_" + "b" * 16, b"evil4")
        zf.writestr("dir/", b"")
        zf.writestr(f"{good[:2]}/id_{good}", b"ok")
    seed_zip.configure(corpus, ZipMode.ON)

    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == [b"ok"]


def _foreign_zip(corpus: Path) -> list[bytes]:
    """seeds.zip as a third party builds it: arbitrary flat/nested names."""
    datas = [b"RIFF-one" * 4, b"OggS-two" * 4, b"\x1aE\xdf\xa3mkv" * 4]
    corpus.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME, "w") as zf:
        zf.writestr("a.avi", datas[0])
        zf.writestr("sub/b.ogg", datas[1])
        zf.writestr("zz/id_nothex", datas[2])
        zf.writestr("a_copy.avi", datas[0])  # duplicate content
    return datas


def test_foreign_members_load_readonly(tmp_path):
    corpus = tmp_path / "c"
    datas = _foreign_zip(corpus)
    before = (corpus / seed_zip.ZIP_NAME).read_bytes()

    got, seen, _ = load_corpus(corpus, add_default=False)
    assert sorted(got) == sorted(datas)
    assert seen == {hash_data(d) for d in datas}
    assert (corpus / seed_zip.ZIP_NAME).read_bytes() == before


def test_foreign_members_adopted_under_zip_mode(tmp_path):
    corpus = tmp_path / "c"
    datas = _foreign_zip(corpus)
    store = seed_zip.configure(corpus, ZipMode.ON)
    try:
        got, _, _ = load_corpus(corpus, add_default=False)
        assert sorted(got) == sorted(datas)
        store.flush()
        names = _members(corpus)
        for d in datas:
            h = hash_data(d)
            assert f"{h[:2]}/id_{h}" in names  # canonical, so prune can retire it

        h0 = hash_data(datas[0])
        assert store.retire(h0)
        store.flush()
        got, _, _ = load_corpus(corpus, add_default=False)
        assert sorted(got) == sorted(datas[1:])  # pruned stays pruned across loads
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_misnamed_member_keyed_by_content(zcorpus):
    """Same as the file loader: content, not name, decides the hash."""
    corpus, _ = zcorpus
    seed_zip.configure(corpus, ZipMode.OFF)
    h = hash_data(b"real")
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME, "w") as zf:
        zf.writestr(f"{h[:2]}/id_{h}", b"forged")
    seed_zip.configure(corpus, ZipMode.ON)

    got, seen, _ = load_corpus(corpus, add_default=False)
    assert got == [b"forged"]
    assert seen == {hash_data(b"forged")}
    assert rehydrate_by_hash(h, corpus) is None  # never hand back wrong bytes


def _zip_names(corpus: Path) -> list[str]:
    path = corpus / seed_zip.ZIP_NAME
    if not path.is_file():
        return []
    with zipfile.ZipFile(path) as zf:
        return sorted(zf.namelist())


def test_zip_mode_does_not_copy_loose_seed_into_zip(tmp_path):
    """seeds/ and seeds.zip are one pool: a seed in either is already held."""
    corpus = tmp_path / "c"
    nested, flat = b"loose-nested" * 3, b"loose-flat" * 3
    hn, hf = hash_data(nested), hash_data(flat)
    (corpus / "seeds" / hn[:2]).mkdir(parents=True)
    (corpus / "seeds" / hn[:2] / f"id_{hn}").write_bytes(nested)
    (corpus / "seeds" / f"id_{hf}").write_bytes(flat)  # what load normalises to
    store = seed_zip.configure(corpus, ZipMode.ON)
    try:
        # fresh seen set: models eviction / a second tracker re-offering them
        assert save_to_corpus(nested, corpus, set())
        assert save_to_corpus(flat, corpus, set())
        store.flush()
        assert _zip_names(corpus) == []
        got, _, _ = load_corpus(corpus, add_default=False)
        assert sorted(got) == sorted([nested, flat])
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_zip_mode_off_does_not_copy_zip_seed_to_files(tmp_path):
    corpus = tmp_path / "c"
    data = b"zip-resident" * 3
    store = seed_zip.configure(corpus, ZipMode.ON)
    save_to_corpus(data, corpus, set())
    store.flush()
    seed_zip.configure(corpus, ZipMode.OFF)

    assert save_to_corpus(data, corpus, set())
    assert _seed_files(corpus) == []
    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == [data]


def test_zip_mode_readmits_pruned_zip_seed(tmp_path):
    """Present means live: a tombstoned seed is still re-admitted."""
    corpus = tmp_path / "c"
    data = b"pruned-then-back" * 3
    store = seed_zip.configure(corpus, ZipMode.ON)
    try:
        save_to_corpus(data, corpus, set())
        store.retire(hash_data(data))
        save_to_corpus(data, corpus, set())
        store.flush()
        got, _, _ = load_corpus(corpus, add_default=False)
        assert got == [data]
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


# ── CLI ──────────────────────────────────────────────────────────────


def _parsed_flag(monkeypatch, *argv: str) -> bool:
    """Run the shipped main() with cmd_fuzz spied (see test_regression_coverage_default)."""
    import sys

    from fuzzer_tool.cli import commands

    captured: dict[str, bool] = {}

    def _spy(args):
        captured["v"] = args.zip_seed_corpus
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", _spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *argv])
    assert commands.main() == 0
    return captured["v"]


def test_cli_flag_default_off(monkeypatch):
    assert _parsed_flag(monkeypatch) is False


def test_cli_flag_parses(monkeypatch):
    assert _parsed_flag(monkeypatch, "--zip-seed-corpus") is True


def test_hail_mary_leaves_layout_alone(monkeypatch):
    assert _parsed_flag(monkeypatch, "--hail-mary") is False


# ── service wiring ───────────────────────────────────────────────────


def test_retire_seed_file_tombstones(zcorpus):
    from fuzzer_tool.services.corpus_manager import _retire_seed_file

    corpus, store = zcorpus
    save_to_corpus(b"old", corpus, set())

    assert _retire_seed_file(corpus, hash_data(b"old"))
    got, _, _ = load_corpus(corpus, add_default=False)
    assert got == []


def test_prune_files_tombstones_unkept(zcorpus):
    from types import SimpleNamespace

    from fuzzer_tool.services.corpus_manager import CorpusManager

    corpus, store = zcorpus
    (corpus / "seeds").mkdir()
    for d in (b"keep", b"drop", b"drop2"):
        save_to_corpus(d, corpus, set())
    save_irreplaceable(b"irep", corpus, set(), set())

    mgr = SimpleNamespace(f=SimpleNamespace(corpus_dir=corpus))
    CorpusManager._prune_files(mgr, {hash_data(b"keep")})

    got, _, _ = load_corpus(corpus, add_default=False)
    assert sorted(got) == [b"irep", b"keep"]
    assert sorted(store.pruned()) == [b"drop", b"drop2"]


def _fuzzer(corpus: Path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    return Fuzzer(
        target="nonexistent",
        corpus_dir=str(corpus),
        crashes_dir=str(corpus.parent / "crashes"),
        **kw,
    )


def test_fuzzer_round_trip(tmp_path):
    corpus = tmp_path / "corpus"
    try:
        f = _fuzzer(corpus, zip_seed_corpus=True)
        f.save_to_corpus(b"discovered input")
        f._save_state()  # flushes the pending block

        assert b"discovered input" in _read_zip(corpus)
        assert _seed_files(corpus) == []

        g = _fuzzer(corpus, zip_seed_corpus=True)
        assert b"discovered input" in g.corpus
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def test_fuzzer_default_is_off(tmp_path):
    corpus = tmp_path / "corpus"
    seed_zip.configure(corpus, ZipMode.ON)  # stale mode from an earlier run
    f = _fuzzer(corpus)
    assert seed_zip.lookup(corpus) is None

    f.save_to_corpus(b"plain")
    assert len(_seed_files(corpus)) == 1


def test_cuckoo_sees_zip_pruned(tmp_path):
    corpus = tmp_path / "corpus"
    store = seed_zip.configure(corpus, ZipMode.ON)
    try:
        save_to_corpus(b"was pruned", corpus, set())
        store.retire(hash_data(b"was pruned"))
        store.flush()

        f = _fuzzer(corpus, zip_seed_corpus=True, cuckoo_seed_filter=True)
        assert f.cuckoo_seed_filter.contains(f._seed_key(b"was pruned"))
        assert b"was pruned" not in f.corpus
    finally:
        seed_zip.configure(corpus, ZipMode.OFF)


def _read_zip(corpus: Path) -> list[bytes]:
    with zipfile.ZipFile(corpus / seed_zip.ZIP_NAME) as zf:
        return [zf.read(n) for n in zf.namelist()]
