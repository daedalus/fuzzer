"""`full-cover`: every edge in an edge dump, fewest seeds, verified.

The dump is the record of coverage (`edge-dump` output), not the corpus: a
corpus pruned last week still has its edges here, and a cover is only useful
if it reaches all of them.
"""

import io
import pickle
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fuzzer_tool.adapters.edge_pickle import EdgePickleWriter, iter_edges_pickle
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.cli import commands
from fuzzer_tool.services import full_cover


def _seed(corpus: Path, data: bytes) -> Path:
    """A seed in the canonical layout: seeds/<hh>/id_<hash>."""
    sid = hash_data(data)
    d = corpus / "seeds" / sid[:2]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"id_{sid}"
    p.write_bytes(data)
    return p


def _dump(path: Path, batches: list[dict[bytes, list[int]]]) -> Path:
    """Write *batches* the way ``edge-dump`` does: one pickle, spliced per batch."""
    writer = EdgePickleWriter(path)
    writer.open()
    for batch in batches:
        writer.write([(hash_data(k), np.array(v, dtype=np.uint32)) for k, v in batch.items()])
    writer.close()
    return path


def _dump_legacy(path: Path, batches: list[dict[bytes, list[int]]]) -> Path:
    """Write *batches* the pre-splice layout: one dict pickle back to back per batch."""
    with open(path, "wb") as f:
        for batch in batches:
            pickle.dump({hash_data(k): np.array(v, dtype=np.uint32) for k, v in batch.items()}, f)
    return path


def test_redundant_seed_dropped_every_edge_kept(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2,3}
    _seed(corpus, b"b")  # {3}     redundant
    _seed(corpus, b"cc")  # {4,5}
    dump = _dump(
        tmp_path / "e.pkl",
        [{b"aaaa": [1, 2, 3], b"b": [3], b"cc": [4, 5]}],
    )

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.kept == 2
    assert rep.union_edges == 5
    assert rep.reachable_edges == 5
    assert rep.unreachable_edges == 0
    assert len(list((tmp_path / "out").rglob("id_*"))) == 2


def test_unreachable_edge_refuses_to_write(tmp_path):
    """Falsification: a dump-only seed holds edge 9; pruning would lose it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3}
    dump = _dump(
        tmp_path / "e.pkl",
        [{b"aaaa": [1, 2], b"bb": [2, 3], b"gone": [9]}],
    )

    with pytest.raises(full_cover.UnreachableEdges):
        full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_allow_unreachable_reports_and_proceeds(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"bb": [2, 3], b"gone": [9]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out", allow_unreachable=True)

    assert rep.unreachable_edges == 1
    assert rep.gone_seeds == 1
    assert rep.reachable_edges == 3


def test_every_batch_of_a_multi_batch_dump_counts(tmp_path):
    """Adversarial: edge 4 lives only in batch 2; a first-pickle read misses it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3,4}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}, {b"bb": [2, 3, 4]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.union_edges == 4
    assert rep.reachable_edges == 4
    assert rep.kept == 2


def test_in_place_moves_redundant_seed_to_pruned(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    keep = _seed(corpus, b"aaaa")  # {1,2}
    drop = _seed(corpus, b"a")  # {1}  redundant
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1]}])

    full_cover.full_cover(dump, corpus)

    assert keep.exists()
    assert not drop.exists()
    assert (corpus / "pruned" / drop.relative_to(corpus)).exists()


def test_seed_absent_from_dump_is_unmeasured_and_kept(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    extra = _seed(corpus, b"zzz")  # never dumped
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}])

    rep = full_cover.full_cover(dump, corpus)

    assert rep.unmeasured == 1
    assert extra.exists()


def test_incomplete_cover_is_rejected(tmp_path, monkeypatch):
    """Falsification of the verifier: a cover that misses an edge must not ship."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"bb")  # {2,3}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"bb": [2, 3]}])
    monkeypatch.setattr(full_cover, "min_cover", lambda edges, sizes: [])

    with pytest.raises(full_cover.IncompleteCover):
        full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")


def test_flat_loose_seeds_kept_by_content_hash(tmp_path):
    """Flat files are seeds when their names key on content hash."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    def loose(data: bytes) -> Path:
        p = corpus / f"id_{hash_data(data)}"
        p.write_bytes(data)
        return p

    loose(b"aaaa")  # {1, 2}
    loose(b"bb")  # {2, 3}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"bb": [2, 3]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.kept == 2
    assert rep.reachable_edges == 3


def _tar(corpus: Path, seeds: dict[bytes, list[int]]) -> Path:
    """seeds.tar.gz with canonical ``seeds/<hh>/id_<hash>`` members."""
    path = corpus / "seeds.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for data in seeds:
            sid = hash_data(data)
            info = tarfile.TarInfo(f"seeds/{sid[:2]}/id_{sid}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def _zip(corpus: Path, seeds: dict[bytes, list[int]]) -> Path:
    path = corpus / "seeds.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for data in seeds:
            sid = hash_data(data)
            zf.writestr(f"{sid[:2]}/id_{sid}", data)
    return path


def _written_hashes(out: Path) -> set[str]:
    """Content hashes of every seed written under *out* (seeds/<hh>/id_*)."""
    return {hash_data(p.read_bytes()) for p in out.rglob("id_*")}


def test_tarred_corpus_covers_and_copies_out(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _tar(corpus, {b"aaaa": [1, 2], b"a": [1], b"cc": [3]})
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1], b"cc": [3]}])
    out = tmp_path / "out"

    rep = full_cover.full_cover(dump, corpus, output_dir=out)

    assert rep.kept == 2
    assert rep.reachable_edges == 3
    assert _written_hashes(out) == {hash_data(b"aaaa"), hash_data(b"cc")}


def test_tarred_corpus_refuses_in_place(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _tar(corpus, {b"aaaa": [1, 2]})
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}])

    with pytest.raises(full_cover.ArchiveReadOnly):
        full_cover.full_cover(dump, corpus)


def test_zipped_corpus_covers_and_copies_out(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _zip(corpus, {b"aaaa": [1, 2], b"a": [1], b"cc": [3]})
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1], b"cc": [3]}])
    out = tmp_path / "out"

    rep = full_cover.full_cover(dump, corpus, output_dir=out)

    assert rep.kept == 2
    assert _written_hashes(out) == {hash_data(b"aaaa"), hash_data(b"cc")}


def test_cli_writes_cover(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"a")  # {1} redundant
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1]}])
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["fuzzer-tool", "full-cover", "-d", str(corpus), "-e", str(dump), "-o", str(out)],
    )
    rc = commands.main()

    assert rc == 0
    assert len(list(out.iterdir())) == 1


def test_cli_unreachable_exits_nonzero(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"gone": [9]}])
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["fuzzer-tool", "full-cover", "-d", str(corpus), "-e", str(dump), "-o", str(out)],
    )
    rc = commands.main()

    err = capsys.readouterr().err
    assert rc == 2
    assert "unreachable" in err


def test_regression_iter_reads_spliced_batches(tmp_path):
    """Falsification: the spliced layout lost its batch MARK and read back empty."""
    path = EdgePickleWriter(tmp_path / "e.pkl").path
    writer = EdgePickleWriter(path)
    writer.open()
    writer.write([("a" * 16, np.array([1, 2], dtype=np.uint32))])
    writer.write(
        [
            ("b" * 16, np.array([3], dtype=np.uint32)),
            ("c" * 16, np.array([4, 5], dtype=np.uint32)),
        ]
    )
    writer.write([("d" * 16, np.array([6], dtype=np.uint32))])  # one item, no MARK
    writer.close()

    got = [{k: v.tolist() for k, v in batch.items()} for batch in iter_edges_pickle(path)]

    assert got == [
        {"a" * 16: [1, 2]},
        {"b" * 16: [3], "c" * 16: [4, 5]},
        {"d" * 16: [6]},
    ]


def test_legacy_back_to_back_dump_still_read(tmp_path):
    """Old exports (one dict pickle per batch) keep working."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"a")  # {1} redundant
    dump = _dump_legacy(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}, {b"a": [1]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.dump_seeds == 2
    assert rep.kept == 1


def test_regression_spliced_dump_minimizes(tmp_path):
    """The regression: a real ``edge-dump`` file was read as empty, so nothing shrank."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")  # {1,2}
    _seed(corpus, b"a")  # {1} redundant
    _seed(corpus, b"bb")  # {2,3}
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2], b"a": [1]}, {b"bb": [2, 3]}])
    out = tmp_path / "out"

    rep = full_cover.full_cover(dump, corpus, output_dir=out)

    assert rep.dump_seeds == 3
    assert rep.reachable_edges == 3
    assert rep.kept == 2
    assert _written_hashes(out) == {hash_data(b"aaaa"), hash_data(b"bb")}


@pytest.mark.parametrize("payload", [b"", b"\x80\x03}."])
def test_dump_without_seeds_refused(tmp_path, payload):
    """An empty dump defines no coverage; keeping every seed hides the failure."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")
    dump = tmp_path / "e.pkl"
    dump.write_bytes(payload)
    out = tmp_path / "out"

    with pytest.raises(full_cover.EmptyDump):
        full_cover.full_cover(dump, corpus, output_dir=out)

    assert not out.exists()


def test_cli_empty_dump_exits_nonzero(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")
    dump = tmp_path / "e.pkl"
    dump.write_bytes(b"\x80\x03}.")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fuzzer-tool",
            "full-cover",
            "-d",
            str(corpus),
            "-e",
            str(dump),
            "-o",
            str(tmp_path / "o"),
        ],
    )
    rc = commands.main()

    assert rc == 2
    assert "no seed" in capsys.readouterr().err


def test_duplicate_id_written_once(tmp_path):
    """Adversarial: one content id under a shard and a protected tree is one seed."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    sid = hash_data(b"aaaa")
    path = corpus / "seeds.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name in (f"seeds/{sid[:2]}/id_{sid}", f"seeds/irreplaceable/id_{sid}"):
            info = tarfile.TarInfo(name)
            info.size = len(b"aaaa")
            tf.addfile(info, io.BytesIO(b"aaaa"))
    dump = _dump(tmp_path / "e.pkl", [{b"aaaa": [1, 2]}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")

    assert rep.kept == 1
    assert len(list((tmp_path / "out").rglob("id_*"))) == 1


BIG_TWIN = b"B" * 100
SMALL_TWIN = b"s"
TWIN_EDGES = [1, 2, 3]


def _kept_twin(tmp_path: Path, batches: list[dict[bytes, list[int]]]) -> set[str]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, BIG_TWIN)
    _seed(corpus, SMALL_TWIN)
    full_cover.full_cover(_dump(tmp_path / "e.pkl", batches), corpus, output_dir=tmp_path / "out")
    return _written_hashes(tmp_path / "out")


@pytest.mark.parametrize(
    "batches",
    [
        [{BIG_TWIN: TWIN_EDGES, SMALL_TWIN: TWIN_EDGES}],  # same batch, big first
        [{SMALL_TWIN: TWIN_EDGES, BIG_TWIN: TWIN_EDGES}],  # control: small first
        [{BIG_TWIN: TWIN_EDGES}, {SMALL_TWIN: TWIN_EDGES}],  # small in a later batch
    ],
    ids=["same-batch", "small-first", "later-batch"],
)
def test_regression_full_cover_keeps_smallest_twin(tmp_path, batches):
    """Identical edge sets: the smallest seed is kept, whatever the dump order."""
    assert _kept_twin(tmp_path, batches) == {hash_data(SMALL_TWIN)}


def test_equal_size_twins_keep_one(tmp_path):
    """Adversarial: same size, same edges -> exactly one kept, every edge covered."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"xx")
    _seed(corpus, b"yy")
    dump = _dump(tmp_path / "e.pkl", [{b"xx": TWIN_EDGES}, {b"yy": TWIN_EDGES}])

    rep = full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")
    assert rep.kept == 1
    assert rep.reachable_edges == len(TWIN_EDGES)


def test_subset_is_not_a_twin(tmp_path):
    """Falsification: a smaller seed with a strict subset of edges never replaces."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, BIG_TWIN)
    _seed(corpus, SMALL_TWIN)
    dump = _dump(tmp_path / "e.pkl", [{BIG_TWIN: TWIN_EDGES}, {SMALL_TWIN: TWIN_EDGES[:2]}])

    full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")
    assert _written_hashes(tmp_path / "out") == {hash_data(BIG_TWIN)}


class _Clock:
    """Scripted ``time`` module: each call advances by *step* seconds.

    ``time()`` (wall) runs backward, ``monotonic()`` forward: the throttle
    must read only the monotonic clock.
    """

    def __init__(self, step: float):
        self.now, self.step = 0.0, step

    def monotonic(self) -> float:
        self.now += self.step
        return self.now

    def time(self) -> float:
        return -self.monotonic()


def _status(out: str) -> list[tuple[int, int]]:
    """``(seeds, edges)`` of every ``Scanned`` line in *out*."""
    lines = [ln for ln in out.split("\r") if ln.startswith("[*] Scanned")]
    return [(int(ln.split()[2]), int(ln.split("edges: ")[1].split()[0])) for ln in lines]


def _running(batches: list[dict[bytes, list[int]]]) -> list[tuple[int, int]]:
    """``(seeds, distinct edges)`` after each batch, derived from the fixture."""
    seen: set[int] = set()
    seeds = 0
    out = []
    for batch in batches:
        seeds += len(batch)
        for ids in batch.values():
            seen.update(ids)
        out.append((seeds, len(seen)))
    return out


_TWO_BATCHES = [{b"aaaa": [1, 2]}, {b"bb": [2, 3, 4]}]


def _cover(tmp_path: Path, batches: list[dict[bytes, list[int]]]) -> full_cover.CoverReport:
    """Seed every fixture key, dump *batches*, run full_cover."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for batch in batches:
        for data in batch:
            _seed(corpus, data)
    dump = _dump(tmp_path / "e.pkl", batches)
    return full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")


def test_scan_prints_status_line_per_interval(tmp_path, capsys, monkeypatch):
    """Monotonic clock past the interval every batch: one line per batch, then the final line.

    The wall clock runs backward here: a throttle reading it prints nothing.
    """
    monkeypatch.setattr(full_cover, "time", _Clock(full_cover._STATUS_EVERY_S))

    _cover(tmp_path, _TWO_BATCHES)

    out = capsys.readouterr().out
    want = _running(_TWO_BATCHES)
    assert _status(out) == want + want[-1:]
    assert "seeds/s" in out and "ETA" not in out


def test_scan_status_throttled_to_final_line(tmp_path, capsys, monkeypatch):
    """Falsification: clock below the interval: only the final line, with full counts."""
    monkeypatch.setattr(full_cover, "time", _Clock(0.0))

    _cover(tmp_path, _TWO_BATCHES)

    assert _status(capsys.readouterr().out) == _running(_TWO_BATCHES)[-1:]


def test_empty_dump_prints_no_status(tmp_path, capsys):
    """Adversarial: no batch, no status line; EmptyDump still raised."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")
    dump = tmp_path / "e.pkl"
    dump.write_bytes(b"\x80\x03}.")

    with pytest.raises(full_cover.EmptyDump):
        full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")
    assert "Scanned" not in capsys.readouterr().out


def test_regression_empty_legacy_batch_prints_no_status(tmp_path, capsys):
    """Adversarial: legacy dump of empty dicts yields batches but no seeds: no `Scanned 0`."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _seed(corpus, b"aaaa")
    dump = tmp_path / "e.pkl"
    dump.write_bytes(pickle.dumps({}) + pickle.dumps({}))

    with pytest.raises(full_cover.EmptyDump):
        full_cover.full_cover(dump, corpus, output_dir=tmp_path / "out")
    assert "Scanned" not in capsys.readouterr().out


def test_regression_status_union_counts_distinct_edges(tmp_path, capsys, monkeypatch):
    """Overlap across seeds and batches never inflates the line: expected = distinct ids."""
    batches = [{b"aa": [1, 2], b"bb": [2, 3]}, {b"cc": [3, 1 << 20]}, {b"dd": [1, 2]}]
    monkeypatch.setattr(full_cover, "time", _Clock(full_cover._STATUS_EVERY_S))

    rep = _cover(tmp_path, batches)

    want = _running(batches)
    assert _status(capsys.readouterr().out) == want + want[-1:]
    assert rep.union_edges == want[-1][1]


class _SlowCount:
    """Clock where each status line takes a full interval; batches take none.

    The first ``monotonic()`` (the throttle's start) reads 0; later reads see
    one interval already passed, so batch 1 is due.
    """

    def __init__(self) -> None:
        self.now = full_cover._STATUS_EVERY_S
        self.started = False
        self.calls = 0

    def monotonic(self) -> float:
        if not self.started:
            self.started = True
            return 0.0
        return self.now

    def time(self) -> float:
        return self.now

    def status(self, *_args: object) -> None:
        self.calls += 1
        self.now += full_cover._STATUS_EVERY_S


def test_regression_slow_status_not_rearmed_per_batch(tmp_path, monkeypatch):
    """A count lasting the interval must not re-trigger on the next batch.

    Interval runs from the end of the last line: batch 1 prints, later batches
    take no time, so only the final line follows.
    """
    clock = _SlowCount()
    monkeypatch.setattr(full_cover, "time", clock)
    monkeypatch.setattr(full_cover, "_scan_status", clock.status)
    batches = [{bytes([65 + i]): [i]} for i in range(4)]

    _cover(tmp_path, batches)

    first_batch, final = 1, 1
    assert clock.calls == first_batch + final


class _NpSpy:
    """numpy proxy: records the size of every array passed to count_nonzero."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    def count_nonzero(self, a: np.ndarray) -> int:
        self.sizes.append(np.asarray(a).size)
        return int(np.count_nonzero(a))


def test_regression_scan_never_rescans_map_per_batch(tmp_path, monkeypatch):
    """Crafted dump: one high edge id, many tiny batches; map counts do not scale with batches."""
    top = 1 << 20
    batches = [{b"hi": [top]}] + [{bytes([65 + i]): [i]} for i in range(8)]
    spy = _NpSpy()
    monkeypatch.setattr(full_cover, "np", spy)
    monkeypatch.setattr(full_cover, "time", _Clock(0.0))

    _cover(tmp_path, batches)

    # Map-sized counts: the final status line and full_cover()'s unreachable check.
    assert sum(size > top for size in spy.sizes) < len(batches)
