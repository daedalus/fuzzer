"""Regressions from the 2026-11-13 audit: corpus loading, archives, state, tmin."""

import pytest

from fuzzer_tool.adapters.filesystem import (
    hash_data,
    load_corpus,
    rehydrate_by_hash,
    save_to_corpus,
)

_PARENT = b"parentparent"
_DELTA_HASH = "00000000000000aa"


def _bad_deltas(parent_hash):
    return {
        "json list": b"[1,2]",
        "non-int offset": b'{"parent":"%s","diff":[["x",1]]}' % parent_hash.encode(),
        "byte out of range": b'{"parent":"%s","diff":[[0,999]]}' % parent_hash.encode(),
        "non-utf8": b"\xff\xfe",
        "diff not a list": b'{"parent":"%s","diff":7}' % parent_hash.encode(),
        "parent not a string": b'{"parent":[1],"diff":[]}',
    }


@pytest.fixture(params=sorted(_bad_deltas("0" * 16)))
def corpus(tmp_path, request):
    save_to_corpus(_PARENT, tmp_path, set())
    deltas = tmp_path / "deltas"
    deltas.mkdir(exist_ok=True)
    body = _bad_deltas(hash_data(_PARENT))[request.param]
    (deltas / f"delta_{_DELTA_HASH}.json").write_bytes(body)
    return tmp_path


class TestMalformedDelta:
    """One bad ``deltas/delta_*.json`` aborted ``load_corpus`` and ``rehydrate_by_hash``."""

    def test_regression_load_corpus_skips_it(self, corpus):
        loaded, seen, _irr = load_corpus(corpus, add_default=False)

        assert loaded == [_PARENT]
        assert _DELTA_HASH not in seen

    def test_regression_rehydrate_skips_it(self, corpus):
        assert rehydrate_by_hash(_DELTA_HASH, corpus) is None


def test_valid_delta_still_loads(tmp_path):
    """Falsification: a well-formed delta chain keeps resolving."""
    seen: set[str] = set()
    child = _PARENT[:-1] + b"X"
    save_to_corpus(_PARENT, tmp_path, seen)
    save_to_corpus(child, tmp_path, seen, parent=_PARENT)

    loaded, _seen, _irr = load_corpus(tmp_path, add_default=False)

    assert sorted(loaded) == sorted([_PARENT, child])


class TestTminStdout:
    """``tmin`` without ``-O`` wrote progress lines into the minimized bytes on stdout."""

    _MIN = b"\x00MIN\xff"

    def _run(self, monkeypatch, tmp_path, *extra):
        import sys

        from fuzzer_tool.cli.commands import main

        def fake_tmin(**_kw):
            print("[*] Crash input: 9 bytes")
            return self._MIN

        monkeypatch.setattr("fuzzer_tool.services.tmin.tmin", fake_tmin)
        monkeypatch.setattr(
            sys, "argv", ["fuzzer-tool", "tmin", "/bin/true", str(tmp_path / "c.bin"), *extra]
        )
        return main()

    def test_regression_stdout_is_only_the_reproducer(self, monkeypatch, tmp_path, capfdbinary):
        assert self._run(monkeypatch, tmp_path) == 0

        out, err = capfdbinary.readouterr()
        assert out == self._MIN
        assert b"[*] Crash input" in err

    def test_output_file_keeps_progress_on_stdout(self, monkeypatch, tmp_path, capfdbinary):
        """Falsification: with -O the file holds the bytes and stdout the progress."""
        dest = tmp_path / "min.bin"
        assert self._run(monkeypatch, tmp_path, "-O", str(dest)) == 0

        out, _err = capfdbinary.readouterr()
        assert dest.read_bytes() == self._MIN
        assert b"[*] Crash input" in out
        assert self._MIN not in out


class TestZipReaderNeverRenames:
    """A read-only ``seeds.zip`` view salvaged (renamed) the archive under a live writer."""

    _SEEDS = [b"seed-%d" % i * 20 for i in range(6)]

    @pytest.fixture
    def torn(self, tmp_path):
        """Writer mid-flush: one member appended over the central directory, no checkpoint yet."""
        from fuzzer_tool.adapters import seed_zip
        from fuzzer_tool.adapters.seed_zip import SeedTree, ZipMode

        writer = seed_zip.configure(tmp_path, ZipMode.ON, block_seeds=2, growth=0)
        for s in self._SEEDS[:2]:
            writer.put(SeedTree.MAIN, hash_data(s), s)
        zf = writer._writer()
        zf.writestr(seed_zip._member("", hash_data(self._SEEDS[2])), self._SEEDS[2])
        zf.fp.flush()
        yield tmp_path, writer, zf
        seed_zip.configure(tmp_path, ZipMode.OFF)

    def test_regression_reader_leaves_archive_in_place(self, torn):
        from fuzzer_tool.adapters import seed_zip
        from fuzzer_tool.adapters.seed_zip import SeedTree, ZipMode

        corpus, writer, zf = torn
        inode = (corpus / seed_zip.ZIP_NAME).stat().st_ino

        list(seed_zip.open_readonly(corpus).live())

        assert sorted(p.name for p in corpus.iterdir()) == [seed_zip.ZIP_NAME]
        assert (corpus / seed_zip.ZIP_NAME).stat().st_ino == inode

        # The writer finishes its flush and keeps going: every seed lands in seeds.zip.
        seed_zip._checkpoint(zf)
        for s in self._SEEDS[3:]:
            writer.put(SeedTree.MAIN, hash_data(s), s)
        writer.close()
        seed_zip.configure(corpus, ZipMode.OFF)
        seed_zip._PEEK.clear()
        loaded, _seen, _irr = load_corpus(corpus, add_default=False)
        assert sorted(loaded) == sorted(self._SEEDS)

    def test_writer_still_salvages(self, tmp_path):
        """Falsification: the registered writer rebuilds a torn archive on open."""
        import zipfile

        from fuzzer_tool.adapters import seed_zip
        from fuzzer_tool.adapters.seed_zip import ZipMode

        path = tmp_path / seed_zip.ZIP_NAME
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(seed_zip._member("", hash_data(_PARENT)), _PARENT)
        raw = path.read_bytes()
        path.write_bytes(raw[: raw.index(b"PK\x01\x02")])  # drop the central directory

        store = seed_zip.configure(tmp_path, ZipMode.ON)
        try:
            assert [d for d, _p in store.live()] == [_PARENT]
            assert (tmp_path / f"{seed_zip.ZIP_NAME}.corrupt.0").is_file()
        finally:
            seed_zip.configure(tmp_path, ZipMode.OFF)


class TestCompactForeign:
    """``compact()`` kept a pruned seed's original foreign member: it came back live."""

    _A, _B = b"foreign-A" * 9, b"foreign-B" * 9

    @pytest.fixture
    def corpus(self, tmp_path):
        import zipfile

        from fuzzer_tool.adapters import seed_zip
        from fuzzer_tool.adapters.seed_zip import ZipMode

        with zipfile.ZipFile(tmp_path / seed_zip.ZIP_NAME, "w") as zf:
            zf.writestr("a.bin", self._A)
            zf.writestr("b.bin", self._B)
        store = seed_zip.configure(tmp_path, ZipMode.ON)
        load_corpus(tmp_path, add_default=False)  # adopts both
        assert store.retire(hash_data(self._B))
        seed_zip.configure(tmp_path, ZipMode.OFF)
        seed_zip._PEEK.clear()
        return tmp_path

    def _live(self, corpus):
        from fuzzer_tool.adapters import seed_zip

        seed_zip._PEEK.clear()
        return sorted(load_corpus(corpus, add_default=False)[0])

    def test_regression_pruned_foreign_stays_pruned(self, corpus):
        from fuzzer_tool.adapters import seed_zip

        assert self._live(corpus) == [self._A]
        seed_zip.compact(corpus)

        assert self._live(corpus) == [self._A]
        h = hash_data(self._B)
        assert (corpus / "seeds" / "pruned" / h[:2] / f"id_{h}").read_bytes() == self._B

    def test_live_foreign_survives_compact(self, corpus):
        """Falsification: compact twice; the unpruned foreign seed is still there."""
        from fuzzer_tool.adapters import seed_zip

        seed_zip.compact(corpus)
        seed_zip.compact(corpus)

        assert self._live(corpus) == [self._A]


class TestMinimizePrunedRehydrates:
    """``fuzzer-tool minimize`` prunes to ``pruned/seeds/<hh>/``; rehydration only read ``seeds/pruned/``."""

    _DATA = b"pruned by minimize"

    @pytest.mark.parametrize("root", ["pruned/seeds", "seeds/pruned"])
    def test_regression_both_pruned_layouts_rehydrate(self, tmp_path, root):
        h = hash_data(self._DATA)
        dest = tmp_path / root / h[:2] / f"id_{h}"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(self._DATA)

        assert rehydrate_by_hash(h, tmp_path) == self._DATA

    def test_pruned_stays_out_of_the_live_corpus(self, tmp_path):
        """Adversarial: readable by hash, never loaded as a live seed."""
        h = hash_data(self._DATA)
        dest = tmp_path / "pruned" / "seeds" / h[:2] / f"id_{h}"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(self._DATA)

        loaded, seen, _irr = load_corpus(tmp_path, add_default=False)

        assert self._DATA not in loaded
        assert h not in seen
