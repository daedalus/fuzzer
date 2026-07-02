"""Tests for services/minimize.py — corpus minimization."""


from fuzzer_tool.services.minimize import _minimize_by_hash, minimize_corpus


class TestMinimizeByHash:
    def test_empty_corpus(self, tmp_path):
        kept, removed = _minimize_by_hash([], None, tmp_path)
        assert kept == 0
        assert removed == 0

    def test_dedup(self, tmp_path):
        files = []
        for name, data in [("a.bin", b"hello"), ("b.bin", b"hello"), ("c.bin", b"world")]:
            p = tmp_path / name
            p.write_bytes(data)
            files.append(p)
        kept, removed = _minimize_by_hash(files, None, tmp_path)
        assert kept == 2  # hello + world
        assert removed == 1

    def test_no_duplicates(self, tmp_path):
        files = []
        for i, data in enumerate([b"a", b"b", b"c"]):
            p = tmp_path / f"f{i}.bin"
            p.write_bytes(data)
            files.append(p)
        kept, removed = _minimize_by_hash(files, None, tmp_path)
        assert kept == 3
        assert removed == 0

    def test_output_dir(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.bin").write_bytes(b"dup")
        (src / "b.bin").write_bytes(b"dup")
        (src / "c.bin").write_bytes(b"unique")
        files = [src / f for f in ["a.bin", "b.bin", "c.bin"]]
        output = tmp_path / "output"
        kept, removed = _minimize_by_hash(files, str(output), tmp_path / "src")
        assert kept == 2
        assert len(list(output.iterdir())) == 2


class TestMinimizeCorpus:
    def test_nonexistent_dir(self, tmp_path):
        kept, removed = minimize_corpus("/fake", str(tmp_path))
        assert kept == 0
        assert removed == 0

    def test_empty_corpus(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        kept, removed = minimize_corpus("/fake", str(corpus))
        assert kept == 0
        assert removed == 0

    def test_no_coverage_dedup(self, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a").write_bytes(b"AAA")
        (corpus / "b").write_bytes(b"AAA")
        (corpus / "c").write_bytes(b"BBB")
        kept, removed = minimize_corpus("/fake", str(corpus))
        assert kept == 2
        assert removed == 1
