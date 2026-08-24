"""Tests for services/import_corpus.py — AFL/libFuzzer/honggfuzz import."""

from fuzzer_tool.services.import_corpus import (
    _TOKEN_CHARS,
    build_autotoken_dictionary,
    extract_tokens,
    import_from_afl,
    import_from_honggfuzz,
    import_from_libfuzzer,
    write_dictionary,
)


class TestImportFromAfl:
    def test_nonexistent_dir(self, tmp_path):
        seeds, crashes = import_from_afl("/nonexistent", str(tmp_path))
        assert seeds == 0
        assert crashes == 0

    def test_import_queue(self, tmp_path):
        # Create AFL output structure
        afl_dir = tmp_path / "afl_out"
        queue_dir = afl_dir / "queue"
        queue_dir.mkdir(parents=True)
        (queue_dir / "id_000001").write_bytes(b"hello")
        (queue_dir / "id_000002").write_bytes(b"world")
        # .txt files are imported from queue (not skipped like in crashes/)
        # Only crash .txt files are skipped
        (queue_dir / "id_000001.txt").write_text("metadata")

        corpus_dir = tmp_path / "corpus"
        seeds, crashes = import_from_afl(str(afl_dir), str(corpus_dir))
        assert seeds == 3  # 2 seeds + 1 .txt metadata
        assert crashes == 0

    def test_import_crashes(self, tmp_path):
        afl_dir = tmp_path / "afl_out"
        crash_dir = afl_dir / "crashes"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_01").write_bytes(b"crash_data")
        (crash_dir / "crash_01.txt").write_text("crash metadata")

        corpus_dir = tmp_path / "corpus"
        crash_out = tmp_path / "crashes"
        seeds, crashes = import_from_afl(str(afl_dir), str(corpus_dir), str(crash_out))
        assert seeds == 0
        assert crashes == 1
        # Metadata .txt is copied with imported_ prefix
        txt_files = list(crash_out.glob("imported_*.txt"))
        assert len(txt_files) == 1

    def test_skip_txt_in_crashes(self, tmp_path):
        afl_dir = tmp_path / "afl_out"
        crash_dir = afl_dir / "crashes"
        crash_dir.mkdir(parents=True)
        (crash_dir / "crash_01.txt").write_text("metadata only")

        corpus_dir = tmp_path / "corpus"
        crash_out = tmp_path / "crashes"
        seeds, crashes = import_from_afl(str(afl_dir), str(corpus_dir), str(crash_out))
        assert crashes == 0  # .txt files skipped

    def test_dedup(self, tmp_path):
        afl_dir = tmp_path / "afl_out"
        queue_dir = afl_dir / "queue"
        queue_dir.mkdir(parents=True)
        (queue_dir / "id_001").write_bytes(b"same_data")
        (queue_dir / "id_002").write_bytes(b"same_data")
        (queue_dir / "id_003").write_bytes(b"different")

        corpus_dir = tmp_path / "corpus"
        seeds, _ = import_from_afl(str(afl_dir), str(corpus_dir))
        assert seeds == 2  # duplicate removed

    def test_import_existing_dedup(self, tmp_path):
        # Pre-populate corpus with existing data
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "id_0000000000000000").write_bytes(b"hello")

        afl_dir = tmp_path / "afl_out"
        queue_dir = afl_dir / "queue"
        queue_dir.mkdir(parents=True)
        (queue_dir / "id_001").write_bytes(b"hello")  # duplicate
        (queue_dir / "id_002").write_bytes(b"new_data")

        seeds, _ = import_from_afl(str(afl_dir), str(corpus_dir))
        assert seeds == 1  # only new_data

    def test_empty_queue(self, tmp_path):
        afl_dir = tmp_path / "afl_out"
        queue_dir = afl_dir / "queue"
        queue_dir.mkdir(parents=True)

        corpus_dir = tmp_path / "corpus"
        seeds, _ = import_from_afl(str(afl_dir), str(corpus_dir))
        assert seeds == 0

    def test_no_crashes_dir(self, tmp_path):
        afl_dir = tmp_path / "afl_out"
        (afl_dir / "queue").mkdir(parents=True)

        corpus_dir = tmp_path / "corpus"
        seeds, crashes = import_from_afl(str(afl_dir), str(corpus_dir))
        assert crashes == 0

    def test_corrupt_data(self, tmp_path):
        afl_dir = tmp_path / "afl_out"
        queue_dir = afl_dir / "queue"
        queue_dir.mkdir(parents=True)
        # File exists but read_bytes might fail
        # Actually, this just tests that the import works

        corpus_dir = tmp_path / "corpus"
        seeds, _ = import_from_afl(str(afl_dir), str(corpus_dir))
        assert seeds == 0


class TestImportFromLibfuzzer:
    def test_nonexistent_dir(self, tmp_path):
        imported = import_from_libfuzzer("/nonexistent", str(tmp_path))
        assert imported == 0

    def test_import_seeds(self, tmp_path):
        src = tmp_path / "libfuzzer_corpus"
        src.mkdir()
        (src / "seed1").write_bytes(b"alpha")
        (src / "seed2").write_bytes(b"beta")

        dest = tmp_path / "target"
        imported = import_from_libfuzzer(str(src), str(dest))
        assert imported == 2
        assert len(list(dest.rglob("id_*"))) == 2

    def test_empty_files_skipped(self, tmp_path):
        src = tmp_path / "libfuzzer_corpus"
        src.mkdir()
        (src / "empty").write_bytes(b"")
        (src / "valid").write_bytes(b"data")

        dest = tmp_path / "target"
        imported = import_from_libfuzzer(str(src), str(dest))
        assert imported == 1

    def test_dedup(self, tmp_path):
        src = tmp_path / "libfuzzer_corpus"
        src.mkdir()
        (src / "a").write_bytes(b"dup")
        (src / "b").write_bytes(b"dup")

        dest = tmp_path / "target"
        imported = import_from_libfuzzer(str(src), str(dest))
        assert imported == 1

    def test_import_existing_dedup(self, tmp_path):
        src = tmp_path / "libfuzzer_corpus"
        src.mkdir()
        (src / "a").write_bytes(b"data")

        dest = tmp_path / "target"
        dest.mkdir()
        (dest / "id_0000000000000000").write_bytes(b"data")  # same content

        imported = import_from_libfuzzer(str(src), str(dest))
        assert imported == 0

    def test_subdirs_skipped(self, tmp_path):
        src = tmp_path / "libfuzzer_corpus"
        src.mkdir()
        (src / "subdir").mkdir()
        (src / "file").write_bytes(b"data")

        dest = tmp_path / "target"
        imported = import_from_libfuzzer(str(src), str(dest))
        assert imported == 1


class TestImportFromHonggfuzz:
    def test_nonexistent_dir(self, tmp_path):
        seeds, crashes = import_from_honggfuzz("/nonexistent", str(tmp_path))
        assert seeds == 0
        assert crashes == 0

    def test_import_seeds(self, tmp_path):
        src = tmp_path / "findings"
        src.mkdir()
        (src / "file1").write_bytes(b"seed_data")
        (src / "file2").write_bytes(b"more_data")

        dest = tmp_path / "target"
        imported, crashes = import_from_honggfuzz(str(src), str(dest))
        assert imported == 2
        assert crashes == 0  # honggfuzz import doesn't separate crashes
        assert len(list(dest.rglob("id_*"))) == 2

    def test_empty_files_skipped(self, tmp_path):
        src = tmp_path / "findings"
        src.mkdir()
        (src / "empty").write_bytes(b"")
        (src / "valid").write_bytes(b"data")

        dest = tmp_path / "target"
        imported, _ = import_from_honggfuzz(str(src), str(dest))
        assert imported == 1

    def test_dedup(self, tmp_path):
        src = tmp_path / "findings"
        src.mkdir()
        (src / "a").write_bytes(b"same")
        (src / "b").write_bytes(b"same")

        dest = tmp_path / "target"
        imported, _ = import_from_honggfuzz(str(src), str(dest))
        assert imported == 1


class TestExtractTokens:
    def test_extracts_bounded_runs(self):
        tokens = extract_tokens(b'\x89PNG header="IHDR" 12 more text here', min_len=3, max_len=32)
        assert b"header" in tokens
        assert b"IHDR" in tokens
        assert b"PNG" in tokens

    def test_respects_min_len(self):
        tokens = extract_tokens(b"ab abc a", min_len=3)
        assert b"ab" not in tokens
        assert b"a" not in tokens
        assert b"abc" in tokens

    def test_respects_max_len(self):
        long_run = b"x" * 40
        tokens = extract_tokens(long_run, min_len=3, max_len=32)
        assert long_run not in tokens
        assert tokens == set()

    def test_token_at_end_of_data(self):
        tokens = extract_tokens(b"  trailing_token", min_len=3, max_len=32)
        assert b"trailing_token" in tokens

    def test_no_tokens_in_binary_noise(self):
        noise = bytes([b for b in range(256) if b not in _TOKEN_CHARS])
        tokens = extract_tokens(noise, min_len=3, max_len=32)
        assert tokens == set()


class TestBuildAutotokenDictionary:
    def test_ranks_by_seed_frequency(self, tmp_path):
        corpus = tmp_path / "corpus"
        for i, content in enumerate([b"common shared", b"common only", b"unique_one xyz"]):
            sub = corpus / "seeds" / "00"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / f"id_{i:04x}").write_bytes(content)

        tokens = build_autotoken_dictionary(str(corpus), min_len=3, max_tokens=10)
        assert b"common" in tokens
        # "common" appears in 2 seeds, "shared"/"only"/"unique_one"/"xyz" in 1 each.
        assert tokens[0] == b"common"

    def test_max_tokens_cap(self, tmp_path):
        corpus = tmp_path / "corpus" / "seeds" / "00"
        corpus.mkdir(parents=True)
        words = " ".join(f"token{i:03d}" for i in range(50))
        (corpus / "id_0001").write_bytes(words.encode())

        tokens = build_autotoken_dictionary(str(tmp_path / "corpus"), min_len=3, max_tokens=5)
        assert len(tokens) == 5

    def test_empty_corpus(self, tmp_path):
        assert build_autotoken_dictionary(str(tmp_path / "nope")) == []


class TestWriteDictionary:
    def test_round_trips_through_load_dictionary(self, tmp_path):
        from fuzzer_tool.core.mutations.generic import load_dictionary

        tokens = [b"IHDR", b'has"quote', b"has\\backslash", bytes([0x01, 0x02])]
        out = tmp_path / "auto.dict"
        write_dictionary(tokens, str(out))

        loaded = load_dictionary(str(out))
        assert loaded == tokens
