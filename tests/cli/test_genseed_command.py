"""Tests for the `genseed` CLI subcommand (cli/commands.py::cmd_genseed)."""

import argparse

from fuzzer_tool.cli.commands import cmd_genseed


def _args(**kw):
    defaults = dict(format="all", corpus=None, count=1, max_len=4096, seed=None)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestGenseedSingleFormat:
    def test_writes_one_seed_file(self, tmp_path):
        corpus = tmp_path / "corpus"
        rc = cmd_genseed(_args(format="flac", corpus=str(corpus), seed=1))
        assert rc == 0
        files = list((corpus / "seeds").rglob("id_*"))
        assert len(files) == 1
        assert files[0].read_bytes()

    def test_count_writes_multiple_distinct_files(self, tmp_path):
        corpus = tmp_path / "corpus"
        rc = cmd_genseed(_args(format="flac", corpus=str(corpus), count=5, seed=1))
        assert rc == 0
        files = list((corpus / "seeds").rglob("id_*"))
        # A rng-driven generator should produce >1 distinct file across 5 draws.
        assert len(files) > 1

    def test_deterministic_with_same_seed(self, tmp_path):
        corpus_a = tmp_path / "a"
        corpus_b = tmp_path / "b"
        cmd_genseed(_args(format="mp3", corpus=str(corpus_a), count=3, seed=7))
        cmd_genseed(_args(format="mp3", corpus=str(corpus_b), count=3, seed=7))
        names_a = sorted(p.name for p in (corpus_a / "seeds").rglob("id_*"))
        names_b = sorted(p.name for p in (corpus_b / "seeds").rglob("id_*"))
        assert names_a == names_b


class TestGenseedConstantFormat:
    def test_minimal_seed_format_dedups_to_one_file_regardless_of_count(self, tmp_path):
        corpus = tmp_path / "corpus"
        rc = cmd_genseed(_args(format="png", corpus=str(corpus), count=4, seed=1))
        assert rc == 0
        files = list((corpus / "seeds").rglob("id_*"))
        assert len(files) == 1


class TestGenseedAllFormats:
    def test_writes_one_file_per_format(self, tmp_path):
        from fuzzer_tool.core.format_generators import FORMATS

        corpus = tmp_path / "corpus"
        rc = cmd_genseed(_args(format="all", corpus=str(corpus), seed=1))
        assert rc == 0
        files = list((corpus / "seeds").rglob("id_*"))
        assert len(files) == len(FORMATS)


class TestGenseedUnknownFormat:
    def test_returns_nonzero(self, tmp_path):
        corpus = tmp_path / "corpus"
        rc = cmd_genseed(_args(format="not-a-format", corpus=str(corpus)))
        assert rc == 1
        assert not (corpus / "seeds").exists()


class TestGenseedMaxLen:
    def test_respects_max_len_for_constant_format(self, tmp_path):
        corpus = tmp_path / "corpus"
        cmd_genseed(_args(format="gzip", corpus=str(corpus), max_len=4, seed=1))
        files = list((corpus / "seeds").rglob("id_*"))
        assert len(files) == 1
        assert len(files[0].read_bytes()) == 4
