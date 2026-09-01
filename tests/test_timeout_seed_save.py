"""Tests for timeout seed saving to corpus/seeds/timeouts/."""

import tempfile
from pathlib import Path

from fuzzer_tool.adapters.filesystem import (
    hash_data,
    load_corpus,
    save_timeout_seed,
)


def test_save_timeout_seed_basic():
    with tempfile.TemporaryDirectory() as tmp:
        data = b"test_timeout_input_12345"
        seen = set()
        irep = set()

        result = save_timeout_seed(data, Path(tmp), seen, irep)
        assert result is True
        assert hash_data(data) in irep
        assert hash_data(data) in seen

        h = hash_data(data)
        expected = Path(tmp) / "seeds" / "timeouts" / h[:2] / f"id_{h}"
        assert expected.is_file()
        assert expected.read_bytes() == data


def test_save_timeout_seed_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        data = b"test_timeout_dedup"
        seen = set()
        irep = set()

        result1 = save_timeout_seed(data, Path(tmp), seen, irep)
        assert result1 is True

        result2 = save_timeout_seed(data, Path(tmp), seen, irep)
        assert result2 is False


def test_save_timeout_seed_different_inputs():
    with tempfile.TemporaryDirectory() as tmp:
        data1 = b"timeout_input_A"
        data2 = b"timeout_input_B"
        seen = set()
        irep = set()

        save_timeout_seed(data1, Path(tmp), seen, irep)
        save_timeout_seed(data2, Path(tmp), seen, irep)

        assert hash_data(data1) in irep
        assert hash_data(data2) in irep

        h1 = hash_data(data1)
        h2 = hash_data(data2)
        assert (Path(tmp) / "seeds" / "timeouts" / h1[:2] / f"id_{h1}").is_file()
        assert (Path(tmp) / "seeds" / "timeouts" / h2[:2] / f"id_{h2}").is_file()


def test_load_corpus_includes_timeouts():
    with tempfile.TemporaryDirectory() as tmp:
        data = b"timeout_for_load"
        save_timeout_seed(data, Path(tmp), set(), set())

        corpus, _, irep = load_corpus(Path(tmp))
        assert data in corpus
        assert hash_data(data) in irep


def test_load_corpus_timeouts_not_irreplaceable_when_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        data = b"timeout_not_irreplicable"
        save_timeout_seed(data, Path(tmp), set(), set())

        corpus, _, irep = load_corpus(Path(tmp), load_irreplaceable=False)
        assert data in corpus
        assert hash_data(data) not in irep
