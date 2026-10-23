"""Negative bloom: seed hashes -> sized bloom file -> skip known inputs while fuzzing."""

import argparse
import gzip
import pickle

import pytest

from fuzzer_tool.adapters.filesystem import iter_seed_batches
from fuzzer_tool.cli.commands import cmd_negative_bloom, load_negative_bloom
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.negative_bloom import (
    NEG_BLOOM_FACTOR,
    NegativeBloomError,
    build,
    build_batched,
    load,
    save,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.state_store import UnsafeStateError
from fuzzer_tool.services.fuzzer import Fuzzer


def _seeds(n, seed=7):
    rng = RandPool(seed=seed)
    return [rng.randbytes(16) + i.to_bytes(4, "big") for i in range(n)]


class TestBuild:
    def test_capacity_is_ten_times_corpus(self):
        bf = build(_seeds(50))
        assert bf.capacity == NEG_BLOOM_FACTOR * 50
        assert NEG_BLOOM_FACTOR == 10

    def test_no_false_negatives(self):
        seeds = _seeds(500)
        bf = build(seeds)
        assert all(bf.query_bytes(s) for s in seeds)

    def test_absent_inputs_not_flagged(self):
        # 10x headroom keeps the realised FPR far below 1e-3.
        bf = build(_seeds(500))
        absent = _seeds(2000, seed=99)
        assert not any(bf.query_bytes(a) for a in absent)

    def test_query_bytes_does_not_mutate(self):
        bf = build(_seeds(10))
        bits, added = bytes(bf._bits), bf.n_added
        bf.query_bytes(b"never-added")
        assert (bytes(bf._bits), bf.n_added) == (bits, added)

    def test_empty_corpus_rejected(self):
        with pytest.raises(NegativeBloomError):
            build([])

    def test_single_seed(self):
        bf = build([b"x"])
        assert bf.capacity == NEG_BLOOM_FACTOR
        assert bf.query_bytes(b"x")

    def test_duplicate_seeds_counted_once(self):
        bf = build([b"a", b"a", b"b"])
        assert bf.n_added == 2


class TestRoundTrip:
    def test_roundtrip_identical(self, tmp_path):
        bf = build(_seeds(200))
        path = tmp_path / "neg.pkl.gz"
        save(path, bf)
        back = load(path)
        # Control: the original equals itself; only then is the reload comparison meaningful.
        assert bytes(bf._bits) == bytes(bf._bits)
        assert bytes(back._bits) == bytes(bf._bits)
        assert (back.capacity, back.m, back._k, back.n_added) == (
            bf.capacity,
            bf.m,
            bf._k,
            bf.n_added,
        )

    def test_file_is_gzip_pickle(self, tmp_path):
        path = tmp_path / "neg.pkl.gz"
        save(path, build(_seeds(5)))
        with gzip.open(path, "rb") as fh:
            assert isinstance(pickle.loads(fh.read()), dict)

    def test_save_leaves_no_temp_files(self, tmp_path):
        save(tmp_path / "neg.pkl.gz", build(_seeds(5)))
        assert [p.name for p in tmp_path.iterdir()] == ["neg.pkl.gz"]

    def test_loaded_filter_membership(self, tmp_path):
        seeds = _seeds(300)
        save(tmp_path / "n.pkl.gz", build(seeds))
        back = load(tmp_path / "n.pkl.gz")
        assert all(back.query_bytes(s) for s in seeds)
        assert not any(back.query_bytes(a) for a in _seeds(500, seed=5))


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned > /tmp/_neg_bloom_pwned",))


def _write(path, obj):
    with gzip.open(path, "wb") as fh:
        fh.write(pickle.dumps(obj))


class TestAdversarialFile:
    def test_malicious_pickle_refused(self, tmp_path):
        path = tmp_path / "evil.pkl.gz"
        _write(path, _Evil())
        with pytest.raises(UnsafeStateError):
            load(path)

    def test_truncated_file(self, tmp_path):
        path = tmp_path / "n.pkl.gz"
        save(path, build(_seeds(50)))
        path.write_bytes(path.read_bytes()[:20])
        with pytest.raises(NegativeBloomError):
            load(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(NegativeBloomError):
            load(tmp_path / "absent.pkl.gz")

    def test_not_a_dict(self, tmp_path):
        _write(tmp_path / "n.pkl.gz", [1, 2, 3])
        with pytest.raises(NegativeBloomError):
            load(tmp_path / "n.pkl.gz")

    def test_wrong_version(self, tmp_path):
        d = build(_seeds(5)).to_dict()
        d["version"] = 999
        _write(tmp_path / "n.pkl.gz", d)
        with pytest.raises(NegativeBloomError):
            load(tmp_path / "n.pkl.gz")

    def test_bits_length_mismatch(self, tmp_path):
        d = build(_seeds(5)).to_dict()
        d["bits"] = d["bits"][:-1]
        _write(tmp_path / "n.pkl.gz", d)
        with pytest.raises(NegativeBloomError):
            load(tmp_path / "n.pkl.gz")

    def test_bad_capacity(self, tmp_path):
        d = build(_seeds(5)).to_dict()
        d["capacity"] = 0
        _write(tmp_path / "n.pkl.gz", d)
        with pytest.raises(NegativeBloomError):
            load(tmp_path / "n.pkl.gz")


class _StubFuzzer:
    """Only what ``_is_repeat`` touches."""

    def __init__(self, neg=None, dedup_execs=True):
        self._exec_bloom = BloomFilter(capacity=64, error_rate=1e-3)
        self._dedup_execs = dedup_execs
        self._orbit_specs = []
        self._orbit_folded = 0
        self.cuckoo_seed_filter = None
        self._neg_bloom = neg
        self._neg_bloom_hits = 0

    _is_repeat = Fuzzer._is_repeat


class TestFuzzerSkip:
    def test_known_seed_is_repeat(self):
        f = _StubFuzzer(build([b"seed-a", b"seed-b"]))
        assert f._is_repeat(b"seed-a", b"parent") is True
        assert f._neg_bloom_hits == 1

    def test_unknown_mutant_passes(self):
        f = _StubFuzzer(build([b"seed-a"]))
        assert f._is_repeat(b"fresh", b"parent") is False
        assert f._neg_bloom_hits == 0

    def test_bytearray_mutant(self):
        f = _StubFuzzer(build([b"seed-a"]))
        assert f._is_repeat(bytearray(b"seed-a"), b"parent") is True

    def test_independent_of_no_dedup_execs(self):
        f = _StubFuzzer(build([b"seed-a"]), dedup_execs=False)
        assert f._is_repeat(b"seed-a", b"parent") is True

    def test_survives_exec_bloom_wipe(self):
        f = _StubFuzzer(build([b"seed-a"]))
        for i in range(200):  # crosses exec-bloom capacity -> generational wipe
            f._is_repeat(i.to_bytes(4, "big"), b"p")
        assert f._is_repeat(b"seed-a", b"parent") is True

    def test_disabled_is_noop(self):
        f = _StubFuzzer(None)
        assert f._is_repeat(b"seed-a", b"parent") is False

    def test_negative_hit_does_not_pollute_exec_bloom(self):
        f = _StubFuzzer(build([b"seed-a"]))
        f._is_repeat(b"seed-a", b"parent")
        assert f._exec_bloom.n_added == 0


class _CountingSha:
    """Counts ``hashlib.sha256`` calls made by ``core.bloom``."""

    def __init__(self, real):
        self._real = real
        self.calls = 0

    def __call__(self, data=b""):
        self.calls += 1
        return self._real(data)


def _count_sha(monkeypatch):
    import fuzzer_tool.core.bloom as bloom_mod

    counter = _CountingSha(bloom_mod.hashlib.sha256)
    monkeypatch.setattr(bloom_mod.hashlib, "sha256", counter)
    return counter


class TestSingleHash:
    def test_regression_neg_bloom_hashes_mutant_once(self, monkeypatch):
        # Control: exec bloom alone hashes once.
        f = _StubFuzzer(None)
        sha = _count_sha(monkeypatch)
        f._is_repeat(b"fresh", b"parent")
        assert sha.calls == 1

        # Negative bloom + exec bloom share that one digest.
        g = _StubFuzzer(build([b"seed-a"]))
        sha.calls = 0
        g._is_repeat(b"fresh", b"parent")
        assert sha.calls == 1

    def test_shared_digest_matches_independent_filter(self):
        # Falsification: reference exec filter fed through update_bytes.
        f = _StubFuzzer(build([b"seed-a"]))
        ref = BloomFilter(capacity=64, error_rate=1e-3)
        mutants = [i.to_bytes(4, "big") for i in range(150)] * 2  # repeats + wipe

        for m in mutants:
            assert f._is_repeat(m, b"parent") == ref.update_bytes(m, reset_on_full=True)
        assert f._exec_bloom._bits == ref._bits

    def test_cuckoo_exec_backend_with_neg_bloom(self):
        # Adversarial: the shared int digest must never reach a cuckoo filter.
        from fuzzer_tool.core.cuckoo import CuckooFilter

        f = _StubFuzzer(build([b"seed-a"]))
        f._exec_bloom = CuckooFilter(capacity=64)
        assert f._is_repeat(b"fresh", b"parent") is False
        assert f._is_repeat(b"fresh", b"parent") is True
        assert f._is_repeat(b"seed-a", b"parent") is True


def _make_corpus(root, seeds):
    sd = root / "seeds"
    sd.mkdir(parents=True)
    for i, s in enumerate(seeds):
        (sd / f"seed{i}").write_bytes(s)


def _args(corpus, out, batch_size=4):
    return argparse.Namespace(corpus=str(corpus), out=str(out), batch_size=batch_size)


def _batches(corpus, size):
    return list(iter_seed_batches(corpus, size))


class TestBatched:
    def test_matches_in_memory_build(self, tmp_path):
        seeds = _seeds(37)
        _make_corpus(tmp_path / "c", seeds)

        # Control: the in-memory reference agrees with itself.
        assert bytes(build(seeds)._bits) == bytes(build(list(reversed(seeds)))._bits)

        bf = build_batched(_batches(tmp_path / "c", 5))
        ref = build(seeds)
        assert bytes(bf._bits) == bytes(ref._bits)
        assert (bf.capacity, bf.n_added) == (ref.capacity, ref.n_added)

    @pytest.mark.parametrize("size", [1, 3, 37, 1000])
    def test_batch_sizes_bounded_and_complete(self, tmp_path, size):
        seeds = _seeds(37)
        _make_corpus(tmp_path / "c", seeds)
        batches = _batches(tmp_path / "c", size)
        assert all(1 <= len(b) <= size for b in batches)
        assert sorted(s for b in batches for s in b) == sorted(seeds)

    def test_files_not_renamed(self, tmp_path):
        _make_corpus(tmp_path / "c", _seeds(4))
        names = sorted(p.name for p in (tmp_path / "c" / "seeds").iterdir())
        _batches(tmp_path / "c", 2)
        assert sorted(p.name for p in (tmp_path / "c" / "seeds").iterdir()) == names

    def test_same_seed_set_as_load_corpus(self, tmp_path):
        # Falsification: pruned/, symlinks, deltas and nested dirs read as load_corpus does.
        import json

        from fuzzer_tool.adapters.filesystem import (
            compute_delta,
            hash_data,
            load_corpus,
            save_to_corpus,
        )

        corpus = tmp_path / "c"
        parent, child = b"P" * 400, b"P" * 399 + b"Q"
        save_to_corpus(parent, corpus, set())
        # Legacy delta record (save_to_corpus no longer writes them).
        (corpus / "deltas").mkdir()
        delta = {"parent": hash_data(parent), "diff": compute_delta(parent, child), "v": 1}
        (corpus / "deltas" / f"delta_{hash_data(child)}.json").write_text(json.dumps(delta))
        (corpus / "seeds" / "pruned").mkdir(parents=True, exist_ok=True)
        (corpus / "seeds" / "pruned" / "gone").write_bytes(b"pruned-seed")
        (corpus / "seeds" / "ab").mkdir(exist_ok=True)
        (corpus / "seeds" / "ab" / "nested").write_bytes(b"nested-seed")
        (tmp_path / "outside").write_bytes(b"outside-seed")
        (corpus / "seeds" / "link").symlink_to(tmp_path / "outside")

        streamed = sorted(s for b in _batches(corpus, 2) for s in b)
        loaded, _, _ = load_corpus(corpus, add_default=False)
        assert streamed == sorted(loaded)
        assert child in streamed
        assert b"pruned-seed" not in streamed and b"outside-seed" not in streamed

    def test_reads_seed_zip(self, tmp_path):
        from fuzzer_tool.adapters import seed_zip
        from fuzzer_tool.adapters.filesystem import save_to_corpus
        from fuzzer_tool.adapters.seed_zip import ZipMode

        corpus = tmp_path / "c"
        corpus.mkdir()
        seed_zip.configure(corpus, ZipMode.ON)
        try:
            save_to_corpus(b"zipped-seed", corpus, set())
            seed_zip.flush_all()
        finally:
            seed_zip.configure(corpus, ZipMode.OFF)

        # Zip mode off: the archive is still read, read-only.
        assert [s for b in _batches(corpus, 8) for s in b] == [b"zipped-seed"]

    def test_extends_base(self):
        base = build([b"a", b"b"])
        bf = build_batched([[b"b", b"c"]], base)
        assert bf is base
        assert bf.query_bytes(b"c") and bf.n_added == 3 and bf.n_rejected == 1

    def test_empty_rejected(self):
        with pytest.raises(NegativeBloomError):
            build_batched([])

    def test_empty_with_base_rejected(self):
        with pytest.raises(NegativeBloomError):
            build_batched([], build([b"a"]))

    def test_bad_batch_size(self, tmp_path):
        _make_corpus(tmp_path / "c", [b"x"])
        with pytest.raises(ValueError):
            _batches(tmp_path / "c", 0)


class TestCli:
    def test_cmd_builds_file(self, tmp_path):
        seeds = _seeds(40)
        _make_corpus(tmp_path / "c", seeds)
        out = tmp_path / "neg.pkl.gz"
        rc = cmd_negative_bloom(argparse.Namespace(corpus=str(tmp_path / "c"), out=str(out)))
        assert rc == 0
        bf = load(out)
        assert bf.capacity == NEG_BLOOM_FACTOR * 40
        assert all(bf.query_bytes(s) for s in seeds)

    def test_cmd_missing_corpus(self, tmp_path):
        rc = cmd_negative_bloom(
            argparse.Namespace(corpus=str(tmp_path / "none"), out=str(tmp_path / "o"))
        )
        assert rc == 1
        assert not (tmp_path / "o").exists()

    def test_cmd_empty_corpus(self, tmp_path):
        (tmp_path / "c").mkdir()
        rc = cmd_negative_bloom(
            argparse.Namespace(corpus=str(tmp_path / "c"), out=str(tmp_path / "o"))
        )
        assert rc == 1
        assert not (tmp_path / "o").exists()

    def test_cmd_does_not_inject_default_seed(self, tmp_path):
        # load_corpus's synthetic b"AAAAAAAA" must not land in the negative set.
        _make_corpus(tmp_path / "c", [b"real-seed"])
        out = tmp_path / "o.pkl.gz"
        cmd_negative_bloom(argparse.Namespace(corpus=str(tmp_path / "c"), out=str(out)))
        assert not load(out).query_bytes(b"AAAAAAAA")

    def test_cmd_extends_existing_file(self, tmp_path):
        old, new = _seeds(30, seed=1), _seeds(20, seed=2)
        _make_corpus(tmp_path / "a", old)
        _make_corpus(tmp_path / "b", new)
        out = tmp_path / "neg.blf"

        assert cmd_negative_bloom(_args(tmp_path / "a", out)) == 0
        assert cmd_negative_bloom(_args(tmp_path / "b", out)) == 0

        bf = load(out)
        assert all(bf.query_bytes(s) for s in old + new)
        assert bf.n_added == len(old) + len(new)
        assert bf.capacity == NEG_BLOOM_FACTOR * len(old)  # sized by the first build

    def test_cmd_rerun_same_corpus_adds_nothing(self, tmp_path):
        seeds = _seeds(25)
        _make_corpus(tmp_path / "c", seeds)
        out = tmp_path / "neg.blf"
        cmd_negative_bloom(_args(tmp_path / "c", out))
        bits = bytes(load(out)._bits)

        assert cmd_negative_bloom(_args(tmp_path / "c", out)) == 0
        bf = load(out)
        assert bytes(bf._bits) == bits
        assert bf.n_added == len(seeds)

    def test_cmd_corrupt_existing_file_kept(self, tmp_path):
        # Adversarial: an unreadable bloom must not be silently replaced.
        _make_corpus(tmp_path / "c", _seeds(5))
        out = tmp_path / "neg.blf"
        out.write_bytes(b"not a bloom")

        assert cmd_negative_bloom(_args(tmp_path / "c", out)) == 1
        assert out.read_bytes() == b"not a bloom"

    def test_cmd_empty_corpus_keeps_existing(self, tmp_path):
        _make_corpus(tmp_path / "a", _seeds(5))
        (tmp_path / "empty").mkdir()
        out = tmp_path / "neg.blf"
        cmd_negative_bloom(_args(tmp_path / "a", out))
        before = out.read_bytes()

        assert cmd_negative_bloom(_args(tmp_path / "empty", out)) == 1
        assert out.read_bytes() == before

    def test_cmd_negative_batch_size(self, tmp_path):
        _make_corpus(tmp_path / "c", [b"x"])
        assert cmd_negative_bloom(_args(tmp_path / "c", tmp_path / "o", batch_size=-1)) == 1
        assert not (tmp_path / "o").exists()

    def test_cmd_legacy_mode(self, tmp_path):
        seeds = _seeds(10)
        _make_corpus(tmp_path / "c", seeds)
        out = tmp_path / "neg.blf"
        assert cmd_negative_bloom(_args(tmp_path / "c", out, batch_size=0)) == 0
        assert all(load(out).query_bytes(s) for s in seeds)

    def test_load_helper_none(self):
        assert load_negative_bloom(None) is None

    def test_load_helper_bad_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_negative_bloom(str(tmp_path / "absent"))
