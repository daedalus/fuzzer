"""Negative bloom: seed hashes -> sized bloom file -> skip known inputs while fuzzing."""

import argparse
import gzip
import pickle

import pytest

from fuzzer_tool.cli.commands import cmd_negative_bloom, load_negative_bloom
from fuzzer_tool.core.bloom import BloomFilter
from fuzzer_tool.core.negative_bloom import (
    NEG_BLOOM_FACTOR,
    NegativeBloomError,
    build,
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


def _make_corpus(root, seeds):
    sd = root / "seeds"
    sd.mkdir(parents=True)
    for i, s in enumerate(seeds):
        (sd / f"seed{i}").write_bytes(s)


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

    def test_load_helper_none(self):
        assert load_negative_bloom(None) is None

    def test_load_helper_bad_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_negative_bloom(str(tmp_path / "absent"))
