"""Merkle site locator + post-mutation repair (``--merkle-sites``).

Covers core/merkle_sites.py and its hook in the mutation round. Expected
trees are built here with hashlib, independently of the code under test.
"""

import hashlib

import pytest

from fuzzer_tool.core import merkle_sites as ms
from fuzzer_tool.core.merkle_sites import (
    Hash,
    Kind,
    Link,
    MerkleBook,
    Tag,
    locate,
    repair,
)
from tests.support.scripted_rng import ScriptedRng

HEADER = b"MRKL\x00\x01\x02\x03"
CHUNK = 64
VARIANTS = [(d, t) for d in (False, True) for t in (False, True)]
IDS = ["sha256", "sha256-tagged", "sha256d", "sha256d-tagged"]


def _body(n, salt=1):
    return bytes((i * 31 + salt * 7 + (i >> 3)) & 0xFF for i in range(n))


def _chunks(k):
    return [_body(CHUNK, i + 1) for i in range(k)]


def _hash(data, double):
    out = hashlib.sha256(data).digest()
    return hashlib.sha256(out).digest() if double else out


def _tree(chunks, double=False, tagged=False):
    """HEADER | chunks | leaf hashes | level 1 | ... | root (odd node pairs itself)."""
    leaf, node = (b"\x00", b"\x01") if tagged else (b"", b"")
    layer = [_hash(leaf + c, double) for c in chunks]
    layers = [layer]
    while len(layer) > 1:
        padded = layer + [layer[-1]] if len(layer) % 2 else layer
        layer = [_hash(node + padded[i] + padded[i + 1], double) for i in range(0, len(padded), 2)]
        layers.append(layer)
    return HEADER + b"".join(chunks) + b"".join(h for ly in layers for h in ly)


def _flip(chunks, i, at=5):
    out = list(chunks)
    out[i] = out[i][:at] + bytes([out[i][at] ^ 0xFF]) + out[i][at + 1 :]
    return out


def _mutate(seed, chunk_idx, at=5):
    buf = bytearray(seed)
    buf[len(HEADER) + chunk_idx * CHUNK + at] ^= 0xFF
    return buf


@pytest.mark.parametrize(("double", "tagged"), VARIANTS, ids=IDS)
class TestTree:
    def test_locate_finds_every_link(self, double, tagged):
        seed = _tree(_chunks(4), double, tagged)
        links = locate(seed)
        assert len(links) == 7  # 4 leaves + 3 nodes
        assert {ln.hash for ln in links} == {Hash.SHA256D if double else Hash.SHA256}
        assert {ln.tag for ln in links} == {Tag.RFC6962 if tagged else Tag.NONE}

    def test_tag_tells_leaf_from_node(self, double, tagged):
        """Untagged, H(64 bytes) is a leaf or a node alike: the tag splits them."""
        kinds = [ln.kind for ln in locate(_tree(_chunks(4), double, tagged))]
        if tagged:
            assert (kinds.count(Kind.LEAF), kinds.count(Kind.NODE)) == (4, 3)
        else:
            assert len(kinds) == 7

    def test_repair_rebuilds_the_tree(self, double, tagged):
        chunks = _chunks(4)
        seed = _tree(chunks, double, tagged)
        buf = _mutate(seed, 1)
        done = repair(buf, locate(seed), len(seed))
        assert bytes(buf) == _tree(_flip(chunks, 1), double, tagged)
        assert done == 3  # one leaf plus two ancestors changed

    def test_valid_seed_is_a_fixed_point(self, double, tagged):
        seed = _tree(_chunks(4), double, tagged)
        buf = bytearray(seed)
        assert repair(buf, locate(seed), len(seed)) == 0
        assert bytes(buf) == seed

    def test_odd_leaf_count_pairs_the_last_node_with_itself(self, double, tagged):
        chunks = _chunks(3)
        seed = _tree(chunks, double, tagged)
        buf = _mutate(seed, 2)
        repair(buf, locate(seed), len(seed))
        assert bytes(buf) == _tree(_flip(chunks, 2), double, tagged)


class TestEdges:
    def test_tail_digest_of_the_body(self):
        body = _body(100, 3)
        seed = body + hashlib.sha256(body).digest()
        buf = bytearray(seed)
        buf[7] ^= 1
        repair(buf, locate(seed), len(seed))
        assert bytes(buf[-32:]) == hashlib.sha256(bytes(buf[:-32])).digest()

    def test_root_inside_a_covered_body_chains(self):
        """Tail digest covers the tree: leaf -> root -> tail, in that order."""
        chunks = _chunks(4)
        tree = _tree(chunks)
        seed = tree + hashlib.sha256(tree).digest()
        buf = _mutate(seed, 0)
        repair(buf, locate(seed), len(seed))
        want = _tree(_flip(chunks, 0))
        assert bytes(buf) == want + hashlib.sha256(want).digest()

    def test_length_change_is_skipped(self):
        seed = _tree(_chunks(4))
        buf = _mutate(seed, 1) + b"\x00"
        before = bytes(buf)
        assert repair(buf, locate(seed), len(seed)) == 0
        assert bytes(buf) == before

    def test_order_drops_cycles(self):
        a = Link(Kind.NODE, Hash.SHA256, Tag.NONE, pos=0, a=32, b=32)
        b = Link(Kind.NODE, Hash.SHA256, Tag.NONE, pos=32, a=0, b=0)
        assert ms.order([a, b]) == []

    def test_max_links_bounds_the_result(self, monkeypatch):
        monkeypatch.setattr(ms, "MAX_LINKS", 3)
        assert len(locate(_tree(_chunks(8)))) <= 3


class TestAdversarial:
    @pytest.mark.parametrize("n", [0, 1, 63, 64, 100])
    def test_short_or_random_input_has_no_links(self, n):
        assert locate(_body(n, 9)) == []

    def test_constant_buffers_have_no_links(self):
        assert locate(b"\x00" * 4096) == []
        assert locate(b"\xff" * 4096) == []

    def test_oversize_is_not_scanned(self):
        seed = _tree(_chunks(4)) + b"\x00" * ms.LOCATE_MAX_LEN
        assert locate(seed) == []

    def test_unhashed_field_is_not_a_link(self):
        seed = _body(300, 5)  # nothing is a digest of anything
        assert locate(seed) == []

    def test_low_entropy_field_is_rejected(self, monkeypatch):
        """A genuine digest is dropped once the distinct-byte floor exceeds 32."""
        body = _body(64, 4)
        seed = body + hashlib.sha256(body).digest()
        assert len(locate(seed)) == 1
        monkeypatch.setattr(ms, "MIN_DISTINCT", 33)
        assert locate(seed) == []


class TestBook:
    def test_apply_repairs_and_counts(self):
        chunks = _chunks(4)
        seed = _tree(chunks)
        buf = _mutate(seed, 3)
        book = MerkleBook()
        done = book.apply(seed, buf, ScriptedRng(randoms=[0.0]))
        assert done == 3
        assert bytes(buf) == _tree(_flip(chunks, 3))
        assert book.repairs == 3
        assert book.located == 7

    def test_ten_percent_stay_broken(self):
        seed = _tree(_chunks(4))
        buf = _mutate(seed, 3)
        before = bytes(buf)
        done = MerkleBook().apply(seed, buf, ScriptedRng(randoms=[ms.REPAIR_P]))
        assert done == 0
        assert bytes(buf) == before

    def test_locate_runs_once_per_parent(self):
        seed = _tree(_chunks(4))
        book = MerkleBook()
        for _ in range(3):
            book.apply(seed, _mutate(seed, 0), ScriptedRng(randoms=[0.0]))
        assert book.located == 7

    def test_negative_result_is_cached(self):
        seed = _body(300, 6)
        book = MerkleBook()
        for _ in range(3):
            assert book.apply(seed, bytearray(seed), ScriptedRng(randoms=[0.0])) == 0
        assert book.seeds_tracked() == 1

    def test_lru_is_bounded(self, monkeypatch):
        monkeypatch.setattr(ms, "MAX_SEEDS", 2)
        book = MerkleBook()
        for s in range(5):
            book.sites_for(_body(300, s))
        assert book.seeds_tracked() == 2


class TestEngineHook:
    @staticmethod
    def _fuzzer(tmp_path, seed, on):
        from fuzzer_tool.services.fuzzer import Fuzzer

        corpus = tmp_path / "corpus"
        (corpus / "seeds").mkdir(parents=True)
        (tmp_path / "crashes").mkdir()
        (corpus / "seeds" / "seed1").write_bytes(seed)
        f = Fuzzer(
            "/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=4096,
            merkle_sites=on,
        )

        def apply(op, buf, data, track_effect):
            buf[len(HEADER) + 5] ^= 0xFF
            return buf

        f._operators._apply_op_once = apply
        return f

    @staticmethod
    def _valid(mutant):
        return mutant == _tree([mutant[8 + i * CHUNK : 8 + (i + 1) * CHUNK] for i in range(4)])

    def test_flag_on_every_mutant_keeps_a_valid_tree(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "REPAIR_P", 1.0)
        seed = _tree(_chunks(4))
        f = self._fuzzer(tmp_path, seed, on=True)
        mutants = [f._operators.mutate(seed) for _ in range(5)]
        assert all(m != seed and self._valid(m) for m in mutants)

    def test_falsification_flag_off_breaks_the_tree(self, tmp_path):
        seed = _tree(_chunks(4))
        f = self._fuzzer(tmp_path, seed, on=False)
        mutants = [f._operators.mutate(seed) for _ in range(5)]
        assert not any(self._valid(m) for m in mutants)
        assert f._merkle_sites is None

    def test_adversarial_seed_without_a_tree_is_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ms, "REPAIR_P", 1.0)
        seed = _body(300, 8)
        f = self._fuzzer(tmp_path, seed, on=True)
        want = bytearray(seed)
        want[len(HEADER) + 5] ^= 0xFF
        assert f._operators.mutate(seed) == bytes(want)
