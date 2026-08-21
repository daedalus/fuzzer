"""Rehydration must work on corpora written by save_to_corpus itself.

tests/test_lineage_replay.py covers rehydrate_by_hash thoroughly, but its
fixture writes delta records at ``deltas/<hh>/delta_<h>.json`` -- the layout
the *reader* expected. save_to_corpus writes them flat at
``deltas/delta_<h>.json``. Nothing in the suite ever went through the writer,
so the two sides disagreed for as long as both existed and every delta was
unrecoverable by hash while six tests stayed green.

These tests deliberately never construct a corpus by hand: they call the
production writer and then ask the production reader to get the bytes back.
That is the only shape of test that can catch writer/reader drift.
"""

from __future__ import annotations

import json

from fuzzer_tool.adapters.filesystem import (
    SNAPSHOT_INTERVAL,
    hash_data,
    load_corpus,
    rehydrate_by_hash,
    save_to_corpus,
)


def _corpus(tmp_path):
    return tmp_path / "corpus"


class TestRehydrateRoundTrip:
    def test_full_snapshot_round_trips(self, tmp_path):
        d, seen = _corpus(tmp_path), set()
        data = b"a full snapshot with no parent"
        assert save_to_corpus(data, d, seen) is True
        assert rehydrate_by_hash(hash_data(data), d) == data

    def test_delta_round_trips(self, tmp_path):
        """The regression: a delta written by save_to_corpus must come back."""
        d, seen = _corpus(tmp_path), set()
        parent = b"A" * 64
        child = b"A" * 32 + b"B" + b"A" * 31
        save_to_corpus(parent, d, seen)
        save_to_corpus(child, d, seen, parent=parent, lineage_depth=1)

        # Confirm the writer really chose the delta path, so this test cannot
        # silently degrade into re-testing the full-snapshot case.
        deltas = list(d.rglob("delta_*.json"))
        assert len(deltas) == 1, f"expected one delta record, got {deltas}"

        assert rehydrate_by_hash(hash_data(child), d) == child

    def test_v2_delta_round_trips(self, tmp_path):
        """Length-changing edit takes the v2 encoder; must also round-trip."""
        d, seen = _corpus(tmp_path), set()
        parent = b"base input for v2 encoding"
        child = parent[:5] + b"XY" + parent[5:]
        save_to_corpus(parent, d, seen)
        save_to_corpus(child, d, seen, parent=parent, lineage_depth=1)

        rec = json.loads(next(d.rglob("delta_*.json")).read_text())
        assert rec["v"] == 2, f"expected a v2 record, got v{rec['v']}"
        assert rehydrate_by_hash(hash_data(child), d) == child

    def test_delta_chain_round_trips(self, tmp_path):
        """A multi-link chain resolves through repeated parent lookups."""
        d, seen = _corpus(tmp_path), set()
        cur = b"Z" * 64
        save_to_corpus(cur, d, seen)
        chain = [cur]
        for i in range(1, min(4, SNAPSHOT_INTERVAL)):
            nxt = bytearray(cur)
            nxt[i] = 0x40 + i
            nxt = bytes(nxt)
            save_to_corpus(nxt, d, seen, parent=cur, lineage_depth=i)
            chain.append(nxt)
            cur = nxt
        for entry in chain:
            assert rehydrate_by_hash(hash_data(entry), d) == entry

    def test_sharded_delta_layout_still_readable(self, tmp_path):
        """Corpora written by a version that sharded deltas must still read.

        This is why the reader accepts both spellings rather than being
        switched to the flat one.
        """
        d, seen = _corpus(tmp_path), set()
        parent = b"Q" * 48
        child = b"Q" * 20 + b"R" + b"Q" * 27
        save_to_corpus(parent, d, seen)
        save_to_corpus(child, d, seen, parent=parent, lineage_depth=1)

        flat = next(d.rglob("delta_*.json"))
        h = hash_data(child)
        sharded = d / "deltas" / h[:2] / flat.name
        sharded.parent.mkdir(parents=True, exist_ok=True)
        flat.rename(sharded)

        assert rehydrate_by_hash(h, d) == child

    def test_unknown_hash_still_returns_none(self, tmp_path):
        """Accepting more paths must not turn a miss into a false hit."""
        d, seen = _corpus(tmp_path), set()
        save_to_corpus(b"something", d, seen)
        assert rehydrate_by_hash("deadbeefdeadbeef", d) is None

    def test_load_corpus_and_rehydrate_agree(self, tmp_path):
        """Every entry load_corpus returns must be rehydratable by hash.

        The two readers went out of sync once already (deltas/ was a sibling
        of seeds/, and load_corpus only scanned seeds/). Pin them together.
        """
        d, seen = _corpus(tmp_path), set()
        cur = b"S" * 40
        save_to_corpus(cur, d, seen)
        for i in range(1, 4):
            nxt = bytes(bytearray(cur[:i]) + b"!" + cur[i + 1 :])
            save_to_corpus(nxt, d, seen, parent=cur, lineage_depth=i)
            cur = nxt

        corpus, _, _ = load_corpus(d, add_default=False)
        assert len(corpus) == 4
        for entry in corpus:
            assert rehydrate_by_hash(hash_data(entry), d) == entry
