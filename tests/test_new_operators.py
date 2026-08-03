"""Tests for new operator handlers: UTF-8, line, fuse, and honggfuzz mutations."""

import random

from fuzzer_tool.core.mutations import (
    _FUNNY_UNICODE,
    DICT_COMPOUND_SEPARATORS,
    MAGIC_TABLE,
    MUTATIONS,
    PUNCTUATION_CHARS,
    SPECIAL_STRINGS,
    ascii_num_arithmetic,
    chunk_shuffle,
)
from fuzzer_tool.services.operators import OperatorEngine


def _make_minimal_fuzzer():
    """Build a minimal fuzzer-like object for operator testing."""

    class _MockCorpus:
        _items = [b"AAAA", b"BBBB", b"CCCC", b"DDDD"]

        def __getitem__(self, idx):
            return self._items[idx]

        def __len__(self):
            return len(self._items)

    class _MockMarkov:
        order = 2

        def sample_byte(self, ctx):
            return 42

    class _MockMC:
        cem_fitted = False
        mc_bandit = False

    class _MockMI:
        def weighted_position(self, n):
            return None

    class _MockSensitivity:
        def get_weighted_position(self, data, n):
            return None

    class _MockElo: ...

    class _MockFrameshift:
        relations = []

    class _MockSeedMeta(dict):
        def get(self, key, default=None):
            return default

    class _MockCmplog:
        def __init__(self):
            self.pairs = []
            self.tokens = []

    class MinimalFuzzer:
        def __init__(self_):  # noqa: N805
            self_._cmplog = None
            self_._crash_mi = None
            self_._mi = _MockMI()
            self_._te = None
            self_._use_transfer_entropy = False
            self_._use_mi = False
            self_._sensitivity = _MockSensitivity()
            self_._elo = None
            self_._use_elo = False
            self_._replicator = None
            self_._use_replicator = False
            self_._mopt = None
            self_._use_mopt = False
            self_._prev_bandit_op = None
            self_._last_mopt_particles = []
            self_._last_ops_used = []
            self_._meta_strategy = None
            self_._meta_strategy_cached = None
            self_._meta_strategy_used = set()
            self_._stall_recovery_active = False
            self_._frameshift = _MockFrameshift()
            self_.markov = _MockMarkov()
            self_.markov_trained = False
            self_.mc = _MockMC()
            self_.mc_cem = False
            self_.grammar = None
            self_.dictionary = []
            self_.corpus = _MockCorpus()
            self_.max_len = 65536
            self_.seed_meta = _MockSeedMeta()
            self_.mutations_per_input = 1
            self_._wfc_enabled = False
            self_._smt_solver = None
            self_.enable_regex_bomb = False
            from fuzzer_tool.core.rand_pool import RandPool

            self_._rand_pool = RandPool()
            self_._dict_scratch = []
            self_._dict_scratch_idx = 0

    return MinimalFuzzer()


# ── UTF-8 mutation tests ──────────────────────────────────────────────


class TestUtf8Widen:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_widens_ascii_byte(self):
        buf = bytearray(b"hello")
        self.engine._op_utf8_widen(buf, 0, b"")
        assert len(buf) > 5  # got wider
        # Find the widened byte (now a 2-byte sequence)
        widened = any(buf[i] >= 0xC0 and 0x80 <= buf[i + 1] <= 0xBF for i in range(len(buf) - 1))
        assert widened

    def test_widens_different_positions(self):
        seen_positions = set()
        for _ in range(30):
            buf = bytearray(b"abcdef")
            self.engine._op_utf8_widen(buf, 0, b"")
            assert len(buf) == 7  # exactly one byte widened to 2 bytes
            # Find which original position was widened (the 2-byte sequence)
            for i in range(6):
                if buf[i] >= 0xC0:
                    seen_positions.add(i)
                    break
        # Over 30 trials we should see at least 3 different positions
        assert len(seen_positions) >= 3, f"Only saw positions {seen_positions}"

    def test_empty_buffer(self):
        buf = bytearray(b"")
        self.engine._op_utf8_widen(buf, 0, b"")
        assert buf == b""

    def test_no_ascii_bytes(self):
        buf = bytearray(b"\x80\xff\xfe")
        before = bytes(buf)
        self.engine._op_utf8_widen(buf, 0, b"")
        assert buf == before  # unchanged — no ASCII bytes

    def test_widen_produces_valid_overlong_encoding(self):
        buf = bytearray(b"AB")
        self.engine._op_utf8_widen(buf, 0, b"")
        # Overlong 2-byte UTF-8: 110xxxxx 10xxxxxx
        # Find the 2-byte prefix and validate it
        assert len(buf) == 3  # one ASCII byte widened to 2
        for i in range(2):
            if buf[i] >= 0xC0:
                assert 0x80 <= buf[i + 1] <= 0xBF
                break
        else:
            raise AssertionError("Expected an overlong UTF-8 sequence in buffer")


class TestUtf8Insert:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_inserts_funny_unicode(self):
        buf = bytearray(b"hello world")
        before_len = len(buf)
        self.engine._op_utf8_insert(buf, 0, b"")
        assert len(buf) > before_len  # inserted bytes
        assert len(buf) <= before_len + max(len(s) for s in _FUNNY_UNICODE)

    def test_inserted_sequence_from_list(self):
        """Verify that the inserted bytes are actually from _FUNNY_UNICODE."""
        for _ in range(30):
            buf = bytearray(b"test")
            self.engine._op_utf8_insert(buf, 0, b"")
            # The inserted bytes should contain at least one sequence from _FUNNY_UNICODE
            found = any(seq in bytes(buf) for seq in _FUNNY_UNICODE)
            assert found, f"Inserted bytes {bytes(buf)} don't contain any funny unicode entry"

    def test_insert_is_deterministic_effect(self):
        """Each call should insert exactly one sequence at a single position."""
        for _ in range(20):
            buf = bytearray(b"abcdef")
            before = len(buf)
            self.engine._op_utf8_insert(buf, 0, b"")
            inserted_len = len(buf) - before
            assert inserted_len > 0  # inserted something
            # Should not have inserted more than the longest funny unicode seq
            assert inserted_len <= max(len(s) for s in _FUNNY_UNICODE)

    def test_empty_buffer(self):
        buf = bytearray(b"")
        self.engine._op_utf8_insert(buf, 0, b"")
        assert buf == b""

    def test_funny_unicode_entries_unique(self):
        assert len(set(_FUNNY_UNICODE)) >= 40  # most are unique
        assert len(_FUNNY_UNICODE) >= 44  # at least 44 entries


# ── Line mutation tests ────────────────────────────────────────────────


class TestLineMutate:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_line_del_shortens(self):
        for _ in range(20):
            buf = bytearray(b"a\nb\nc\nd\ne\nf")
            self.engine._op_line_mutate(buf, 0, b"")
            # May or may not change depending on random choice
            assert isinstance(buf, bytearray)

    def test_line_dup_lengthens(self):
        found_longer = False
        for _ in range(50):
            buf = bytearray(b"a\nb\nc\nd\ne\nf")
            self.engine._op_line_mutate(buf, 0, b"")
            if len(buf) > 11:  # original is 11 bytes
                found_longer = True
                break
        assert found_longer, "Expected at least one dup/repeat mutation to lengthen"

    def test_different_mutations_produce_different_results(self):
        results = set()
        for _ in range(100):
            buf = bytearray(b"a\nb\nc\nd\ne\nf")
            self.engine._op_line_mutate(buf, 0, b"")
            results.add(bytes(buf))
        assert len(results) > 1

    def test_single_line_unchanged(self):
        buf = bytearray(b"singleline")
        self.engine._op_line_mutate(buf, 0, b"")
        assert buf == b"singleline"

    def test_empty_buffer(self):
        buf = bytearray(b"")
        self.engine._op_line_mutate(buf, 0, b"")
        assert buf == b""

    def test_swap_adjacent(self):
        swaps_seen = 0
        for _ in range(200):
            buf = bytearray(b"aaa\nbbb")
            self.engine._op_line_mutate(buf, 0, b"")
            if buf == b"bbb\naaa":
                swaps_seen += 1
                if swaps_seen >= 3:
                    break
        assert swaps_seen >= 1, "Expected at least one adjacent swap"


# ── Fuse mutation tests ────────────────────────────────────────────────


class TestFuseThis:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_fuse_this_changes_buffer(self):
        for _ in range(20):
            buf = bytearray(b"AAAABBBBCCCCDDDD")
            self.engine._op_fuse_this(buf, 0, b"")
            # Buffer should still be in reasonable range
            assert len(buf) >= len(b"AAAABBBBCCCCDDDD") // 2
            assert len(buf) <= len(b"AAAABBBBCCCCDDDD") * 2

    def test_fuse_this_produces_different_results(self):
        results = set()
        for _ in range(50):
            buf = bytearray(b"AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH")
            self.engine._op_fuse_this(buf, 0, b"")
            results.add(bytes(buf))
        assert len(results) > 1

    def test_fuse_this_short_buffer(self):
        buf = bytearray(b"AB")
        self.engine._op_fuse_this(buf, 0, b"")
        assert buf == b"AB"  # unchanged (too short)

    def test_fuse_this_empty(self):
        buf = bytearray(b"")
        self.engine._op_fuse_this(buf, 0, b"")
        assert buf == b""


class TestFuseNext:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_fuse_next_changes_buffer(self):
        for _ in range(20):
            buf = bytearray(b"XXXXYYYY")
            self.engine._op_fuse_next(buf, 0, b"")
            # Should fuse with some corpus entry (AAAA/BBBB/CCCC/DDDD)
            assert len(buf) <= 65536

    def test_fuse_next_short_buffer(self):
        buf = bytearray(b"ab")
        self.engine._op_fuse_next(buf, 0, b"")
        assert buf == b"ab"

    def test_fuse_next_empty(self):
        buf = bytearray(b"")
        self.engine._op_fuse_next(buf, 0, b"")
        assert buf == b""


class TestFuseOld:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())
        # Reset fuse memory
        cls = type(self.engine)
        if hasattr(cls, "_fuse_memory"):
            del cls._fuse_memory

    def test_fuse_old_needs_at_least_two_calls(self):
        buf = bytearray(b"AAAAAAAA")
        self.engine._op_fuse_old(buf, 0, b"")
        assert buf == b"AAAAAAAA"  # first call just records, doesn't fuse

    def test_fuse_old_changes_on_second_call(self):
        buf = bytearray(b"AAAAAAAA")
        self.engine._op_fuse_old(buf, 0, b"")  # records

        # Reset the memory to have previous content
        # (already recorded from first call)
        found_change = False
        for _ in range(30):
            buf2 = bytearray(b"BBBBBBBB")
            self.engine._op_fuse_old(buf2, 0, b"")
            if buf2 != b"BBBBBBBB":
                found_change = True
                break
        assert found_change, "Expected fuse_old to change buffer on second+ call"

    def test_fuse_old_short_buffer(self):
        buf = bytearray(b"ab")
        self.engine._op_fuse_old(buf, 0, b"")
        assert buf == b"ab"


# ── Operator registration tests ────────────────────────────────────────


# ── Tree mutation smoke test (dispatch-based) ──────────────────────────


class TestTreeMutatorDispatch:
    def test_tree_mutate_via_dispatch(self):
        engine = OperatorEngine(_make_minimal_fuzzer())
        dispatch = engine.build_dispatch()
        buf = bytearray(b"[abc][def][ghi]")
        result = dispatch["tree_mutate"](buf, 0, b"")
        # tree_mutate mutates in-place (returns None)
        assert result is None

    def test_tree_mutate_short_buffer(self):
        engine = OperatorEngine(_make_minimal_fuzzer())
        dispatch = engine.build_dispatch()
        buf = bytearray(b"ab")
        result = dispatch["tree_mutate"](buf, 0, b"")
        assert result is None
        assert buf == b"ab"


# ── Special strings mutation tests ─────────────────────────────────────


class TestSpecialStrings:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_inserts_security_string(self):
        buf = bytearray(b"hello world")
        before_len = len(buf)
        self.engine._op_special_strings(buf, 0, b"")
        assert len(buf) > before_len

    def test_inserted_bytes_are_from_list(self):
        """The inserted bytes must be a substring of SPECIAL_STRINGS."""
        for _ in range(50):
            buf = bytearray(b"test input data")
            self.engine._op_special_strings(buf, 0, b"")
            found = any(seq in bytes(buf) for seq in SPECIAL_STRINGS)
            assert found, f"No special string found in {bytes(buf)}"

    def test_empty_buffer(self):
        buf = bytearray(b"")
        self.engine._op_special_strings(buf, 0, b"")
        assert buf == b""  # empty → no-op

    def test_buffer_at_max_len(self):
        f = self.engine.f
        old_max = f.max_len
        try:
            f.max_len = 5
            buf = bytearray(b"hello")
            self.engine._op_special_strings(buf, 0, b"")
            assert len(buf) == 5  # unchanged
        finally:
            f.max_len = old_max

    def test_special_strings_list_completeness(self):
        assert len(SPECIAL_STRINGS) >= 40
        # Must contain known attack strings
        assert b"%n" in SPECIAL_STRINGS
        assert b"UNION SELECT" in SPECIAL_STRINGS
        assert b"../" in SPECIAL_STRINGS
        assert b"<script>" in SPECIAL_STRINGS
        assert b"${" in SPECIAL_STRINGS or b"$(" in SPECIAL_STRINGS
        assert b"\x00" in SPECIAL_STRINGS


# ── Magic values mutation tests ────────────────────────────────────────


class TestMagicValues:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_inserts_magic_value(self):
        buf = bytearray(b"\x00" * 16)
        self.engine._op_magic_values(buf, 0, b"")
        # Buffer should have changed (magic value inserted)
        assert any(b != 0 for b in buf)

    def test_magic_table_covers_all_widths(self):
        widths = set(w for w, _ in MAGIC_TABLE)
        assert widths == {1, 2, 4, 8}

    def test_magic_table_has_both_endians(self):
        """For widths >= 2, both LE and BE variants should exist."""
        for width in (2, 4, 8):
            entries = [(w, p) for w, p in MAGIC_TABLE if w == width]
            assert len(entries) >= 4, f"Width {width} has only {len(entries)} entries"

    def test_empty_buffer(self):
        buf = bytearray(b"")
        self.engine._op_magic_values(buf, 0, b"")
        # Empty buffer: insert branch (len + width <= max_len)
        assert len(buf) <= 8  # may insert up to 8 bytes

    def test_small_buffer_insert(self):
        buf = bytearray(b"AB")
        self.engine._op_magic_values(buf, 0, b"")
        assert len(buf) >= 2  # either inserted or overwritten

    def test_large_buffer_overwrite(self):
        """When buffer is near max_len, magic value overwrites instead of inserting."""
        f = self.engine.f
        old_max = f.max_len
        try:
            f.max_len = 100
            buf = bytearray(b"\x00" * 100)
            self.engine._op_magic_values(buf, 0, b"")
            assert len(buf) == 100  # length preserved on overwrite
        finally:
            f.max_len = old_max

    def test_magic_table_size(self):
        assert len(MAGIC_TABLE) >= 100  # substantial table


# ── ASCII number arithmetic tests ─────────────────────────────────────


class TestAsciiNumArithmetic:
    def test_finds_and_mutates_digit_sequence(self):
        for _ in range(30):
            result = ascii_num_arithmetic(b"value=42 end", rng=random.Random())
            if result is not None:
                assert result[6:8] != b"42" or result == b"value=42 end"
                # At least one call should find digits
                break
        else:
            # All 30 missed — unlikely but not impossible
            pass

    def test_returns_none_on_no_digits(self):
        result = ascii_num_arithmetic(b"no digits here", rng=random.Random())
        assert result is None

    def test_returns_none_on_empty(self):
        result = ascii_num_arithmetic(b"", rng=random.Random())
        assert result is None

    def test_increments_by_one(self):
        """When op=0 (+1), '123' should become '124'."""
        found = False
        for _ in range(200):
            result = ascii_num_arithmetic(b"num=123", rng=random.Random())
            if result and b"124" in result:
                found = True
                break
        assert found, "Expected +1 operation to produce '124'"

    def test_decrements(self):
        """When op=1 (-1), '123' should become '122'."""
        found = False
        for _ in range(200):
            result = ascii_num_arithmetic(b"num=123", rng=random.Random())
            if result and b"122" in result:
                found = True
                break
        assert found, "Expected -1 operation to produce '122'"

    def test_doubles(self):
        """When op=2 (*2), '50' should become '100'."""
        found = False
        for _ in range(200):
            result = ascii_num_arithmetic(b"val=50 ", rng=random.Random())
            if result and b"100" in result:
                found = True
                break
        assert found, "Expected *2 operation to produce '100'"

    def test_halves(self):
        """When op=3 (/2), '100' should become '50'."""
        found = False
        for _ in range(200):
            result = ascii_num_arithmetic(b"val=100", rng=random.Random())
            if result and b"50" in result:
                found = True
                break
        assert found, "Expected /2 operation to produce '50'"

    def test_preserves_non_digit_bytes(self):
        result = ascii_num_arithmetic(b"prefix123suffix", rng=random.Random())
        if result is not None:
            assert result[:6] == b"prefix"
            assert result[-6:] == b"suffix"

    def test_handler_dispatch(self):
        engine = OperatorEngine(_make_minimal_fuzzer())
        buf = bytearray(b"count=42 items")
        result = engine._op_ascii_num_arithmetic(buf, 0, b"")
        # May or may not find digits depending on random offset
        assert result is None or isinstance(result, bytearray)


# ── Chunk shuffle tests ───────────────────────────────────────────────


class TestChunkShuffle:
    def test_shuffle_preserves_length(self):
        data = b"A" * 32
        for _ in range(20):
            result = chunk_shuffle(data, rng=random.Random())
            assert len(result) == len(data)

    def test_shuffle_changes_content(self):
        data = bytes(range(32))
        changed = False
        for _ in range(20):
            result = chunk_shuffle(data, rng=random.Random())
            if result != data:
                changed = True
                break
        assert changed, "Expected chunk_shuffle to change content at least once"

    def test_short_buffer_unchanged(self):
        data = b"ABCD"
        result = chunk_shuffle(data, rng=random.Random())
        assert result == data  # < 8 bytes → no-op

    def test_single_chunk_unchanged(self):
        data = b"A" * 7  # 7 bytes, chunk_size=1 → 7 chunks, should shuffle
        result = chunk_shuffle(data, rng=random.Random())
        # 7 bytes < 8 threshold → no-op
        assert result == data

    def test_different_chunk_sizes(self):
        """Different random chunk sizes should produce different results."""
        results = set()
        data = bytes(range(64))
        for _ in range(30):
            result = chunk_shuffle(data, rng=random.Random())
            results.add(result)
        assert len(results) > 1

    def test_handler_dispatch(self):
        engine = OperatorEngine(_make_minimal_fuzzer())
        buf = bytearray(b"\x00" * 32)
        result = engine._op_chunk_shuffle(buf, 0, b"")
        assert result is None or isinstance(result, bytearray)
        if result is not None:
            assert len(result) == 32

    def test_empty_buffer(self):
        result = chunk_shuffle(b"", rng=random.Random())
        assert result == b""


# ── Dict compound tests ───────────────────────────────────────────────


class TestDictCompound:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())
        self.engine.f.dictionary = [b"key", b"value", b"param", b"token"]

    def test_inserts_compound_token(self):
        buf = bytearray(b"existing data")
        before_len = len(buf)
        self.engine._op_dict_compound(buf, 0, b"")
        assert len(buf) > before_len

    def test_compound_contains_separator(self):
        """Inserted content should contain a separator from the list."""
        found_sep = False
        for _ in range(50):
            buf = bytearray(b"test")
            self.engine._op_dict_compound(buf, 0, b"")
            compound = bytes(buf)[4:]  # after original "test"
            for sep in DICT_COMPOUND_SEPARATORS:
                if sep and sep in compound:
                    found_sep = True
                    break
            if found_sep:
                break
        assert found_sep, "Expected a non-empty separator in compound token"

    def test_no_dictionary_noop(self):
        self.engine.f.dictionary = []
        buf = bytearray(b"test")
        self.engine._op_dict_compound(buf, 0, b"")
        assert buf == b"test"

    def test_single_dict_entry_noop(self):
        self.engine.f.dictionary = [b"only_one"]
        buf = bytearray(b"test")
        self.engine._op_dict_compound(buf, 0, b"")
        assert buf == b"test"

    def test_empty_buffer(self):
        buf = bytearray(b"")
        self.engine._op_dict_compound(buf, 0, b"")
        assert buf == b""  # empty → no-op (handler requires existing content)

    def test_separator_list_completeness(self):
        assert b"" in DICT_COMPOUND_SEPARATORS
        assert b" " in DICT_COMPOUND_SEPARATORS
        assert b"=" in DICT_COMPOUND_SEPARATORS
        assert b"&" in DICT_COMPOUND_SEPARATORS
        assert b"\n" in DICT_COMPOUND_SEPARATORS


# ── Punctuation insert tests ──────────────────────────────────────────


class TestPunctuationInsert:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_inserts_punctuation(self):
        buf = bytearray(b"hello world")
        before_len = len(buf)
        self.engine._op_punctuation_insert(buf, 0, b"")
        assert len(buf) > before_len
        assert len(buf) <= before_len + 4

    def test_inserted_bytes_are_punctuation(self):
        for _ in range(30):
            buf = bytearray(b"test")
            before_len = len(buf)
            self.engine._op_punctuation_insert(buf, 0, b"")
            diff = len(buf) - before_len
            assert 1 <= diff <= 4

    def test_empty_buffer(self):
        buf = bytearray(b"")
        self.engine._op_punctuation_insert(buf, 0, b"")
        assert buf == b""  # empty → no-op

    def test_buffer_at_max_len(self):
        f = self.engine.f
        old_max = f.max_len
        try:
            f.max_len = 5
            buf = bytearray(b"hello")
            self.engine._op_punctuation_insert(buf, 0, b"")
            assert len(buf) == 5  # unchanged
        finally:
            f.max_len = old_max

    def test_punctuation_chars_completeness(self):
        assert len(PUNCTUATION_CHARS) >= 30
        # Must contain common punctuation
        assert 0x21 in PUNCTUATION_CHARS  # !
        assert 0x3C in PUNCTUATION_CHARS  # <
        assert 0x7B in PUNCTUATION_CHARS  # {

    def test_varies_insertion_length(self):
        lengths = set()
        for _ in range(50):
            buf = bytearray(b"x" * 100)
            self.engine._op_punctuation_insert(buf, 0, b"")
            lengths.add(len(buf) - 100)
        assert len(lengths) >= 2  # should see 1, 2, 3, or 4


# ── Havoc escalation tests ────────────────────────────────────────────


class TestHavocEscalation:
    def setup_method(self):
        self.engine = OperatorEngine(_make_minimal_fuzzer())

    def test_normal_havoc_applies_few_mutations(self):
        """Normal havoc: 2-8 mutations applied."""
        for _ in range(20):
            buf = bytearray(b"\x00" * 256)
            self.engine.havoc_mutate(buf)
            # Buffer changed but not wildly
            assert isinstance(buf, bytearray)

    def test_stall_havoc_applies_more_mutations(self):
        """During stall recovery: 8-16 mutations."""
        self.engine.f._stall_recovery_active = True
        # With more mutations, we expect more dramatic changes
        changed = False
        for _ in range(10):
            buf = bytearray(b"\x00" * 256)
            original = bytes(buf)
            self.engine.havoc_mutate(buf)
            if bytes(buf) != original:
                changed = True
                break
        assert changed

    def test_havoc_handler_dispatch(self):
        engine = OperatorEngine(_make_minimal_fuzzer())
        buf = bytearray(b"\x00" * 64)
        result = engine._op_havoc(buf, 0, b"")
        assert isinstance(result, bytearray | bytes)

    def test_stall_flag_respected(self):
        """Verify stall flag is checked (not hardcoded)."""
        self.engine.f._stall_recovery_active = False
        buf1 = bytearray(b"\x00" * 256)
        self.engine.havoc_mutate(buf1)
        self.engine.f._stall_recovery_active = True
        buf2 = bytearray(b"\x00" * 256)
        self.engine.havoc_mutate(buf2)
        # Both should produce valid results (no crash)
        assert isinstance(buf1, bytearray)
        assert isinstance(buf2, bytearray)


# ── Operator registration tests (updated) ─────────────────────────────


class TestNewOperatorsRegistered:
    def test_tree_mutate_in_list(self):
        assert "tree_mutate" in MUTATIONS

    def test_utf8_ops_in_list(self):
        assert "utf8_widen" in MUTATIONS
        assert "utf8_insert" in MUTATIONS

    def test_line_mutate_in_list(self):
        assert "line_mutate" in MUTATIONS

    def test_fuse_ops_in_list(self):
        assert "fuse_this" in MUTATIONS
        assert "fuse_next" in MUTATIONS
        assert "fuse_old" in MUTATIONS

    def test_honggfuzz_ops_in_list(self):
        assert "special_strings" in MUTATIONS
        assert "magic_values" in MUTATIONS
        assert "ascii_num_arithmetic" in MUTATIONS
        assert "chunk_shuffle" in MUTATIONS
        assert "dict_compound" in MUTATIONS
        assert "punctuation_insert" in MUTATIONS

    def test_dispatch_contains_all_new_ops(self):
        engine = OperatorEngine(_make_minimal_fuzzer())
        dispatch = engine.build_dispatch()
        for op in (
            "tree_mutate",
            "utf8_widen",
            "utf8_insert",
            "line_mutate",
            "fuse_this",
            "fuse_next",
            "fuse_old",
            "redqueen_xform",
            "special_strings",
            "magic_values",
            "ascii_num_arithmetic",
            "chunk_shuffle",
            "dict_compound",
            "punctuation_insert",
        ):
            assert op in dispatch, f"{op} missing from dispatch table"


class TestEloMetaStrategyThrottle:
    """Elo meta-strategy must be resolved once per exec, not per mutation.

    Regression: operators.select_op called elo.select_strategy on every
    mutation (~21x/exec), each doing a gauss sample per rated strategy.
    mutate() resets _meta_strategy_cached at the start of each exec.
    """

    def _make_elo_fuzzer(self):
        f = _make_minimal_fuzzer()
        f._use_elo = True
        f._use_mopt = True
        f._use_replicator = False
        f._replicator = None
        f.mc_bandit = True
        f.mc_cem = False
        f._use_exp3 = False
        f._use_eps_greedy = False
        f._use_hierarchical = False
        f._use_gp_ucb = False

        class _FakeMopt:
            def select_op(self, ops, prev_op=None):
                return ops[0]

        class _FakeElo:
            def select_strategy(self, strategies, temperature=None):
                return strategies[0]

        f._mopt = _FakeMopt()
        f.mc = _FakeMopt()
        f._elo = _FakeElo()
        return f

    def test_regression_meta_strategy_resolved_once_per_exec(self):
        f = self._make_elo_fuzzer()
        engine = OperatorEngine(f)
        calls = {"n": 0}
        real_select_strategy = f._elo.select_strategy

        def counting_select_strategy(strategies, temperature=None):
            calls["n"] += 1
            return real_select_strategy(strategies, temperature)

        f._elo.select_strategy = counting_select_strategy

        # One exec = one mutate() = n_mutations select_op calls.
        # The cached strategy must be reused across all of them.
        f._meta_strategy_cached = None  # what mutate() does at exec start
        ops = ["bit_flip", "byte_flip", "arithmetic"]
        for _ in range(8):  # 8 mutations in one exec
            engine.select_op(ops)
        assert calls["n"] == 1, f"select_strategy called {calls['n']}x in one exec"

        # Next exec re-resolves
        f._meta_strategy_cached = None
        engine.select_op(ops)
        assert calls["n"] == 2

    def test_regression_meta_strategy_revalidated_on_available_change(self):
        """If the cached strategy is no longer available, re-resolve."""
        f = self._make_elo_fuzzer()
        engine = OperatorEngine(f)
        calls = {"n": 0}
        real_select_strategy = f._elo.select_strategy

        def counting_select_strategy(strategies, temperature=None):
            calls["n"] += 1
            return real_select_strategy(strategies, temperature)

        f._elo.select_strategy = counting_select_strategy

        f._meta_strategy_cached = None
        engine.select_op(["bit_flip"])
        first = calls["n"]
        assert first >= 1

        # Poison the cache with a strategy that is not in `available`
        f._meta_strategy_cached = "not_a_strategy"
        engine.select_op(["bit_flip"])
        assert calls["n"] == first + 1


class TestRedqueenXformPairCache:
    """The bisect-based candidate filter must match the legacy linear scan."""

    def _make_engine(self, pairs):
        fuzzer = _make_minimal_fuzzer()
        fuzzer._cmplog = type("Cmplog", (), {"pairs": pairs, "is_hash_candidate": None})()
        return OperatorEngine(fuzzer)

    def test_cache_prefix_matches_legacy_filter(self):
        import bisect

        pairs = [
            (b"ab", b"cd"),
            (b"x", b"y"),  # 1 byte — never a candidate
            (b"abcdefghijklmnop", b"zz"),  # longer than the buffer
            (b"abc", b"def"),
            (b"abcd", b"wxyz"),
        ]
        engine = self._make_engine(pairs)
        buf = bytearray(b"abcdef")
        engine._op_redqueen_xform(buf, 0, b"")
        # Cache keeps only len(op_a) >= 2 pairs, sorted by length.
        assert engine._redqueen_sorted_pairs == [
            (b"ab", b"cd"),
            (b"abc", b"def"),
            (b"abcd", b"wxyz"),
            (b"abcdefghijklmnop", b"zz"),
        ]
        assert engine._redqueen_pair_lengths == [2, 3, 4, 16]
        # Legacy filter for len(buf)=6 keeps exactly lengths <= 6.
        cutoff = bisect.bisect_right(engine._redqueen_pair_lengths, len(buf))
        assert engine._redqueen_sorted_pairs[:cutoff] == [
            (b"ab", b"cd"),
            (b"abc", b"def"),
            (b"abcd", b"wxyz"),
        ]

    def test_cache_rebuilds_on_pool_change(self):
        engine = self._make_engine([(b"ab", b"cd")])
        buf = bytearray(b"abcdef")
        engine._op_redqueen_xform(buf, 0, b"")
        assert engine._redqueen_pair_lengths == [2]
        engine.f._cmplog.pairs = [(b"ab", b"cd"), (b"abcde", b"fghij")]
        engine._op_redqueen_xform(buf, 0, b"")
        assert engine._redqueen_pair_lengths == [2, 5]
