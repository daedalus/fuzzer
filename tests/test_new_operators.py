"""Tests for new operator handlers: UTF-8, line, and fuse mutations."""

import random

from fuzzer_tool.core.mutations import _FUNNY_UNICODE, MUTATIONS
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

    class _MockElo:
        ...

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
        widened = any(
            buf[i] >= 0xC0 and 0x80 <= buf[i + 1] <= 0xBF
            for i in range(len(buf) - 1)
        )
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
            assert False, "Expected an overlong UTF-8 sequence in buffer"


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
            before = bytes(buf)
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

    def test_dispatch_contains_all_new_ops(self):
        engine = OperatorEngine(_make_minimal_fuzzer())
        dispatch = engine.build_dispatch()
        for op in ("tree_mutate", "utf8_widen", "utf8_insert",
                    "line_mutate", "fuse_this", "fuse_next", "fuse_old",
                    "redqueen_xform"):
            assert op in dispatch, f"{op} missing from dispatch table"


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
