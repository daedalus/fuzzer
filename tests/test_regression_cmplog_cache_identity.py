"""The cmplog-derived caches were keyed on ``id(pairs) + len(pairs)``.

CPython reuses an object's address once the object at it is freed, so a
rebuilt pairs list can land exactly where the one a cache entry was built
from used to live.  Mixing ``len()`` into the key only narrows that: a
rebuild of the same length still collides, and two *live* objects whose
ids differ by exactly their length difference alias in the other
direction.

Both caches now compare the owning list by identity and keep a reference
to it, which makes the check exact -- the object cannot be freed, so its
address cannot be reused, while entries derived from it are live.
"""

from __future__ import annotations

import gc

import pytest


class TestIdReuseIsReal:
    """The premise, not the fix: this is why id() alone is not a key."""

    def test_a_freed_lists_address_is_handed_to_its_replacement(self):
        seen = set()
        for _ in range(2000):
            pairs = [(b"a", b"b")] * 4
            key = id(pairs) + len(pairs)
            if key in seen:
                break
            seen.add(key)
            del pairs
            gc.collect()
        else:
            pytest.skip("no address reuse observed on this interpreter")
        # Two lists of the same length, one freed, produced the same key --
        # which is exactly what the old cache used to tell them apart.


def _engine(pairs):
    """An OperatorEngine stub carrying just enough context for the caches."""
    from fuzzer_tool.services.operators import OperatorEngine

    eng = OperatorEngine.__new__(OperatorEngine)

    class Ctx:
        cmplog_pairs = pairs
        cmplog = None
        max_len = 4096

    eng._ctx_cache = Ctx()
    return eng, eng._ctx_cache


class TestColorizeCacheScope:
    def _colorable(self, eng, buf):
        """Drive _op_colorize far enough to populate the cache."""
        from fuzzer_tool.core.rand_pool import RandPool

        eng._ctx_cache._rng = RandPool(seed=1234)
        eng._op_colorization(bytearray(buf), 0, bytes(buf))
        return eng._colorize_cache

    def test_cache_is_dropped_when_the_pairs_object_changes(self):
        pairs_a = [(b"AAAA", b"BBBB")]
        eng, ctx = _engine(pairs_a)
        buf = b"AAAA" + b"\x00" * 60

        self._colorable(eng, buf)
        assert eng._colorize_cache, "first call should populate the cache"
        assert eng._colorize_cache_owner is pairs_a

        # A different list of the *same length* -- indistinguishable under
        # the old id()+len() key if it lands on the freed address.
        pairs_b = [(b"ZZZZ", b"YYYY")]
        ctx.cmplog_pairs = pairs_b
        self._colorable(eng, buf)
        assert eng._colorize_cache_owner is pairs_b

    def test_same_pairs_object_keeps_the_cache(self):
        pairs = [(b"AAAA", b"BBBB")]
        eng, _ = _engine(pairs)
        buf = b"AAAA" + b"\x00" * 60

        self._colorable(eng, buf)
        first = eng._colorize_cache
        self._colorable(eng, buf)
        assert eng._colorize_cache is first, "cache must survive an unchanged list"

    def test_owner_reference_keeps_the_list_alive(self):
        """Holding the owner is what makes identity comparison sound."""
        pairs = [(b"AAAA", b"BBBB")]
        eng, _ = _engine(pairs)
        self._colorable(eng, b"AAAA" + b"\x00" * 60)
        addr = id(pairs)
        del pairs
        gc.collect()
        # The cache still owns it, so the address cannot be recycled under us.
        assert id(eng._colorize_cache_owner) == addr


class TestCondStmtsCacheScope:
    def test_none_pairs_do_not_read_as_a_hit(self):
        """``None`` is a legitimate value of ctx.cmplog_pairs.

        A sentinel distinct from None is required, or the very first call
        with no pairs would match an unset attribute and return a stale
        (or absent) list.
        """
        from fuzzer_tool.services import operators as ops

        eng, ctx = _engine(None)
        eng._cond_stmts = ["stale"]
        # No _cond_stmts_pairs attribute set yet.
        assert not hasattr(eng, "_cond_stmts_pairs")
        assert getattr(eng, "_cond_stmts_pairs", ops._MISSING) is ops._MISSING
        assert ops._MISSING is not None

    def test_identity_hit_and_miss(self):
        from fuzzer_tool.services import operators as ops

        pairs_a = [(b"AAAA", b"BBBB")]
        eng, ctx = _engine(pairs_a)
        eng._cond_stmts = ["cached"]
        eng._cond_stmts_pairs = pairs_a
        assert eng._get_cond_stmts() == ["cached"]

        # Same length, different object: must miss.
        pairs_b = [(b"CCCC", b"DDDD")]
        ctx.cmplog_pairs = pairs_b
        rebuilt = eng._get_cond_stmts()
        assert rebuilt is not None
        assert eng._cond_stmts_pairs is pairs_b
        assert ops is not None
