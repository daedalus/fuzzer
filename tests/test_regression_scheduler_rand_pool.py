"""The shared ``RandPool`` must exist before any scheduler asks for it.

``Fuzzer.__init__`` passed ``rng=self._rand_pool`` to ``CMAESScheduler``
roughly 9,000 characters before ``self._rand_pool`` was assigned, so
``--cma-es`` raised ``AttributeError`` at construction. The comment on that
argument explains why the pool has to be threaded in (Hard Rule 16: an
unseeded scheduler means a crash found under it cannot be replayed from
``--seed``); nothing checked that it existed yet.

Position, not presence, is what is asserted -- a source-order check rather
than a construction test, because constructing a real ``Fuzzer`` needs a
target binary and the defect is purely one of ordering.
"""

import inspect
import re

import pytest


class TestRandPoolAvailableBeforeSchedulers:
    """``rng=self._rand_pool`` read before the assignment that creates it."""

    def test_rand_pool_assigned_before_first_use(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        src = inspect.getsource(Fuzzer.__init__)
        assign = src.find("self._rand_pool = RandPool")
        assert assign != -1, "Fuzzer.__init__ no longer constructs its RandPool"

        first_use = min(
            (m.start() for m in re.finditer(r"rng=self\._rand_pool", src)),
            default=None,
        )
        if first_use is None:
            pytest.skip("no scheduler currently receives the shared pool")
        assert assign < first_use, (
            "self._rand_pool is read at char "
            f"{first_use} but only assigned at char {assign}: "
            "constructing that scheduler raises AttributeError"
        )
