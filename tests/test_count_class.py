"""Tests for core/count_class.py that are *not* enumerations.

The classification ladders, the u16 packing and ``new_bits`` all have
domains of at most 2**16 and are walked exhaustively in
``test_count_class_exhaustive.py`` (port item P2-6). Forty-four hand-picked
examples used to live here; they are gone rather than kept alongside,
because keeping both means the sparse copy is what fails first on an edit
and the sparse copy is the one whose failure says least.

Two of them were worse than redundant. ``test_8byte_boundary_overlap`` and
``test_8byte_new_coverage`` read as coverage of the word loop in
``new_bits`` while choosing byte values on which the word loop and the byte
tail happened to agree -- so the file's most reassuring test names sat
directly on top of its only bug.

What stays here is what enumeration cannot express: when the u16 table is
built, and what it costs.
"""

import sys

import pytest

import fuzzer_tool.core.count_class as mod
from fuzzer_tool.core.count_class import _build_u16_table


class TestLookupU16Laziness:
    """LOOKUP_U16 must not be built at import time.

    It is 65 536 entries and, with numpy an unconditional import, it is
    unreachable in normal operation -- ``classify_counts`` only falls
    through to it for an empty buffer. Building it eagerly cost ~128 KB
    retained per process for a path that never runs, and the fuzzer forks
    per exec.
    """

    def setup_method(self):
        self._saved = mod.__dict__.pop("LOOKUP_U16", None)

    def teardown_method(self):
        if self._saved is not None:
            mod.__dict__["LOOKUP_U16"] = self._saved
        else:
            mod.__dict__.pop("LOOKUP_U16", None)

    def test_absent_until_first_access(self):
        assert "LOOKUP_U16" not in mod.__dict__

    def test_first_access_builds_and_caches(self):
        table = mod.LOOKUP_U16
        assert len(table) == 65536
        assert mod.__dict__["LOOKUP_U16"] is table
        assert mod.LOOKUP_U16 is table  # second access hits the cache, not __getattr__

    def test_unknown_attribute_still_raises(self):
        """The PEP 562 hook must not swallow typos into None."""
        with pytest.raises(AttributeError, match="LOOKUP_U32"):
            _ = mod.LOOKUP_U32

    def test_uses_array_not_list(self):
        """array('H') over list[int]: ~128 KB retained instead of ~2.6 MB.

        Asserted on the type rather than on a measured size, because the
        per-object overhead this guards against is interpreter-specific
        while the storage choice is the actual decision.
        """
        table = _build_u16_table()
        assert table.typecode == "H"
        assert table.itemsize == 2
        assert sys.getsizeof(table) < 200_000
