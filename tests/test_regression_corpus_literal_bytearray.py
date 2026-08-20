"""Regression: corpus-learning helpers raised on the live corpus type.

`extract_corpus_literals` and `_build_verse` are both annotated as taking
`bytes`, and both were called with `Fuzzer.corpus` entries, which are
`bytearray` from startup onward. Slicing a bytearray yields a bytearray,
which is unhashable, so the first `lit not in seen_int` membership test
raised `TypeError: unhashable type: 'bytearray'`.

That meant `corpus_literal_insert` and `versifier_generate` could never do
anything in a real campaign -- not degrade, not work occasionally, but throw
on every single invocation from the moment a corpus existed.

It stayed hidden because the no-op sweep in
`tests/test_regression_no_op_mutations.py` caught and discarded operator
exceptions, so both operators were reported as "available but never changed
any input". That is a true statement pointing at a completely wrong cause:
it reads as a missing state gate, and there was no missing state gate.

The type annotations were not wrong about intent, so the fix is to coerce at
the boundary rather than to loosen the annotations -- these helpers hash
their slices, and the whole class of bug comes from a mutable type reaching
code that needs a hashable one.
"""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.mutations import extract_corpus_literals
from fuzzer_tool.core.mutations.generic import _build_verse

TEXT = b"key = value\nname_field = 12345\nother = abcdef\n"


class TestExtractCorpusLiterals:
    def test_accepts_bytearray_the_live_corpus_type(self):
        """The regression itself. Fuzzer.corpus holds bytearray, so this is
        the only call shape that ever happens in production."""
        ints, strs = extract_corpus_literals([bytearray(TEXT)])
        assert ints or strs

    def test_bytes_and_bytearray_give_identical_results(self):
        a = extract_corpus_literals([bytes(TEXT)])
        b = extract_corpus_literals([bytearray(TEXT)])
        assert a == b

    def test_mixed_corpus_does_not_raise(self):
        """A live corpus can hold both: seeds loaded from disk and seeds
        appended after mutation do not necessarily share a type."""
        ints, strs = extract_corpus_literals([bytes(TEXT), bytearray(TEXT), memoryview(TEXT)])
        assert ints or strs

    def test_returned_literals_are_hashable(self):
        """Callers put these in sets and use them as dict keys; returning
        bytearray would move the same failure downstream."""
        ints, strs = extract_corpus_literals([bytearray(TEXT)])
        assert set(ints) is not None
        assert set(strs) is not None
        for lit in list(ints) + list(strs):
            assert isinstance(lit, bytes)

    def test_finds_the_expected_literals(self):
        """Guard against 'fixed' by making it return nothing."""
        ints, strs = extract_corpus_literals([bytearray(TEXT)])
        assert b"12345" in ints
        assert any(b"value" in s or s == b"value" for s in strs)

    def test_empty_corpus_is_not_an_error(self):
        assert extract_corpus_literals([]) == ([], [])


class TestBuildVerse:
    def test_accepts_bytearray(self):
        assert _build_verse(bytearray(TEXT), random.Random(1)) is not None

    def test_bytes_and_bytearray_agree_on_reachability(self):
        rng = random.Random(1)
        assert (_build_verse(bytes(TEXT), rng) is None) == (
            _build_verse(bytearray(TEXT), rng) is None
        )

    def test_binary_input_declines_rather_than_raises(self):
        """Non-text input is legitimately not versifiable; that must be a
        None, not an exception, or the operator is unusable on binary
        corpora."""
        binary = bytearray(range(256))
        assert _build_verse(binary, random.Random(1)) is None

    def test_generated_output_is_usable(self):
        verse = _build_verse(bytearray(TEXT), random.Random(7))
        assert verse is not None
        out = verse.Rhyme()
        assert isinstance(out, bytes | bytearray)


@pytest.mark.parametrize("op", ["corpus_literal_insert", "versifier_generate"])
def test_operators_run_against_a_bytearray_corpus(op):
    """End-to-end at the operator layer, which is where the impact was.

    Asserts only that the operator does not raise: whether it changes this
    particular input is the no-op sweep's job, and conflating the two is
    what hid the bug in the first place.
    """
    import tempfile
    from unittest.mock import patch

    from fuzzer_tool.core.operator_registry import REGISTRY
    from fuzzer_tool.services.fuzzer import Fuzzer

    tmpdir = tempfile.mkdtemp(prefix="noop_regression_")
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )
    f.corpus = [bytearray(TEXT), bytearray(TEXT + b"more_text = 999\n")]
    table = REGISTRY.dispatch(f._operators)
    buf = bytearray(TEXT)
    table[op](buf, len(buf) // 2, bytes(TEXT))
