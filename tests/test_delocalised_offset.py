"""The mutation offset published for a whole-buffer reordering operator.

`Fuzzer.fuzz_one` hands `_last_mutation_offset` to `record_coverage_diff` as
"the byte position the mutation touched". For an operator that rewrites the
whole buffer no such position exists, so the loop must publish None and let
the caller's existing `is not None` guard skip the observation.

Falsification: reverting the fix (publishing `byte_idx` unconditionally)
fails `test_delocalised_ops_publish_no_offset`. Adversarial: an operator name
that is not delocalised must keep publishing its offset, so the fix cannot be
"return None always", which would silently disable liveness entirely.
"""

import inspect

import pytest

from fuzzer_tool.services.operators import _DELOCALISED_OPS, OperatorEngine


class TestDelocalisedOps:
    def test_every_listed_op_exists_and_ignores_byte_idx(self):
        """Each listed name must have a handler that declares byte_idx unused.

        A handler that grew a real use of byte_idx would still be skipped by
        the list, silently losing observations it could have contributed.
        """
        for name in _DELOCALISED_OPS:
            handler = getattr(OperatorEngine, f"_op_{name}", None)
            assert handler is not None, name
            params = list(inspect.signature(handler).parameters)
            assert params[2].startswith("_"), (name, params)

    def test_list_is_not_every_delocalised_handler(self):
        """The list is deliberately narrow.

        139 of 156 handlers declare `_byte_idx`; most of them pick their own
        single site and do have a true offset to report. Widening this set to
        all of them would disable the liveness estimator for ~89% of
        mutations, which is a different change from fixing the fabricated
        attribution for the four operators that have no offset at all.
        """
        ignoring = [
            n
            for n in dir(OperatorEngine)
            if n.startswith("_op_")
            and len(list(inspect.signature(getattr(OperatorEngine, n)).parameters)) >= 3
            and list(inspect.signature(getattr(OperatorEngine, n)).parameters)[2].startswith("_")
        ]
        assert len(_DELOCALISED_OPS) < len(ignoring) / 10

    def test_source_publishes_none_for_delocalised_ops(self):
        """Pin the loop line itself.

        The behaviour is one assignment inside `_apply_mutations`; reaching it
        through a real mutation round needs a whole Fuzzer. Asserting on the
        source is the cheap guard against the line being reverted, which is
        exactly the regression this file exists for.
        """
        src = inspect.getsource(OperatorEngine)
        assert "None if op in _DELOCALISED_OPS else byte_idx" in src
        assert "f._last_mutation_offset = byte_idx" not in src


class TestDelocalisedOpsMembership:
    @pytest.mark.parametrize("name", ["byte_shuffle", "chunk_shuffle", "token_shuffle"])
    def test_whole_buffer_reorderings_are_listed(self, name):
        assert name in _DELOCALISED_OPS

    @pytest.mark.parametrize("name", ["bit_flip", "arithmetic", "region_shuffle"])
    def test_localisable_ops_are_not_listed(self, name):
        assert name not in _DELOCALISED_OPS
