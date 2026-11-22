"""Graded validity energy (``--validity-energy``): AFLSmart's validity-based
power schedule, with the parsed-byte fraction of the Weizz map as validity."""

from __future__ import annotations

import inspect

import pytest

from fuzzer_tool.core.validity import VALIDITY_ENERGY_GAIN, graded_validity_factor
from fuzzer_tool.core.weizz_tags import (
    attach_tags_to_meta,
    build_tag_map_from_cmplog,
    parsed_fraction_from_meta,
)

_PAIRS = [(b"HEAD", b"XXXX"), (b"BODY", b"YYYY")]


def _meta(data: bytes, pairs=_PAIRS) -> dict:
    return attach_tags_to_meta({}, build_tag_map_from_cmplog(data, pairs))


class TestParsedFraction:
    def test_counts_tagged_bytes_over_length(self):
        # HEAD + BODY tagged, the 2 filler bytes are not.
        assert parsed_fraction_from_meta(_meta(b"HEAD..BODY")) == pytest.approx(0.8)

    def test_fully_explained_is_one(self):
        assert parsed_fraction_from_meta(_meta(b"HEADBODY")) == pytest.approx(1.0)

    def test_nothing_tagged_is_zero(self):
        assert parsed_fraction_from_meta(_meta(b"zzzzzzzz")) == 0.0

    def test_more_structure_ranks_higher(self):
        dense = parsed_fraction_from_meta(_meta(b"HEADBODY"))
        sparse = parsed_fraction_from_meta(_meta(b"HEAD........BODY"))
        assert dense > sparse

    @pytest.mark.parametrize("meta", [None, {}, {"weizz_tags_rle": [], "weizz_tags_len": 0}])
    def test_absent_map_is_none(self, meta):
        assert parsed_fraction_from_meta(meta) is None

    def test_dirty_map_is_none_not_zero(self):
        meta = _meta(b"HEAD..BODY")
        meta["weizz_tags_dirty"] = True
        assert parsed_fraction_from_meta(meta) is None

    def test_never_exceeds_one(self):
        meta = {"weizz_tags_rle": [(0, 10, 7, 0, 0, 0)], "weizz_tags_len": 4}
        assert parsed_fraction_from_meta(meta) == 1.0


class TestGradedFactor:
    def test_unknown_and_empty_are_neutral(self):
        assert graded_validity_factor(None) == 1.0
        assert graded_validity_factor(0.0) == 1.0

    def test_linear_in_fraction(self):
        assert graded_validity_factor(0.5) == pytest.approx(1.0 + 0.5 * VALIDITY_ENERGY_GAIN)
        assert graded_validity_factor(1.0) == pytest.approx(1.0 + VALIDITY_ENERGY_GAIN)

    def test_clamps_above_one_and_honours_gain(self):
        assert graded_validity_factor(7.0) == graded_validity_factor(1.0)
        assert graded_validity_factor(0.5, gain=3.0) == pytest.approx(2.5)

    def test_boost_not_gate(self):
        # Never below 1.0: an unparsed seed is still fuzzed at base energy.
        assert all(graded_validity_factor(f / 10) >= 1.0 for f in range(0, 11))


def test_flag_and_wiring_exist():
    from fuzzer_tool.cli import commands
    from fuzzer_tool.services.fuzzer import Fuzzer

    assert "--validity-energy" in inspect.getsource(commands)
    assert "validity_energy" in inspect.signature(Fuzzer.__init__).parameters
