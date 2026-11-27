"""Tests for CLI flag registration of new uniform schedulers."""

from fuzzer_tool.cli import commands


def test_seed_uniform_scheduler_flag_in_hail_mary():
    """seed_uniform_scheduler should be in _HAIL_MARY_FLAGS."""
    assert "seed_uniform_scheduler" in commands._HAIL_MARY_FLAGS


def test_op_uniform_flag_in_hail_mary():
    """op_uniform should be in _HAIL_MARY_FLAGS."""
    assert "uniform" in commands._HAIL_MARY_FLAGS


def test_pos_uniform_flag_in_hail_mary():
    """pos_uniform should be in _HAIL_MARY_FLAGS."""
    assert "pos_uniform" in commands._HAIL_MARY_FLAGS
