"""The cross-target gap is computed once per weight pass, not per seed.

``_weight_length_and_cross_target`` rebuilt ``{target: len(edges)}`` plus
min/max for every seed, though the result is the same for the whole pass.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.services.seed_picker import SeedPicker

_SEED = b"seed"
_EDGES = {"a": set(range(10)), "b": set(range(40)), "c": set(range(25))}


class _Exploding(dict):
    """target_cumulative_edges that fails if read during a hoisted pass."""

    def items(self):
        raise AssertionError("per-seed recompute")

    def __len__(self):
        raise AssertionError("per-seed recompute")


def _fuzzer(edges, seed_targets):
    tracker = SimpleNamespace(target_cumulative_edges=edges, seed_target_edges=seed_targets)
    return SimpleNamespace(
        multi_targets=True,
        _edge_tracker=tracker,
        _length_tracker=None,
        _seed_key=lambda s: s.hex(),
    )


def _picker() -> SeedPicker:
    return SeedPicker(type("o", (object,), {"__init__": lambda s: None})())


def _expected_factor(edges) -> float:
    counts = sorted(len(e) for e in edges.values())
    return 1.0 + min((counts[-1] - counts[0]) / max(counts[0], 1), 1.0)


def test_gap_matches_inline_formula():
    """Falsification: hoisted (target, factor) equals the old per-seed math."""
    assert _picker()._cross_target_gap(_fuzzer(_EDGES, {})) == ("a", _expected_factor(_EDGES))


@pytest.mark.parametrize(
    "edges",
    [{}, {"a": {1}}, {"a": {1, 2}, "b": {3, 4}}],
    ids=["none", "single", "no-gap"],
)
def test_no_gap_is_none(edges):
    """Adversarial: zero/one target or equal coverage -> no bonus."""
    assert _picker()._cross_target_gap(_fuzzer(edges, {})) is None


def test_hoisted_pass_skips_per_seed_recompute():
    """Falsification: inside a pass the helper never re-reads target edges."""
    sp = _picker()
    f = _fuzzer(_EDGES, {_SEED.hex(): {"a": {1}}})
    sp._pass_xtarget = sp._cross_target_gap(f)
    f._edge_tracker.target_cumulative_edges = _Exploding(_EDGES)

    w = sp._weight_length_and_cross_target(_SEED, {}, 1.0, f)
    assert w == pytest.approx(_expected_factor(_EDGES))


@pytest.mark.parametrize(
    "seed_targets", [{}, {_SEED.hex(): {"b": {1}}}, {_SEED.hex(): {"a": set()}}]
)
def test_bonus_only_for_seeds_on_least_covered_target(seed_targets):
    """Adversarial: no entry, another target, or an empty edge set -> no bonus."""
    sp = _picker()
    f = _fuzzer(_EDGES, seed_targets)
    sp._pass_xtarget = sp._cross_target_gap(f)
    assert sp._weight_length_and_cross_target(_SEED, {}, 1.0, f) == 1.0


def test_outside_a_pass_still_computes():
    """Adversarial: a direct call (no pass running) computes the gap itself."""
    sp = _picker()
    f = _fuzzer(_EDGES, {_SEED.hex(): {"a": {1}}})
    w = sp._weight_length_and_cross_target(_SEED, {}, 1.0, f)
    assert w == pytest.approx(_expected_factor(_EDGES))
