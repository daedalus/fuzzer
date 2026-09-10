"""Eviction must take every pair-keyed map with it, not just some.

`CmplogCollector` caps `pairs` at `_max_pairs` and evicts the lowest-value
entries when it overflows. The eviction site drops the pair from
`_pair_set`, `_pair_value`, `_pair_cmp` and `_pair_pc` -- and, before this,
left `_pair_occurrence` alone. Nothing else bounds that map, so the cap was
cosmetic for it: at `max_pairs=8` with 40 distinct pairs it held all 40, and
`high_confidence_pairs()` named 32 pairs the collector no longer held.

This is the second time in this codebase: `_maybe_prune` evicted seeds from
`seed_edges` and eight companion maps while never adjusting
`_edge_owner_count`, and the counts only went up. So the test below is
written against the *set* of pair-keyed maps discovered from the instance
rather than against a list of names, which is the only version that fails
when the next map is added and forgotten.
"""

from __future__ import annotations

from fuzzer_tool.core.cmplog import CmplogCollector

_CAP = 8
_DISTINCT = 40


def _feed(collector, n=_DISTINCT, runs=1):
    """Push *n* distinct pairs through the parser, *runs* times."""
    lines = [f"CMP {i:08x} {i + 1:08x} 0 4" for i in range(n)]
    for _ in range(runs):
        collector._parse_lines(lines)
    return collector


def _pair_keyed_maps(collector):
    """Every dict/set on the instance keyed by a pair tuple.

    Discovered, not listed: a map added later and forgotten at the eviction
    site is exactly the defect, and a hand-written list cannot see it.
    """
    found = {}
    for name, value in vars(collector).items():
        if not isinstance(value, (dict, set)):
            continue
        keys = value if isinstance(value, set) else value.keys()
        sample = next(iter(keys), None)
        if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[0], bytes):
            found[name] = value
    return found


def test_the_cap_is_enforced_on_pairs():
    c = _feed(CmplogCollector(max_tokens=100_000, max_pairs=_CAP))

    assert len(c.pairs) == _CAP
    assert len(c._pair_set) == _CAP
    assert c.evicted_pair_count == _DISTINCT - _CAP


def test_discovery_found_the_companion_maps():
    """Guard: an empty set would make the assertion below vacuous."""
    maps = _pair_keyed_maps(_feed(CmplogCollector(max_tokens=100_000, max_pairs=_CAP)))

    assert len(maps) >= 3, f"only found {sorted(maps)}"


def test_no_pair_keyed_map_outlives_the_cap():
    c = _feed(CmplogCollector(max_tokens=100_000, max_pairs=_CAP))

    oversized = {
        name: len(value) for name, value in _pair_keyed_maps(c).items() if len(value) > _CAP
    }
    assert not oversized, (
        f"kept entries for evicted pairs, so the cap does not bound them: {oversized}"
    )


def test_high_confidence_pairs_never_names_an_evicted_pair():
    """The public accessor is where the leak became visible."""
    c = _feed(CmplogCollector(max_tokens=100_000, max_pairs=_CAP), runs=2)

    ghosts = [p for p in c.high_confidence_pairs(min_occurrences=2) if p not in c._pair_set]
    assert not ghosts, f"{len(ghosts)} pairs reported as high-confidence are already evicted"


def test_confidence_still_accumulates_for_surviving_pairs():
    """Falsification: pruning the map must not clear counts for pairs kept.

    A fix that emptied `_pair_occurrence` wholesale would pass every
    assertion above and destroy the signal the map exists for.
    """
    c = CmplogCollector(max_tokens=100_000, max_pairs=_CAP)
    _feed(c, n=_CAP, runs=3)  # under the cap: nothing is evicted

    assert len(c.pairs) == _CAP
    assert c.evicted_pair_count == 0
    assert sorted(c._pair_occurrence.values()) == [3] * _CAP
    assert len(c.high_confidence_pairs(min_occurrences=3)) == _CAP
