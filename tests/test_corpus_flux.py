"""Tests for gross corpus flux (P4-T6 of the thermo handover).

The item exists because ``len(corpus)`` is a *net* quantity and a flat net
figure has two causes it cannot distinguish: nothing happening, or additions
and evictions running at matched rates. The falsifier stated in the handover
is the first test below -- two synthetic runs whose final corpus size is
*identical* must be distinguishable from the flux log.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.analyzer_registry import REGISTRY
from fuzzer_tool.core.analyzers.analyzer_corpus_flux import CorpusFlux


# --- the stated falsifier --------------------------------------------------


def test_quiet_and_balanced_plateaus_are_distinguishable() -> None:
    """Same net size, same net flux, opposite verdicts.

    This is the whole item. If this test can be made to pass by a tracker
    that only records net change, the tracker is not doing anything.
    """
    quiet = CorpusFlux()
    balanced = CorpusFlux()
    for _ in range(50):
        # Quiet: nothing enters, nothing leaves.
        quiet.tick()
        # Balanced: three in, three out, every tick.
        balanced.record_addition(3)
        balanced.record_eviction(3)
        balanced.tick()

    # Indistinguishable on everything net.
    assert quiet.net() == balanced.net() == 0

    # Distinguishable on gross.
    assert quiet.gross() == 0
    assert balanced.gross() == 300

    # And on the derived verdict: turnover is None for quiet *by design*,
    # because |net| / gross is 0/0 there and reporting 1.0 would call a dead
    # corpus perfectly balanced.
    assert quiet.turnover() is None
    assert balanced.turnover() == pytest.approx(1.0)


def test_turnover_is_none_rather_than_one_when_nothing_happened() -> None:
    """The 0/0 case must not read as balance."""
    flux = CorpusFlux()
    flux.tick()
    assert flux.turnover() is None
    flux.record_addition()
    flux.tick()
    assert flux.turnover() == pytest.approx(0.0)  # one-directional


def test_turnover_spans_one_way_to_balanced() -> None:
    growth = CorpusFlux()
    growth.record_addition(10)
    growth.tick()
    assert growth.turnover() == pytest.approx(0.0)

    decay = CorpusFlux()
    decay.record_eviction(10)
    decay.tick()
    assert decay.turnover() == pytest.approx(0.0)

    lopsided = CorpusFlux()
    lopsided.record_addition(9)
    lopsided.record_eviction(1)
    lopsided.tick()
    assert lopsided.turnover() == pytest.approx(0.2)


# --- rejections are a third channel, not part of either --------------------


def test_rejections_do_not_inflate_gross_flux() -> None:
    """A corpus that rejects everything must not look busy.

    A rejected candidate never entered and displaced nothing, so folding it
    into additions (or into gross) would make admission pressure read as
    turnover.
    """
    flux = CorpusFlux()
    flux.record_rejection(1000)
    flux.tick()
    assert flux.gross() == 0
    assert flux.net() == 0
    assert flux.turnover() is None
    assert flux.windowed() == (0, 0, 1000)
    assert flux.total_rejections == 1000


# --- bucketing -------------------------------------------------------------


def test_batched_evictions_and_trickled_additions_share_a_bucket() -> None:
    """Evictions arrive in batches, additions one at a time.

    Bucketing by tick is what makes the two rates comparable; without it a
    single 40-seed minimize pass would dominate any instantaneous ratio.
    """
    flux = CorpusFlux()
    for _ in range(40):
        flux.record_addition()
    flux.record_eviction(40)
    flux.tick()
    rates = flux.rates()
    assert rates["additions_per_tick"] == pytest.approx(40.0)
    assert rates["evictions_per_tick"] == pytest.approx(40.0)
    assert rates["net_per_tick"] == pytest.approx(0.0)
    assert rates["gross_per_tick"] == pytest.approx(80.0)


def test_counts_before_the_first_tick_are_pending_not_lost() -> None:
    flux = CorpusFlux()
    flux.record_addition(5)
    assert flux.ticks == 0
    assert flux.rates() == {}
    assert flux.windowed() == (0, 0, 0)  # not in the window yet
    assert flux.total_additions == 5  # but counted lifetime
    flux.tick()
    assert flux.windowed() == (5, 0, 0)


def test_window_evicts_old_buckets_but_totals_are_lifetime() -> None:
    flux = CorpusFlux(window=10)
    for _ in range(25):
        flux.record_addition(2)
        flux.tick()
    assert flux.ticks == 10
    assert flux.windowed()[0] == 20  # last ten ticks only
    assert flux.total_additions == 50  # all of them


# --- persistence -----------------------------------------------------------


def test_save_load_roundtrip_preserves_window_and_totals() -> None:
    flux = CorpusFlux(window=7)
    for i in range(5):
        flux.record_addition(i)
        flux.record_eviction(1)
        flux.record_rejection(2)
        flux.tick()
    restored = CorpusFlux()
    restored.load(flux.save())
    assert restored.window == 7
    assert restored.windowed() == flux.windowed()
    assert restored.gross() == flux.gross()
    assert restored.total_additions == flux.total_additions
    assert restored.total_evictions == flux.total_evictions
    assert restored.total_rejections == flux.total_rejections


def test_load_from_empty_state_is_inert() -> None:
    flux = CorpusFlux()
    flux.load({})
    assert flux.ticks == 0
    assert flux.total_additions == 0
    assert flux.turnover() is None


# --- registration ----------------------------------------------------------


def test_corpus_flux_is_registered_and_always_on() -> None:
    """No ``available`` gate: the counters must cover every campaign.

    An analyzer that is only sometimes constructed produces a figure whose
    absence is indistinguishable from zero flux, which is the exact
    confusion this item exists to remove.
    """
    spec = REGISTRY._specs["corpus_flux"]
    assert spec.available is None, "corpus_flux must not be gated behind a flag"
    assert spec.activate is not None


def test_state_is_a_plain_store_section_not_a_legacy_file() -> None:
    """corpus_flux must NOT be in LEGACY_JSON_FILES.

    That map exists to migrate the eleven per-component JSON files that
    predate the single consolidated store. A key added today has no legacy
    file to migrate, so listing it there would make the loader look for an
    on-disk file that never existed. The modern store is a plain
    section-keyed dict, so ``set``/``get`` need no registration at all --
    which is exactly the trap, since adding the key to the legacy map
    appears to work.
    """
    from fuzzer_tool.core.state_store import LEGACY_JSON_FILES

    assert "corpus_flux" not in LEGACY_JSON_FILES


def test_state_roundtrips_through_the_store_section() -> None:
    from fuzzer_tool.core.state_store import StateStore

    store = StateStore.__new__(StateStore)
    store._data = {}
    flux = CorpusFlux()
    flux.record_addition(4)
    flux.record_eviction(2)
    flux.tick()
    store.set("corpus_flux", flux.save())
    restored = CorpusFlux()
    restored.load(store.get("corpus_flux"))
    assert restored.windowed() == (4, 2, 0)
    assert restored.gross() == 6


# --- z_score: is the net drift distinguishable from a random walk? --------
#
# net and gross alone don't say whether a nonzero net is a real directional
# trend or just the imbalance you'd expect from a handful of coin flips.
# Model each admission/eviction as an i.i.d. +-1 step (X_i); under the null
# of undirected churn, mu=0 and sigma=1, so by the CLT
# Z_n = sum(X_i - mu) / (sigma * sqrt(n)) = net / sqrt(gross).


def test_z_score_is_none_when_there_has_been_no_flux() -> None:
    """0/0 must not read as a z-score of 0 (which would claim confidence)."""
    flux = CorpusFlux()
    flux.tick()
    assert flux.z_score() is None
    assert flux.is_significant_drift() is None


def test_z_score_matches_the_closed_form() -> None:
    flux = CorpusFlux()
    flux.record_addition(7)
    flux.record_eviction(3)
    flux.tick()
    # net=4, gross=10 -> z = 4 / sqrt(10)
    assert flux.z_score() == pytest.approx(4 / (10**0.5))


def test_single_event_imbalance_is_not_significant() -> None:
    """The falsifier: bare turnover would call one lone addition 100%
    one-directional, but a single +-1 step carries no statistical power --
    it is indistinguishable from a coin flip. gross must temper net.
    """
    flux = CorpusFlux()
    flux.record_addition()
    flux.tick()
    assert flux.turnover() == pytest.approx(0.0)  # looks maximally directional
    assert flux.z_score() == pytest.approx(1.0)  # but |z|=1 is unremarkable
    assert flux.is_significant_drift() is False


def test_large_one_directional_run_is_significant() -> None:
    """Sustained growth with no evictions at all: gross is large enough for
    the same directionality to actually be improbable under the null."""
    flux = CorpusFlux(window=200)
    for _ in range(100):
        flux.record_addition()
        flux.tick()
    assert flux.z_score() == pytest.approx(10.0)  # 100 / sqrt(100)
    assert flux.is_significant_drift() is True


def test_small_residual_imbalance_in_balanced_churn_is_not_significant() -> None:
    """Balanced turnover (per the module's own thesis) can still leave a
    small nonzero net just from noise; that residual should not trip the
    significance flag merely for being nonzero."""
    flux = CorpusFlux()
    flux.record_addition(5)
    flux.record_eviction(4)
    flux.tick()
    # net=1, gross=9 -> z = 1/sqrt(9) = 0.33
    assert flux.z_score() == pytest.approx(1 / (9**0.5))
    assert flux.is_significant_drift() is False


def test_is_significant_drift_respects_custom_threshold() -> None:
    flux = CorpusFlux()
    flux.record_addition()
    flux.tick()
    assert flux.z_score() == pytest.approx(1.0)
    assert flux.is_significant_drift(threshold=1.96) is False
    assert flux.is_significant_drift(threshold=0.5) is True
