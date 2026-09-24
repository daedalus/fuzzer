"""Tests for the entropy-gradient seed strategy (core/schedulers/seed_entropy_gradient.py).

Proposal 4 of docs/handover/handover_entropy_seed_schedulers_2026-09-19.md.

Unlike its §1-3 siblings, this strategy is credited off a stream of
admission events (:meth:`record_child`) rather than scored purely off the
current corpus snapshot, so most of these tests drive it through a
scripted admission sequence and check the resulting credit against an
independently-computed marginal-entropy oracle, rather than sampling.
Draws are captured via a fixed-index RNG, never retried-until-hit (Hard
Rule 39).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.byte_entropy import CumulativeByteEntropy
from fuzzer_tool.core.schedulers.seed_entropy_gradient import (
    DEFAULT_DECAY,
    MIN_OBSERVATIONS,
    MIN_WEIGHT,
    EntropyGradientSeedStrategy,
)


class CapturingRng:
    """Records the weights it is handed and returns a fixed index."""

    def __init__(self, index: int = 0):
        self._index = index
        self.weights: list[float] = []

    def weighted_choice(self, seq, weights):
        self.weights = list(weights)
        return seq[self._index]


def _strategy(rng=None, **kwargs) -> EntropyGradientSeedStrategy:
    return EntropyGradientSeedStrategy(rng or CapturingRng(), **kwargs)


def _marginal_oracle(pool_seeds: list[bytes], child: bytes) -> float:
    """Δbits() from folding `child` into a pool of `pool_seeds`, spelled out
    independently of the strategy via two fresh CumulativeByteEntropy instances."""
    before = CumulativeByteEntropy()
    for s in pool_seeds:
        before.add(s)
    after = CumulativeByteEntropy()
    for s in pool_seeds:
        after.add(s)
    after.add(child)
    return after.bits() - before.bits()


def _fill_root(strategy: EntropyGradientSeedStrategy, corpus: list[bytes]) -> None:
    """Admit `corpus` as parentless roots (initial corpus load), crediting nothing."""
    for i in range(1, len(corpus) + 1):
        strategy.record_child(None, corpus[i - 1], corpus[:i])


class TestWarmup:
    def test_not_warmed_before_min_observations(self):
        strategy = _strategy()
        root = [b"seed-root"]
        _fill_root(strategy, root)
        corpus = list(root)
        for i in range(MIN_OBSERVATIONS - 1):
            child = f"child-{i}".encode()
            corpus.append(child)
            strategy.record_child(root[0], child, corpus)
        assert not strategy.warmed

    def test_warmed_at_min_observations(self):
        strategy = _strategy()
        root = [b"seed-root"]
        _fill_root(strategy, root)
        corpus = list(root)
        for i in range(MIN_OBSERVATIONS):
            child = f"child-{i}".encode()
            corpus.append(child)
            strategy.record_child(root[0], child, corpus)
        assert strategy.warmed

    def test_select_declines_while_cold(self):
        strategy = _strategy()
        assert strategy.select([b"a", b"b"]) is None

    def test_select_declines_on_empty_seed_list(self):
        strategy = _strategy()
        corpus = [b"p"]
        for i in range(MIN_OBSERVATIONS):
            child = f"c{i}".encode()
            corpus.append(child)
            strategy.record_child(b"p", child, corpus)
        assert strategy.warmed
        assert strategy.select([]) is None


class TestCreditMath:
    def test_credit_matches_marginal_oracle_single_child(self):
        strategy = _strategy(decay=1.0)
        root = [b"A" * 40]
        _fill_root(strategy, root)
        child = b"B" * 20 + b"A" * 20
        corpus = root + [child]
        strategy.record_child(root[0], child, corpus)

        expected = _marginal_oracle(root, child)
        assert strategy._credit(root[0]) == pytest.approx(expected)

    def test_no_credit_when_parent_is_none(self):
        strategy = _strategy(decay=1.0)
        corpus = [b"root"]
        strategy.record_child(None, b"root", corpus)
        # Nothing crediting-worthy happened: no parent, no positive credit
        # for anyone, and the credited counter (which gates `warmed`) must
        # not advance off a rootless admission.
        assert strategy._credited == 0
        assert strategy._credit(b"root") == 0.0

    def test_credit_accumulates_across_multiple_children_undecayed(self):
        strategy = _strategy(decay=1.0)
        root = [b"seed-A" * 5]
        _fill_root(strategy, root)
        corpus = list(root)
        total_expected = 0.0
        for i in range(3):
            pool_before = list(corpus)
            child = f"varied-child-{i}-{'x' * i}".encode()
            corpus.append(child)
            total_expected += _marginal_oracle(pool_before, child)
            strategy.record_child(root[0], child, corpus)

        assert strategy._credit(root[0]) == pytest.approx(total_expected)

    def test_two_parents_credited_independently(self):
        strategy = _strategy(decay=1.0)
        roots = [b"root-one" * 3, b"root-two" * 3]
        _fill_root(strategy, roots)
        corpus = list(roots)

        child_a = b"child-of-one" + b"z" * 10
        corpus.append(child_a)
        expected_a = _marginal_oracle(roots + [], child_a)
        strategy.record_child(roots[0], child_a, corpus)

        pool_before_b = list(corpus)
        child_b = b"child-of-two" + b"q" * 10
        corpus.append(child_b)
        expected_b = _marginal_oracle(pool_before_b, child_b)
        strategy.record_child(roots[1], child_b, corpus)

        assert strategy._credit(roots[0]) == pytest.approx(expected_a)
        assert strategy._credit(roots[1]) == pytest.approx(expected_b)


class TestDecay:
    def test_decay_shrinks_older_credit_relative_to_newer(self):
        # Exercises _credit_update directly (bypassing record_child's pool
        # dynamics, which confound this: a corpus that has grown more
        # diverse in between gives a *later*, same-shaped child a smaller
        # raw marginal purely from diminishing returns, which can swamp
        # the decay effect this test wants to isolate). Same delta for A
        # and B; only the number of decay rounds in between differs.
        strategy = _strategy(decay=0.9)
        strategy._credit_update(b"A", 1.0)
        for _ in range(5):
            strategy._credit_update(b"C", 0.0)  # advances the discount only
        strategy._credit_update(b"B", 1.0)

        assert strategy._credit(b"A") == pytest.approx(0.9**6)
        assert strategy._credit(b"B") == pytest.approx(1.0)
        assert strategy._credit(b"A") < strategy._credit(b"B")

    def test_decay_one_is_a_pure_undiscounted_sum(self):
        strategy_decay = _strategy(decay=0.5)
        strategy_flat = _strategy(decay=1.0)
        root = [b"root" * 8]
        _fill_root(strategy_decay, root)
        _fill_root(strategy_flat, root)
        corpus = list(root)

        for i in range(4):
            child = f"child-{i}-{'y' * i}".encode()
            corpus.append(child)
            strategy_decay.record_child(root[0], child, corpus)
            strategy_flat.record_child(root[0], child, corpus)

        # Same admissions, but the decayed run must end up with strictly
        # less accumulated credit than the undiscounted one once more than
        # one admission has landed (decay < 1 always shrinks history).
        assert strategy_decay._credit(root[0]) < strategy_flat._credit(root[0])

    def test_invalid_decay_rejected(self):
        with pytest.raises(ValueError):
            _strategy(decay=0.0)
        with pytest.raises(ValueError):
            _strategy(decay=1.5)


class TestPoolSync:
    def test_full_rebuild_on_corpus_shrink(self):
        strategy = _strategy(decay=1.0)
        root = [b"root-seed" * 3]
        _fill_root(strategy, root)
        corpus = list(root)
        for i in range(3):
            child = f"child-{i}".encode()
            corpus.append(child)
            strategy.record_child(root[0], child, corpus)

        assert len(strategy._live) == 4  # root + 3 children

        # Corpus shrinks (a prune/minimize event) -- next sync must rebuild
        # from scratch rather than treating this as a fresh append.
        pruned_corpus = [root[0], corpus[1]]
        new_child = b"post-prune-child" * 2
        pruned_corpus.append(new_child)
        strategy.record_child(root[0], new_child, pruned_corpus)

        assert strategy._live == set(pruned_corpus)
        assert strategy._synced_len == len(pruned_corpus)

    def test_pending_child_not_double_folded(self):
        # record_child must fold `child` exactly once into the pool even
        # though it appears in `corpus` at call time.
        strategy = _strategy(decay=1.0)
        root = [b"root-once" * 3]
        _fill_root(strategy, root)
        corpus = root + [b"child-once" * 3]
        strategy.record_child(root[0], corpus[-1], corpus)

        oracle = CumulativeByteEntropy()
        for s in corpus:
            oracle.add(s)
        assert strategy._pool.bits() == pytest.approx(oracle.bits())


class TestSelection:
    def test_selects_seed_with_highest_credit(self):
        strategy = _strategy(decay=1.0)
        roots = [b"low-root" * 3, b"high-root" * 3, b"filler-root" * 3]
        _fill_root(strategy, roots)
        corpus = list(roots)

        # Give roots[1] a large, clearly bigger marginal than roots[0].
        small_child = roots[0][:1]  # near-identical to an existing byte, tiny delta
        corpus.append(small_child)
        strategy.record_child(roots[0], small_child, corpus)

        big_child = bytes(range(256))  # maximal-diversity payload, big delta
        corpus.append(big_child)
        strategy.record_child(roots[1], big_child, corpus)

        # Filler admissions credited to a third seed just to clear the
        # warm-up floor -- parentless admissions don't count (see
        # test_no_credit_when_parent_is_none), so these need a parent too.
        for i in range(MIN_OBSERVATIONS - 2):
            filler = f"filler-{i}".encode()
            corpus.append(filler)
            strategy.record_child(roots[2], filler, corpus)

        assert strategy.warmed
        rng = CapturingRng(index=corpus.index(roots[1]))
        strategy._rng = rng
        chosen = strategy.select(corpus)
        assert chosen == roots[1]
        assert rng.weights[corpus.index(roots[1])] > rng.weights[corpus.index(roots[0])]

    def test_uncredited_seed_gets_min_weight_floor(self):
        strategy = _strategy(decay=1.0)
        root = [b"root-floor" * 3]
        _fill_root(strategy, root)
        corpus = list(root)
        for i in range(MIN_OBSERVATIONS):
            child = f"child-{i}".encode()
            corpus.append(child)
            strategy.record_child(root[0], child, corpus)

        never_credited = b"bystander-seed"
        corpus.append(never_credited)
        rng = CapturingRng(index=0)
        strategy._rng = rng
        strategy.select(corpus)
        idx = corpus.index(never_credited)
        assert rng.weights[idx] == pytest.approx(MIN_WEIGHT)


class TestStats:
    def test_stats_reflect_activity(self):
        strategy = _strategy(decay=1.0)
        root = [b"root-stats" * 3]
        _fill_root(strategy, root)
        corpus = list(root)
        for i in range(MIN_OBSERVATIONS):
            child = f"child-{i}".encode()
            corpus.append(child)
            strategy.record_child(root[0], child, corpus)

        st = strategy.stats()
        assert st["credited"] == MIN_OBSERVATIONS
        assert st["warmed"] is True
        assert st["pooled"] == len(corpus)

    def test_stats_on_untouched_strategy(self):
        strategy = _strategy()
        st = strategy.stats()
        assert st == {
            "credited": 0,
            "selected": 0,
            "pooled": 0,
            "warmed": False,
            "mean_credit": 0.0,
        }


class TestDefaults:
    def test_default_decay_is_the_module_constant(self):
        strategy = _strategy()
        assert strategy._decay == DEFAULT_DECAY


# ── Falsification / adversarial (Hard Rule 23) ──────────────────────────


class TestFalsification:
    """A control that must NOT show the effect under test -- if it does,
    the oracle/comparison itself is broken, not the code (Hard Rule 46)."""

    def test_flat_children_produce_zero_credit(self):
        """Children with a byte distribution identical to the existing pool
        contribute (near-)zero marginal entropy -- the oracle's control."""
        strategy = _strategy(decay=1.0)
        root = [b"\x00" * 64]
        _fill_root(strategy, root)
        corpus = list(root)
        for _i in range(5):
            child = b"\x00" * 64  # exact duplicate distribution
            corpus.append(child)
            strategy.record_child(root[0], child, corpus)

        assert strategy._credit(root[0]) == pytest.approx(0.0, abs=1e-9)

    def test_self_comparison_of_marginal_oracle_is_stable(self):
        """Running the independent oracle twice on identical input must
        agree with itself -- if this fails, the oracle is broken, not the
        strategy (Hard Rule 46)."""
        pool = [b"alpha-bytes" * 4, b"beta-bytes" * 4]
        child = b"gamma-bytes" * 4
        first = _marginal_oracle(pool, child)
        second = _marginal_oracle(pool, child)
        assert first == second


class TestAdversarial:
    def test_record_child_with_empty_bytes_does_not_crash(self):
        strategy = _strategy(decay=1.0)
        corpus = [b"root", b""]
        strategy.record_child(b"root", b"", corpus)
        assert strategy._credit(b"root") == pytest.approx(0.0)

    def test_repeated_identical_child_key_is_idempotent_in_pool_membership(self):
        # A child key equal to an already-live seed (e.g. a re-admitted
        # duplicate after a prune/reload race) must not be folded twice.
        strategy = _strategy(decay=1.0)
        root = [b"dup-root" * 3]
        _fill_root(strategy, root)
        corpus = list(root)
        child = b"dup-child" * 3
        corpus.append(child)
        strategy.record_child(root[0], child, corpus)
        pool_bits_once = strategy._pool.bits()

        # Same corpus handed again (no growth) -- must be a no-op.
        strategy.record_child(root[0], child, corpus)
        assert strategy._pool.bits() == pytest.approx(pool_bits_once)
        assert strategy._synced_len == len(corpus)

    def test_parent_not_in_corpus_still_credited(self):
        # A parent pruned from the corpus between being mutated and its
        # child's admission (a real race in a fuzzer with concurrent
        # minimize) must still receive credit -- credit is keyed by the
        # parent's bytes, not by current corpus membership.
        strategy = _strategy(decay=1.0)
        root = [b"vanished-parent" * 3]
        _fill_root(strategy, root)
        corpus = [b"other-seed" * 3]  # root already pruned out
        child = b"child-of-vanished" * 3
        corpus_with_child = corpus + [child]
        strategy.record_child(root[0], child, corpus_with_child)
        assert strategy._credited == 1
        assert strategy._credit(root[0]) != 0.0


class FakeFuzzer(SimpleNamespace):
    """Minimal stand-in for wiring-level checks that only need attribute
    presence, mirroring the shape sibling entropy-arm tests use."""


def test_scores_aligned_with_input_order():
    strategy = _strategy(decay=1.0)
    root = [b"order-root" * 3]
    _fill_root(strategy, root)
    corpus = list(root)
    child = b"order-child" * 3
    corpus.append(child)
    strategy.record_child(root[0], child, corpus)

    seeds = [child, root[0]]
    scores = strategy.scores(seeds)
    assert scores[1] == pytest.approx(strategy._credit(root[0]))
    assert scores[0] == pytest.approx(0.0)


def test_regression_negative_credit_never_reaches_weighted_choice():
    """A low-diversity child lowers pooled entropy, so its parent's credit
    goes negative. Negative weights break RandPool.weighted_choice: an
    all-negative set raises IndexError, a mixed set makes the cumulative
    sum non-monotone. Weights must be floored at MIN_WEIGHT."""
    from fuzzer_tool.core.rand_pool import RandPool

    diverse = bytes(range(256))
    flat = b"\x00" * 256
    corpus = [diverse]
    strategy = _strategy(rng=RandPool(seed=1), min_observations=1)
    _fill_root(strategy, corpus)
    corpus.append(flat)
    assert _marginal_oracle([diverse], flat) < 0
    strategy.record_child(diverse, flat, corpus)
    assert strategy.scores([diverse])[0] < 0

    assert strategy.select([diverse]) == diverse

    capture = CapturingRng()
    strategy._rng = capture
    strategy.select([diverse, flat])
    assert min(capture.weights) >= MIN_WEIGHT
