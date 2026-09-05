"""Regression: JS-vs-aggregate dropped half its sum and measured the wrong thing.

``EdgeTracker._js_divergence_vs_aggregate`` iterated only the seed's own edges
while comparing against the *unconditioned* corpus aggregate, on the stated
grounds that "edges where the seed is zero contribute 0 to KL(P || M)". True of
that term, false of the other one: JS = 0.5*KL(P||M) + 0.5*KL(Q||M), and where
p = 0 the mixture is m = q/2, so the Q term contributes q*ln2. The omitted mass
was exactly 0.5 * ln2 * (1 - Q_seed) -- measured 0.204 as-coded against a true
JS of 0.506 on a 40-seed, 2000-edge corpus, the two differing by precisely the
predicted amount.

The obvious repair -- restore the missing term -- was measured and rejected. The
omitted quantity is a monotone function of how much of the corpus the seed does
not cover, so putting it back turns this into a coverage-breadth proxy. Across a
60-seed corpus with mixed edge counts and a third of the seeds carrying hot
loops:

    metric        corr w/ edge count   corr w/ loopiness
    as-coded                  -0.965              -0.129
    full JS                   -0.988              -0.161
    conditional JS            -0.026              +0.789

Both the original and the full-JS repair rank a narrow seed with a flat profile
above a seed with three edges hit 500 times. compute_hitcount_diversity_weight
documents this weight as being for "the same edges as the corpus but with a very
different frequency profile", and breadth is already carried by
compute_subsumption_weight, so the fix conditions the aggregate on the seed's
support: both arguments then live on the same set, the result is a proper
divergence in [0, ln 2], and it measures profile alone.
"""

import math
import random

import pytest

from fuzzer_tool.core.edge_tracker import EdgeTracker, _js_divergence


def _corpus(seed=11, n_seeds=40, n_edges=2000, per_seed=200):
    rng = random.Random(seed)
    et = EdgeTracker()
    for i in range(n_seeds):
        edges = rng.sample(range(n_edges), per_seed)
        hc = {e: rng.randint(1, 20) for e in edges}
        et.seed_hit_counts[f"s{i}"] = hc
        for e, c in hc.items():
            et._aggregate_totals[e] = et._aggregate_totals.get(e, 0) + c
    et._aggregate_total_count = sum(et._aggregate_totals.values())
    et._aggregate_cache = None
    return et, rng


def _dist(hc):
    total = sum(hc.values())
    return {e: c / total for e, c in hc.items()}


def _weight(et, hc):
    """The scaling compute_hitcount_diversity_weight applies, without needing a
    registered seed key."""
    js = et._js_divergence_vs_aggregate(_dist(hc))
    return 0.5 + 1.5 * min(js / math.log(2), 1.0)


def _conditional_reference(et, seed_dist):
    """Independent construction of the intended quantity.

    Renormalises the aggregate onto the seed's support and defers to the
    module's general _js_divergence, which is the known-good implementation.
    Checking the specialisation against the general routine is the point: the
    old code was a hand-rolled partial sum that no oracle compared to anything.
    """
    mass = sum(et._aggregate_totals.get(e, 0) for e in seed_dist)
    if mass <= 0:
        return 0.0
    q = {e: et._aggregate_totals.get(e, 0) / mass for e in seed_dist}
    return _js_divergence(seed_dist, q)


class TestMatchesTheGeneralImplementation:
    def test_agrees_with_js_divergence_on_the_conditioned_pair(self):
        et, rng = _corpus()
        for i in range(10):
            sd = _dist(et.seed_hit_counts[f"s{i}"])
            assert et._js_divergence_vs_aggregate(sd) == pytest.approx(
                _conditional_reference(et, sd), abs=1e-12
            )

    def test_is_a_bounded_divergence(self):
        et, rng = _corpus()
        for i in range(40):
            sd = _dist(et.seed_hit_counts[f"s{i}"])
            js = et._js_divergence_vs_aggregate(sd)
            assert 0.0 <= js <= math.log(2) + 1e-12

    def test_identical_profile_is_zero(self):
        """A seed whose profile equals the corpus on its own edges."""
        et = EdgeTracker()
        et._aggregate_totals = {1: 40, 2: 20, 3: 10}
        et._aggregate_total_count = 70
        sd = {1: 40 / 70, 2: 20 / 70, 3: 10 / 70}
        assert et._js_divergence_vs_aggregate(sd) == pytest.approx(0.0, abs=1e-12)

    def test_partial_support_still_scores_zero_when_shape_matches(self):
        """The old code could not do this: the corpus edges the seed misses
        contributed the bulk of its value, so a shape-identical seed on a
        subset scored high."""
        et = EdgeTracker()
        et._aggregate_totals = {1: 40, 2: 20, 3: 10, 4: 900, 5: 900}
        et._aggregate_total_count = 1870
        sd = {1: 40 / 70, 2: 20 / 70, 3: 10 / 70}
        assert et._js_divergence_vs_aggregate(sd) == pytest.approx(0.0, abs=1e-12)


class TestMeasuresProfileNotBreadth:
    @staticmethod
    def _mixed_corpus(seed=5):
        rng = random.Random(seed)
        et = EdgeTracker()
        meta = {}
        for i in range(60):
            n_edges = rng.choice([30, 80, 200, 600, 1200])
            edges = rng.sample(range(3000), n_edges)
            loopy = rng.random() < 0.3
            hc = {
                e: (rng.choice([200, 800]) if (loopy and j < 3) else rng.randint(1, 15))
                for j, e in enumerate(edges)
            }
            et.seed_hit_counts[f"s{i}"] = hc
            meta[f"s{i}"] = (n_edges, loopy)
            for e, c in hc.items():
                et._aggregate_totals[e] = et._aggregate_totals.get(e, 0) + c
        et._aggregate_total_count = sum(et._aggregate_totals.values())
        et._aggregate_cache = None
        return et, meta

    @staticmethod
    def _corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
        den = math.sqrt(
            sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)
        )
        return num / den if den else 0.0

    def test_weight_is_not_a_breadth_proxy(self):
        et, meta = self._mixed_corpus()
        widths, loops, values = [], [], []
        for key, (n_edges, loopy) in meta.items():
            values.append(et._js_divergence_vs_aggregate(_dist(et.seed_hit_counts[key])))
            widths.append(float(n_edges))
            loops.append(1.0 if loopy else 0.0)

        breadth = abs(self._corr(widths, values))
        profile = self._corr(loops, values)
        assert breadth < 0.35, f"still tracks edge count (r={breadth:.3f})"
        assert profile > 0.5, f"does not track hit-count profile (r={profile:.3f})"
        assert profile > breadth

    def test_hot_loops_outrank_a_narrow_flat_seed(self):
        """The specific inversion: both the old code and the full-JS repair put
        a small flat seed above a seed with three edges hit 500 times."""
        et, rng = _corpus()
        pool = sorted(et._aggregate_totals)
        base = rng.sample(pool, 200)

        flat_narrow = {e: 5 for e in rng.sample(pool, 40)}
        loopy = {e: (500 if i < 3 else 5) for i, e in enumerate(base)}

        assert _weight(et, loopy) > _weight(et, flat_narrow)

    def test_broad_and_narrow_flat_seeds_score_alike(self):
        et, rng = _corpus()
        pool = sorted(et._aggregate_totals)
        narrow = et._js_divergence_vs_aggregate(_dist({e: 5 for e in rng.sample(pool, 40)}))
        broad = et._js_divergence_vs_aggregate(_dist({e: 5 for e in rng.sample(pool, 1200)}))
        # Both are flat: the metric should barely separate them.
        assert abs(narrow - broad) < 0.05


class TestWeightWiring:
    def test_weight_stays_in_band(self):
        et, _ = _corpus()
        for i in range(40):
            w = et.compute_hitcount_diversity_weight(f"s{i}")
            assert 0.5 <= w <= 2.0

    def test_missing_seed_is_neutral(self):
        et, _ = _corpus()
        assert et.compute_hitcount_diversity_weight("nope") == 1.0

    def test_band_top_is_reachable(self):
        """A profile extreme enough must be able to climb well above neutral,
        or the weight is decorative."""
        et, rng = _corpus()
        pool = sorted(et._aggregate_totals)
        base = rng.sample(pool, 200)
        extreme = {e: (5000 if i < 1 else 5) for i, e in enumerate(base)}
        assert _weight(et, extreme) > 1.2
