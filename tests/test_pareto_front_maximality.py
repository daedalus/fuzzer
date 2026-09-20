"""``_pareto_front`` must return the maximal points, not a rolling-max sweep.

The previous ``dims <= 3`` path tested candidates against the componentwise
maximum of the accepted set rather than against the 2-D staircase, so it
rejected points that nothing dominated. A test asserting only *soundness*
(everything returned is non-dominated) passes against that implementation and
pins nothing -- the sweep never returns a dominated point, it returns too few.
Completeness is the property that falsifies it, so every property test below
asserts both halves.
"""

import itertools
import random

from fuzzer_tool.services.seed_picker import SeedPicker


def _reference_front(scores, window=100):
    """Maximal elements under strict Pareto dominance, brute force."""
    idx = list(range(max(0, len(scores) - window), len(scores)))
    dims = range(len(scores[idx[0]]))
    out = set()
    for i in idx:
        if not any(
            j != i
            and all(scores[j][d] >= scores[i][d] for d in dims)
            and any(scores[j][d] > scores[i][d] for d in dims)
            for j in idx
        ):
            out.add(i)
    return out


def _legacy_sweep(scores, window=100):
    """Verbatim copy of the removed path, as a falsification oracle."""
    idx = list(range(max(0, len(scores) - window), len(scores)))
    idx.sort(key=lambda i: (-scores[i][0], -scores[i][1], -scores[i][2]))
    result, max_b, max_c = [], float("-inf"), float("-inf")
    for i in idx:
        _a, b, c = scores[i][0], scores[i][1], scores[i][2]
        if b > max_b or c > max_c:
            result.append(i)
            max_b = max(max_b, b)
            max_c = max(max_c, c)
    return set(result)


class TestWitness:
    def test_three_point_witness(self):
        # (0.9, 0.5, 0.5) loses on c to the first point and on b to the
        # second, so neither dominates it. The sweep returned {0, 1}.
        scores = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.9, 0.5, 0.5)]
        assert SeedPicker._pareto_front(scores) == {0, 1, 2}

    def test_witness_falsifies_the_removed_path(self):
        scores = [(1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.9, 0.5, 0.5)]
        assert _legacy_sweep(scores) == {0, 1}


class TestMaximality:
    def test_soundness_and_completeness_continuous(self):
        rng = random.Random(1234)
        for _ in range(60):
            scores = [tuple(rng.random() for _ in range(3)) for _ in range(40)]
            assert SeedPicker._pareto_front(scores) == _reference_front(scores)

    def test_soundness_and_completeness_quantised(self):
        # Coarse scores are the realistic regime and produce ties, which is
        # where the front's tie rule is exercised. Compare against the
        # reference restricted to the documented non-strict semantics: one
        # representative per group of identical points.
        rng = random.Random(99)
        for _ in range(60):
            scores = [tuple(round(rng.random(), 1) for _ in range(3)) for _ in range(40)]
            front = SeedPicker._pareto_front(scores)
            reference = _reference_front(scores)
            # Every returned index is genuinely maximal.
            assert front <= reference
            # Every maximal point is represented, by itself or by a twin.
            kept = {scores[i] for i in front}
            assert {scores[i] for i in reference} == kept

    def test_four_dimensional_path_unchanged(self):
        rng = random.Random(7)
        for _ in range(40):
            scores = [tuple(rng.random() for _ in range(4)) for _ in range(30)]
            assert SeedPicker._pareto_front(scores) == _reference_front(scores)

    def test_exhaustive_small_grids(self):
        # Every 4-point configuration over a 3-value grid in 3 dimensions.
        grid = (0.0, 0.5, 1.0)
        points = list(itertools.product(grid, repeat=3))
        rng = random.Random(3)
        for _ in range(400):
            scores = [rng.choice(points) for _ in range(4)]
            front = SeedPicker._pareto_front(scores)
            reference = _reference_front(scores)
            assert front <= reference
            assert {scores[i] for i in front} == {scores[i] for i in reference}


class TestAgreementControl:
    def test_three_and_four_dims_agree_when_the_fourth_is_constant(self):
        # Hard Rule 46: run the two dimensionalities against each other on
        # data where they must agree. Before this change they did not -- the
        # 3-D path took a different branch and returned a smaller set.
        rng = random.Random(555)
        for _ in range(60):
            three = [tuple(round(rng.random(), 2) for _ in range(3)) for _ in range(40)]
            four = [s + (0.5,) for s in three]
            assert SeedPicker._pareto_front(three) == SeedPicker._pareto_front(four)


class TestContract:
    def test_empty_input(self):
        assert SeedPicker._pareto_front([]) == set()

    def test_window_limits_the_candidate_range(self):
        scores = [(1.0, 1.0, 1.0)] + [(0.0, 0.0, 0.0)] * 120
        # The dominating point at index 0 is outside a 100-wide window, so it
        # neither appears in nor suppresses the front.
        front = SeedPicker._pareto_front(scores, window=100)
        assert 0 not in front
        assert all(i >= len(scores) - 100 for i in front)

    def test_identical_points_yield_one_representative(self):
        # Documented non-strict tie rule: equal points dominate each other and
        # the lowest index survives. Pinned so that a later switch to strict
        # dominance is a deliberate change with a failing test attached.
        scores = [(0.5, 0.5, 0.5)] * 10
        assert SeedPicker._pareto_front(scores) == {0}

    def test_saturated_scores_collapse_to_one_seed(self):
        # Under _saturation_gated, sub and spa are constant 1.0 and only the
        # burst factor varies. Recorded rather than endorsed -- see the
        # docstring; the single survivor takes x2.0 and the rest x0.5.
        scores = [(1.0, bf, 1.0) for bf in (0.2, 0.9, 0.4, 0.9, 0.1)]
        front = SeedPicker._pareto_front(scores)
        assert len(front) == 1
        assert scores[next(iter(front))][1] == 0.9
