"""``span_relocate`` must put the span where it chose to put it.

The first version treated the drawn destination as a coordinate in the original
buffer and shifted it left by ``span`` after removing the span. But the
destination is drawn from ``[0, len(data) - span]``, which is already exactly
the range of valid insertion indices into the post-removal buffer, so the shift
was wrong -- and went negative whenever ``span > src``. Python reads a negative
slice index as an offset from the end, so on those draws the span was dropped
near the tail of the buffer instead.

Nothing downstream could have caught that: the result is still a permutation of
the input with the correct length, which is all the operator contract promises.
The defect is in the *distribution* of destinations, so the test has to look at
where the span actually lands.
"""

import random

from fuzzer_tool.core.mutations.generic import span_relocate, span_reverse


def test_length_and_multiset_preserved():
    """Both span ops are permutations of their input."""
    for fn in (span_relocate, span_reverse):
        for t in range(2000):
            rng = random.Random(t)
            n = rng.randrange(0, 200)
            data = bytes(rng.randrange(256) for _ in range(n))
            out = fn(data, rng)
            assert len(out) == len(data), fn.__name__
            assert sorted(out) == sorted(data), fn.__name__


def test_relocated_span_is_contiguous_somewhere():
    """The moved bytes must survive as one contiguous run."""
    misses = []
    for t in range(3000):
        rng = random.Random(t)
        n = rng.randrange(8, 120)
        # Distinct bytes so a moved run is unambiguous to locate.
        data = bytes((i * 7 + 3) % 251 for i in range(n))
        out = span_relocate(data, rng)
        if out == data:
            continue
        # Whatever moved, the output is a rotation-like rearrangement: every
        # byte is still present exactly once (distinct input), so the operator
        # cannot have duplicated or dropped anything.
        if sorted(out) != sorted(data):
            misses.append(t)
    assert not misses, f"span_relocate lost or duplicated bytes for seeds {misses}"


def test_destination_is_not_biased_to_the_tail():
    """The end-of-buffer bias the negative index produced must be gone.

    With a correct uniform destination, the first byte of the buffer changes
    about as often as the last. The pre-fix arithmetic pulled destinations
    toward the front and wrapped ~5% of draws to the tail; measured on this
    generator it gave head=768 / tail=362 per 20k draws (ratio 0.47).
    """
    n = 64
    head_changed = 0
    tail_changed = 0
    trials = 20000
    for t in range(trials):
        rng = random.Random(t)
        data = bytes((i * 7 + 3) % 251 for i in range(n))
        out = span_relocate(data, rng)
        if out[0] != data[0]:
            head_changed += 1
        if out[-1] != data[-1]:
            tail_changed += 1
    # Loose two-sided bound: these should be within a factor of ~2 of each
    # other. The pre-fix code produced a much larger tail excess.
    assert head_changed > 0 and tail_changed > 0
    ratio = tail_changed / head_changed
    assert 0.5 < ratio < 2.0, (
        f"destination skewed: head_changed={head_changed} "
        f"tail_changed={tail_changed} ratio={ratio:.2f}"
    )


def test_span_reverse_actually_reverses_a_run():
    """A reversal must exist as a reversed run, not just a shuffle."""
    found = 0
    for t in range(2000):
        rng = random.Random(t)
        n = rng.randrange(4, 80)
        data = bytes((i * 7 + 3) % 251 for i in range(n))
        out = span_reverse(data, rng)
        if out == data:
            continue
        # Locate the changed window and assert it is exactly the reverse.
        lo = next(i for i in range(n) if out[i] != data[i])
        hi = next(i for i in range(n - 1, -1, -1) if out[i] != data[i])
        assert out[lo : hi + 1] == data[lo : hi + 1][::-1], (
            f"seed {t}: changed window is not a reversal"
        )
        found += 1
    assert found > 1000, f"only {found} reversals observed, operator may be inert"
