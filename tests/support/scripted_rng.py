"""Iterator-scripted fake RNG for Hard Rule 39 determinized tests.

Lineal descendant of the inline ``_FakeRand``/``_FakeRng`` classes from
commits f00927a/f5d599a. Serves both injection seams in the codebase —
operators taking a ``random.Random``-style ``rng=`` argument and handlers
reading the ``RandPool``-API ``f._rand_pool`` — because both are duck-typed.

Each method consumes the next value of its own iterator; bounds arguments
are accepted but ignored, matching the reference fakes. Exhausting an
iterator raises StopIteration, which doubles as the tripwire against code
that draws more or fewer times than scripted, draws the wrong kind of value,
or falls back to the global ``random`` stream. Nothing here touches
``random`` or numpy.
"""


class ScriptedRng:
    """Fully scripted stand-in for ``random.Random`` / ``RandPool``.

    Args:
        randints: Values returned one per scalar ``randint()`` call.
        randoms: Floats returned one per ``random()`` call.
        choice_idxs: Indices consumed one per ``choice()`` call
            (index-based like ``random.choice``/``RandPool.choice``).
        counts: Values returned (wrapped in a list) per single-value
            ``randint_list(a, b, 1)`` draw — e.g. havoc's mutation-count roll.
        batch_value: Pinned value filling every multi-value ``randint_list``
            draw, so batch consumers (havoc sub-mutations) stay fully
            scripted instead of delegating to a real RNG.
    """

    def __init__(self, randints=(), randoms=(), choice_idxs=(), counts=(), batch_value=0):
        self._randints = iter(randints)
        self._randoms = iter(randoms)
        self._choice_idxs = iter(choice_idxs)
        self._counts = iter(counts)
        self._batch_value = batch_value

    def randint(self, _a, _b):
        return next(self._randints)

    def random(self):
        return next(self._randoms)

    def choice(self, seq):
        return seq[next(self._choice_idxs)]

    def randint_list(self, _a, _b, count):
        if count == 1:
            return [next(self._counts)]
        return [self._batch_value] * count

    def shuffle(self, seq):
        seq.reverse()
