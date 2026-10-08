"""Iterator-scripted fake RNG for Hard Rule 39 determinized tests.

Lineal descendant of the inline ``_FakeRand``/``_FakeRng`` classes from
commits f00927a/f5d599a. Serves both injection seams in the codebase —
operators taking a ``random.Random``-style ``rng=`` argument and handlers
reading the ``RandPool``-API ``f._rng`` — because both are duck-typed.

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
        randoms: Floats returned one per ``random()`` call, or ``count`` at
            a time per ``random_list(count)`` call.
        choice_idxs: Indices consumed one per ``choice()`` call
            (index-based like ``random.choice``/``RandPool.choice``).
        counts: Values returned (wrapped in a list) per single-value
            ``randint_list(a, b, 1)`` draw — e.g. havoc's mutation-count roll.
        batch_value: Pinned value filling every multi-value ``randint_list``
            draw, so batch consumers (havoc sub-mutations) stay fully
            scripted instead of delegating to a real RNG.
        randbytes: Blobs returned one per ``randbytes()`` call. The
            requested length is ignored like every other bound here, so a
            blob of the wrong size is a scripting error the consumer will
            surface rather than something this class papers over.
        gauss_lists: Lists returned one per ``gauss_list()`` call (mu,
            sigma and count ignored).
        beta_arrays: Arrays returned one per ``betavariate_array()`` call
            (parameters ignored).
        binomial_arrays: Arrays returned one per ``binomial_array()`` call
            (counts and p ignored).
        gamma_arrays: Arrays returned one per ``gammavariate_array()`` call
            (shapes and rates ignored).
    """

    def __init__(
        self,
        randints=(),
        randoms=(),
        choice_idxs=(),
        counts=(),
        batch_value=0,
        randbytes=(),
        gauss_lists=(),
        beta_arrays=(),
        binomial_arrays=(),
        gamma_arrays=(),
    ):
        self._randints = iter(randints)
        self._randoms = iter(randoms)
        self._choice_idxs = iter(choice_idxs)
        self._counts = iter(counts)
        self._batch_value = batch_value
        self._randbytes = iter(randbytes)
        self._gauss_lists = iter(gauss_lists)
        self._beta_arrays = iter(beta_arrays)
        self._binomial_arrays = iter(binomial_arrays)
        self._gamma_arrays = iter(gamma_arrays)

    def randint(self, _a, _b):
        return next(self._randints)

    def random(self):
        return next(self._randoms)

    def random_list(self, count):
        return [next(self._randoms) for _ in range(count)]

    def choice(self, seq):
        return seq[next(self._choice_idxs)]

    def randint_list(self, _a, _b, count):
        if count == 1:
            return [next(self._counts)]
        return [self._batch_value] * count

    def randbytes(self, _n):
        return next(self._randbytes)

    def gauss_list(self, _mu, _sigma, _count):
        return next(self._gauss_lists)

    def betavariate_array(self, _alphas, _betas):
        return next(self._beta_arrays)

    def binomial_array(self, _counts, _p):
        return next(self._binomial_arrays)

    def gammavariate_array(self, _alphas, _betas):
        return next(self._gamma_arrays)

    def shuffle(self, seq):
        seq.reverse()
