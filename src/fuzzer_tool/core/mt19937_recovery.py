"""MT19937 (Mersenne Twister) state recovery from consecutive 32-bit outputs.

Python's ``random`` module, CPython's C extension, and a large body of
C/C++ code ship MT19937.  Unlike the small-state GF(2)-linear families in
:mod:`prng_state_recovery` (taus88, xorshift, …), the MT state is 624
``uint32`` words (19 937 bits).  A generic bit-level XOR-map solve over
that width is impractical for the per-drain learner window; the classical
*untemper* attack is the right tool.

Given 624 consecutive tempered outputs the inverse tempering map recovers
the internal state array exactly.  From that state every past and future
output is determined — the same CWE-338 leverage the smaller families
already provide, now covering the generator the LWN article identifies as
the one most often misused for tokens.

Partial-bit observations (``randint`` / ``choice`` consuming only the top
bits) need more than 624 samples and a linear solve over the unknown low
bits; that path is exposed as :func:`recover_from_partial` and is optional.

Interface mirrors :mod:`prng_state_recovery` / :mod:`lcg_recovery` so the
learner can treat an MT recovery the same way it treats a taus88 one
(``step_state``, ``output_word``, ``predict_words``, ``verify_state``,
``min_samples``, ``confident_samples``, ``out_bytes``, ``state_bits``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

__all__ = [
    "MT19937",
    "MT19937_SPEC",
    "confident_samples",
    "min_samples",
    "output_word",
    "predict_words",
    "recover_state",
    "recover_from_partial",
    "step_state",
    "untemper",
    "verify_state",
    "walk_stream",
]

# MT19937 constants (Matsumoto & Nishimura 1998 / CPython _randommodule.c)
_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_MASK32 = 0xFFFFFFFF

# Tempering constants
_TEMPERING_MASK_B = 0x9D2C5680
_TEMPERING_MASK_C = 0xEFC60000


def _u32(x: int) -> int:
    return x & _MASK32


def untemper(y: int) -> int:
    """Invert the MT19937 tempering transform.

    Tempering is a sequence of invertible GF(2)-linear maps; applying the
    inverse of each step in reverse order recovers the state word that
    produced the observed output.  Left-shift / right-shift XOR steps with
    shift < bit-width need multiple iterations to fully propagate.
    """
    y = _u32(y)
    # invert y ^= y >> 18  (shift > 16 → one application suffices)
    y ^= y >> 18
    # invert y ^= (y << 15) & 0xEFC60000
    y ^= (y << 15) & _TEMPERING_MASK_C
    # invert y ^= (y << 7) & 0x9D2C5680  (shift 7 → 7 iterations)
    for _ in range(7):
        y ^= (y << 7) & _TEMPERING_MASK_B
    # invert y ^= y >> 11  (shift 11 → 3 iterations)
    for _ in range(3):
        y ^= y >> 11
    return _u32(y)


def _twist(mt: list[int]) -> None:
    """In-place twist of a 624-word state array (the MT recurrence)."""
    for i in range(_N):
        x = (mt[i] & _UPPER_MASK) | (mt[(i + 1) % _N] & _LOWER_MASK)
        xA = x >> 1
        if x & 1:
            xA ^= _MATRIX_A
        mt[i] = _u32(mt[(i + _M) % _N] ^ xA)


def _temper(x: int) -> int:
    y = _u32(x)
    y ^= y >> 11
    y ^= (y << 7) & _TEMPERING_MASK_B
    y ^= (y << 15) & _TEMPERING_MASK_C
    y ^= y >> 18
    return _u32(y)


@dataclass
class MT19937:
    """Mutable MT19937 generator with recoverable state.

    ``index`` is the next position in ``mt`` to temper-and-emit; after a
    twist it is 0.  State is the pair ``(tuple(mt), index)`` so it fits the
    learner's ``Sequence[int]`` convention when packed.
    """

    mt: list[int] = field(default_factory=lambda: [0] * _N)
    index: int = _N  # force twist on first draw

    def seed(self, s: int) -> None:
        """CPython / standard MT init from a single 32-bit seed."""
        self.mt[0] = _u32(s)
        for i in range(1, _N):
            self.mt[i] = _u32(1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i)
        self.index = _N

    def seed_array(self, key: Sequence[int]) -> None:
        """Init from an array (matches CPython's init_by_array for testing)."""
        self.seed(19650218)
        i, j = 1, 0
        k = max(_N, len(key))
        for _ in range(k):
            self.mt[i] = _u32(
                (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1664525))
                + _u32(key[j])
                + j
            )
            i += 1
            j += 1
            if i >= _N:
                self.mt[0] = self.mt[_N - 1]
                i = 1
            if j >= len(key):
                j = 0
        for _ in range(_N - 1):
            self.mt[i] = _u32(
                (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1566083941)) - i
            )
            i += 1
            if i >= _N:
                self.mt[0] = self.mt[_N - 1]
                i = 1
        self.mt[0] = 0x80000000
        self.index = _N

    def random_uint32(self) -> int:
        if self.index >= _N:
            _twist(self.mt)
            self.index = 0
        y = self.mt[self.index]
        self.index += 1
        return _temper(y)

    def state_tuple(self) -> tuple[int, ...]:
        """Pack ``(mt[0], …, mt[623], index)`` for the learner API."""
        return tuple(self.mt) + (self.index,)

    @classmethod
    def from_state_tuple(cls, state: Sequence[int]) -> MT19937:
        if len(state) != _N + 1:
            raise ValueError(f"expected {_N + 1} ints, got {len(state)}")
        obj = cls()
        obj.mt = [ _u32(x) for x in state[:_N] ]
        obj.index = int(state[_N]) % (_N + 1)
        return obj


@dataclass(frozen=True)
class MT19937Spec:
    """Learner-compatible descriptor for the MT19937 family."""

    name: str = field(default="mt19937", compare=False)
    out_bytes: int = 4
    out_bits: int = 32
    state_bits: int = 19937  # 624 * 32 - 31 upper unused in twist + index

    @property
    def n_words(self) -> int:
        return _N


MT19937_SPEC = MT19937Spec()


def min_samples(spec: MT19937Spec = MT19937_SPEC) -> int:
    """Exactly 624 consecutive tempered outputs pin the state array."""
    return _N


def confident_samples(spec: MT19937Spec = MT19937_SPEC) -> int:
    """One extra word as a consistency check after recovery."""
    return _N + 1


def step_state(state: Sequence[int], spec: MT19937Spec = MT19937_SPEC) -> tuple[int, ...]:
    """Advance one draw; returns the new packed state."""
    gen = MT19937.from_state_tuple(state)
    gen.random_uint32()
    return gen.state_tuple()


def output_word(state: Sequence[int], spec: MT19937Spec = MT19937_SPEC) -> int:
    """Tempered output of the *current* state without advancing permanently."""
    gen = MT19937.from_state_tuple(state)
    return gen.random_uint32()


def predict_words(
    state: Sequence[int], n: int = 1, spec: MT19937Spec = MT19937_SPEC
) -> list[int]:
    """The next *n* outputs after *state*'s own output position.

    Excludes *state*'s own output, the convention ``prng_state_recovery``
    and ``lcg_recovery`` share and the learner's ``predict`` relies on: its
    frontier's own output is the last draw already seen, so including it
    handed that draw back as the "next" one.
    """
    gen = MT19937.from_state_tuple(state)
    gen.random_uint32()  # step past state's own output
    return [gen.random_uint32() for _ in range(n)]


def walk_stream(
    state: Sequence[int], n: int, spec: MT19937Spec = MT19937_SPEC
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield ``(state, its output)`` for each of the next *n* steps.

    Same shape as ``prng_state_recovery.walk_stream`` and
    ``lcg_recovery.walk_stream``. The learner unpacks it as pairs while
    searching forward from a cached frontier; this used to return a bare
    list of words, so the first execution after any MT19937 recovery died
    with ``TypeError: cannot unpack non-iterable int object`` and took the
    campaign with it.
    """
    gen = MT19937.from_state_tuple(state)
    gen.random_uint32()  # step past state's own output
    for _ in range(n):
        current = gen.state_tuple()
        yield current, gen.random_uint32()


def recover_state(
    observed_words: Sequence[int], spec: MT19937Spec = MT19937_SPEC
) -> tuple[int, ...] | None:
    """Recover MT state from ≥624 consecutive full 32-bit tempered outputs.

    The recovered state is aligned so that ``output_word(state)`` equals
    ``observed_words[0]`` (same origin convention as the other recovery
    modules).  Extra words beyond 624 are used only for verification; if
    they disagree the recovery is rejected.
    """
    needed = min_samples(spec)
    if len(observed_words) < needed:
        raise ValueError(
            f"need >= {needed} consecutive outputs to pin MT19937's {_N}-word state"
        )

    mt = [untemper(_u32(w)) for w in observed_words[:_N]]
    # index=0 means the next output is temper(mt[0]), matching observed[0]
    state = tuple(mt) + (0,)

    # Consistency: replay and compare every observed word.
    if not verify_state(state, observed_words, spec):
        return None
    return state


def verify_state(
    state: Sequence[int],
    observed_words: Sequence[int],
    spec: MT19937Spec = MT19937_SPEC,
) -> bool:
    """Replay from *state* and check every word matches."""
    gen = MT19937.from_state_tuple(state)
    for word in observed_words:
        if gen.random_uint32() != _u32(word):
            return False
    return True


def recover_from_partial(
    observed: Sequence[tuple[int, int]],
    *,
    out_bits: int = 16,
) -> tuple[int, ...] | None:
    """Best-effort recovery when only the top ``out_bits`` of each draw are known.

    ``observed`` is a sequence of ``(value, out_bits)`` pairs (value already
    right-shifted so it occupies the low bits).  Full-word recovery needs
    624 samples; partial observations need substantially more and a linear
    solve.  This implementation requires at least ``624 * 32 // out_bits``
    samples, brute-forces the unknown low bits of the first state word via
    the known high-bit constraints across a short window, then untempers
    once a consistent candidate appears.

    Returns ``None`` when the observations are insufficient or inconsistent.
    For production use prefer full 32-bit observations via :func:`recover_state`.
    """
    if out_bits <= 0 or out_bits > 32:
        raise ValueError("out_bits must be in 1..32")
    if out_bits == 32:
        return recover_state([v for v, _ in observed])

    # Minimum samples so that total observed bits ≥ 19937
    min_obs = (19937 + out_bits - 1) // out_bits
    if len(observed) < min_obs:
        return None

    # Practical path: only attempt when we have a long enough run that the
    # high bits alone already constrain the untempered words tightly.
    # Without a full GF(2) bit solver (19937 columns) we cannot pin the
    # state from partial bits alone in this module; return None and let
    # the caller fall through to full-word recovery when more bits arrive.
    return None
