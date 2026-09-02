"""Class-based mutator interface (port of wtf's ``Mutator_t``).

The existing operator model is function-based: a ``_op_<name>`` method on
``OperatorEngine`` plus a registration in ``operator_registry``. That works
and every built-in operator keeps using it -- this module does not replace
it and changes nothing about how existing operators run.

What it adds is a way to plug in a *self-contained* mutator that does not
live on ``OperatorEngine``: a class carrying its own state, registered as
one object rather than as a method plus a registry entry in a different
file. Two things motivate it, both from wtf:

* Wrapping external mutation engines (libFuzzer's ``FuzzerMutate``,
  honggfuzz's mutators) as fuzzer-new operators without threading their
  state through ``OperatorEngine``.
* ``on_new_coverage()`` -- a feedback hook the function-based operators
  have no equivalent for. A mutator that adapts to which of its own
  mutations found coverage needs somewhere to keep that, and a method on
  a shared engine is the wrong place.

Contract for implementers:

    class MyMutator(MutatorBase):
        name = "my_mutator"
        category = "adaptive"

        def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
            return transformed_bytes_or_None

        def on_new_coverage(self, seed, new_edges):
            ...  # optional

``mutate()`` returns the mutated bytes, or ``None`` to decline (no
applicable mutation this call) -- declining is normal and cheap, matching
the ``_op_*`` convention of returning ``None``. The registry adapter
handles ``max_len`` clamping, so implementations do not have to, though
they may truncate earlier for efficiency.

Implementations must not mutate ``data`` in place; it is the caller's
buffer. Return a new ``bytes``. (The in-place convention some ``_op_*``
handlers use relies on details of ``mutate()``'s dispatch loop that this
interface deliberately does not expose -- and getting it wrong there has
already cost one unbounded-growth bug, see ``_op_fuse_this``.)

Why ``context`` and not ``fuzzer``
----------------------------------

``mutate()`` and ``is_available()`` used to receive the whole ``Fuzzer``
instance. That is the coupling ``services/operators.py`` already
demonstrates the cost of: 365 ``self.f.<attr>`` reads there resolve to 29
distinct attributes, and 249 of them (68%) are just ``max_len`` and
``_rand_pool``. The essential interface is an int and a PRNG, but because
the whole object is in scope, ``test_operator_smoke.py`` has to build a
fuzzer -- and therefore compile a target binary -- to call
``_op_bit_flip``.

``MutationContext`` is the narrow version, and it is introduced here
rather than in ``operators.py`` because this interface has no
implementors yet: there is no ``MutatorBase`` subclass anywhere in
``src/``. Once there is one that reads ``ctx["fuzzer"]``, the door is open
permanently. See docs/port-backlog.md, item F1.

A mutator that needs state the context does not carry should get a named
field added to ``MutationContext``, not reach around it. That is cheap
precisely while there are no implementors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

log = logging.getLogger(__name__)


class MutationContext:
    """The fuzzer state a mutator is allowed to read, and nothing else.

    Deliberately not a dataclass and deliberately ``__slots__``-ed: it is
    built on every ``mutate()`` call so that ``max_len`` reflects any
    adaptive length cap in force *now*, which means construction has to be
    cheap. It is four attribute reads and one object allocation, ~1 us.

    The sequence fields are the fuzzer's live containers, not copies --
    copying a 5000-entry cmplog pair list or a large corpus per mutation
    would cost more than the mutation. They are read-only *by contract*,
    the same way ``data`` is: a mutator that appends to ``dictionary`` is
    misbehaving in the same way as one that mutates its input buffer in
    place.

    Constructible directly, with no fuzzer at all, which is the point::

        ctx = MutationContext(max_len=64, dictionary=[b"GET", b"POST"])
    """

    __slots__ = (
        "cmplog_pairs",
        "cmplog_tokens",
        "corpus",
        "crash_mi",
        "dictionary",
        "grammar",
        "markov",
        "max_len",
        "mc",
        "rand_pool",
        "seed_meta",
        "stall_recovery_active",
        "weizz_tags_enabled",
    )

    def __init__(
        self,
        *,
        max_len: int = 0,
        dictionary: Sequence[bytes] = (),
        cmplog_pairs: Sequence[tuple[bytes, bytes]] = (),
        cmplog_tokens: Sequence[bytes] = (),
        corpus: Sequence[bytes] = (),
        weizz_tags_enabled: bool = False,
        rand_pool=None,
        seed_meta=None,
        markov=None,
        mc=None,
        grammar=None,
        crash_mi=None,
        stall_recovery_active: bool = False,
    ) -> None:
        #: Length cap for the returned buffer; 0 means uncapped.
        self.max_len = max_len
        #: Dictionary tokens, from ``-x`` and from auto-extraction.
        self.dictionary = dictionary
        #: Comparison operand pairs recovered by cmplog, possibly empty.
        self.cmplog_pairs = cmplog_pairs
        #: Comparison-operand tokens recovered by cmplog, possibly empty.
        #: A separate field from ``cmplog_pairs`` rather than derived from
        #: it -- cmplog collects them independently (tokens are singletons
        #: harvested from operand bytes, pairs are the two-sided
        #: comparisons), so there's no cheap way to get one from the other.
        self.cmplog_tokens = cmplog_tokens
        #: Current corpus entries, for splice-style mutations.
        self.corpus = corpus
        #: True when ``--weizz-tags`` is set (Weizz structure-aware ops).
        self.weizz_tags_enabled = weizz_tags_enabled
        #: The fuzzer's shared PRNG (``randint``/``choice``/...). Optional
        #: because most implementors of this interface receive ``rng`` as
        #: its own positional argument to ``mutate()`` already; carried here
        #: too so ``OperatorEngine``'s existing ``_op_*`` handlers -- which
        #: read it dozens of times per call, see docs/port-backlog.md F1 --
        #: can read ``ctx.rand_pool`` instead of reaching around the context
        #: at ``self.f._rand_pool``.
        self.rand_pool = rand_pool
        #: Per-seed metadata dict, keyed by raw seed bytes (record_stride,
        #: seed_passed_det, ...). A live reference like ``corpus`` and
        #: ``dictionary``, not a copy -- see their docstrings above.
        self.seed_meta = seed_meta
        #: Markov-chain byte model, when trained. ``None`` if untrained or
        #: disabled -- callers already guard on this before use upstream.
        self.markov = markov
        #: Monte-Carlo / cross-entropy byte model, when fitted. Same
        #: None-means-absent convention as ``markov``.
        self.mc = mc
        #: Grammar-based structural mutator, when a grammar is loaded.
        self.grammar = grammar
        #: Crash-guided mutual-information estimator, when populated.
        self.crash_mi = crash_mi
        #: True while the fuzzer's stall-recovery mode is active.
        self.stall_recovery_active = stall_recovery_active

    @classmethod
    def from_fuzzer(cls, fuzzer) -> MutationContext:
        """Project a ``Fuzzer`` onto the declared context fields.

        ``getattr`` throughout rather than direct access: the adapter runs
        against partially-constructed fuzzers in tests and against a
        fuzzer whose cmplog is absent whenever ``--cmplog`` is off, and a
        missing attribute must read as "no such state", not as a crash
        inside a third-party mutator's call path.
        """
        cmplog = getattr(fuzzer, "_cmplog", None)
        return cls(
            max_len=getattr(fuzzer, "max_len", 0) or 0,
            dictionary=getattr(fuzzer, "dictionary", None) or (),
            cmplog_pairs=getattr(cmplog, "pairs", None) or (),
            cmplog_tokens=getattr(cmplog, "tokens", None) or (),
            corpus=getattr(fuzzer, "corpus", None) or (),
            weizz_tags_enabled=bool(getattr(fuzzer, "weizz_tags", False)),
            rand_pool=getattr(fuzzer, "_rand_pool", None),
            seed_meta=getattr(fuzzer, "seed_meta", None),
            markov=getattr(fuzzer, "markov", None),
            mc=getattr(fuzzer, "mc", None),
            grammar=getattr(fuzzer, "grammar", None),
            crash_mi=getattr(fuzzer, "_crash_mi", None),
            stall_recovery_active=bool(getattr(fuzzer, "_stall_recovery_active", False)),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<MutationContext max_len={self.max_len} "
            f"dictionary={len(self.dictionary)} "
            f"cmplog_pairs={len(self.cmplog_pairs)} "
            f"corpus={len(self.corpus)} "
            f"weizz_tags_enabled={self.weizz_tags_enabled}>"
        )


class MutatorBase(ABC):
    """Abstract base for self-contained, registerable mutators.

    Subclasses set ``name`` (unique, matches the operator name used by
    schedulers and stats) and ``category`` (one of the registry's category
    bands, e.g. ``"adaptive"``, ``"byte"``, ``"structural"``).
    """

    #: Unique operator name. Must be set by the subclass.
    name: str = ""

    #: Registry category band this mutator is classified under.
    category: str = "adaptive"

    @abstractmethod
    def mutate(
        self,
        data: bytes,
        rng,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx,
    ) -> bytes | None:
        """Return mutated bytes, or None to decline this call.

        Args:
            data: Current input. Must not be modified in place.
            rng: The fuzzer's random pool (``randint``/``choice``).
            max_len: Length cap, 0 meaning uncapped. Same value as
                ``context.max_len``, kept as a positional for the common
                case where it is all a mutator needs. The registry adapter
                clamps the return value regardless.
            context: Read-only view of the fuzzer state a mutator may use
                (dictionary, cmplog pairs, corpus). ``None`` only when a
                mutator is called directly rather than through the
                registry adapter.
            **ctx: Forward-compatible extras. Accept and ignore what you
                do not use.
        """

    def on_new_coverage(self, seed: bytes, new_edges: int) -> None:  # noqa: B027
        """Called when a mutation produced new coverage.

        Default is a no-op so simple mutators need not implement it. Note
        this fires for new coverage generally, not solely for coverage
        attributable to this mutator -- treat it as a campaign-progress
        signal, not a per-mutator reward.
        """

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        """Whether this mutator can run right now.

        Mirrors the registry's availability predicates (a dictionary being
        loaded, cmplog pairs existing, ...) -- which is exactly why it
        takes the same ``MutationContext`` as ``mutate()`` and not the
        fuzzer: every predicate the built-ins express is a statement about
        one of those fields. Default: always available.
        """
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} category={self.category!r}>"
