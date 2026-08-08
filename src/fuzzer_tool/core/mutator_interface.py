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

        def mutate(self, data, rng, max_len=0, **ctx):
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
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


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
    def mutate(self, data: bytes, rng, max_len: int = 0, **ctx) -> bytes | None:
        """Return mutated bytes, or None to decline this call.

        Args:
            data: Current input. Must not be modified in place.
            rng: The fuzzer's random pool (``randint``/``choice``).
            max_len: Length cap, 0 meaning uncapped. The registry adapter
                clamps the return value regardless.
            **ctx: Forward-compatible extras (currently ``fuzzer``, the
                Fuzzer instance, for mutators needing cmplog/dictionary
                access). Accept and ignore what you do not use.
        """

    def on_new_coverage(self, seed: bytes, new_edges: int) -> None:  # noqa: B027
        """Called when a mutation produced new coverage.

        Default is a no-op so simple mutators need not implement it. Note
        this fires for new coverage generally, not solely for coverage
        attributable to this mutator -- treat it as a campaign-progress
        signal, not a per-mutator reward.
        """

    def is_available(self, fuzzer, data: bytes) -> bool:
        """Whether this mutator can run right now.

        Mirrors the registry's availability predicates (a dictionary being
        loaded, cmplog pairs existing, ...). Default: always available.
        """
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} category={self.category!r}>"
