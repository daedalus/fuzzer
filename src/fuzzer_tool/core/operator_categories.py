"""Shared operator→category taxonomy used by multiple schedulers.

Plain data module (not a scheduler): the category bandit schedulers group
operators by (bit, byte, block, dict, structural, radamsa, format, adaptive).
Schedulers must stay independent of each other and may only share neutral
dependencies like this one; Elo arbitration sits on top in the services layer.

The taxonomy is derived from the operator dispatcher
(``fuzzer_tool.core.operator_registry``), the single source of truth for
operator names and categories — never edit this mapping by hand.

``OPERATOR_CATEGORIES`` is a *snapshot* taken at import. That is correct for
the built-in table, which is fully registered by the time this module is
imported, but not for ``REGISTRY.register_mutator()`` — the documented
extension path (``core/mutator_interface``), which registers operators at
runtime. A snapshot taken before such a registration silently omits it, and
``HierarchicalBanditScheduler`` then drops the arm entirely: measured at 0
pulls out of 60,000 selections. Consumers must therefore resolve an
individual operator through :func:`category_of`, which consults the live
registry, rather than looking it up in the snapshot dict.
"""

from fuzzer_tool.core.operator_registry import REGISTRY

#: Category name given to an operator the registry has never heard of. Not a
#: registry category and never returned for a registered operator — it exists
#: so a scheduler handed an unknown name (a test harness, an operator list
#: from a resumed state file written by a different build) still has somewhere
#: to put it. Silently dropping the arm instead makes the operator permanently
#: unselectable, which is the failure mode this module exists to prevent.
UNCATEGORIZED = "uncategorized"

OPERATOR_CATEGORIES: dict[str, set[str]] = REGISTRY.categories()


def refresh() -> dict[str, set[str]]:
    """Re-sync the snapshot with the live registry, in place.

    Mutates ``OPERATOR_CATEGORIES`` rather than rebinding it, so the aliases
    other modules hold (notably ``HierarchicalBanditScheduler.CATEGORIES``)
    observe the update instead of pointing at a stale dict.
    """
    live = REGISTRY.categories()
    OPERATOR_CATEGORIES.clear()
    OPERATOR_CATEGORIES.update({cat: set(ops) for cat, ops in live.items()})
    return OPERATOR_CATEGORIES


def category_of(name: str) -> str:
    """Category for operator *name*, resolved against the live registry.

    Returns :data:`UNCATEGORIZED` for a name the registry does not know,
    never ``None`` — callers use the result as a bandit-level key, and a
    missing key is exactly what caused the arm to be dropped before.
    """
    try:
        return REGISTRY.category_of(name)
    except KeyError:
        return UNCATEGORIZED


#: The "bitflip family": operators whose per-application transform is a pure
#: fixed-width XOR against the buffer -- ``buf[i] ^= K`` for some constant
#: ``K`` that does not depend on the buffer's current contents, at a byte
#: position that does not move and without changing the buffer's length.
#: Per ``docs/handover/handover_skittercreek_tailslayer_port.md`` item 3:
#: this is the boundary for when composing consecutive mutation steps into
#: one combined map (:func:`fuzzer_tool.core.gf2_common.compose_bitmask_maps`
#: via :func:`fuzzer_tool.core.gf2_common.compose_linear_runs`) is valid.
#:
#: Deliberately narrow. Most of the ``"bit"`` category (``bit_offset_flip``,
#: ``bit_rotate``, ``bit_shift``, ``bit_transpose_*``, ``span_invert``,
#: ``bit_repack``) is excluded even though several of those are, in the
#: strict linear-algebra sense, also GF(2)-linear (a rotation is a
#: permutation matrix) -- because this set is deliberately scoped to what
#: item 3 actually asked for and verified: consecutive constant-XOR
#: mutations that commute and compose by simply XOR-ing their masks
#: together. Widening this set to permutation-like operators is a separate,
#: later change that needs its own composition proof and its own tests, not
#: an assumption to fold in here. Splices, insertions, deletions, and any
#: other length-changing or position-shifting operator are never
#: XOR-linear in this sense and must never be added.
XOR_LINEAR_OPS: frozenset[str] = frozenset({"bit_flip", "byte_flip"})


def is_xor_linear(name: str) -> bool:
    """Whether operator *name* is in the XOR-linear family (:data:`XOR_LINEAR_OPS`)."""
    return name in XOR_LINEAR_OPS
