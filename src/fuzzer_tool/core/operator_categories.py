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
