"""Read-only catalog over the op, analyzer and scheduler registries.

Backs ``fuzzer-tool registry``. One flat view per registry:

    kind        source                                   category
    ops         core.operator_registry.REGISTRY          op category band
    analyzers   core.analyzer_registry.REGISTRY          analyzer category
    schedulers  op/seed/position/target strategy names   family
"""

import fnmatch
from dataclasses import dataclass
from enum import StrEnum

from fuzzer_tool.core.analyzer_registry import REGISTRY as _ANALYZERS
from fuzzer_tool.core.operator_registry import REGISTRY as _OPS
from fuzzer_tool.services import fuzzer as _fuzzer_mod
from fuzzer_tool.services.position_arena import POSITION_STRATEGY_NAMES
from fuzzer_tool.services.target_arena import TARGET_STRATEGY_NAMES

_GLOB_CHARS = frozenset("*?[")


class RegistryKind(StrEnum):
    """Which registry to list."""

    OPS = "ops"
    ANALYZERS = "analyzers"
    SCHEDULERS = "schedulers"


@dataclass(frozen=True)
class Entry:
    """One registered item: name, category and extra ``key=value`` details."""

    name: str
    category: str
    details: tuple[tuple[str, str], ...] = ()


def _gated(predicate) -> str:
    return "yes" if predicate is not None else "no"


def _op_entries() -> list[Entry]:
    out = []
    for s in _OPS.specs():
        # Function ops name an OperatorEngine method; class mutators their type.
        handler = s.handler_name or type(s.mutator).__name__
        out.append(
            Entry(s.name, s.category, (("handler", handler), ("gated", _gated(s.available))))
        )
    return out


def _analyzer_entries() -> list[Entry]:
    return [
        Entry(
            s.name,
            s.category,
            (
                ("phase", s.phase),
                ("gated", _gated(s.available)),
                ("swallow_errors", str(s.swallow_errors).lower()),
            ),
        )
        for s in _ANALYZERS.specs()
    ]


def _scheduler_entries() -> list[Entry]:
    families = (
        ("op", _fuzzer_mod._OPERATOR_STRATEGY_NAMES),
        ("seed", _fuzzer_mod._SEED_STRATEGY_NAMES),
        ("position", POSITION_STRATEGY_NAMES),
        ("target", TARGET_STRATEGY_NAMES),
    )
    return [Entry(n, fam) for fam, names in families for n in names]


_BUILDERS = {
    RegistryKind.OPS: _op_entries,
    RegistryKind.ANALYZERS: _analyzer_entries,
    RegistryKind.SCHEDULERS: _scheduler_entries,
}


def entries(kind: RegistryKind) -> list[Entry]:
    """Every item of *kind*, in registration order."""
    return _BUILDERS[RegistryKind(kind)]()


def categories(kind: RegistryKind) -> dict[str, int]:
    """Category -> item count, in first-seen order."""
    counts: dict[str, int] = {}
    for e in entries(kind):
        counts[e.category] = counts.get(e.category, 0) + 1
    return counts


def _matcher(pattern: str):
    """Glob if *pattern* has ``*?[``, else substring; case-insensitive, ``-`` == ``_``."""
    pat = pattern.lower().replace("-", "_")
    if _GLOB_CHARS.isdisjoint(pat):
        return lambda name: pat in name.lower()
    return lambda name: fnmatch.fnmatchcase(name.lower(), pat)


def query(
    kind: RegistryKind, pattern: str | None = None, category: str | None = None
) -> list[Entry]:
    """Items of *kind* whose name matches *pattern* and category equals *category*.

    E.g. ``query(RegistryKind.OPS, "png*")`` or ``query(SCHEDULERS, category="seed")``.
    """
    hits = entries(kind)
    if category is not None:
        hits = [e for e in hits if e.category == category]
    if pattern:
        match = _matcher(pattern)
        hits = [e for e in hits if match(e.name)]
    return hits
