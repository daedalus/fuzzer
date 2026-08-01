"""Shared operator→category taxonomy used by multiple schedulers.

Plain data module (not a scheduler): the category bandit schedulers group
operators by (bit, byte, block, dict, structural, radamsa, format, adaptive).
Schedulers must stay independent of each other and may only share neutral
dependencies like this one; Elo arbitration sits on top in the services layer.

The taxonomy is derived from the operator dispatcher
(``fuzzer_tool.core.operator_registry``), the single source of truth for
operator names and categories — never edit this mapping by hand.
"""

from fuzzer_tool.core.operator_registry import REGISTRY

OPERATOR_CATEGORIES: dict[str, set[str]] = REGISTRY.categories()
