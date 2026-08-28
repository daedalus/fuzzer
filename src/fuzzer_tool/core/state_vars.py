"""SGFuzz enum-state extraction and source instrumentation.

A parser's state almost always lives in an enum-typed variable, and the
assignments to it are the state transitions. Edge coverage cannot see
those transitions as such: it records the *code* that performs one, so
two runs visiting the same blocks in a different order collapse into one
bitmap. SGFuzz (USENIX Sec '22) recovers the state machine by finding
enum-typed variables in the source and instrumenting every assignment.

This module is the source half: scan for enum types, their constants and
the variables declared with them, then rewrite each assignment to call
``__sfuzz_state(var_id, value)`` alongside it. The runtime half is in
``adapters/afl_shim.c``, which folds ``(var_id, previous, current)`` into
the edge map -- so a state sequence no input produced before shows up as
a new edge and every existing consumer of coverage (scoring, scheduling,
admission) picks it up with no new feedback plumbing.

Deliberately a regex scan and not a parse. Upstream ships a regex script
too, and the cost of being wrong is bounded in both directions: a missed
variable loses signal the fuzzer never had, and a spurious one costs one
inert call. Anything that would change program semantics -- compound
assignment, comparison, a constant belonging to a different enum -- is
excluded by construction below rather than left to chance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# The runtime packs the variable id into the low bits of an edge hash.
_MAX_VAR_ID = 0xFFFF

# `typedef enum [tag] { A, B } name;` and `enum name { A, B }`.
_TYPEDEF_ENUM = re.compile(
    r"typedef\s+enum\s+(?:\w+\s*)?\{(?P<body>[^}]*)\}\s*(?P<name>\w+)\s*;",
    re.DOTALL,
)
_NAMED_ENUM = re.compile(
    r"(?<!typedef\s)enum\s+(?P<name>\w+)\s*\{(?P<body>[^}]*)\}",
    re.DOTALL,
)

# One enumerator: `NAME`, `NAME = 3`, `NAME = 1 << 2`.
_ENUMERATOR = re.compile(r"^\s*(\w+)\s*(?:=[^,]*)?$")

_PROTOTYPE = "void __sfuzz_state(unsigned var_id, unsigned long long value);"


@dataclass
class EnumState:
    """One enum type: its constants, its variables, and its runtime id."""

    MAX_VAR_ID = _MAX_VAR_ID

    type_name: str
    constants: list[str]
    variables: list[str] = field(default_factory=list)
    var_id: int = 0


def _enumerators(body: str) -> list[str]:
    """Names from an enum body, dropping values and comments."""
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    body = re.sub(r"//[^\n]*", "", body)

    out = []
    for part in body.split(","):
        m = _ENUMERATOR.match(part.strip() and part or "")
        if m:
            out.append(m.group(1))
    return out


def _stable_id(type_name: str) -> int:
    """Id derived from the type name, not from scan order.

    The id is baked into instrumented source. Numbering by discovery
    order would renumber every state when an unrelated enum is added
    above it, and a rebuilt object would disagree with a still-cached
    one about what state 3 means.
    """
    h = 0
    for ch in type_name.encode():
        h = (h * 131 + ch) & _MAX_VAR_ID
    return h


def scan_enum_states(source: str) -> list[EnumState]:
    """Find enum types, their constants, and variables declared with them."""
    if not source or not source.strip():
        return []

    found: dict[str, EnumState] = {}

    for m in _TYPEDEF_ENUM.finditer(source):
        name = m.group("name")
        constants = _enumerators(m.group("body"))
        if constants:
            found[name] = EnumState(name, constants, var_id=_stable_id(name))

    for m in _NAMED_ENUM.finditer(source):
        name = m.group("name")
        if name in found:
            continue
        constants = _enumerators(m.group("body"))
        if constants:
            found[name] = EnumState(name, constants, var_id=_stable_id(name))

    # Variables: `<type> name` for a typedef, `enum <type> name` for a tag.
    for state in found.values():
        decl = re.compile(
            rf"(?:enum\s+)?\b{re.escape(state.type_name)}\s+(\w+)\s*(?:=|;|\))",
        )
        for m in decl.finditer(source):
            var = m.group(1)
            if var not in state.variables:
                state.variables.append(var)

    log.debug("scan_enum_states: %d enum types", len(found))
    return list(found.values())


def instrument_source(source: str) -> str:
    """Rewrite enum assignments to report the transition.

    ``st = ST_BODY;`` becomes ``st = ST_BODY, __sfuzz_state(id, ST_BODY);``
    -- a comma expression, so the assignment's own value and side effects
    are untouched and the statement stays one statement (an `if` without
    braces keeps working).
    """
    if not source or not source.strip():
        return source
    if "__sfuzz_state(" in source:
        return source  # already instrumented; rebuilds run over both

    states = scan_enum_states(source)
    if not states:
        return source

    out = source
    hits = 0
    for state in states:
        if not state.variables:
            continue
        variables = "|".join(re.escape(v) for v in state.variables)
        constants = "|".join(re.escape(c) for c in state.constants)
        # `=` not preceded or followed by another operator character keeps
        # `==`, `!=`, `+=` and friends out: those are not transitions.
        #
        # `decl` catches `parse_state st = ST_INIT` and is then left alone.
        # In a declaration the comma separates declarators, so the comma
        # expression below would read as a second declarator named
        # __sfuzz_state -- which does not compile. The initial state is
        # therefore not reported; the first real transition still is, and
        # its predecessor reads as 0.
        assign = re.compile(
            rf"(?P<decl>(?:enum\s+)?\b{re.escape(state.type_name)}\s+)?"
            rf"\b(?P<var>{variables})\s*(?<![=!<>+\-*/%&|^])=(?!=)\s*"
            rf"(?P<val>{constants})\b"
        )

        def _rewrite(m, _id=state.var_id):
            if m.group("decl"):
                return m.group(0)
            return f"{m.group('var')} = {m.group('val')}, __sfuzz_state({_id}, {m.group('val')})"

        before = out.count(", __sfuzz_state(")
        out = assign.sub(_rewrite, out)
        hits += out.count(", __sfuzz_state(") - before

    if not hits:
        return source

    return f"{_PROTOTYPE}\n{out}"
