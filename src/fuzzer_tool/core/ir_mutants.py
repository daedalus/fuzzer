"""Mutation sites in textual LLVM IR (mutation analysis, Görz et al. 2023).

One site = one instruction line + one operator. Applying a site yields a
first-order mutant: one source-level fault, e.g. ``n < 4`` -> ``n <= 4``.
Pure text in, text out; building and replaying mutants live in
``adapters/mutant_build.py``.

    base.ll ──find_sites──> [Site(line, op), ...] ──sample_sites──> subset
       │                                                              │
       └──────────────────── apply_site(base.ll, site) <──────────────┘
                                     │
                                 mutant.ll

Expects ``-O0``-shaped IR (``clang -O2 -Xclang -disable-llvm-passes``):
allocas, loads and stores still mirror source statements.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuzzer_tool.core.rand_pool import RandPool

# Paper Table 10 (MALLOC / CALLOC): allocations shrink by 16 bytes.
SHRINK_BYTES = 16

# Size argument index per allocator: malloc(size), calloc(n, size), realloc(p, size).
_ALLOC_SIZE_ARG = {"malloc": 0, "calloc": 1, "realloc": 1}

_SHRINK_NAME = "%__mut_sz"

_ID = r"%[-\w.$]+"

_ICMP = re.compile(rf"^(\s+{_ID} = icmp )(eq|ne|[su](?:lt|le|gt|ge))( .*)$")
_BR = re.compile(rf"^(\s+br i1 [^,]+, )label ({_ID}), label ({_ID})(.*)$")
_ARITH = re.compile(rf"^(\s+{_ID} = )(add|sub)( .*)$")
_STORE = re.compile(rf"^\s+store \S+ (\S+), ptr ({_ID}),")
_CALL_VOID = re.compile(r"^\s+(?:tail |notail )?call (?:\w+ )*void @(?!llvm\.)[-\w.$]+\(")
_ALLOC = re.compile(rf"^(\s+{_ID} = (?:tail )?call [^@]*@(malloc|calloc|realloc)\()(.*)(\)[^)]*)$")

# Compiler bookkeeping slots: deleting their stores is not a source fault.
_BOOKKEEPING_SLOTS = frozenset({"%cleanup.dest.slot"})
_SPILL_SUFFIX = ".addr"

_BOUNDARY = {"slt": "sle", "sle": "slt", "sgt": "sge", "sge": "sgt"}
_BOUNDARY |= {"u" + k[1:]: "u" + v[1:] for k, v in _BOUNDARY.items()}
_EQ_NE = {"eq": "ne", "ne": "eq"}


class MutOp(enum.Enum):
    """Line-level mutation operators (subset of the paper's Table 10)."""

    ICMP_BOUNDARY = 1  # slt <-> sle, sgt <-> sge (and unsigned)
    ICMP_SIGN = 2  # slt <-> ult ...      (SIGNED_TO_UNSIGNED and back)
    ICMP_EQ_NE = 3  # eq <-> ne
    BR_NEGATE = 4  # swap conditional branch targets (REDIRECT_BRANCH)
    STORE_DELETE = 5  # drop a store       (DELETE_STORE_PATTERN)
    ARITH_SWAP = 6  # add <-> sub          (SWITCH_PLUS_MINUS)
    ALLOC_SHRINK = 7  # size - 16          (MALLOC / CALLOC)
    CALL_DELETE = 8  # drop a void call    (DELETE_CALL_INSTRUCTION)


@dataclass(frozen=True, slots=True)
class Site:
    """One mutable instruction: 0-based line index and operator."""

    line: int
    op: MutOp


# A rewrite maps one instruction line to its replacement lines, or None
# when the operator does not apply to that line.
_Rewrite = Callable[[str], "list[str] | None"]


def _icmp(body: str, table: dict[str, str]) -> list[str] | None:
    """Swap the icmp predicate through *table*."""
    m = _ICMP.match(body)
    if m is None or m[2] not in table:
        return None
    return [m[1] + table[m[2]] + m[3]]


def _flip_sign(pred: str) -> str:
    """slt -> ult, ult -> slt; eq/ne have no signedness."""
    return ("u" if pred[0] == "s" else "s") + pred[1:]


def _icmp_sign(body: str) -> list[str] | None:
    m = _ICMP.match(body)
    if m is None or m[2] in _EQ_NE:
        return None
    return [m[1] + _flip_sign(m[2]) + m[3]]


def _br_negate(body: str) -> list[str] | None:
    """``br i1 %c, label %a, label %b`` -> targets swapped, metadata kept."""
    m = _BR.match(body)
    if m is None:
        return None
    return [f"{m[1]}label {m[3]}, label {m[2]}{m[4]}"]


def _arith_swap(body: str) -> list[str] | None:
    m = _ARITH.match(body)
    if m is None:
        return None
    return [m[1] + ("sub" if m[2] == "add" else "add") + m[3]]


def _store_delete(body: str) -> list[str] | None:
    """Drop a source-level store; argument spills and bookkeeping stay."""
    m = _STORE.match(body)
    if m is None:
        return None

    value, ptr = m[1], m[2]
    if ptr == value + _SPILL_SUFFIX or ptr in _BOOKKEEPING_SLOTS:
        return None
    return []


def _call_delete(body: str) -> list[str] | None:
    return [] if _CALL_VOID.match(body) else None


def _alloc_shrink(body: str) -> list[str] | None:
    """Insert ``%__mut_sz = sub i64 SIZE, 16`` and pass it as the size.

    Example: ``@malloc(i64 noundef %n)`` -> ``@malloc(i64 noundef %__mut_sz)``.
    Constant-expression arguments (nested parens) are skipped.
    """
    m = _ALLOC.match(body)
    if m is None or "(" in m[3]:
        return None

    args = m[3].split(", ")
    idx = _ALLOC_SIZE_ARG[m[2]]
    if idx >= len(args):
        return None

    parts = args[idx].split(" ")
    if parts[0] != "i64":
        return None

    indent = body[: len(body) - len(body.lstrip())]
    sub = f"{indent}{_SHRINK_NAME} = sub i64 {parts[-1]}, {SHRINK_BYTES}"
    args[idx] = " ".join([*parts[:-1], _SHRINK_NAME])
    return [sub, m[1] + ", ".join(args) + m[4]]


# Ifs up (Hard Rule 57): one bound rewrite per operator.
_REWRITES: dict[MutOp, _Rewrite] = {
    MutOp.ICMP_BOUNDARY: lambda b: _icmp(b, _BOUNDARY),
    MutOp.ICMP_SIGN: _icmp_sign,
    MutOp.ICMP_EQ_NE: lambda b: _icmp(b, _EQ_NE),
    MutOp.BR_NEGATE: _br_negate,
    MutOp.STORE_DELETE: _store_delete,
    MutOp.ARITH_SWAP: _arith_swap,
    MutOp.ALLOC_SHRINK: _alloc_shrink,
    MutOp.CALL_DELETE: _call_delete,
}

# Cheap substring gate per operator: most lines match none.
_GATES: dict[MutOp, str] = {
    MutOp.ICMP_BOUNDARY: " icmp ",
    MutOp.ICMP_SIGN: " icmp ",
    MutOp.ICMP_EQ_NE: " icmp ",
    MutOp.BR_NEGATE: "br i1 ",
    MutOp.STORE_DELETE: "store ",
    MutOp.ARITH_SWAP: " = ",
    MutOp.ALLOC_SHRINK: "alloc(",
    MutOp.CALL_DELETE: "call ",
}


# Union of the gates: one search skips the ~75% of lines no operator wants
# (4.9 -> 3.9 ms on fuzzgoatNoVulns, min-of-7).
_ANY_GATE = re.compile(r" icmp |br i1 |store | = (?:add|sub) |call ")


def _split_cr(line: str) -> tuple[str, str]:
    """``'x\\r'`` -> ``('x', '\\r')``: CRLF input keeps its line endings."""
    body = line.rstrip("\r")
    return body, line[len(body) :]


def find_sites(ll: str) -> list[Site]:
    """Every (line, operator) pair that yields a mutant, in line order."""
    sites: list[Site] = []
    for i, line in enumerate(ll.split("\n")):
        # Instructions are indented; globals, labels and metadata are not.
        if not line[:1].isspace() or _ANY_GATE.search(line) is None:
            continue

        body, _ = _split_cr(line)
        for op, rewrite in _REWRITES.items():
            if _GATES[op] in body and rewrite(body) is not None:
                sites.append(Site(i, op))
    return sites


def apply_site(ll: str, site: Site) -> str:
    """*ll* with *site* mutated; every other line byte-identical.

    Raises:
        ValueError: *site* does not apply to that line (stale site list).
    """
    lines = ll.split("\n")
    body, cr = _split_cr(lines[site.line])
    new = _REWRITES[site.op](body)
    if new is None:
        raise ValueError(f"line {site.line} is not a {site.op.name} site: {body!r}")

    lines[site.line : site.line + 1] = [ln + cr for ln in new]
    return "\n".join(lines)


def sample_sites(sites: list[Site], cap: int, rng: RandPool) -> list[Site]:
    """At most *cap* sites, uniformly chosen by *rng* (``RandPool``), in line order.

    Raises:
        ValueError: *cap* is negative.
    """
    if cap < 0:
        raise ValueError(f"cap must be >= 0, got {cap}")
    if cap >= len(sites):
        return list(sites)

    idx = list(range(len(sites)))
    rng.shuffle(idx)
    return [sites[i] for i in sorted(idx[:cap])]
