"""Compiler-fuzzer source mutations: mutation-testing rules over C-like text.

Port of AFL++ ``custom_mutators/compiler_fuzzer/compiler_mutator.c``
(afl-compiler-fuzzer; Groce, van Tonder, Kalburgi, Le Goues, "Making No-Fuss
Compiler Fuzzing Effective", CC 2022). Mutants of valid programs mostly stay
valid, so the target spends its time past the parser without a grammar.

Each attempt picks a random position and one of 64 rules and applies it at
the first match at or after that position; 16 attempts, then give up.

    simple  replace the first `from` with `to`         "==" -> "!="
    rep     replace l ... r (inclusive) with `to`      ";x = 1;" -> ";"
    swap    l A m B r  ->  l B m A r                   "(a,b)" -> "(b,a)"
    dup     duplicate l ... r (inclusive)              "\\nb;\\n" twice

Delimiter matches are textual (not nesting-aware) and bounded to
MAX_MUTANT_CHANGE bytes past the left delimiter, as upstream.
"""

from __future__ import annotations

from typing import NamedTuple

MAX_MUTANT_CHANGE = 1024
MAX_MUTANT_TRIES = 16


class Rule(NamedTuple):
    kind: str  # "simple" | "rep" | "swap" | "dup"
    a: bytes  # simple: from; else the left delimiter
    b: bytes = b""  # simple: to; rep/dup: right delimiter; swap: middle
    c: bytes = b""  # rep: replacement; swap: right delimiter


def _s(frm: bytes, to: bytes) -> Rule:
    return Rule("simple", frm, to)


# Order and content follow the upstream table (compiler_mutator.c rules[]).
RULES: tuple[Rule, ...] = (
    _s(b"\n", b"\nif (0==1)\n"),  # semantic statement deletion
    _s(b"(", b"(!"),
    _s(b"==", b"!="),
    _s(b"!=", b"=="),
    _s(b"==", b"<"),
    _s(b"<", b"=="),
    _s(b"==", b">"),
    _s(b">", b"=="),
    _s(b"=", b"<"),
    _s(b"=", b">"),
    _s(b"<", b">"),
    _s(b">", b"<"),
    _s(b"++", b"--"),
    _s(b"--", b"++"),
    *(
        _s(x, y)
        for x, ys in (
            (b"+", (b"-", b"*", b"/", b"%")),
            (b"*", (b"-", b"+", b"/", b"%")),
            (b"-", (b"+", b"*", b"/", b"%")),
            (b"/", (b"-", b"*", b"+", b"%")),
            (b"%", (b"-", b"*", b"/", b"+")),
        )
        for y in ys
    ),
    _s(b"\n", b"\nbreak;\n"),
    _s(b"\n", b"\ncontinue;\n"),
    _s(b"0", b"1"),
    _s(b"1", b"0"),
    _s(b"if", b"while"),
    _s(b"while", b"if"),
    _s(b"\n", b"\nwhile(1==1)\n"),
    _s(b"&&", b"||"),
    _s(b"||", b"&&"),
    _s(b"!", b""),
    Rule("rep", b";", b";", b";"),  # remove a statement after a semicolon
    Rule("rep", b"}", b";", b"}"),  # remove a statement after a right brace
    Rule("rep", b"{", b"}", b""),  # remove a (non-nested) brace construct
    Rule("rep", b"{", b"}", b"{}"),  # empty a (non-nested) brace construct
    Rule("swap", b";", b";", b";"),  # swap adjacent statements
    Rule("swap", b"}", b";", b";"),
    Rule("swap", b"(", b",", b")"),  # swap comma-delimited things
    Rule("swap", b"(", b",", b","),
    Rule("swap", b",", b",", b","),
    Rule("swap", b",", b",", b")"),
    Rule("rep", b"\n", b"\n", b""),  # delete a line
    Rule("rep", b" ", b" ", b""),  # delete something like "const"
    Rule("rep", b"\n", b" ", b""),
    Rule("rep", b"(", b" ", b""),
    Rule("swap", b" ", b" ", b" "),  # swap space-delimited things
    Rule("swap", b" ", b" ", b")"),
    Rule("swap", b"(", b" ", b" "),
    Rule("swap", b"(", b" ", b")"),
    Rule("dup", b"\n", b"\n"),  # duplicate a line
    Rule("dup", b"\n", b"}"),  # duplicate a construct, e.g. a for loop
)


def _splice(data: bytes, at: int, olen: int, rep: bytes, max_len: int) -> bytes | None:
    """data with [at, at+olen) replaced by rep; None if empty or over max_len."""
    new_len = len(data) - olen + len(rep)
    if not new_len or new_len > max_len:
        return None
    return data[:at] + rep + data[at + olen :]


def _apply_delim(rule: Rule, data: bytes, a: int, max_len: int) -> bytes | None:
    limit = min(len(data), a + MAX_MUTANT_CHANGE)
    if rule.kind == "swap":
        b = data.find(rule.b, a + 1, limit)
        c = data.find(rule.c, b + 1, limit) if b >= 0 else -1
        if c < 0:
            return None
        # data[a+1 .. c] = A m B r  ->  B m A r
        swapped = data[b + 1 : c] + rule.b + data[a + 1 : b] + rule.c
        return _splice(data, a + 1, len(swapped), swapped, max_len)

    b = data.find(rule.b, a + 1, limit)
    if b < 0:
        return None
    span = data[a : b + 1]
    rep = span + span if rule.kind == "dup" else rule.c
    return _splice(data, a, len(span), rep, max_len)


def apply_rule(rule: Rule, data: bytes, pos: int, max_len: int) -> bytes | None:
    """*rule* at the first match at/after *pos*; None when it does not apply."""
    if rule.kind == "simple":
        hit = data.find(rule.a, pos)
        if hit < 0:
            return None
        return _splice(data, hit, len(rule.a), rule.b, max_len)

    a = data.find(rule.a, pos)
    if a < 0 or a >= len(data) - 2:
        return None
    return _apply_delim(rule, data, a, max_len)


def source_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """One source-level mutation of *data*, or None after MAX_MUTANT_TRIES misses."""
    n = len(data)
    if n < 2:
        return None

    for _ in range(MAX_MUTANT_TRIES):
        pos = rng.randint(0, n - 2)
        out = apply_rule(RULES[rng.randint(0, len(RULES) - 1)], data, pos, max_len)
        if out is not None:
            return out
    return None
