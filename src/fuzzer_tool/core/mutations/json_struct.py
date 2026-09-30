"""Token-level JSON mutations (fuzzgoat and any other JSON parser).

``tree_mutate`` sees only brackets and ``special_strings`` splices at random
offsets, so nothing edits a JSON document *as* JSON. This module lexes the
input with one regex (tolerant: unknown bytes are skipped, an unterminated
string runs to EOF) and applies one token-aware edit:

    mode            example                        parser path it reaches
    --------------  -----------------------------  -------------------------
    type_swap       {"a":1}   -> {"a":[[]]}        type dispatch
    number_edge     [7]       -> [1e309]           strtod / int overflow
    string_edge     {"k":1}   -> {"\\ud800":1}     escape + surrogate decode
    dup_member      {"a":1}   -> {"a":1,"a":1}     duplicate-key handling
    trailing_comma  [1]       -> [1,]              separator state machine
    drop_punct      {"a":1}   -> {"a"1}            missing-separator errors
    truncate        {"ab":1}  -> {"a               EOF inside a token

Deep nesting is left to ``nest_bomb``. Returns None to decline when the mode
has no candidate token or the result would exceed ``max_len``.
"""

import itertools
import re
from enum import IntEnum

# One pass, linear: the string alternatives are disjoint and the closing
# quote is optional, so the match never backtracks.
_TOKEN = re.compile(
    rb'(?P<s>"(?:[^"\\]+|\\.)*"?)'
    rb"|(?P<n>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    rb"|(?P<l>[A-Za-z]+)"
    rb"|(?P<p>[{}\[\]:,])",
    re.S,
)
# Bounds lexing cost on huge inputs; edits land in the first N tokens.
_MAX_TOKENS = 4096


class Kind(IntEnum):
    STR = 0
    NUM = 1
    LIT = 2
    PUNCT = 3


_GROUP_KIND = {"s": Kind.STR, "n": Kind.NUM, "l": Kind.LIT, "p": Kind.PUNCT}
_OPENERS = frozenset(b"{[")
_CLOSERS = frozenset(b"}]")
_SEPARATORS = frozenset(b":,")
_COLON = b":"[0]
_COMMA = b","[0]

VALUE_SWAPS = (
    b"null",
    b"true",
    b"false",
    b"0",
    b'""',
    b"[]",
    b"{}",
    b"[[]]",
    b'{"":{}}',
    b"nul",
    b"tru",
    b"None",
    b"undefined",
)

JSON_NUMBERS = (
    b"-0",
    b"-0.0",
    b"0e0",
    b"1e309",
    b"-1e309",
    b"1E-400",
    b"9223372036854775807",
    b"9223372036854775808",
    b"-9223372036854775809",
    b"18446744073709551616",
    b"2.2250738585072011e-308",
    b"4.9e-324",
    b"1" + b"0" * 400,
    b"00",
    b"01",
    b"-",
    b"1.",
    b".5",
    b"1e",
    b"+1",
    b"0x10",
    b"1e+-1",
    b"NaN",
    b"Infinity",
)

STRING_EDGES = (
    b'"\\ud800"',
    b'"\\udc00"',
    b'"\\ud800\\u0041"',
    b'"\\u0000"',
    b'"\\uZZZZ"',
    b'"\\u12"',
    b'"\\x41"',
    b'"\\"',
    b'"\xed\xa0\x80"',
    b'"\xc0\xaf"',
    b'"\x00"',
    b'"\n"',
    b'""',
    b'"' + b"A" * 1024 + b'"',
)


class _Tok:
    __slots__ = ("kind", "start", "end")

    def __init__(self, kind: Kind, start: int, end: int):
        self.kind = kind
        self.start = start
        self.end = end


def _lex(data: bytes) -> list[_Tok]:
    return [
        _Tok(_GROUP_KIND[m.lastgroup], m.start(), m.end())
        for m in itertools.islice(_TOKEN.finditer(data), _MAX_TOKENS)
    ]


def _is_key(data: bytes, toks: list[_Tok], i: int) -> bool:
    """STR token immediately followed by a ':' token."""
    return (
        toks[i].kind == Kind.STR
        and i + 1 < len(toks)
        and toks[i + 1].kind == Kind.PUNCT
        and data[toks[i + 1].start] == _COLON
    )


def _punct(data: bytes, toks: list[_Tok], chars: frozenset[int]) -> list[int]:
    return [i for i, t in enumerate(toks) if t.kind == Kind.PUNCT and data[t.start] in chars]


def _replace(data: bytes, tok: _Tok, repl: bytes) -> bytes:
    return data[: tok.start] + repl + data[tok.end :]


def _insert(data: bytes, pos: int, repl: bytes) -> bytes:
    return data[:pos] + repl + data[pos:]


# ── modes: (data, toks, rng) -> bytes | None ───────────────────────────


def type_swap(data, toks, rng):
    cands = [
        i
        for i, t in enumerate(toks)
        if t.kind in (Kind.NUM, Kind.LIT) or (t.kind == Kind.STR and not _is_key(data, toks, i))
    ]
    if not cands:
        return None
    return _replace(data, toks[rng.choice(cands)], rng.choice(VALUE_SWAPS))


def number_edge(data, toks, rng):
    cands = [t for t in toks if t.kind == Kind.NUM]
    if not cands:
        return None
    return _replace(data, rng.choice(cands), rng.choice(JSON_NUMBERS))


def string_edge(data, toks, rng):
    cands = [t for t in toks if t.kind == Kind.STR]
    if not cands:
        return None
    return _replace(data, rng.choice(cands), rng.choice(STRING_EDGES))


def _value_end(data: bytes, toks: list[_Tok], key: int) -> int:
    """Byte end of the value following toks[key] and its ':'."""
    last = key + 1
    depth = 0
    for j in range(key + 2, len(toks)):
        t = toks[j]
        b = data[t.start] if t.kind == Kind.PUNCT else -1
        if b in _OPENERS:
            depth += 1
        elif b in _CLOSERS:
            if depth == 0:
                break
            depth -= 1
        elif b == _COMMA and depth == 0:
            break
        last = j
    return toks[last].end


def dup_member(data, toks, rng):
    cands = [i for i in range(len(toks)) if _is_key(data, toks, i)]
    if not cands:
        return None

    key = rng.choice(cands)
    end = _value_end(data, toks, key)
    return _insert(data, end, b"," + data[toks[key].start : end])


def trailing_comma(data, toks, rng):
    cands = _punct(data, toks, _CLOSERS)
    if not cands:
        return None
    return _insert(data, toks[rng.choice(cands)].start, b",")


def drop_punct(data, toks, rng):
    cands = _punct(data, toks, _SEPARATORS)
    if not cands:
        return None
    return _replace(data, toks[rng.choice(cands)], b"")


def truncate(data, toks, rng):
    """Cut inside a token: mid-string, mid-number, or just after a bracket."""
    t = rng.choice(toks)
    return data[: t.start + max(1, (t.end - t.start) // 2)]


MODES = (type_swap, number_edge, string_edge, dup_member, trailing_comma, drop_punct, truncate)


def json_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one token-level JSON edit; None when there is nothing to do."""
    toks = _lex(data)
    if not toks:
        return None

    mode = rng.choice(MODES)
    out = mode(data, toks, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
