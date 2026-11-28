#!/usr/bin/env python3
"""Translate the TLA+ fragment tools/tla_extract.py emits into a Lean 4 model.

    <Class>.tla + Syscalls.tla ─parse─> AST ─emit─> <Class>.lean ─lake─> check (BFS)
                                                     imports tools/tla/lean/TlaRt.lean

Semantics (TLC's, for this fragment):

    w = e (Init), w' = e    one successor state: [e]
    A \\/ B                  successors of both: A ++ B
    g /\\ A                  pure guards, then exactly one action: if g then A else []
    \\E x \\in S : A          flatMap over S
    FALSE                   no successor: []
    CASE                    first true arm (no arm and no OTHER: evaluation error)

Every value is a dynamic ``Tla.Val``. LET operators and LAMBDAs are lambda-lifted
to top-level defs (their free locals become leading parameters), so no Lean term
nests deeper than one CPS step. Anything outside the fragment raises
TranslateError with its line; nothing is written.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

_RT = Path(__file__).resolve().parent / "tla" / "lean" / "TlaRt.lean"
_LEAN_VERSION = "4.36.0"
_BUILD_TIMEOUT = 3600
_RUN_TIMEOUT = 600
_EXIT_PANIC = 3

_KEYWORDS = {"IF", "THEN", "ELSE", "CASE", "OTHER", "LET", "IN", "LAMBDA", "EXCEPT"}
_UNSUPPORTED = {
    "CHOOSE", "SUBSET", "UNION", "DOMAIN", "ENABLED", "UNCHANGED", "RECURSIVE",
    "INSTANCE", "ASSUME", "THEOREM", "LOCAL",
}  # fmt: skip
_BUILTINS = {"ToString": "Tla.toStr"}
_DECLS = {"EXTENDS", "CONSTANT", "CONSTANTS", "VARIABLE", "VARIABLES"}

# Infix: token -> (precedence, kind). TLA+ precedences; /\ binds tighter than \/.
_INFIX = {
    "=>": (1, "implies"),
    "\\/": (2, "or"),
    "/\\": (3, "and"),
    "=": (5, "eq"),
    "#": (5, "ne"),
    "<": (5, "lt"),
    ">": (5, "gt"),
    "<=": (5, "le"),
    ">=": (5, "ge"),
    "\\in": (5, "in"),
    "\\notin": (5, "notin"),
    "\\cup": (8, "cup"),
    "..": (9, "range"),
    "+": (10, "add"),
    "-": (10, "sub"),
}
_NOT_PREC = 4
_RIGHT_ASSOC = {"implies"}
_JUNCTIONS = {"and", "or"}

_TOKEN_RE = re.compile(
    r"""(?P<ws>[ \t\r]+)|(?P<nl>\n)|(?P<lcom>\\\*[^\n]*)|(?P<bcom>\(\*.*?\*\))
    |(?P<rule>-{4,}[^\n]*|={4,}[^\n]*)
    |(?P<str>"(?:\\.|[^"\\])*")|(?P<num>\d+)
    |(?P<op>==|\|->|->|=>|<=|>=|/\\|\\/|\\(?:notin|in|cup|E|A)\b|\.\.|\[\]|[=\#<>+\-@!.,:()\[\]{}~'])
    |(?P<id>[A-Za-z_][A-Za-z0-9_]*)""",
    re.VERBOSE | re.DOTALL,
)


class TranslateError(Exception):
    """A construct outside the supported TLA+ fragment."""


class Sort(Enum):
    """What an expression denotes in Lean."""

    VAL = "Tla.Val"
    ACTS = "Tla.Acts"


@dataclass
class Tok:
    kind: str
    text: str
    line: int
    col: int


@dataclass
class Node:
    kind: str
    args: list = field(default_factory=list)
    line: int = 0


@dataclass
class Def:
    """`Name(params) == body`; a param's arity is 0 for a value, n for an operator."""

    name: str
    params: list[tuple[str, int]]
    body: Node
    line: int


@dataclass
class Module:
    defs: list[Def]
    constants: list[str]
    variables: list[str]


# ── Lexer ───────────────────────────────────────────────────────────


def _tokens(text: str) -> list[Tok]:
    out, line, start, pos = [], 1, 0, 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise TranslateError(f"line {line}: unexpected character {text[pos]!r}")
        kind, tok = m.lastgroup, m.group()
        if kind not in ("ws", "nl", "lcom", "bcom", "rule"):
            out.append(Tok(kind, tok, line, pos - start))
        if kind == "nl":
            line, start = line + 1, m.end()
        elif kind == "bcom":
            line += tok.count("\n")
            start = pos + tok.rfind("\n") + 1 if "\n" in tok else start
        pos = m.end()
    out.append(Tok("eof", "", line, -1))
    return out


# ── Parser ──────────────────────────────────────────────────────────


class Parser:
    """Precedence climbing over the fragment. A bullet list (`/\\` or `\\/` first on
    an item) fences its items: a token at or left of the bullet column ends one."""

    def __init__(self, text: str):
        self.toks = _tokens(text)
        self.i = 0
        self.fence = -2

    # Token access ---------------------------------------------------

    def raw(self) -> Tok:
        return self.toks[self.i]

    def peek(self) -> Tok:
        tok = self.toks[self.i]
        if tok.kind != "eof" and tok.col <= self.fence:
            return Tok("eof", "", tok.line, tok.col)
        return tok

    def next(self) -> Tok:
        tok = self.peek()
        if tok.kind == "eof":
            raise TranslateError(f"line {tok.line}: unexpected end of expression")
        self.i += 1
        return tok

    def at(self, text: str) -> bool:
        tok = self.peek()
        return tok.text == text and tok.kind in ("op", "id")

    def expect(self, text: str) -> Tok:
        tok = self.peek()
        if tok.text != text:
            raise TranslateError(f"line {tok.line}: expected {text!r}, got {tok.text!r}")
        return self.next()

    def ident(self) -> str:
        tok = self.next()
        if tok.kind != "id" or tok.text in _KEYWORDS:
            raise TranslateError(f"line {tok.line}: expected a name, got {tok.text!r}")
        return tok.text

    # Module ---------------------------------------------------------

    def module(self) -> Module:
        mod = Module([], [], [])
        if self.at("MODULE"):
            self.next()
            self.ident()
        while self.raw().kind != "eof":
            tok = self.raw()
            if tok.text in _DECLS:
                self._decl(mod)
                continue
            mod.defs.append(self.definition())
        return mod

    def _decl(self, mod: Module) -> None:
        word = self.next().text
        names = [self.ident()]
        while self.at(","):
            self.next()
            names.append(self.ident())
        if word.startswith("CONSTANT"):
            mod.constants += names
        elif word.startswith("VARIABLE"):
            mod.variables += names

    def definition(self) -> Def:
        tok = self.peek()
        name = self.ident()
        params: list[tuple[str, int]] = []
        if self.at("("):
            self.next()
            params.append(self._param())
            while self.at(","):
                self.next()
                params.append(self._param())
            self.expect(")")
        self.expect("==")
        return Def(name, params, self.expr(), tok.line)

    def _param(self) -> tuple[str, int]:
        name = self.ident()
        if not self.at("("):
            return name, 0
        self.next()
        arity = 1
        self.expect("_")
        while self.at(","):
            self.next()
            self.expect("_")
            arity += 1
        self.expect(")")
        return name, arity

    # Expressions ----------------------------------------------------

    def expr(self, min_prec: int = 0) -> Node:
        left = self.prefix()
        while True:
            tok = self.peek()
            spec = _INFIX.get(tok.text) if tok.kind == "op" else None
            if spec is None or spec[0] < min_prec:
                return left
            self.next()
            prec, kind = spec
            right = self.expr(prec if kind in _RIGHT_ASSOC else prec + 1)
            left = self._join(kind, left, right, tok.line)

    @staticmethod
    def _join(kind: str, left: Node, right: Node, line: int) -> Node:
        # Flatten chains of one junction so emitted guards stay one `&&`.
        if kind in _JUNCTIONS and left.kind == kind:
            return Node(kind, [*left.args, right], left.line)
        return Node(kind, [left, right], line)

    def prefix(self) -> Node:
        tok = self.peek()
        if tok.kind == "id" and tok.text in _UNSUPPORTED:
            raise TranslateError(f"line {tok.line}: {tok.text} is outside the fragment")
        handler = _PREFIX.get(tok.text) if tok.kind in ("op", "id") else None
        if handler is not None:
            return handler(self)
        return self.postfix(self.primary())

    def primary(self) -> Node:
        tok = self.next()
        if tok.kind == "num":
            return Node("num", [int(tok.text)], tok.line)
        if tok.kind == "str":
            return Node("str", [_unquote(tok.text)], tok.line)
        if tok.text == "@":
            return Node("at", [], tok.line)
        if tok.kind != "id" or tok.text in _KEYWORDS:
            raise TranslateError(f"line {tok.line}: unexpected {tok.text!r}")
        if self.peek().text == "(" and self.peek().line == tok.line:
            return Node("apply", [tok.text, self._args()], tok.line)
        return Node("ident", [tok.text], tok.line)

    def _args(self) -> list[Node]:
        self.expect("(")
        args = [self.expr()]
        while self.at(","):
            self.next()
            args.append(self.expr())
        self.expect(")")
        return args

    def postfix(self, node: Node) -> Node:
        while True:
            tok = self.peek()
            if tok.text == "'":
                self.next()
                node = Node("prime", [node], tok.line)
            elif tok.text == ".":
                self.next()
                node = Node("dot", [node, self.ident()], tok.line)
            elif tok.text == "[" and tok.kind == "op":
                self.next()
                key = self.expr()
                self.expect("]")
                node = Node("app", [node, key], tok.line)
            else:
                return node

    # Prefix forms ---------------------------------------------------

    def p_not(self) -> Node:
        tok = self.next()
        return Node("not", [self.expr(_NOT_PREC + 1)], tok.line)

    def p_bullets(self) -> Node:
        first = self.peek()
        kind = _INFIX[first.text][1]
        items = []
        while self.raw().text == first.text and self.raw().col == first.col:
            self.i += 1
            saved, self.fence = self.fence, first.col
            items.append(self.expr())
            self.fence = saved
        return items[0] if len(items) == 1 else Node(kind, items, first.line)

    def p_paren(self) -> Node:
        tok = self.next()
        saved, self.fence = self.fence, -2  # parens bracket their own layout
        inner = self.expr()
        self.fence = saved
        self.expect(")")
        return self.postfix(Node("paren", [inner], tok.line))

    def p_if(self) -> Node:
        tok = self.next()
        cond = self.expr()
        self.expect("THEN")
        then = self.expr()
        self.expect("ELSE")
        return Node("if", [cond, then, self.expr()], tok.line)

    def p_case(self) -> Node:
        tok = self.next()
        arms, other = [], None
        while True:
            if self.at("OTHER"):
                self.next()
                self.expect("->")
                other = self.expr()
                break
            guard = self.expr()
            self.expect("->")
            arms.append((guard, self.expr()))
            if not self.at("[]"):
                break
            self.next()
        return Node("case", [arms, other], tok.line)

    def p_let(self) -> Node:
        tok = self.next()
        defs = []
        while not self.at("IN"):
            defs.append(self.definition())
        self.next()
        return Node("let", [defs, self.expr()], tok.line)

    def p_lambda(self) -> Node:
        tok = self.next()
        params = [self.ident()]
        while self.at(","):
            self.next()
            params.append(self.ident())
        self.expect(":")
        return Node("lambda", [params, self.expr()], tok.line)

    def p_quant(self) -> Node:
        tok = self.next()
        var = self.ident()
        self.expect("\\in")
        dom = self.expr()
        self.expect(":")
        kind = "exists" if tok.text == "\\E" else "forall"
        return Node(kind, [var, dom, self.expr()], tok.line)

    def p_set(self) -> Node:
        tok = self.next()
        saved, self.fence = self.fence, -2
        node = self._set_body(tok.line)
        self.fence = saved
        self.expect("}")
        return self.postfix(node)

    def _set_body(self, line: int) -> Node:
        if self.at("}"):
            return Node("setlit", [[]], line)
        first = self.expr()
        if self.at(":"):
            self.next()
            return self._set_comprehension(first, line)
        items = [first]
        while self.at(","):
            self.next()
            items.append(self.expr())
        return Node("setlit", [items], line)

    def _set_comprehension(self, first: Node, line: int) -> Node:
        nxt = self.toks[self.i + 1]
        if self.peek().kind == "id" and nxt.text == "\\in":
            var = self.ident()
            self.expect("\\in")
            return Node("setmap", [first, var, self.expr()], line)
        if first.kind != "in" or first.args[0].kind != "ident":
            raise TranslateError(f"line {line}: set filter needs `x \\in S : P`")
        return Node("setfilter", [first.args[0].args[0], first.args[1], self.expr()], line)

    def p_bracket(self) -> Node:
        tok = self.next()
        saved, self.fence = self.fence, -2
        node = self._bracket_body(tok.line)
        self.fence = saved
        self.expect("]")
        return self.postfix(node)

    def _bracket_body(self, line: int) -> Node:
        if self.peek().kind == "id" and self.toks[self.i + 1].text == "|->":
            return self._record(line)
        first = self.expr()
        if self.at("|->"):
            if first.kind != "in" or first.args[0].kind != "ident":
                raise TranslateError(f"line {line}: function needs `[x \\in S |-> e]`")
            self.next()
            return Node("fnof", [first.args[0].args[0], first.args[1], self.expr()], line)
        if self.at("->"):
            self.next()
            return Node("funset", [first, self.expr()], line)
        self.expect("EXCEPT")
        return Node("except", [first, self._updates()], line)

    def _record(self, line: int) -> Node:
        fields = []
        while True:
            name = self.ident()
            self.expect("|->")
            fields.append((name, self.expr()))
            if not self.at(","):
                return Node("record", [fields], line)
            self.next()

    def _updates(self) -> list[tuple[list, Node]]:
        ups = []
        while True:
            self.expect("!")
            path: list = []
            while self.at(".") or self.at("["):
                if self.next().text == ".":
                    path.append(Node("str", [self.ident()]))
                    continue
                path.append(self.expr())
                self.expect("]")
            self.expect("=")
            ups.append((path, self.expr()))
            if not self.at(","):
                return ups
            self.next()


_PREFIX = {
    "~": Parser.p_not,
    "/\\": Parser.p_bullets,
    "\\/": Parser.p_bullets,
    "(": Parser.p_paren,
    "IF": Parser.p_if,
    "CASE": Parser.p_case,
    "LET": Parser.p_let,
    "LAMBDA": Parser.p_lambda,
    "\\E": Parser.p_quant,
    "\\A": Parser.p_quant,
    "{": Parser.p_set,
    "[": Parser.p_bracket,
}


def _unquote(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text[1:-1])


def parse_expr(text: str) -> Node:
    """Parse one expression (tests and debugging)."""
    p = Parser(text)
    node = p.expr()
    if p.raw().kind != "eof":
        raise TranslateError(f"line {p.raw().line}: trailing {p.raw().text!r}")
    return node


def parse_module(text: str) -> Module:
    return Parser(text).module()


# ── Emitter ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sym:
    """A name in scope. `params` arities for an operator; VAL/ACTS result."""

    kind: str  # "val" | "op" | "state" | "global" | "const"
    params: tuple[int, ...] = ()
    sort: Sort = Sort.VAL
    state: bool = False  # global reads the state variable: takes it first


def _q(name: str) -> str:
    return f"«{name}»"


def _lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _op_type(arity: int, sort: Sort = Sort.ACTS) -> str:
    return " → ".join(["Tla.Val"] * arity + [sort.value])


class Emitter:
    """AST -> Lean source. One instance per model."""

    def __init__(self, variable: str):
        self.var = variable
        self.globals: dict[str, Sym] = {}
        self.out: list[str] = []
        self.lifted = 0
        self.init_mode = False

    # Scope ----------------------------------------------------------

    def look(self, name: str, scope: dict[str, Sym], line: int) -> Sym:
        sym = scope.get(name) or self.globals.get(name)
        if sym is None:
            raise TranslateError(f"line {line}: unknown name {name!r}")
        return sym

    def sort(self, n: Node, scope: dict[str, Sym]) -> Sort:
        fn = _SORT.get(n.kind)
        return fn(self, n, scope) if fn else Sort.VAL

    def _sort_eq(self, n: Node, scope) -> Sort:
        return Sort.ACTS if self._produce_lhs(n.args[0], scope) else Sort.VAL

    def _produce_lhs(self, lhs: Node, scope) -> bool:
        if lhs.kind == "prime":
            return True
        if not (self.init_mode and lhs.kind == "ident"):
            return False
        return self.look(lhs.args[0], scope, lhs.line).kind == "state"

    def _sort_any(self, nodes, scope) -> Sort:
        acts = any(self.sort(x, scope) is Sort.ACTS for x in nodes if x is not None)
        return Sort.ACTS if acts else Sort.VAL

    def _sort_call(self, n: Node, scope) -> Sort:
        name = n.args[0]
        if name in _BUILTINS:
            return Sort.VAL
        sym = self.look(name, scope, n.line)
        return sym.sort if sym.kind in ("op", "global") else Sort.VAL

    def _sort_let(self, n: Node, scope) -> Sort:
        return self.sort(n.args[1], self._let_scope(n.args[0], scope))

    def _let_scope(self, defs: list[Def], scope) -> dict[str, Sym]:
        inner = dict(scope)
        for d in defs:
            inner[d.name] = self._def_sym(d, inner, "op" if d.params else "val")
        return inner

    def _def_sym(self, d: Def, scope, kind: str) -> Sym:
        body_scope = {**scope, **self._param_syms(d.params)}
        return Sym(kind, tuple(a for _, a in d.params), self.sort(d.body, body_scope))

    @staticmethod
    def _param_syms(params) -> dict[str, Sym]:
        return {p: Sym("op", (0,) * a, Sort.ACTS) if a else Sym("val") for p, a in params}

    # Free locals (lambda lifting) -----------------------------------

    def free(self, n, bound: frozenset, scope: dict[str, Sym], acc: dict[str, Sym]) -> None:
        if isinstance(n, Node):
            self._free_node(n, bound, scope, acc)
        elif isinstance(n, (list, tuple)):
            for x in n:
                self.free(x, bound, scope, acc)

    def _free_node(self, n: Node, bound, scope, acc) -> None:
        name = n.args[0] if n.kind in ("ident", "apply") else None
        sym = scope.get(name) if name not in bound else None
        # Init only assigns the variable: it is never captured.
        if sym is not None and not (self.init_mode and sym.kind == "state"):
            acc.setdefault(name, sym)
        binder = _BINDERS.get(n.kind)
        if binder is None:
            self.free(n.args, bound, scope, acc)
            return
        binder(self, n, bound, scope, acc)

    def _free_let(self, n: Node, bound, scope, acc) -> None:
        for d in n.args[0]:
            self.free(d.body, bound | {p for p, _ in d.params}, scope, acc)
            bound = bound | {d.name}
        self.free(n.args[1], bound, scope, acc)

    def _free_lambda(self, n: Node, bound, scope, acc) -> None:
        self.free(n.args[1], bound | set(n.args[0]), scope, acc)

    def _free_bind3(self, n: Node, bound, scope, acc) -> None:
        # exists/forall/setfilter/fnof: [var, dom, body]; setmap: [body, var, dom]
        if n.kind == "setmap":
            body, var, dom = n.args
        else:
            var, dom, body = n.args
        self.free(dom, bound, scope, acc)
        self.free(body, bound | {var}, scope, acc)

    def lift(self, params: list[tuple[str, int]], body: Node, scope, line: int) -> str:
        """Hoist an operator body to a top-level def; return the closure expression."""
        fvs: dict[str, Sym] = {}
        self.free(body, frozenset(p for p, _ in params), scope, fvs)
        assigned = {self.var: Sym("state")} if self.init_mode else {}
        inner = {**assigned, **fvs, **self._param_syms(params)}
        sort = self.sort(body, inner)
        self.lifted += 1
        name = f"L{self.lifted}"
        code = self.emit(body, inner, sort)
        sig = [f"({_q(v)} : {self._type(s)})" for v, s in fvs.items()]
        sig += [f"({_q(p)} : {_op_type(a) if a else 'Tla.Val'})" for p, a in params]
        self.out.append(f"def {_q(name)} {' '.join(sig)} : {sort.value} :=\n  {code}\n")
        return f"({_q(name)} {' '.join(_q(v) for v in fvs)})" if fvs else _q(name)

    @staticmethod
    def _type(sym: Sym) -> str:
        if sym.kind == "op":
            return _op_type(len(sym.params), sym.sort)
        return "Tla.Val"

    # Emission -------------------------------------------------------

    def emit(self, n: Node, scope: dict[str, Sym], want: Sort) -> str:
        return self.acts(n, scope) if want is Sort.ACTS else self.val(n, scope)

    def val(self, n: Node, scope) -> str:
        fn = _VAL.get(n.kind)
        if fn is not None:
            return fn(self, n, scope)
        if n.kind in _BOOLISH:
            return f"(Tla.Val.bool {self.boolean(n, scope)})"
        raise TranslateError(f"line {n.line}: {n.kind} is not a value here")

    def boolean(self, n: Node, scope) -> str:
        fn = _BOOL.get(n.kind)
        return fn(self, n, scope) if fn else f"(Tla.b {self.val(n, scope)})"

    def acts(self, n: Node, scope) -> str:
        if self.sort(n, scope) is Sort.VAL:
            if n.kind == "ident" and n.args[0] == "FALSE":
                return "[]"
            raise TranslateError(f"line {n.line}: a value where an action is needed")
        fn = _ACTS.get(n.kind)
        if fn is None:
            raise TranslateError(f"line {n.line}: {n.kind} is not an action")
        return fn(self, n, scope)

    # Values ---------------------------------------------------------

    def v_num(self, n: Node, _scope) -> str:
        return f"(Tla.Val.int {n.args[0]})"

    def v_str(self, n: Node, _scope) -> str:
        return f"(Tla.Val.str {_lean_str(n.args[0])})"

    def v_at(self, _n: Node, _scope) -> str:
        return _q("@")

    def v_ident(self, n: Node, scope) -> str:
        name = n.args[0]
        if name in ("TRUE", "FALSE"):
            return f"(Tla.Val.bool {'true' if name == 'TRUE' else 'false'})"
        return self.call(name, [], scope, n.line)

    def v_apply(self, n: Node, scope) -> str:
        return self.call(n.args[0], n.args[1], scope, n.line)

    def call(self, name: str, args: list[Node], scope, line: int) -> str:
        if name in _BUILTINS:
            return f"({_BUILTINS[name]} {' '.join(self.val(a, scope) for a in args)})"
        sym = self.look(name, scope, line)
        if len(args) != len(sym.params):
            raise TranslateError(f"line {line}: {name} takes {len(sym.params)} arguments")
        parts = [_q(name)]
        if sym.state:
            parts.append(_q(self.var))
        parts += [self._arg(a, arity, scope) for a, arity in zip(args, sym.params, strict=True)]
        return parts[0] if len(parts) == 1 else f"({' '.join(parts)})"

    def _arg(self, a: Node, arity: int, scope) -> str:
        if arity == 0:
            return self.val(a, scope)
        if a.kind == "lambda":
            return self.lift([(p, 0) for p in a.args[0]], a.args[1], scope, a.line)
        if a.kind == "ident":
            return _q(a.args[0])
        raise TranslateError(f"line {a.line}: operator argument must be a name or LAMBDA")

    def v_paren(self, n: Node, scope) -> str:
        return self.val(n.args[0], scope)

    def v_if(self, n: Node, scope) -> str:
        c, t, e = n.args
        return f"(if {self.boolean(c, scope)} then {self.val(t, scope)} else {self.val(e, scope)})"

    def v_case(self, n: Node, scope) -> str:
        return self._case(n, scope, self.val)

    def _case(self, n: Node, scope, arm) -> str:
        arms, other = n.args
        tail = arm(other, scope) if other is not None else '(panic! "CASE: no arm matched")'
        for guard, body in reversed(arms):
            tail = f"(if {self.boolean(guard, scope)} then {arm(body, scope)} else\n  {tail})"
        return tail

    def v_let(self, n: Node, scope) -> str:
        return self._let(n, scope, self.val)

    def _let(self, n: Node, scope, body) -> str:
        defs, inner = n.args
        scope = dict(scope)
        binds = []
        for d in defs:
            sym = self._def_sym(d, scope, "op" if d.params else "val")
            if d.params:
                code = self.lift(d.params, d.body, scope, d.line)
            elif sym.sort is Sort.VAL:
                code = self.val(d.body, scope)
            else:
                raise TranslateError(f"line {d.line}: LET {d.name} is an action without params")
            binds.append(f"let {_q(d.name)} : {self._type(sym)} := {code}")
            scope[d.name] = sym
        return "(" + ";\n  ".join([*binds, body(inner, scope)]) + ")"

    def v_dot(self, n: Node, scope) -> str:
        return f"(Tla.dot {self.val(n.args[0], scope)} {_lean_str(n.args[1])})"

    def v_app(self, n: Node, scope) -> str:
        return f"(Tla.app {self.val(n.args[0], scope)} {self.val(n.args[1], scope)})"

    def v_bin(self, n: Node, scope) -> str:
        a, b = (self.val(x, scope) for x in n.args)
        return f"(Tla.{_VAL_BIN[n.kind]} {a} {b})"

    def v_setlit(self, n: Node, scope) -> str:
        items = ", ".join(self.val(x, scope) for x in n.args[0])
        return f"(Tla.setOf #[{items}])"

    def v_setmap(self, n: Node, scope) -> str:
        body, var, dom = n.args
        inner = {**scope, var: Sym("val")}
        return f"(Tla.setMap {self.val(dom, scope)} (fun {_q(var)} => {self.val(body, inner)}))"

    def v_setfilter(self, n: Node, scope) -> str:
        var, dom, body = n.args
        inner = {**scope, var: Sym("val")}
        pred = self.boolean(body, inner)
        return f"(Tla.setFilter {self.val(dom, scope)} (fun {_q(var)} => {pred}))"

    def v_fnof(self, n: Node, scope) -> str:
        var, dom, body = n.args
        inner = {**scope, var: Sym("val")}
        return f"(Tla.fnOf {self.val(dom, scope)} (fun {_q(var)} => {self.val(body, inner)}))"

    def v_record(self, n: Node, scope) -> str:
        fields = ", ".join(f"({_lean_str(k)}, {self.val(v, scope)})" for k, v in n.args[0])
        return f"(Tla.record #[{fields}])"

    def v_except(self, n: Node, scope) -> str:
        code = self.val(n.args[0], scope)
        inner = {**scope, "@": Sym("val")}
        for path, value in n.args[1]:
            keys = ", ".join(self.val(k, scope) for k in path)
            code = f"(Tla.exc {code} [{keys}] (fun {_q('@')} => {self.val(value, inner)}))"
        return code

    def v_prime(self, n: Node, _scope) -> str:
        raise TranslateError(f"line {n.line}: a primed variable is only a successor (w' = e)")

    # Booleans -------------------------------------------------------

    def b_and(self, n: Node, scope) -> str:
        return "(" + " && ".join(self.boolean(x, scope) for x in n.args) + ")"

    def b_or(self, n: Node, scope) -> str:
        return "(" + " || ".join(self.boolean(x, scope) for x in n.args) + ")"

    def b_implies(self, n: Node, scope) -> str:
        a, c = n.args
        return f"(!{self.boolean(a, scope)} || {self.boolean(c, scope)})"

    def b_not(self, n: Node, scope) -> str:
        return f"(!{self.boolean(n.args[0], scope)})"

    def b_eq(self, n: Node, scope) -> str:
        a, c = (self.val(x, scope) for x in n.args)
        return f"({a} == {c})" if n.kind == "eq" else f"({a} != {c})"

    def b_in(self, n: Node, scope) -> str:
        code = f"(Tla.b (Tla.mem {self.val(n.args[0], scope)} {self.val(n.args[1], scope)}))"
        return code if n.kind == "in" else f"(!{code})"

    def b_quant(self, n: Node, scope) -> str:
        var, dom, body = n.args
        inner = {**scope, var: Sym("val")}
        fn = "any" if n.kind == "exists" else "all"
        pred = self.boolean(body, inner)
        return f"(Tla.b (Tla.{fn} {self.val(dom, scope)} (fun {_q(var)} => {pred})))"

    def b_paren(self, n: Node, scope) -> str:
        return self.boolean(n.args[0], scope)

    # Actions --------------------------------------------------------

    def a_eq(self, n: Node, scope) -> str:
        return f"[{self.val(n.args[1], scope)}]"

    def a_or(self, n: Node, scope) -> str:
        return "(" + " ++ ".join(self.acts(x, scope) for x in n.args) + ")"

    def a_and(self, n: Node, scope) -> str:
        acts = [x for x in n.args if self.sort(x, scope) is Sort.ACTS]
        if len(acts) != 1:
            raise TranslateError(f"line {n.line}: a conjunction needs exactly one action")
        guards = [self.boolean(x, scope) for x in n.args if x is not acts[0]]
        return f"(if {' && '.join(guards)} then {self.acts(acts[0], scope)} else [])"

    def a_if(self, n: Node, scope) -> str:
        c, t, e = n.args
        return (
            f"(if {self.boolean(c, scope)} then {self.acts(t, scope)}\n"
            f"  else {self.acts(e, scope)})"
        )

    def a_case(self, n: Node, scope) -> str:
        return self._case(n, scope, self.acts)

    def a_let(self, n: Node, scope) -> str:
        return self._let(n, scope, self.acts)

    def a_exists(self, n: Node, scope) -> str:
        var, dom, body = n.args
        inner = {**scope, var: Sym("val")}
        return f"(Tla.flatMapS {self.val(dom, scope)} (fun {_q(var)} => {self.acts(body, inner)}))"

    def a_paren(self, n: Node, scope) -> str:
        return self.acts(n.args[0], scope)

    def a_call(self, n: Node, scope) -> str:
        return self.v_apply(n, scope) if n.kind == "apply" else self.v_ident(n, scope)

    # Module ---------------------------------------------------------

    def module_def(self, d: Def) -> None:
        self.init_mode = d.name == "Init"
        state = bool(self.var) and self._reads_state(d)
        scope = {self.var: Sym("state")} if state or self.init_mode else {}
        scope.update(self._param_syms(d.params))
        sort = self.sort(d.body, scope)
        self.globals[d.name] = Sym("global", tuple(a for _, a in d.params), sort, state)
        code = self.emit(d.body, scope, sort)
        sig = [f"({_q(self.var)} : Tla.Val)"] if state else []
        sig += [f"({_q(p)} : {_op_type(a) if a else 'Tla.Val'})" for p, a in d.params]
        self.out.append(f"def {_q(d.name)} {' '.join(sig)} : {sort.value} :=\n  {code}\n")
        self.init_mode = False

    def _reads_state(self, d: Def) -> bool:
        """Body reads the variable unprimed (Init only assigns it), or calls a def that does."""
        refs: dict[str, Sym] = {}
        probe = {self.var: Sym("state")} if not self.init_mode else {}
        probe.update({k: v for k, v in self.globals.items() if v.state})
        self.free(d.body, frozenset(p for p, _ in d.params), probe, refs)
        return bool(refs)


_BOOLISH = {"and", "or", "implies", "not", "eq", "ne", "in", "notin", "exists", "forall"}
_VAL_BIN = {
    "lt": "lt", "gt": "gt", "le": "le", "ge": "ge", "add": "add", "sub": "sub",
    "cup": "cup", "range": "range", "funset": "funSet",
}  # fmt: skip
_VAL = {
    "num": Emitter.v_num, "str": Emitter.v_str, "at": Emitter.v_at, "ident": Emitter.v_ident,
    "apply": Emitter.v_apply, "paren": Emitter.v_paren, "if": Emitter.v_if,
    "case": Emitter.v_case, "let": Emitter.v_let, "dot": Emitter.v_dot, "app": Emitter.v_app,
    "setlit": Emitter.v_setlit, "setmap": Emitter.v_setmap, "setfilter": Emitter.v_setfilter,
    "fnof": Emitter.v_fnof, "record": Emitter.v_record, "except": Emitter.v_except,
    "prime": Emitter.v_prime, **{k: Emitter.v_bin for k in _VAL_BIN},
}  # fmt: skip
_BOOL = {
    "and": Emitter.b_and, "or": Emitter.b_or, "implies": Emitter.b_implies,
    "not": Emitter.b_not, "eq": Emitter.b_eq, "ne": Emitter.b_eq, "in": Emitter.b_in,
    "notin": Emitter.b_in, "exists": Emitter.b_quant, "forall": Emitter.b_quant,
    "paren": Emitter.b_paren,
}  # fmt: skip
_ACTS = {
    "eq": Emitter.a_eq, "or": Emitter.a_or, "and": Emitter.a_and, "if": Emitter.a_if,
    "case": Emitter.a_case, "let": Emitter.a_let, "exists": Emitter.a_exists,
    "paren": Emitter.a_paren, "apply": Emitter.a_call, "ident": Emitter.a_call,
}  # fmt: skip
_SORT = {
    "eq": Emitter._sort_eq,
    "apply": Emitter._sort_call,
    "ident": lambda self, n, scope: (
        Sort.VAL if n.args[0] in ("TRUE", "FALSE") else Emitter._sort_call(self, n, scope)
    ),
    "or": lambda self, n, scope: self._sort_any(n.args, scope),
    "and": lambda self, n, scope: self._sort_any(n.args, scope),
    "if": lambda self, n, scope: self._sort_any(n.args[1:], scope),
    "case": lambda self, n, scope: self._sort_any([b for _, b in n.args[0]] + [n.args[1]], scope),
    "let": Emitter._sort_let,
    "exists": lambda self, n, scope: self.sort(n.args[2], {**scope, n.args[0]: Sym("val")}),
    "paren": lambda self, n, scope: self.sort(n.args[0], scope),
}
_BINDERS = {
    "let": Emitter._free_let,
    "lambda": Emitter._free_lambda,
    **{k: Emitter._free_bind3 for k in ("exists", "forall", "setfilter", "setmap", "fnof")},
}


# Standard-module constants the fragment uses.
_PRELUDE = ["def «BOOLEAN» : Tla.Val := Tla.setOf #[Tla.Val.bool false, Tla.Val.bool true]\n"]


def translate(texts: list[str], consts: dict[str, int], invariants: list[str]) -> str:
    """Lean source for the modules in *texts* (EXTENDS order: dependencies first),
    with each CONSTANT bound from *consts* and a `main` checking *invariants*."""
    mods = [parse_module(t) for t in texts]
    variables = [v for m in mods for v in m.variables]
    if len(variables) > 1:
        raise TranslateError(f"one VARIABLE supported, got {variables}")
    em = Emitter(variables[0] if variables else "")
    head = ["import TlaRt", "", "namespace TlaModel", "", *_PRELUDE]
    em.globals["BOOLEAN"] = Sym("const")
    for name in (c for m in mods for c in m.constants):
        if name not in consts:
            raise TranslateError(f"CONSTANT {name} has no value")
        em.globals[name] = Sym("const")
        head.append(f"def {_q(name)} : Tla.Val := Tla.Val.int {consts[name]}\n")
    for m in mods:
        for d in m.defs:
            em.module_def(d)
    return "\n".join([*head, *em.out, *_main(em, invariants)])


def _main(em: Emitter, invariants: list[str]) -> list[str]:
    for name in ("Init", "Next", *invariants):
        if name not in em.globals:
            raise TranslateError(f"missing definition {name}")
    invs = [f"({_lean_str(inv)}, fun s => Tla.b {_ref(em, inv)})" for inv in invariants]
    return [
        "end TlaModel",
        "",
        "def main : IO UInt32 :=",
        f"  Tla.check TlaModel.{_q('Init')} (fun s => {_ref(em, 'Next')})",
        "    #[" + ",\n      ".join(invs) + "]",
        "",
    ]


def _ref(em: Emitter, name: str) -> str:
    """A global applied to the state `s` when it reads it."""
    code = f"TlaModel.{_q(name)}"
    return f"({code} s)" if em.globals[name].state else code


# ── Build + run ─────────────────────────────────────────────────────


@dataclass
class LeanResult:
    completed: bool
    violations: dict[str, str]
    states: int


def find_lean() -> Path | None:
    """The bin/ of the vendored toolchain ($FUZZ_LEAN_HOME or tools/vendor_lean.sh)."""
    env = os.environ.get("FUZZ_LEAN_HOME")
    root = Path(os.environ.get("FUZZ_VENDOR_ROOT", Path.home() / "fuzzing" / "vendoring"))
    for cand in (Path(env) if env else None, root / "lean" / _LEAN_VERSION):
        if cand is not None and (cand / "bin" / "lake").is_file():
            return cand / "bin"
    return None


_LAKEFILE = """name = "tlamodel"
defaultTargets = ["check"]

[[lean_lib]]
name = "TlaRt"

[[lean_lib]]
name = "{name}"

[[lean_exe]]
name = "check"
root = "{name}"
"""


def write(name: str, src: str, out: Path) -> None:
    """Lake project in *out*: TlaRt.lean, <name>.lean (with main), lakefile.toml."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(_RT, out / _RT.name)
    (out / f"{name}.lean").write_text(src)
    (out / "lakefile.toml").write_text(_LAKEFILE.format(name=name))


def _parse_run(log: str, rc: int) -> LeanResult:
    m = re.search(r"^STATES (\d+)$", log, re.M)
    states = int(m.group(1)) if m else 0
    violations = {
        inv: trace.strip()
        for inv, trace in re.findall(r"^TRACE (\S+)\n(.*?)^END$", log, re.M | re.S)
    }
    completed = m is not None and rc in (0, 1)
    return LeanResult(completed, violations, states)


def check(name: str, src: str, out: Path, lean_bin: Path) -> LeanResult:
    """Build the model with lake and run its checker."""
    write(name, src, out)
    env = {**os.environ, "PATH": f"{lean_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    build = subprocess.run(
        [str(lean_bin / "lake"), "build"], cwd=out, env=env, capture_output=True, text=True,
        timeout=_BUILD_TIMEOUT, check=False,
    )  # fmt: skip
    if build.returncode != 0:
        err = (build.stdout + build.stderr)[-4000:]
        return LeanResult(False, {"LEAN_ERROR": err}, 0)
    run = subprocess.run(
        [str(out / ".lake" / "build" / "bin" / "check")], cwd=out, capture_output=True,
        text=True, timeout=_RUN_TIMEOUT, check=False,
    )  # fmt: skip
    res = _parse_run(run.stdout, run.returncode)
    if "PANIC" in run.stderr or run.returncode == _EXIT_PANIC or not res.completed:
        res.completed = False
        res.violations.setdefault("LEAN_ERROR", (run.stdout + run.stderr)[-4000:])
    return res
