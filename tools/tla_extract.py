#!/usr/bin/env python3
"""Extract a TLA+ model from a ``@tla.machine`` class and check it with TLC.

    python tools/tla_extract.py forkserver --check
    python tools/tla_extract.py pkg.module:Class --out DIR

Pipeline:

    class source ─ast─> Compiler (CPS) ─> <Class>.tla/.cfg ─> TLC, one run per invariant
         │                                     │
    @tla.machine spec                  EXTENDS Syscalls.tla
    (state, model, ignore, replies)    (hand-written I/O outcomes)

The compiler walks each entry method in continuation-passing style: every
statement receives "what happens next" (fall through, return, raise) and
emits a TLA+ expression. A call with several outcomes becomes a
Syscalls.tla operator applied to LAMBDA continuations; a continuation that
is reached from more than one place is emitted once as a LET operator.
Anything outside the subset raises ExtractError naming file:line.
"""

import argparse
import ast
import builtins
import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Runnable from a checkout without `pip install -e`: src/ for fuzzer_tool, root for module:Class.
_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from fuzzer_tool.core import tla  # noqa: E402

MACHINES = {
    "forkserver": "fuzzer_tool.adapters.forkserver:ForkserverRunner",
    "persistent": "fuzzer_tool.adapters.persistent_subprocess:PersistentLoader",
    "gdbjob": "fuzzer_tool.core.analyzers.analyzer_trace:GdbJob",
    "push": "fuzzer_tool.services.push:PushNotifier",
}
BUILTIN_INVARIANTS = (
    "NoOrphan",
    "NoHang",
    "NoUncaught",
    "RecursionBound",
    "NoDesync",
    "SetupCleanup",
)

_SYSCALLS = Path(__file__).resolve().parent / "tla" / "Syscalls.tla"
_JAR_NAME = "tla2tools.jar"
_MAX_INLINE = 3  # nested activations of one method before "Overflow"
_RECURSION_LIMIT = 20_000
_TLC_TIMEOUT = 600
_UNCATCHABLE = ('"Hang"', '"Overflow"')
_NO_DEADLINE = "NO_DEADLINE"
_NONE_HANDLE = "0"
_NONE_STR = '"None"'

# Exception classes the model can raise, child -> parent.
_PARENT = {
    "BrokenPipeError": "OSError",
    "ProcessLookupError": "OSError",
    "FileNotFoundError": "OSError",
    "OSError": "Exception",
    "ValueError": "Exception",
    "AttributeError": "Exception",
    "TimeoutExpired": "SubprocessError",
    "SubprocessError": "Exception",
    "Exception": "BaseException",
}


class ExtractError(Exception):
    """Source falls outside the supported subset."""


class VKind(Enum):
    """Abstract value kind at compile time."""

    BOOL = "bool"
    HANDLE = "handle"  # int id, 0 = None
    STR = "str"  # TLA string
    NONE = "none"  # the None literal, coerced on use
    HAVOC = "havoc"  # not modeled


class HType(Enum):
    """What a handle refers to."""

    PROC = "proc"
    READER = "reader"
    FD = "fd"
    FILE = "file"
    SET = "set"  # a PROCSET field: a TLA set of proc ids


_FIELD_HTYPE = {
    tla.Kind.PROC: HType.PROC,
    tla.Kind.READER: HType.READER,
    tla.Kind.FILE: HType.FILE,
    tla.Kind.PROCSET: HType.SET,
}
_HANDLE_KINDS = (tla.Kind.PROC, tla.Kind.READER, tla.Kind.FILE)
# Context managers whose body runs once and whose exit has no modeled effect.
_PASSTHROUGH_CTX = ("contextlib.ExitStack",)
_SUPPRESS_CTX = ("contextlib.suppress", "suppress")
_EXC = "__exc__"  # env slot: the exception a handler is running for (bare `raise`)


@dataclass(frozen=True)
class Val:
    """A compiled value: TLA text (None when unmodeled) plus kind."""

    tla: str | None
    kind: VKind
    htype: HType | None = None
    tags: frozenset[str] = frozenset()  # leading words of literal payloads, e.g. {"RUN"}


HAVOC = Val(None, VKind.HAVOC)
NONE_VAL = Val(None, VKind.NONE)

Env = dict[str, Val]
NextK = Callable[[str, Env], str]
RetK = Callable[[str, Val], str]
ExcK = Callable[[str, str, Env], str]
ValK = Callable[[str, Val], str]
RaiseK = Callable[[str, str], str]
WorldK = Callable[[str], str]


@dataclass
class Cont:
    """Where control goes after a statement."""

    nxt: NextK
    ret: RetK
    exc: ExcK


class CallKind(Enum):
    """How a call site is compiled."""

    IGNORE = "ignore"
    INLINE = "inline"
    OPAQUE = "opaque"
    MODEL = "model"
    TYPED = "typed"
    PURE = "pure"


# (receiver type, attribute path) -> (operator, result kind, result htype)
_TYPED_OPS = {
    (HType.PROC, "poll"): ("Poll", VKind.STR, None),
    (HType.PROC, "kill"): ("Kill", VKind.STR, None),
    (HType.PROC, "wait"): ("Wait", VKind.STR, None),
    (HType.PROC, "stdin.write"): ("Write", VKind.STR, None),
    (HType.PROC, "stdin.flush"): ("Flush", VKind.STR, None),
    (HType.PROC, "stdin.close"): ("CloseStdin", VKind.STR, None),
    (HType.PROC, "stdout.fileno"): ("Fileno", VKind.HANDLE, HType.FD),
    (HType.READER, "readline"): ("ReadLineT", VKind.STR, None),
    (HType.READER, "read_exact"): ("ReadExact", VKind.STR, None),
    (HType.FILE, "seek"): ("FileUse", VKind.STR, None),
    (HType.FILE, "read"): ("FileUse", VKind.STR, None),
    (HType.FILE, "write"): ("FileUse", VKind.STR, None),
    (HType.FILE, "flush"): ("FileUse", VKind.STR, None),
    (HType.FILE, "close"): ("FileClose", VKind.STR, None),
}

_MODEL_RESULT = {
    tla.Op.SPAWN: (VKind.HANDLE, HType.PROC),
    tla.Op.READER: (VKind.HANDLE, HType.READER),
    tla.Op.CLOSE_STREAMS: (VKind.STR, None),
    tla.Op.MAYBE: (VKind.STR, None),
    tla.Op.OPEN_FILE: (VKind.HANDLE, HType.FILE),
    tla.Op.KILLPG: (VKind.STR, None),
}
_MODEL_ARGS = {
    tla.Op.SPAWN: 0,
    tla.Op.READER: 1,
    tla.Op.CLOSE_STREAMS: 1,
    tla.Op.MAYBE: 0,
    tla.Op.OPEN_FILE: 0,
    tla.Op.KILLPG: 1,
}


@dataclass
class Model:
    """Generated TLA+ module and TLC config."""

    name: str
    tla: str
    invariants: list[str]
    havoc: list[str] = field(default_factory=list)

    def cfg(self, invariant: str, bounds: "Bounds") -> str:
        """TLC config checking one invariant."""
        return (
            "INIT Init\nNEXT Next\n"
            f"CONSTANTS MaxProcs = {bounds.max_procs}\n          MaxCalls = {bounds.max_calls}\n"
            f"INVARIANT {invariant}\n"
        )


@dataclass(frozen=True)
class Bounds:
    """State-space bounds."""

    max_procs: int = 3
    max_calls: int = 4


@dataclass
class CheckResult:
    """TLC outcome over every invariant."""

    completed: bool
    violations: dict[str, str]
    states: int


def _handlers_match(names: list[str]) -> set[str]:
    """Known exception classes an ``except names`` clause catches."""
    out = set()
    for exc in _PARENT:
        cur: str | None = exc
        while cur is not None:
            if cur in names:
                out.add(exc)
                break
            cur = _PARENT.get(cur)
    return out


def _first_word(text: str | bytes) -> frozenset[str]:
    """Leading word of a literal payload: b"RUN 3\\n" -> {"RUN"}."""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    words = text.split()
    return frozenset(words[:1])


def _tla_set(items) -> str:
    return "{" + ", ".join(f'"{i}"' for i in sorted(items)) + "}"


class _Join:
    """A continuation reached from several places, emitted once as a LET operator.

    Call sites go through ``stub``; ``wrap`` then compiles the body with the
    kinds merged over every call site and prefixes the LET.
    """

    def __init__(
        self,
        comp: "Compiler",
        body: Callable[..., str],
        extra: tuple[str, ...] = (),
        names: list[str] | None = None,
    ):
        self._comp = comp
        self._body = body
        self._extra = extra  # extra params besides world and locals, e.g. ("v",)
        self._names = sorted(comp.frame_names()) if names is None else names
        self._calls: list[tuple[Env, tuple[Val, ...]]] = []
        self.name = comp.fresh("K")

    def stub(self, w: str, env: Env, *extra: Val) -> str:
        self._calls.append((env, extra))
        args = (
            [w]
            + [self._arg(env.get(n, HAVOC)) for n in self._names]
            + [self._arg(v) for v in extra]
        )
        return f"{self.name}({', '.join(args)})"

    @staticmethod
    def _arg(v: Val) -> str:
        return v.tla if v.tla is not None else _NONE_HANDLE

    def _merge(self, vals: list[Val]) -> tuple[VKind, HType | None, frozenset[str]]:
        tags = frozenset().union(*(v.tags for v in vals))
        first = vals[0]
        if all(v.kind == first.kind and v.htype == first.htype for v in vals):
            return first.kind, first.htype, tags
        return VKind.HAVOC, None, tags

    def _param(self, vals: list[Val], base: str) -> tuple[str, Val]:
        kind, htype, tags = self._merge(vals)
        p = self._comp.fresh(base)
        modeled = kind not in (VKind.HAVOC, VKind.NONE)
        return p, Val(p if modeled else None, kind, htype, tags)

    def wrap(self, text: str) -> str:
        if not self._calls:
            return text
        wp = self._comp.fresh("w")
        params, env = [wp], {}
        for n in self._names:
            p, v = self._param([c[0].get(n, HAVOC) for c in self._calls], "l_" + n.lstrip("_"))
            params.append(p)
            env[n] = v
        extras = []
        for i, base in enumerate(self._extra):
            p, v = self._param([c[1][i] for c in self._calls], base)
            params.append(p)
            extras.append(v)
        body = self._body(wp, env, *extras)
        return f"LET {self.name}({', '.join(params)}) ==\n{body}\nIN {text}"


class Compiler:
    """CPS compiler from a ``@tla.machine`` class to one TLA+ module."""

    def __init__(self, cls: type):
        spec = getattr(cls, tla.MACHINE_ATTR, None)
        if spec is None:
            raise ExtractError(f"{cls.__qualname__}: not decorated with @tla.machine")
        self.cls = cls
        self.spec: tla.MachineSpec = spec
        self.file = inspect.getsourcefile(cls) or "?"
        lines, self._line0 = inspect.getsourcelines(cls)
        tree = ast.parse(textwrap.dedent("".join(lines)))
        self._methods = {n.name: n for n in tree.body[0].body if isinstance(n, ast.FunctionDef)}
        self._globals = vars(sys.modules[cls.__module__])
        self._fields = {f: f.lstrip("_") for f in spec.state}
        self._counter = 0
        self._frames: list[set[str]] = []
        self._stack: list[str] = []
        self.havoc: list[str] = []
        self._pending_wraps: list[_Join] = []
        self._stmt = {
            ast.Expr: self._s_expr,
            ast.Assign: self._s_assign,
            ast.AnnAssign: self._s_annassign,
            ast.AugAssign: self._s_augassign,
            ast.If: self._s_if,
            ast.Return: self._s_return,
            ast.Pass: lambda n, w, env, k: k.nxt(w, env),
            ast.Raise: self._s_raise,
            ast.Try: self._s_try,
            ast.With: self._s_with,
            ast.For: self._s_for,
            ast.Assert: lambda n, w, env, k: k.nxt(w, env),  # stripped under -O
        }

    # ── helpers ────────────────────────────────────────────────────

    def fresh(self, base: str) -> str:
        self._counter += 1
        return f"{base}{self._counter}"

    def frame_names(self) -> set[str]:
        return self._frames[-1] if self._frames else set()

    def where(self, node: ast.AST) -> str:
        return f"{os.path.basename(self.file)}:{self._line0 + node.lineno - 1}"

    def fail(self, node: ast.AST, msg: str) -> ExtractError:
        return ExtractError(f"{self.where(node)}: {msg}: {ast.unparse(node)[:80]}")

    def _note_havoc(self, node: ast.AST) -> None:
        entry = f"{self.where(node)}: {ast.unparse(node)[:80]}"
        if entry not in self.havoc:
            self.havoc.append(entry)

    @staticmethod
    def _locals_of(fn: ast.FunctionDef) -> set[str]:
        names = {a.arg for a in fn.args.args[1:]} | {a.arg for a in fn.args.kwonlyargs}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            if isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)
            if isinstance(node, ast.Raise) and node.exc is None:
                names.add(_EXC)
        return names

    def _let(self, expr: str, then: WorldK) -> str:
        w2 = self.fresh("w")
        return f"LET {w2} == {expr} IN\n{then(w2)}"

    # ── statements ─────────────────────────────────────────────────

    def stmts(self, body: list[ast.stmt], w: str, env: Env, k: Cont) -> str:
        if not body:
            return k.nxt(w, env)
        head, rest = body[0], body[1:]
        if not rest:
            return self.stmt(head, w, env, k)
        k2 = Cont(lambda w2, e2: self.stmts(rest, w2, e2, k), k.ret, k.exc)
        return self.stmt(head, w, env, k2)

    def stmt(self, node: ast.stmt, w: str, env: Env, k: Cont) -> str:
        handler = self._stmt.get(type(node))
        if handler is None:
            raise self.fail(node, f"unsupported statement {type(node).__name__}")
        return handler(node, w, env, k)

    def _s_expr(self, node: ast.Expr, w: str, env: Env, k: Cont) -> str:
        if isinstance(node.value, ast.Constant):
            return k.nxt(w, env)  # docstring
        return self.expr(node.value, w, env, lambda w2, _v: k.nxt(w2, env), self._raise_to(k, env))

    def _raise_to(self, k: Cont, env: Env) -> RaiseK:
        return lambda w2, e: k.exc(w2, e, env)

    def _s_assign(self, node: ast.Assign, w: str, env: Env, k: Cont) -> str:
        if len(node.targets) != 1:
            raise self.fail(node, "chained assignment")
        return self._assign(node.targets[0], node.value, w, env, k)

    def _s_annassign(self, node: ast.AnnAssign, w: str, env: Env, k: Cont) -> str:
        if node.value is None:
            return k.nxt(w, env)
        return self._assign(node.target, node.value, w, env, k)

    def _s_augassign(self, node: ast.AugAssign, w: str, env: Env, k: Cont) -> str:
        if self._modeled_field(node.target):
            raise self.fail(node, "augmented assignment to a modeled field")
        self._assert_pure(node.value, env)
        if isinstance(node.target, ast.Name):
            env = {**env, node.target.id: HAVOC}
        return k.nxt(w, env)

    def _modeled_field(self, node: ast.AST) -> str | None:
        is_self = (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )
        if is_self and node.attr in self._fields:
            return node.attr
        return None

    def _assign(self, target: ast.expr, value: ast.expr, w: str, env: Env, k: Cont) -> str:
        fld = self._modeled_field(target)
        if fld is not None and self.spec.state[fld] is tla.Kind.PROCSET:
            return self._rebind_set(target, fld, value, w, env, k)

        def bind(w2: str, v: Val) -> str:
            return self._bind(target, v, w2, env, k)

        return self.expr(value, w, env, bind, self._raise_to(k, env))

    def _rebind_set(self, target, fld: str, value: ast.expr, w: str, env: Env, k: Cont) -> str:
        """``self.<procset> = []`` empties it; any computed list keeps the unreaped
        members (the reap-filter idiom, e.g. ``self._senders = alive``)."""
        self._assert_pure(value, env)
        name = self._fields[fld]
        empty = isinstance(value, ast.List) and not value.elts
        kept = "{}" if empty else f'{{p \\in {w}.s.{name} : {w}.procs[p] \\in {{"live", "dead"}}}}'
        return self._let(f"[{w} EXCEPT !.s.{name} = {kept}]", lambda w2: k.nxt(w2, env))

    def _bind(self, target: ast.expr, v: Val, w: str, env: Env, k: Cont) -> str:
        if isinstance(target, ast.Name):
            return k.nxt(w, {**env, target.id: v})
        fld = self._modeled_field(target)
        if fld is not None:
            return self._store(target, fld, v, w, lambda w2: k.nxt(w2, env))
        if isinstance(target, ast.Attribute | ast.Subscript):
            if any(self._modeled_field(n) for n in ast.walk(target)):
                raise self.fail(target, "indirect store into a modeled field")
            return k.nxt(w, env)
        if isinstance(target, ast.Tuple) and not any(self._modeled_field(e) for e in target.elts):
            # Unpacking an unmodeled value: locals go havoc, unmodeled fields are dropped.
            names = {e.id: HAVOC for e in target.elts if isinstance(e, ast.Name)}
            return k.nxt(w, {**env, **names})
        raise self.fail(target, "unsupported assignment target")

    def _store(self, node: ast.AST, fld: str, v: Val, w: str, then: WorldK) -> str:
        kind = self.spec.state[fld]
        name = self._fields[fld]
        if kind is tla.Kind.BOOL and v.kind is VKind.BOOL:
            return self._let(f"[{w} EXCEPT !.s.{name} = {v.tla}]", then)
        if kind is tla.Kind.BOOL and v.kind is VKind.HAVOC:
            self._note_havoc(node)
            b = self.fresh("b")
            return f"\\E {b} \\in BOOLEAN :\n" + self._let(f"[{w} EXCEPT !.s.{name} = {b}]", then)
        if kind in _HANDLE_KINDS and v.kind is VKind.NONE:
            return self._let(f"[{w} EXCEPT !.s.{name} = {_NONE_HANDLE}]", then)
        if kind in _HANDLE_KINDS and v.kind is VKind.HANDLE:
            return self._let(f"[{w} EXCEPT !.s.{name} = {v.tla}]", then)
        raise self.fail(node, f"cannot store {v.kind.value} into {kind.value} field")

    def _s_return(self, node: ast.Return, w: str, env: Env, k: Cont) -> str:
        value = node.value
        if value is None:
            return k.ret(w, NONE_VAL)
        if isinstance(value, ast.Tuple):
            for extra in value.elts[1:]:
                self._assert_pure(extra, env)
            value = value.elts[0]
        return self.expr(value, w, env, k.ret, self._raise_to(k, env))

    def _s_raise(self, node: ast.Raise, w: str, env: Env, k: Cont) -> str:
        exc = node.exc
        if exc is None:
            cur = env.get(_EXC)
            if cur is None or cur.tla is None:
                raise self.fail(node, "bare raise outside an except handler")
            return k.exc(w, cur.tla, env)
        name = exc.func if isinstance(exc, ast.Call) else exc
        text = ast.unparse(name).rsplit(".", 1)[-1] if name is not None else ""
        if text not in _PARENT:
            raise self.fail(node, "raise of an exception class the model does not know")
        return k.exc(w, f'"{text}"', env)

    def _s_if(self, node: ast.If, w: str, env: Env, k: Cont) -> str:
        join = _Join(self, k.nxt)
        inner = Cont(join.stub, k.ret, k.exc)
        then = _Join(self, lambda w2, e2: self.stmts(node.body, w2, e2, inner))
        other = _Join(self, lambda w2, e2: self.stmts(node.orelse, w2, e2, inner))
        text = self.cond(
            node.test, w, env, lambda w2: then.stub(w2, env), lambda w2: other.stub(w2, env), k
        )
        return join.wrap(then.wrap(other.wrap(text)))

    def _s_with(self, node: ast.With, w: str, env: Env, k: Cont) -> str:
        if len(node.items) != 1:
            raise self.fail(node, "only single-item `with` is supported")
        item = node.items[0].context_expr
        text = ast.unparse(item)
        if self._passthrough(text):
            return self._with_body(node, w, env, k)

        func = ast.unparse(item.func) if isinstance(item, ast.Call) else ""
        if func not in _SUPPRESS_CTX or node.items[0].optional_vars:
            raise self.fail(
                node, "`with` supports contextlib.suppress and ignore=/ExitStack contexts"
            )
        caught = _handlers_match([ast.unparse(a).rsplit(".", 1)[-1] for a in item.args])
        join = _Join(self, k.nxt)

        def on_exc(w2: str, e: str, e2: Env) -> str:
            return (
                f"IF {e} \\in {_tla_set(caught)} THEN {join.stub(w2, e2)} ELSE {k.exc(w2, e, e2)}"
            )

        body = self._with_pending(
            lambda: self.stmts(node.body, w, env, Cont(join.stub, k.ret, self._share_exc(on_exc)))
        )
        return join.wrap(body)

    def _passthrough(self, text: str) -> bool:
        """A context whose enter/exit has no modeled effect: a lock, an ExitStack."""
        prefixes = (*self.spec.ignore, *_PASSTHROUGH_CTX)
        return any(text.startswith(p) for p in prefixes)

    def _with_body(self, node: ast.With, w: str, env: Env, k: Cont) -> str:
        var = node.items[0].optional_vars
        if var is not None and not isinstance(var, ast.Name):
            raise self.fail(node, "unsupported `with ... as` target")
        if var is not None:
            env = {**env, var.id: HAVOC}
        return self.stmts(node.body, w, env, k)

    def _s_for(self, node: ast.For, w: str, env: Env, k: Cont) -> str:
        """A loop with no modeled effect is skipped; anything else is rejected.

        Sound for the state the model tracks: the body may only call pure or
        ignored functions and never writes a modeled field or returns.
        """
        if node.orelse:
            raise self.fail(node, "for/else is unsupported")
        bound = {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        loop_env = {**env, **dict.fromkeys(bound, HAVOC)}
        for sub in ast.walk(node):
            self._check_inert(node, sub, loop_env)
        return k.nxt(w, loop_env)

    def _check_inert(self, loop: ast.For, sub: ast.AST, env: Env) -> None:
        if isinstance(sub, ast.Return | ast.Raise | ast.Yield):
            raise self.fail(loop, "loop that leaves the method is unsupported")
        targets = sub.targets if isinstance(sub, ast.Assign) else []
        targets = [sub.target] if isinstance(sub, ast.AugAssign | ast.AnnAssign) else targets
        if any(self._modeled_field(t) for t in targets):
            raise self.fail(loop, "loop writes a modeled field")
        if not isinstance(sub, ast.Call) or ast.unparse(sub.func) in _SUPPRESS_CTX:
            return
        if self.classify(sub, env) not in (CallKind.PURE, CallKind.IGNORE, CallKind.OPAQUE):
            raise self.fail(loop, "loop calls modeled I/O (bounded unrolling is not implemented)")

    def _share_exc(self, fn: ExcK) -> ExcK:
        """Exception continuation emitted once; raise sites call the stub."""
        holder = _Join(self, lambda w2, e2, ev: fn(w2, ev.tla or '"?"', e2), ("e",))
        self._pending_wraps.append(holder)
        return lambda w2, e, e2: holder.stub(w2, e2, Val(e, VKind.STR))

    def _s_try(self, node: ast.Try, w: str, env: Env, k: Cont) -> str:
        outer = self._finally_cont(node, env, k) if node.finalbody else (k, [])
        kf, wraps = outer

        def build() -> str:
            body_k = self._handler_cont(node, kf)
            if node.orelse:
                body_k = Cont(
                    lambda w2, e2: self.stmts(node.orelse, w2, e2, kf), body_k.ret, body_k.exc
                )
            return self.stmts(node.body, w, env, body_k)

        text = self._with_pending(build)
        for j in wraps:
            text = j.wrap(text)
        return text

    def _with_pending(self, build: Callable[[], str]) -> str:
        """Run *build*, then wrap its text in the joins it registered."""
        saved, self._pending_wraps = self._pending_wraps, []
        try:
            text = build()
            for j in reversed(self._pending_wraps):
                text = j.wrap(text)
        finally:
            self._pending_wraps = saved
        return text

    def _handler_cont(self, node: ast.Try, kf: Cont) -> Cont:
        if not node.handlers:
            return kf
        arms = []
        for h in node.handlers:
            names = self._handler_names(h)
            arms.append((h, _handlers_match(names)))

        def dispatch(w2: str, e: str, e2: Env) -> str:
            return self._dispatch(arms, w2, e, e2, kf)

        return Cont(kf.nxt, kf.ret, self._share_exc(dispatch))

    def _handler_names(self, h: ast.ExceptHandler) -> list[str]:
        if h.type is None:
            return ["BaseException"]
        elts = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        names = [ast.unparse(t).rsplit(".", 1)[-1] for t in elts]
        unknown = [n for n in names if n not in _PARENT]
        if unknown:
            raise self.fail(h, f"except of unmodeled class {unknown}")
        return names

    def _dispatch(self, arms, w: str, e: str, env: Env, kf: Cont) -> str:
        if not arms:
            return kf.exc(w, e, env)
        (h, caught), rest = arms[0], arms[1:]
        henv = {**env, _EXC: Val(e, VKind.STR)}
        if h.name:
            henv[h.name] = HAVOC
        here = self._with_pending(lambda: self.stmts(h.body, w, henv, kf))
        return f"IF {e} \\in {_tla_set(caught)} THEN\n{here}\nELSE {self._dispatch(rest, w, e, env, kf)}"

    def _finally_cont(self, node: ast.Try, env: Env, k: Cont) -> tuple[Cont, list[_Join]]:
        fin = node.finalbody

        def after(w2: str, e2: Env, then: NextK) -> str:
            return self._with_pending(lambda: self.stmts(fin, w2, e2, Cont(then, k.ret, k.exc)))

        nxt = _Join(self, lambda w2, e2: after(w2, e2, k.nxt))
        ret = _Join(self, lambda w2, e2, v: after(w2, e2, lambda w3, _e3: k.ret(w3, v)), ("v",))

        def on_exc(w2: str, e2: Env, ev: Val) -> str:
            e = ev.tla or '"?"'
            again = after(w2, e2, lambda w3, e3: k.exc(w3, e, e3))
            return (
                f"IF {e} \\in {{{', '.join(_UNCATCHABLE)}}} THEN {k.exc(w2, e, e2)} ELSE\n{again}"
            )

        exc = _Join(self, on_exc, ("e",))
        kf = Cont(
            nxt.stub,
            lambda w2, v: ret.stub(w2, env, v),
            lambda w2, e, e2: exc.stub(w2, e2, Val(e, VKind.STR)),
        )
        return kf, [exc, ret, nxt]

    # ── conditions ─────────────────────────────────────────────────

    def cond(self, node: ast.expr, w: str, env: Env, t: WorldK, f: WorldK, k: Cont) -> str:
        if isinstance(node, ast.BoolOp):
            return self._cond_boolop(node, node.values, w, env, t, f, k)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.cond(node.operand, w, env, f, t, k)

        def branch(w2: str, v: Val) -> str:
            truth = self.truth(v)
            if truth is None:
                self._note_havoc(node)
                return f"(\n{t(w2)}\n) \\/ (\n{f(w2)}\n)"
            return f"IF {truth} THEN\n{t(w2)}\nELSE\n{f(w2)}"

        return self.expr(node, w, env, branch, self._raise_to(k, env))

    def _cond_boolop(self, node, values, w, env, t, f, k) -> str:
        head, rest = values[0], values[1:]
        if not rest:
            return self.cond(head, w, env, t, f, k)
        tail = lambda w2: self._cond_boolop(node, rest, w2, env, t, f, k)  # noqa: E731
        if isinstance(node.op, ast.And):
            return self.cond(head, w, env, tail, f, k)
        return self.cond(head, w, env, t, tail, k)

    @staticmethod
    def truth(v: Val) -> str | None:
        if v.kind is VKind.BOOL:
            return v.tla
        if v.kind is VKind.HANDLE:
            return f"({v.tla} # {_NONE_HANDLE})"
        if v.kind is VKind.STR:
            return f"Truthy({v.tla})"
        if v.kind is VKind.NONE:
            return "FALSE"
        return None

    # ── expressions ────────────────────────────────────────────────

    def expr(self, node: ast.expr, w: str, env: Env, kv: ValK, kx: RaiseK) -> str:
        if isinstance(node, ast.Constant):
            return kv(w, self._const(node.value))
        if isinstance(node, ast.Name):
            return kv(w, self._name(node, env))
        if isinstance(node, ast.Attribute):
            return self._attr(node, w, env, kv, kx)
        if isinstance(node, ast.Call):
            return self.call(node, w, env, kv, kx)
        if isinstance(node, ast.Compare):
            return self._compare(node, w, env, kv, kx)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.expr(node.operand, w, env, lambda w2, v: kv(w2, self._not(v)), kx)
        if self._negative_int(node):
            return kv(w, self._const(-node.operand.value))
        self._assert_pure(node, env)
        return kv(w, Val(None, VKind.HAVOC, tags=self._tags(node, env)))

    @staticmethod
    def _negative_int(node: ast.expr) -> bool:
        """``-2`` parses as USub(Constant 2)."""
        is_neg = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
        return (
            is_neg
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
        )

    def _const(self, value: object) -> Val:
        if value is None:
            return NONE_VAL
        if isinstance(value, bool):
            return Val("TRUE" if value else "FALSE", VKind.BOOL)
        if isinstance(value, int):
            return Val(f'"{value}"', VKind.STR)
        if isinstance(value, str | bytes):
            return Val(None, VKind.HAVOC, tags=_first_word(value))
        return HAVOC

    def _name(self, node: ast.Name, env: Env) -> Val:
        if node.id in env:
            return env[node.id]
        if node.id == "self":
            raise self.fail(node, "bare self escapes the model")
        return HAVOC  # module constant

    def _not(self, v: Val) -> Val:
        truth = self.truth(v)
        return Val(f"~{truth}", VKind.BOOL) if truth is not None else HAVOC

    def _attr(self, node: ast.Attribute, w: str, env: Env, kv: ValK, kx: RaiseK) -> str:
        fld = self._modeled_field(node)
        if fld is not None:
            return kv(w, self._field_val(fld, w))
        if node.attr == "closed":
            # f.closed on a modeled file.
            def closed(w2: str, b: Val) -> str:
                if b.htype is not HType.FILE:
                    return kv(w2, HAVOC)
                return kv(w2, Val(f'({w2}.files[{b.tla}] = "closed")', VKind.BOOL))

            return self.expr(node.value, w, env, closed, kx)
        if node.attr == "pid":
            # proc.pid names the proc (and, after setsid, its group).
            def pid(w2: str, b: Val) -> str:
                return kv(w2, b if b.htype is HType.PROC else HAVOC)

            return self.expr(node.value, w, env, pid, kx)
        self._assert_pure(node.value, env)
        return kv(w, HAVOC)

    def _field_val(self, fld: str, w: str) -> Val:
        kind = self.spec.state[fld]
        ref = f"{w}.s.{self._fields[fld]}"
        if kind is tla.Kind.BOOL:
            return Val(ref, VKind.BOOL)
        if kind is tla.Kind.PROCSET:
            return Val(ref, VKind.HAVOC, HType.SET)
        return Val(ref, VKind.HANDLE, _FIELD_HTYPE[kind])

    def _compare(self, node: ast.Compare, w: str, env: Env, kv: ValK, kx: RaiseK) -> str:
        if len(node.ops) != 1:
            self._assert_pure(node, env)
            return kv(w, HAVOC)
        op = node.ops[0]

        def right(w2: str, a: Val) -> str:
            return self.expr(
                node.comparators[0], w2, env, lambda w3, b: kv(w3, self._cmp(op, a, b)), kx
            )

        return self.expr(node.left, w, env, right, kx)

    def _cmp(self, op: ast.cmpop, a: Val, b: Val) -> Val:
        if isinstance(op, ast.Is | ast.IsNot) and b.kind is VKind.NONE:
            res = self._is_none(a)
            if res is None:
                return HAVOC
            return Val(res if isinstance(op, ast.Is) else f"~{res}", VKind.BOOL)
        same = a.kind is b.kind and a.kind in (VKind.STR, VKind.BOOL, VKind.HANDLE)
        if isinstance(op, ast.Eq | ast.NotEq) and same:
            sym = "=" if isinstance(op, ast.Eq) else "#"
            return Val(f"({a.tla} {sym} {b.tla})", VKind.BOOL)
        return HAVOC

    @staticmethod
    def _is_none(a: Val) -> str | None:
        table = {
            VKind.HANDLE: f"({a.tla} = {_NONE_HANDLE})",
            VKind.STR: f"({a.tla} = {_NONE_STR})",
            VKind.NONE: "TRUE",
            VKind.BOOL: "FALSE",
        }
        return table.get(a.kind)

    def _tags(self, node: ast.AST, env: Env) -> frozenset[str]:
        """Literal request words a payload expression carries."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
            return _first_word(node.value)
        if (
            isinstance(node, ast.JoinedStr)
            and node.values
            and isinstance(node.values[0], ast.Constant)
        ):
            return _first_word(node.values[0].value)
        if isinstance(node, ast.BinOp):
            return self._tags(node.left, env) | self._tags(node.right, env)
        if isinstance(node, ast.Name):
            return env.get(node.id, HAVOC).tags
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return self._tags(node.func.value, env)
        return frozenset()

    # ── calls ──────────────────────────────────────────────────────

    def _chain(self, func: ast.expr) -> tuple[ast.expr, list[str]]:
        """``self._proc.stdin.write`` -> (Name self, ["_proc", "stdin", "write"])."""
        attrs = []
        while isinstance(func, ast.Attribute):
            attrs.append(func.attr)
            func = func.value
        return func, attrs[::-1]

    def classify(self, node: ast.Call, env: Env) -> CallKind:
        text = ast.unparse(node.func)
        if any(text.startswith(p) for p in self.spec.ignore):
            return CallKind.IGNORE
        if text in self.spec.model:
            return CallKind.MODEL
        root, attrs = self._chain(node.func)
        if isinstance(root, ast.Name) and root.id == "self":
            return self._classify_self(node, attrs)
        if isinstance(root, ast.Name):
            return self._classify_name(node, root, attrs, env)
        return CallKind.PURE  # method on a literal or a computed value

    def _classify_self(self, node: ast.Call, attrs: list[str]) -> CallKind:
        if len(attrs) == 1 and attrs[0] in self._methods:
            fn = getattr(self.cls, attrs[0])
            return CallKind.OPAQUE if hasattr(fn, tla.OPAQUE_ATTR) else CallKind.INLINE
        if attrs[0] in self._fields and self.spec.state[attrs[0]] in _FIELD_HTYPE:
            return CallKind.TYPED
        raise self.fail(node, "call on an unmodeled field (model it or add it to ignore=)")

    def _classify_name(
        self, node: ast.Call, root: ast.Name, attrs: list[str], env: Env
    ) -> CallKind:
        local = env.get(root.id)
        if local is not None:
            return CallKind.TYPED if local.kind is VKind.HANDLE and attrs else CallKind.PURE
        if root.id in self._globals and root.id not in vars(builtins):
            raise self.fail(node, "unmodeled call (add it to model= or ignore=)")
        return CallKind.PURE

    def _assert_pure(self, node: ast.AST, env: Env) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and self.classify(sub, env) not in (
                CallKind.PURE,
                CallKind.IGNORE,
            ):
                raise self.fail(sub, "effectful call nested in an expression")

    def call(self, node: ast.Call, w: str, env: Env, kv: ValK, kx: RaiseK) -> str:
        kind = self.classify(node, env)
        if kind is CallKind.IGNORE:
            return kv(w, HAVOC)
        if kind is CallKind.PURE:
            self._assert_pure(node, env)
            return kv(w, Val(None, VKind.HAVOC, tags=self._tags(node, env)))
        handler = {
            CallKind.INLINE: self._inline,
            CallKind.OPAQUE: self._opaque,
            CallKind.MODEL: self._model_call,
            CallKind.TYPED: self._typed_call,
        }[kind]
        return handler(node, w, env, kv, kx)

    def _args(
        self, nodes: list[ast.expr], w: str, env: Env, then: Callable[[str, list[Val]], str], kx
    ) -> str:
        """Evaluate *nodes* left to right, then call *then* with their values."""

        def step(i: int, w2: str, acc: list[Val]) -> str:
            if i == len(nodes):
                return then(w2, acc)
            return self.expr(nodes[i], w2, env, lambda w3, v: step(i + 1, w3, acc + [v]), kx)

        return step(0, w, [])

    def _apply(
        self, op: str, w: str, args: list[str], kind: VKind, htype: HType | None, kv: ValK, kx
    ) -> str:
        w2, v2, w3, e3 = self.fresh("w"), self.fresh("v"), self.fresh("w"), self.fresh("e")
        lead = ", ".join([w, *args])
        ok = kv(w2, Val(v2, kind, htype))
        return f"{op}({lead},\n LAMBDA {w2}, {v2}:\n{ok},\n LAMBDA {w3}, {e3}: {kx(w3, e3)})"

    def _handle_arg(self, node: ast.AST, v: Val) -> str:
        if v.kind is VKind.HANDLE:
            return v.tla or _NONE_HANDLE
        if v.kind is VKind.NONE:
            return _NONE_HANDLE
        raise self.fail(node, "argument must be a modeled handle")

    def _model_call(self, node: ast.Call, w: str, env: Env, kv: ValK, kx) -> str:
        op = self.spec.model[ast.unparse(node.func)]
        if op is tla.Op.WRAP:
            return self._args(
                list(node.args), w, env, lambda w2, vals: kv(w2, self._wrapped(node, vals)), kx
            )
        kind, htype = _MODEL_RESULT[op]
        n_args = _MODEL_ARGS[op]

        def then(w2: str, vals: list[Val]) -> str:
            handles = [self._handle_arg(node, v) for v in vals[:n_args]]
            return self._apply(op.value, w2, handles, kind, htype, kv, kx)

        return self._args(list(node.args) + [kw.value for kw in node.keywords], w, env, then, kx)

    def _wrapped(self, node: ast.Call, vals: list[Val]) -> Val:
        """Op.WRAP: the record stands for the first proc among its arguments."""
        procs = [v for v in vals if v.htype is HType.PROC]
        if not procs:
            raise self.fail(node, "Op.WRAP needs a proc argument")
        return procs[0]

    def _hold(self, node: ast.Call, r: Val, w: str, env: Env, kv: ValK, kx) -> str:
        """``self.<procset>.append(proc)``: the object now owns (and will reap) proc."""
        if ast.unparse(node.func).rsplit(".", 1)[-1] != "append" or len(node.args) != 1:
            raise self.fail(node, "a PROCSET field supports only .append(proc)")

        def then(w2: str, vals: list[Val]) -> str:
            h = self._handle_arg(node, vals[0])
            field = r.tla.split(".s.", 1)[1]
            return self._let(
                f"[{w2} EXCEPT !.s.{field} = @ \\cup {{{h}}}]", lambda w3: kv(w3, NONE_VAL)
            )

        return self._args(list(node.args), w, env, then, kx)

    def _typed_call(self, node: ast.Call, w: str, env: Env, kv: ValK, kx) -> str:
        root, attrs = self._chain(node.func)
        recv = (
            ast.Attribute(value=root, attr=attrs[0], ctx=ast.Load()) if root.id == "self" else root
        )
        path = ".".join(attrs[1:] if root.id == "self" else attrs)

        def with_recv(w2: str, r: Val) -> str:
            if r.htype is HType.SET:
                return self._hold(node, r, w2, env, kv, kx)
            entry = _TYPED_OPS.get((r.htype, path))
            if entry is None:
                raise self.fail(node, f"no Syscalls.tla model for {r.htype} .{path}()")
            op, kind, htype = entry
            op = self._variant(op, node)

            def then(w3: str, vals: list[Val]) -> str:
                extra = [str(self._replies_in(vals))] if op == "Write" else []
                return self._apply(op, w3, [self._handle_arg(node, r), *extra], kind, htype, kv, kx)

            return self._args(
                list(node.args) + [kw.value for kw in node.keywords], w2, env, then, kx
            )

        return self.expr(recv, w, env, with_recv, kx)

    @staticmethod
    def _variant(op: str, node: ast.Call) -> str:
        if op == "Wait" and (node.args or any(kw.arg == "timeout" for kw in node.keywords)):
            return "WaitT"
        if op == "ReadLineT" and node.args and ast.unparse(node.args[0]) == _NO_DEADLINE:
            return "ReadLineBlock"
        return op

    def _replies_in(self, vals: list[Val]) -> int:
        tags = frozenset().union(*(v.tags for v in vals)) if vals else frozenset()
        return int(bool(tags & set(self.spec.replies)))

    def _opaque(self, node: ast.Call, w: str, env: Env, kv: ValK, kx) -> str:
        name = self._chain(node.func)[1][0]
        fn = self._methods[name]
        if any(
            self._modeled_field(t)
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            for t in n.targets
        ):
            raise self.fail(node, f"@tla.opaque {name} writes a modeled field")
        values = getattr(self.cls, name).__tla_opaque__
        choices = "{" + ", ".join(f'"{v}"' for v in values) + "}"
        v = self.fresh("v")

        def then(w2: str, _vals: list[Val]) -> str:
            return f"\\E {v} \\in {choices} :\n{kv(w2, Val(v, VKind.STR))}"

        return self._args(list(node.args), w, env, then, kx)

    def _inline(self, node: ast.Call, w: str, env: Env, kv: ValK, kx) -> str:
        name = self._chain(node.func)[1][0]
        if self._stack.count(name) >= _MAX_INLINE:
            return kx(w, '"Overflow"')
        fn = self._methods[name]

        def then(w2: str, vals: list[Val]) -> str:
            callee_env = self._bind_params(fn, node, vals)
            return self.body(fn, w2, callee_env, kv, lambda w3, e, _e: kx(w3, e))

        return self._args(list(node.args) + [kw.value for kw in node.keywords], w, env, then, kx)

    def _bind_params(self, fn: ast.FunctionDef, node: ast.Call, vals: list[Val]) -> Env:
        params = [a.arg for a in fn.args.args[1:]]
        n_pos = len(node.args)
        env = dict(zip(params, vals[:n_pos], strict=False))
        for kw, v in zip(node.keywords, vals[n_pos:], strict=False):
            env[kw.arg or ""] = v
        defaults = fn.args.defaults
        for p, d in zip(params[len(params) - len(defaults) :], defaults, strict=False):
            env.setdefault(p, self._const(d.value) if isinstance(d, ast.Constant) else HAVOC)
        return {p: env.get(p, HAVOC) for p in params} | {
            a.arg: env.get(a.arg, HAVOC) for a in fn.args.kwonlyargs
        }

    def body(self, fn: ast.FunctionDef, w: str, env: Env, kv: ValK, kexc: ExcK) -> str:
        """Compile *fn*'s body in a fresh frame; returns go to *kv*."""
        self._stack.append(fn.name)
        self._frames.append(self._locals_of(fn))
        saved, self._pending_wraps = self._pending_wraps, []
        try:
            ret = _Join(self, lambda w2, _e, v: kv(w2, v), ("v",), names=[])
            k = Cont(
                lambda w2, _e: ret.stub(w2, {}, NONE_VAL), lambda w2, v: ret.stub(w2, {}, v), kexc
            )
            text = self.stmts(fn.body, w, env, k)
            for j in reversed(self._pending_wraps):
                text = j.wrap(text)
            return ret.wrap(text)
        finally:
            self._pending_wraps = saved
            self._frames.pop()
            self._stack.pop()

    # ── invariants (pure) ──────────────────────────────────────────

    def pure(self, src: str) -> str:
        """Translate an invariant/guard expression over ``self`` fields."""
        node = ast.parse(src, mode="eval").body
        return self._pure(node)

    def _pure(self, node: ast.expr) -> str:
        if isinstance(node, ast.BoolOp):
            sym = " /\\ " if isinstance(node.op, ast.And) else " \\/ "
            return "(" + sym.join(self._pure(v) for v in node.values) + ")"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"~{self._pure(node.operand)}"
        val = self._pure_val(node)
        truth = self.truth(val)
        if truth is None:
            raise ExtractError(f"invariant/guard uses unmodeled state: {ast.unparse(node)}")
        return truth

    def _pure_val(self, node: ast.expr) -> Val:
        if isinstance(node, ast.Constant):
            return self._const(node.value)
        fld = self._modeled_field(node)
        if fld is not None:
            return self._field_val(fld, "w")
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return self._cmp(
                node.ops[0], self._pure_val(node.left), self._pure_val(node.comparators[0])
            )
        poll = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "poll"
        )
        if poll:
            recv = self._pure_val(node.func.value)
            if recv.htype is HType.PROC:
                return Val(f"PollValue(w, {recv.tla})", VKind.STR)
        raise ExtractError(f"invariant/guard uses unmodeled state: {ast.unparse(node)}")


# ── module assembly ────────────────────────────────────────────────


class Emitter:
    """Assemble the TLA+ module from compiled entries."""

    def __init__(self, comp: Compiler):
        self.comp = comp
        self.spec = comp.spec
        self.cls = comp.cls

    def _entries(self) -> list[tuple[str, tla.EntrySpec]]:
        out = []
        for name in self.comp._methods:
            es = getattr(getattr(self.cls, name, None), tla.ENTRY_ATTR, None)
            if es is not None:
                out.append((name, es))
        return out

    _DEFAULT = {tla.Kind.BOOL: "FALSE", tla.Kind.PROCSET: "{}"}

    def _initial(self) -> str:
        fields = ", ".join(
            f"{n} |-> {self._DEFAULT.get(self.spec.state[f], _NONE_HANDLE)}"
            for f, n in self.comp._fields.items()
        )
        return (
            f"W0 == [s |-> [{fields}],\n"
            '        procs |-> [p \\in 1..MaxProcs |-> "free"], pend |-> [p \\in 1..MaxProcs |-> 0],\n'
            "        stale |-> [p \\in 1..MaxProcs |-> FALSE],\n"
            '        files |-> [f \\in 1..MaxProcs |-> "free"], nfile |-> 0,\n'
            '        closed |-> {}, nproc |-> 0, ret |-> "", calls |-> 0, last |-> "",\n'
            '        hung |-> FALSE, overflow |-> FALSE, uncaught |-> "", desync |-> FALSE,\n'
            '        setup |-> ""]'
        )

    @staticmethod
    def _render(v: Val) -> str:
        if v.kind is VKind.STR:
            return v.tla or '"?"'
        if v.kind in (VKind.BOOL, VKind.HANDLE):
            return f"ToString({v.tla})"
        return _NONE_STR if v.kind is VKind.NONE else '"?"'

    def _done(self, name: str) -> RetK:
        def done(w: str, v: Val) -> str:
            return (
                f'w\' = [{w} EXCEPT !.ret = {self._render(v)}, !.calls = @ + 1, !.last = "{name}"]'
            )

        return done

    def _crash(self, name: str, es: tla.EntrySpec) -> ExcK:
        # Uncatchable pseudo-exceptions and the ones the caller handles are not NoUncaught.
        quiet = ", ".join(
            [*_UNCATCHABLE, *(f'"{c}"' for c in sorted(_handlers_match(list(es.raises))))]
        )

        def crash(w: str, e: str, _env: Env) -> str:
            return (
                f'w\' = [{w} EXCEPT !.ret = {e}, !.calls = @ + 1, !.last = "{name}",\n'
                f'   !.hung = @ \\/ {e} = "Hang", !.overflow = @ \\/ {e} = "Overflow",\n'
                f"   !.uncaught = IF {e} \\in {{{quiet}}} THEN @ ELSE {e}]"
            )

        return crash

    def _setup(self, setup: str | None) -> str:
        """Init: __init__ then the SETUP entry. A raise or a falsy return is kept as
        a failed setup (w.setup), which the caller drops: SetupCleanup checks it
        left nothing running or open."""
        init_fn = self.comp._methods.get("__init__")

        def failed(w: str, e: str, _env: Env | None = None) -> str:
            return f'w = [{w} EXCEPT !.setup = {e}, !.hung = {e} = "Hang"]'

        def keep(w: str, v: Val) -> str:
            truth = self.comp.truth(v)
            if truth is None:
                return f"w = {w}"
            return f"IF {truth} THEN w = {w} ELSE {failed(w, chr(34) + 'False' + chr(34))}"

        def run_start(w: str, _v: Val) -> str:
            if setup is None:
                return f"w = {w}"
            fn = self.comp._methods[setup]
            return self.comp.body(fn, w, self._havoc_params(fn), keep, failed)

        if init_fn is None:
            return run_start("W0", NONE_VAL)
        return self.comp.body(init_fn, "W0", self._havoc_params(init_fn), run_start, failed)

    @staticmethod
    def _havoc_params(fn: ast.FunctionDef) -> Env:
        return {a.arg: HAVOC for a in fn.args.args[1:] + fn.args.kwonlyargs}

    def _call(self, name: str, es: tla.EntrySpec) -> str:
        fn = self.comp._methods[name]
        guard = f" /\\ {self.comp.pure(es.requires)}" if es.requires else ""
        body = self.comp.body(
            fn, "w", self._havoc_params(fn), self._done(name), self._crash(name, es)
        )
        live = '~w.hung /\\ w.setup = "" /\\ w.calls < MaxCalls'
        return f"Call_{name} ==\n  {live}{guard} /\\ (\n{body}\n)\n"

    def _held(self) -> str:
        """Proc ids the object can still reach: PROC fields plus PROCSET members."""
        kinds = self.spec.state
        procs = [f"w.s.{n}" for f, n in self.comp._fields.items() if kinds[f] is tla.Kind.PROC]
        sets = [f"w.s.{n}" for f, n in self.comp._fields.items() if kinds[f] is tla.Kind.PROCSET]
        return " \\cup ".join(["{" + ", ".join(procs) + "}", *sets])

    def _invariants(self) -> list[str]:
        unreaped = '{"live", "dead"}'
        out = [
            # A running or zombie proc nobody holds is never reaped.
            f"NoOrphan == \\A p \\in 1..MaxProcs : w.procs[p] \\in {unreaped} => p \\in {self._held()}",
            "NoHang == ~w.hung",
            'NoUncaught == w.uncaught = ""',
            "RecursionBound == ~w.overflow",
            "NoDesync == ~w.desync",
            # A failed setup is dropped by its caller: it must leave nothing behind.
            'SetupCleanup == w.setup # "" => '
            f"(\\A p \\in 1..MaxProcs : w.procs[p] \\notin {unreaped}) /\\ "
            '(\\A f \\in 1..MaxProcs : w.files[f] # "open")',
        ]
        # A hung call is blocked mid-method; NoHang reports it, field invariants skip it.
        out += [
            f"{n} == ~w.hung => {self.comp.pure(src)}" for n, src in self.spec.invariants.items()
        ]
        return out

    def emit(self) -> Model:
        entries = self._entries()
        setups = [n for n, es in entries if es.phase is tla.Phase.SETUP]
        if len(setups) > 1:
            raise ExtractError(
                f"{self.cls.__qualname__}: at most one Phase.SETUP entry, got {setups}"
            )
        calls = [(n, es) for n, es in entries if es.phase is tla.Phase.CALL]
        name = self.cls.__name__
        init = self._setup(setups[0] if setups else None)
        actions = [self._call(n, es) for n, es in calls]
        invariants = self._invariants()
        nxt = " \\/ ".join([f"Call_{n}" for n, _ in calls] + ["Die"])
        havoc = "\n".join(f"\\*   {h}" for h in self.comp.havoc) or "\\*   none"
        text = "\n".join(
            [
                f"---- MODULE {name} ----",
                f"\\* Generated by tools/tla_extract.py from {self.cls.__module__}.{name}. Do not edit.",
                "\\* Conditions over unmodeled values, explored both ways:",
                havoc,
                "EXTENDS Naturals, TLC, Syscalls",
                "CONSTANT MaxCalls",
                "VARIABLE w",
                self._initial(),
                f"Init == {init}",
                *actions,
                'Die == ~w.hung /\\ \\E w2 \\in Died(w) : w\' = [w2 EXCEPT !.last = "<loader died>"]',
                f"Next == {nxt}",
                *invariants,
                "====",
            ]
        )
        inv_names = [i.split(" ==", 1)[0] for i in invariants]
        return Model(name, text, inv_names, list(self.comp.havoc))


def extract(cls: type) -> Model:
    """Compile *cls* into a TLA+ model.

    CPS recurses once per statement along every inlined call chain, so the
    default limit (1000) is exhausted by a few nested methods.
    """
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, _RECURSION_LIMIT))
    try:
        return Emitter(Compiler(cls)).emit()
    finally:
        sys.setrecursionlimit(limit)


# ── TLC ────────────────────────────────────────────────────────────


def find_jar() -> Path | None:
    """tla2tools.jar from $FUZZ_TLA_JAR or $FUZZ_VENDOR_ROOT/tla/ (tools/vendor_tla.sh)."""
    env = os.environ.get("FUZZ_TLA_JAR")
    root = Path(os.environ.get("FUZZ_VENDOR_ROOT", Path.home() / "fuzzing" / "vendoring"))
    for cand in (Path(env) if env else None, root / "tla" / _JAR_NAME):
        if cand is not None and cand.is_file() and shutil.which("java"):
            return cand
    return None


def write(model: Model, out: Path, bounds: Bounds) -> None:
    """Write the module, Syscalls.tla and one cfg per invariant into *out*."""
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model.name}.tla").write_text(model.tla)
    shutil.copy(_SYSCALLS, out / _SYSCALLS.name)
    for inv in model.invariants:
        (out / f"{model.name}_{inv}.cfg").write_text(model.cfg(inv, bounds))


_STATES_RE = re.compile(r"(\d+) distinct states found")


def _run_tlc(model: Model, out: Path, jar: Path, inv: str) -> tuple[str, str]:
    cmd = [
        "java", "-XX:+UseParallelGC", "-cp", str(jar), "tlc2.TLC",
        "-deadlock", "-workers", "1", "-cleanup",
        "-metadir", str(out / "states"),
        "-config", f"{model.name}_{inv}.cfg", model.name,
    ]  # fmt: skip
    proc = subprocess.run(
        cmd, cwd=out, capture_output=True, text=True, timeout=_TLC_TIMEOUT, check=False
    )
    return proc.stdout, proc.stderr


def check(model: Model, out: Path, jar: Path, bounds: Bounds) -> CheckResult:
    """Run TLC once per invariant; collect each violated one's trace."""
    write(model, out, bounds)
    violations, states, completed = {}, 0, True
    for inv in model.invariants:
        log, err = _run_tlc(model, out, jar, inv)
        found = _STATES_RE.findall(log)
        states = max(states, int(found[-1])) if found else states
        if f"Invariant {inv} is violated" in log:
            trace = log.split("The behavior up to this point is:", 1)[-1]
            violations[inv] = re.split(r"\n\d+ states generated", trace, maxsplit=1)[0]
            continue
        if "Model checking completed. No error has been found." not in log:
            completed = False
            violations.setdefault("TLC_ERROR", (log + err)[-4000:])
    return CheckResult(completed, violations, states)


def _summarize(trace: str) -> str:
    """One line per state: the step taken and what it changed."""
    lines = []
    for chunk in re.split(r"\nState \d+: ", trace):
        flat = " ".join(chunk.split())
        if "w = " in flat:
            lines.append("  " + flat[flat.find("w = ") + 4 :][:400])
    return "\n".join(lines)


def _mark_push(cls: type) -> None:
    """PushNotifier's spec, attached here: push.py also runs as ``python -I push.py``
    (the sender child), where the package is not importable, so it cannot
    import fuzzer_tool.core.tla itself."""
    tla.machine(
        state={"_senders": tla.Kind.PROCSET},
        model={"subprocess.Popen": tla.Op.SPAWN, "_Sender": tla.Op.WRAP},
        ignore=("time.monotonic", "os.path.", "replace", "_encode"),
    )(cls)
    tla.entry(raises=("OSError",))(cls.notify)  # corpus_manager._push_crash catches OSError
    tla.entry()(cls.close)
    tla.opaque(None)(cls._report)


# Machines whose module cannot import fuzzer_tool.core.tla: name -> spec attacher.
_EXTERNAL_SPECS: dict[str, Callable[[type], None]] = {"push": _mark_push}


def _resolve(target: str) -> type:
    spec = MACHINES.get(target, target)
    if ":" not in spec:
        raise ExtractError(f"unknown machine {target!r}; known: {sorted(MACHINES)} or module:Class")
    mod, _, name = spec.partition(":")
    cls = getattr(importlib.import_module(mod), name)
    attach = _EXTERNAL_SPECS.get(target)
    if attach is not None and not hasattr(cls, tla.MACHINE_ATTR):
        attach(cls)
    return cls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("machine", help=f"{sorted(MACHINES)} or module:Class")
    ap.add_argument("--out", type=Path, help="output dir (default ~/fuzzing/tla/<machine>/)")
    ap.add_argument("--check", action="store_true", help="run TLC on every invariant")
    ap.add_argument("--max-procs", type=int, default=Bounds.max_procs)
    ap.add_argument("--max-calls", type=int, default=Bounds.max_calls)
    ap.add_argument("--jar", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.out or Path.home() / "fuzzing" / "tla" / args.machine.replace(":", "_")
    bounds = Bounds(args.max_procs, args.max_calls)
    try:
        model = extract(_resolve(args.machine))
    except ExtractError as e:
        print(f"[!] {e}", file=sys.stderr)
        return 2
    for h in model.havoc:
        print(f"[havoc] {h}")
    if not args.check:
        write(model, out, bounds)
        print(f"[*] wrote {out}/{model.name}.tla")
        return 0

    jar = args.jar or find_jar()
    if jar is None:
        print("[!] tla2tools.jar or java missing: run tools/vendor_tla.sh", file=sys.stderr)
        return 2
    res = check(model, out, jar, bounds)
    print(
        f"[*] {model.name}: {res.states} states, bounds procs={bounds.max_procs} calls={bounds.max_calls}"
    )
    for inv in model.invariants:
        print(f"  {'FAIL' if inv in res.violations else 'ok  '} {inv}")
    for inv, trace in res.violations.items():
        print(f"\n[{inv}] counterexample:\n{_summarize(trace) or trace}")
    return 1 if res.violations else 0


if __name__ == "__main__":
    sys.exit(main())
