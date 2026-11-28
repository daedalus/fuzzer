"""Thread backend for tools/tla_extract.py: classes with LOCK / COND / THREAD fields.

The CPS compiler turns a whole call into one TLA+ action, so a thread cannot
stop at a lock and resume later. Here every statement is a labeled step:

    method ─> blocks of statements ─> Step_<label>(t)   (one atomic step each)
                                        │
             blocking steps are disabled until they can proceed:
             `with lock` (lock free), Condition.wait (notified, lock free),
             Thread.join (thread done)

Python switches threads between bytecodes; statement granularity coarsens that
but keeps every lock-ordering interleaving. Expressions inside a statement are
compiled by the CPS compiler. Per thread t the state holds:

    pc[t]      label; "idle" (main, between calls), "new" / "done" (workers), "dead"
    loc[t]     "<method>.<local>" -> value (modeled locals only)
    tret[t][m] where m returns: a caller label, "entry", "init" or "thread"
    rv[t][m]   m's return value          rstat[t] / exc[t]  raise across a call
    fin[t][F]  pending completion (next / break / continue / return / raise) of finally F

Thread 1 is the caller (entries); threads 2.. are one per Thread(target=self.m).
Extra invariants: NoDeadlock (the caller is mid-call and no thread can step),
NoThreadCrash (a worker died of an exception), NoHeldLock (a call returned holding a lock).
"""

import ast
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from fuzzer_tool.core import tla

MAIN = 1
_LOCK_CTORS = ("threading.Lock", "threading.RLock", "threading.Condition")
_QUEUE_CTORS = ("collections.deque", "deque")
_THREAD_CTOR = "threading.Thread"
_IDLE, _NEW, _DONE, _DEAD = '"idle"', '"new"', '"done"', '"dead"'

# A jump: (world, env, extra EXCEPT clauses) -> TLA text ending the step.
Jump = Callable[[str, dict, list[str]], str]


class ScopeKind(Enum):
    """Static context of a statement, innermost last."""

    LOOP = "loop"
    TRY_BODY = "try_body"  # handlers and finally apply
    TRY_REST = "try_rest"  # in a handler / else: only the finally applies
    LOCKED = "locked"  # inside `with <lock>`: leaving releases it


class Xfer(Enum):
    """Control transfers that may cross finally blocks and locks."""

    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"
    RAISE = "raise"


@dataclass(frozen=True)
class Scope:
    kind: ScopeKind
    head: str = ""  # LOOP: `continue` label
    after: Jump | None = None  # LOOP: `break` target
    handlers: tuple = ()  # TRY_BODY: ((caught names, Jump), ...)
    fin: str = ""  # TRY_*: finally entry label, "" if none
    lock: str = ""  # LOCKED: lock name


@dataclass
class _Frame:
    """What is being compiled: the method and its static scopes."""

    method: str
    ctx: tuple[Scope, ...] = field(default_factory=tuple)


class ThreadModel:
    """Labeled-step (PlusCal-style) compilation of a thread-using class."""

    def __init__(self, emitter) -> None:
        self.em = emitter
        self.comp = emitter.comp
        self.spec = self.comp.spec
        self.x = sys.modules[type(self.comp).__module__]  # tla_extract, as loaded
        self.methods = self.comp._methods
        self.locks = [f for f, k in self.spec.state.items() if k is tla.Kind.LOCK]
        self.conds = self._cond_locks()
        self.comp.cond_lock = self.conds
        self.targets = self._thread_targets()
        self.nt = MAIN + len(self.targets)
        self.entries = dict(self.em._entries())
        self.init = "__init__" in self.methods and self.spec.init is tla.Init.RUN
        self.procs = self._procedures()
        self.lkind: dict[tuple[str, str], tuple | None] = {}
        # Pinned __init__ arguments start in their loc slots (tla.machine init_args).
        self.pinned = {
            ("__init__", k): self.comp.arg_val(v) for k, v in self.spec.init_args.items()
        }
        for key, v in self.pinned.items():
            self.lkind[key] = (v.kind, v.htype)
        self.rkind: dict[str, tuple | None] = {}
        self.steps: list[str] = []
        self.fins: list[str] = []
        self._n = 0

    # ── analysis ───────────────────────────────────────────────────

    def _cond_locks(self) -> dict[str, str]:
        """COND field -> the LOCK it wraps (``Condition(self._lock)``), else itself."""
        out = {f: f for f, k in self.spec.state.items() if k is tla.Kind.COND}
        for node in ast.walk(self.methods.get("__init__", ast.Pass())):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            fld = self.comp._modeled_field(node.targets[0])
            arg = node.value.args[0] if node.value.args else None
            lock = self.comp._modeled_field(arg) if arg is not None else None
            if fld in out and lock:
                out[fld] = lock
        for c, lock in out.items():
            if lock == c:
                self.locks.append(c)  # a Condition() with its own lock
        return out

    def _thread_targets(self) -> dict[str, int]:
        """Thread(target=self.m) sites -> thread id per target method."""
        out: dict[str, int] = {}
        for fn in self.methods.values():
            for node in ast.walk(fn):
                target = self._thread_target(node)
                if target and target not in out:
                    out[target] = MAIN + 1 + len(out)
        return out

    @staticmethod
    def _thread_target(node: ast.AST) -> str | None:
        if not (isinstance(node, ast.Call) and ast.unparse(node.func) == _THREAD_CTOR):
            return None
        for kw in node.keywords:
            v = kw.value
            if (
                kw.arg == "target"
                and isinstance(v, ast.Attribute)
                and isinstance(v.value, ast.Name)
            ):
                return v.attr
        return None

    def _sync_construct(self, node: ast.AST) -> bool:
        """with lock/cond, .wait/.join/.start, Thread(): needs labeled steps."""
        if isinstance(node, ast.With):
            return any(self._lock_of(i.context_expr) for i in node.items)
        if not isinstance(node, ast.Call):
            return False
        if ast.unparse(node.func) == _THREAD_CTOR:
            return True
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in ("wait", "join", "start")):
            return False
        kind = self.spec.state.get(self.comp._modeled_field(fn.value) or "")
        return kind in (tla.Kind.COND, tla.Kind.THREAD)

    def _lock_of(self, expr: ast.AST) -> str | None:
        fld = self.comp._modeled_field(expr)
        kind = self.spec.state.get(fld or "")
        if kind is tla.Kind.LOCK:
            return fld
        if kind is tla.Kind.COND:
            return self.conds[fld]
        return None

    @staticmethod
    def _self_method(node: ast.AST) -> str | None:
        """``self.<name>(...)`` -> name, else None."""
        f = getattr(node, "func", None)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
            return f.attr
        return None

    def _self_calls(self, fn: ast.FunctionDef) -> set[str]:
        return {m for node in ast.walk(fn) if (m := self._self_method(node)) in self.methods}

    def _procedures(self) -> set[str]:
        """Methods compiled as labeled steps: entries, thread targets, __init__, and any
        method using a sync construct or calling one that does (fixed point)."""
        roots = set(self.entries) | set(self.targets) | ({"__init__"} if self.init else set())
        reach = set(roots)
        while True:  # methods the roots can call
            more = set().union(*(self._self_calls(self.methods[m]) for m in reach)) - reach
            if not more:
                break
            reach |= more
        procs = roots | {
            m for m in reach if any(map(self._sync_construct, ast.walk(self.methods[m])))
        }
        while True:
            more = {m for m in reach if self._self_calls(self.methods[m]) & procs} - procs
            if not more:
                break
            procs |= more
        procs -= {m for m in procs if hasattr(getattr(self.comp.cls, m, None), tla.OPAQUE_ATTR)}
        self._no_recursion(procs)
        return procs

    def _no_recursion(self, procs: set[str]) -> None:
        def visit(m: str, path: tuple[str, ...]) -> None:
            for callee in self._self_calls(self.methods[m]) & procs:
                if callee in path:
                    raise self.x.ExtractError(
                        f"recursive thread-model call {' -> '.join((*path, callee))}"
                    )
                visit(callee, (*path, callee))

        for m in procs:
            visit(m, (m,))

    # ── labels, locals, jumps ──────────────────────────────────────

    def _label(self, m: str) -> str:
        self._n += 1
        return f"{m.strip('_') or 'm'}_{self._n}"

    @staticmethod
    def _entry(m: str) -> str:
        return f"E_{m.strip('_') or 'm'}"

    def _slot(self, m: str, name: str) -> str:
        return f'"{m}.{name}"'

    def _env(self, m: str, w: str) -> dict:
        """Locals of m restored from the state: those with one storable kind."""
        env = {}
        for name in self.comp._locals_of(self.methods[m]):
            kind = self.lkind.get((m, name))
            if kind:
                env[name] = self.x.Val(f"{w}.loc[t][{self._slot(m, name)}]", kind[0], kind[1])
        return env

    def _storable(self, v) -> bool:
        return v.tla is not None and v.kind in (
            self.x.VKind.BOOL,
            self.x.VKind.HANDLE,
            self.x.VKind.STR,
        )

    def _note(self, table: dict, key, v) -> None:
        """Record a slot's kind (first pass); a conflict makes it unstorable."""
        kind = (v.kind, v.htype)
        if key in table and table[key] != kind:
            table[key] = None
        else:
            table.setdefault(key, kind)

    def _stores(self, m: str, env: dict) -> list[str]:
        out = []
        for name, v in env.items():
            if not self._storable(v) or name.startswith("__") and name != self.x._EXC:
                continue
            self._note(self.lkind, (m, name), v)
            if self.lkind.get((m, name)) == (v.kind, v.htype):
                out.append(f"!.loc[t][{self._slot(m, name)}] = {v.tla}")
        return out

    def _write(self, fr: _Frame, w: str, env: dict, pc: str, extra: list[str]) -> str:
        clauses = [f"!.pc[t] = {pc}", *self._stores(fr.method, env), *extra]
        return f"w' = [{w} EXCEPT {', '.join(clauses)}]"

    def _goto(self, fr: _Frame, label: str) -> Jump:
        return lambda w, env, extra: self._write(fr, w, env, f'"{label}"', extra)

    def _step(self, label: str, body: str) -> None:
        self.steps.append(
            f'Step_{label}(t) ==\n  ~w.bound /\\ w.pc[t] = "{label}" /\\ (\n{body}\n)\n'
        )

    # ── control transfers ──────────────────────────────────────────

    def _xfer(self, fr: _Frame, kind: Xfer, w: str, env: dict, extra: list[str], **kw) -> str:
        """Leave the current position: through locks (released) and finally blocks.

        A "Bound" raise (a model bound was hit) freezes the path instead (w.bound).
        """
        if kind is Xfer.RAISE and not kw.get("bounded"):
            rest = self._xfer(fr, kind, w, env, extra, bounded=True, **kw)
            return f'IF {kw["exc"]} = "Bound" THEN w\' = [{w} EXCEPT !.bound = TRUE] ELSE\n{rest}'
        kw.pop("bounded", None)
        if kind is Xfer.RETURN:
            extra = [*extra, *self._store_rv(fr, kw.get("val"))]
        for i in range(len(fr.ctx) - 1, -1, -1):
            sc = fr.ctx[i]
            if sc.kind is ScopeKind.LOCKED:
                extra = [*extra, f'!.lock["{sc.lock}"] = 0']
                continue
            text = self._cross(
                fr, sc, _Frame(fr.method, fr.ctx[:i]), kind, w, env, extra, kw.get("exc")
            )
            if text is not None:
                return text
        if kind in (Xfer.BREAK, Xfer.CONTINUE):
            raise self.x.ExtractError(f"{fr.method}: {kind.value} outside a loop")
        return self._boundary(fr, kind, w, env, extra, kw.get("exc"))

    def _store_rv(self, fr: _Frame, val) -> list[str]:
        """Save a return value now: a finally block may run before the frame pops."""
        if val is None or not self._storable(val):
            return []
        self._note(self.rkind, fr.method, val)
        if self.rkind.get(fr.method) != (val.kind, val.htype):
            return []
        return [f'!.rv[t]["{fr.method}"] = {val.tla}']

    def _cross(self, fr, sc: Scope, outer: _Frame, kind: Xfer, w, env, extra, exc) -> str | None:
        """The text where *kind* stops at scope *sc*, or None to keep unwinding."""
        if sc.kind is ScopeKind.LOOP and kind in (Xfer.BREAK, Xfer.CONTINUE):
            jump = sc.after if kind is Xfer.BREAK else self._goto(fr, sc.head)
            return jump(w, env, extra)
        if sc.kind is ScopeKind.TRY_BODY and kind is Xfer.RAISE and sc.handlers:
            return self._dispatch(fr, sc, outer, w, env, extra, exc)
        if sc.kind in (ScopeKind.TRY_BODY, ScopeKind.TRY_REST) and sc.fin:
            return self._to_finally(fr, sc, w, env, extra, kind, exc)
        return None

    def _to_finally(self, fr, sc: Scope, w, env, extra, kind: Xfer, exc) -> str:
        rec = f'[k |-> "{kind.value}", e |-> {exc or chr(34) * 2}]'
        return self._goto(fr, sc.fin)(w, env, [*extra, f'!.fin[t]["{sc.fin}"] = {rec}'])

    def _dispatch(self, fr, sc: Scope, outer: _Frame, w, env, extra, exc: str) -> str:
        """except clauses in order; unmatched goes to the finally or outward."""
        if sc.fin:
            rest = self._to_finally(fr, sc, w, env, extra, Xfer.RAISE, exc)
        else:
            rest = self._xfer(outer, Xfer.RAISE, w, env, extra, exc=exc)
        text = rest
        for caught, target in reversed(sc.handlers):
            env_h = {**env, self.x._EXC: self.x.Val(exc, self.x.VKind.STR)}
            text = f"IF {exc} \\in {self.x._tla_set(caught)} THEN\n{target(w, env_h, extra)}\nELSE {text}"
        return text

    def _boundary(self, fr: _Frame, kind: Xfer, w: str, env: dict, extra: list[str], exc) -> str:
        """Leave the method: back to its caller, or end of an entry / __init__ / thread."""
        m = fr.method
        raising = kind is Xfer.RAISE
        arms = []
        if m in self.entries:
            arms.append(("entry", self._end_entry(m, w, extra, exc if raising else None)))
        if m == "__init__" and self.init:
            pc = _DEAD if raising else _IDLE
            setup = f", !.setup = {exc}" if raising else ", !.inited = TRUE"
            arms.append(("init", f"w' = [{w} EXCEPT !.pc[t] = {pc}{setup}{self._join(extra)}]"))
        if m in self.targets:
            crash = (
                f", !.thcrash = @ \\/ {exc} \\notin {{{', '.join(self.x._UNCATCHABLE)}}}"
                if raising
                else ""
            )
            arms.append(
                ("thread", f"w' = [{w} EXCEPT !.pc[t] = {_DONE}{crash}{self._join(extra)}]")
            )
        status = f'!.rstat[t] = "raise", !.exc[t] = {exc}' if raising else '!.rstat[t] = "ok"'
        caller = f'w\' = [{w} EXCEPT !.pc[t] = {w}.tret[t]["{m}"], {status}{self._join(extra)}]'
        cases = [f'{w}.tret[t]["{m}"] = "{tag}" -> {text}' for tag, text in arms]
        if not cases:
            return caller
        return "(CASE " + "\n  [] ".join([*cases, f"OTHER -> {caller}"]) + ")"

    @staticmethod
    def _join(extra: list[str]) -> str:
        return "".join(f", {c}" for c in extra)

    def _end_entry(self, m: str, w: str, extra: list[str], exc) -> str:
        if exc is None:
            return f"w' = [{w} EXCEPT !.pc[t] = {_IDLE}{self._join(extra)}]"
        quiet = [
            *self.x._UNCATCHABLE,
            *(f'"{c}"' for c in sorted(self.x._handlers_match(list(self.entries[m].raises)))),
        ]
        flag = f"!.uncaught = IF {exc} \\in {{{', '.join(quiet)}}} THEN @ ELSE {exc}"
        return f"w' = [{w} EXCEPT !.pc[t] = {_IDLE}, !.ret = {exc}, {flag}{self._join(extra)}]"

    def _raiser(self, fr: _Frame) -> Callable:
        """CPS exception continuation: (w, e, env) -> raise transfer."""
        return lambda w, e, env: self._xfer(fr, Xfer.RAISE, w, env, [], exc=e)

    def _cont(self, fr: _Frame, nxt: Jump):
        def no_return(w, v):
            raise self.x.ExtractError(f"{fr.method}: return inside an atomic statement")

        return self.x.Cont(lambda w, env: nxt(w, env, []), no_return, self._raiser(fr))

    # ── statements ─────────────────────────────────────────────────

    def _block(self, fr: _Frame, stmts: list[ast.stmt], after: Jump) -> Jump:
        """Compile a statement list; returns the jump into its first statement."""
        if not stmts:
            return after
        labels = [self._label(fr.method) for _ in stmts]
        for i, st in enumerate(stmts):
            nxt = self._goto(fr, labels[i + 1]) if i + 1 < len(stmts) else after
            self._stmt(fr, st, labels[i], nxt)
        return self._goto(fr, labels[0])

    def _stmt(self, fr: _Frame, st: ast.stmt, label: str, nxt: Jump) -> None:
        handler = {
            ast.If: self._s_if,
            ast.While: self._s_while,
            ast.Try: self._s_try,
            ast.With: self._s_with,
            ast.Return: self._s_return,
            ast.Raise: self._s_raise,
            ast.Break: self._s_jump,
            ast.Continue: self._s_jump,
            ast.Expr: self._s_expr,
            ast.Assign: self._s_assign,
            ast.AnnAssign: self._s_annassign,
        }.get(type(st), self._s_plain)
        handler(fr, st, label, nxt)

    def _s_plain(self, fr: _Frame, st: ast.stmt, label: str, nxt: Jump) -> None:
        """An atomic statement compiled by the CPS compiler in one step."""
        self._no_blocking(st)
        env = self._env(fr.method, "w")
        self._step(label, self.comp.stmt(st, "w", env, self._cont(fr, nxt)))

    def _no_blocking(self, node: ast.AST) -> None:
        for sub in ast.walk(node):
            if self._self_method(sub) in self.procs:
                raise self.comp.fail(
                    sub, "blocking method called inside an expression (call it as a statement)"
                )
            if self._sync_construct(sub):
                raise self.comp.fail(sub, "thread operation inside an expression")

    def _proc_call(self, node: ast.AST) -> ast.Call | None:
        """``self.m(...)`` where m is a procedure, else None."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._proc_call(node.operand)
        f = getattr(node, "func", None)
        is_self = (
            isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self"
        )
        return node if is_self and f.attr in self.procs else None

    def _call(self, fr: _Frame, call: ast.Call, label: str, after_call: Callable) -> None:
        """Procedure call: store params, jump to the callee; resume at <label>_r."""
        callee = call.func.attr
        fn = self.methods[callee]
        back = f"{label}_r"
        params = [a.arg for a in fn.args.args[1:]]
        env = self._env(fr.method, "w")

        def then(w, vals) -> str:
            stores = []
            for p, v in zip(params, vals, strict=False):
                if self._storable(v):
                    self._note(self.lkind, (callee, p), v)
                    if self.lkind.get((callee, p)) == (v.kind, v.htype):
                        stores.append(f"!.loc[t][{self._slot(callee, p)}] = {v.tla}")
            extra = [f'!.tret[t]["{callee}"] = "{back}"', *stores]
            return self._write(fr, w, env, f'"{self._entry(callee)}"', extra)

        self._step(
            label,
            self.comp._args(
                list(call.args),
                "w",
                env,
                then,
                lambda w, e: self._xfer(fr, Xfer.RAISE, w, env, [], exc=e),
            ),
        )
        env_r = self._env(fr.method, "w")
        kind = self.rkind.get(callee)
        rv = self.x.Val(f'w.rv[t]["{callee}"]', kind[0], kind[1]) if kind else self.x.HAVOC
        raised = self._xfer(fr, Xfer.RAISE, "w", env_r, [], exc="w.exc[t]")
        self._step(
            back, f'IF w.rstat[t] = "raise" THEN\n{raised}\nELSE\n{after_call("w", env_r, rv)}'
        )

    def _branch(self, fr: _Frame, test: ast.expr, label: str, yes: Jump, no: Jump) -> None:
        """A condition: a procedure call (two steps) or a CPS condition (one step)."""
        call = self._proc_call(test)
        if call is not None:
            negate = isinstance(test, ast.UnaryOp)

            def after(w, env, rv) -> str:
                truth = self.comp.truth(rv)
                t, f = (no, yes) if negate else (yes, no)
                if truth is None:
                    return f"({t(w, env, [])}) \\/ ({f(w, env, [])})"
                return f"IF {truth} THEN\n{t(w, env, [])}\nELSE\n{f(w, env, [])}"

            self._call(fr, call, label, after)
            return
        self._no_blocking(test)
        env = self._env(fr.method, "w")
        k = self._cont(fr, yes)
        text = self.comp.cond(
            test, "w", env, lambda w: yes(w, env, []), lambda w: no(w, env, []), k
        )
        self._step(label, text)

    def _s_if(self, fr: _Frame, st: ast.If, label: str, nxt: Jump) -> None:
        yes = self._block(fr, st.body, nxt)
        no = self._block(fr, st.orelse, nxt)
        self._branch(fr, st.test, label, yes, no)

    def _s_while(self, fr: _Frame, st: ast.While, label: str, nxt: Jump) -> None:
        if st.orelse:
            raise self.comp.fail(st, "while/else is unsupported")
        inner = _Frame(fr.method, (*fr.ctx, Scope(ScopeKind.LOOP, head=label, after=nxt)))
        body = self._block(inner, st.body, self._goto(fr, label))
        if isinstance(st.test, ast.Constant) and st.test.value is True:
            env = self._env(fr.method, "w")
            self._step(label, body("w", env, []))
            return
        self._branch(fr, st.test, label, body, nxt)

    def _s_try(self, fr: _Frame, st: ast.Try, label: str, nxt: Jump) -> None:
        fin = f"{label}_F" if st.finalbody else ""
        if fin:
            self.fins.append(fin)
            self._finally(fr, st, fin, nxt)
        rest = _Frame(fr.method, (*fr.ctx, Scope(ScopeKind.TRY_REST, fin=fin)))
        done = self._fin_next(fr, fin, nxt)
        handlers = tuple(
            (self.x._handlers_match(self.comp._handler_names(h)), self._block(rest, h.body, done))
            for h in st.handlers
        )
        body_fr = _Frame(
            fr.method, (*fr.ctx, Scope(ScopeKind.TRY_BODY, handlers=handlers, fin=fin))
        )
        orelse = self._block(rest, st.orelse, done)
        body = self._block(body_fr, st.body, orelse)
        env = self._env(fr.method, "w")
        self._step(label, body("w", env, []))

    def _fin_next(self, fr: _Frame, fin: str, nxt: Jump) -> Jump:
        """Falling out of a try region: through the finally (completion "next") or on."""
        if not fin:
            return nxt
        rec = '[k |-> "next", e |-> ""]'
        return lambda w, env, extra: self._goto(fr, fin)(
            w, env, [*extra, f'!.fin[t]["{fin}"] = {rec}']
        )

    def _finally(self, fr: _Frame, st: ast.Try, fin: str, nxt: Jump) -> None:
        """The finally body, then resume the pending completion stored in fin[t][F]."""
        exit_label = f"{fin}x"
        body = self._block(fr, st.finalbody, self._goto(fr, exit_label))
        env = self._env(fr.method, "w")
        self._step(fin, body("w", env, []))
        pend = f'w.fin[t]["{fin}"]'
        arms = [f'{pend}.k = "next" -> {nxt("w", env, [])}']
        if any(sc.kind is ScopeKind.LOOP for sc in fr.ctx):
            for kind in (Xfer.BREAK, Xfer.CONTINUE):
                arms.append(f'{pend}.k = "{kind.value}" -> {self._xfer(fr, kind, "w", env, [])}')
        arms.append(f'{pend}.k = "return" -> {self._xfer(fr, Xfer.RETURN, "w", env, [])}')
        arms.append(f"OTHER -> {self._xfer(fr, Xfer.RAISE, 'w', env, [], exc=f'{pend}.e')}")
        self._step(exit_label, "(CASE " + "\n  [] ".join(arms) + ")")

    def _s_with(self, fr: _Frame, st: ast.With, label: str, nxt: Jump) -> None:
        if len(st.items) != 1:
            raise self.comp.fail(st, "only single-item `with` is supported")
        item = st.items[0].context_expr
        env = self._env(fr.method, "w")
        lock = self._lock_of(item)
        if lock is not None:
            inner = _Frame(fr.method, (*fr.ctx, Scope(ScopeKind.LOCKED, lock=lock)))
            release = f'!.lock["{lock}"] = 0'
            body = self._block(inner, st.body, lambda w, e, extra: nxt(w, e, [*extra, release]))
            # Blocks (disabled) while another thread holds the lock; a held Lock is not re-entrant.
            self._step(
                label,
                f'w.lock["{lock}"] = 0 /\\\n{body("w", env, [f"!.lock[{chr(34)}{lock}{chr(34)}] = t"])}',
            )
            return
        func = ast.unparse(item.func) if isinstance(item, ast.Call) else ""
        if func in self.x._SUPPRESS_CTX:
            caught = self.x._handlers_match([ast.unparse(a).rsplit(".", 1)[-1] for a in item.args])
            inner = _Frame(
                fr.method, (*fr.ctx, Scope(ScopeKind.TRY_BODY, handlers=((caught, nxt),)))
            )
            self._step(label, self._block(inner, st.body, nxt)("w", env, []))
            return
        if self.comp._passthrough(ast.unparse(item)):
            self._step(label, self._block(fr, st.body, nxt)("w", env, []))
            return
        raise self.comp.fail(st, "`with` supports locks, conditions, suppress and ignore= contexts")

    def _s_return(self, fr: _Frame, st: ast.Return, label: str, nxt: Jump) -> None:
        value = st.value.elts[0] if isinstance(st.value, ast.Tuple) else st.value
        call = self._proc_call(value) if value is not None else None
        if call is not None and not isinstance(value, ast.UnaryOp):
            self._call(
                fr, call, label, lambda w, env, rv: self._xfer(fr, Xfer.RETURN, w, env, [], val=rv)
            )
            return
        if value is None:
            env = self._env(fr.method, "w")
            self._step(label, self._xfer(fr, Xfer.RETURN, "w", env, [], val=self.x.NONE_VAL))
            return
        self._no_blocking(value)
        env = self._env(fr.method, "w")
        text = self.comp.expr(
            value,
            "w",
            env,
            lambda w, v: self._xfer(fr, Xfer.RETURN, w, env, [], val=v),
            lambda w, e: self._xfer(fr, Xfer.RAISE, w, env, [], exc=e),
        )
        self._step(label, text)

    def _s_raise(self, fr: _Frame, st: ast.Raise, label: str, nxt: Jump) -> None:
        env = self._env(fr.method, "w")
        k = self.x.Cont(lambda w, e: nxt(w, e, []), lambda w, v: "FALSE", self._raiser(fr))
        self._step(label, self.comp._s_raise(st, "w", env, k))

    def _s_jump(self, fr: _Frame, st: ast.stmt, label: str, nxt: Jump) -> None:
        kind = Xfer.BREAK if isinstance(st, ast.Break) else Xfer.CONTINUE
        env = self._env(fr.method, "w")
        self._step(label, self._xfer(fr, kind, "w", env, []))

    def _s_expr(self, fr: _Frame, st: ast.Expr, label: str, nxt: Jump) -> None:
        call = st.value if isinstance(st.value, ast.Call) else None
        op = self._sync_method(call)
        if op is not None:
            {"wait": self._wait, "join": self._join_thread, "start": self._start}[op](
                fr, call, label, nxt
            )
            return
        if call is not None and self._proc_call(call) is not None:
            self._call(fr, call, label, lambda w, env, rv: nxt(w, env, []))
            return
        self._s_plain(fr, st, label, nxt)

    def _sync_method(self, call: ast.Call | None) -> str | None:
        if (
            call is None
            or not isinstance(call.func, ast.Attribute)
            or not self._sync_construct(call)
        ):
            return None
        return call.func.attr

    def _wait(self, fr: _Frame, call: ast.Call, label: str, nxt: Jump) -> None:
        """Condition.wait(): release and sleep; wake when notified and the lock is free."""
        cond = self.comp._modeled_field(call.func.value)
        lock = self.conds[cond]
        env = self._env(fr.method, "w")
        held = f'w.lock["{lock}"] = t'
        sleep = [f'!.lock["{lock}"] = 0', f'!.waiters["{cond}"] = @ \\cup {{t}}']
        raised = self._xfer(fr, Xfer.RAISE, "w", env, [], exc='"RuntimeError"')
        self._step(
            label, f"IF {held} THEN\n{self._goto(fr, label + '_w')('w', env, sleep)}\nELSE {raised}"
        )
        wake = f't \\notin w.waiters["{cond}"] /\\ w.lock["{lock}"] = 0'
        self._step(
            f"{label}_w", f"{wake} /\\\n{nxt('w', env, [f'!.lock[{chr(34)}{lock}{chr(34)}] = t'])}"
        )

    def _thread_cases(
        self, fr: _Frame, call: ast.Call, env: dict, ready: Callable[[str], str]
    ) -> str:
        """CASE over the THREAD field's value: None / not started / ready(th)."""
        th = f"w.s.{self.comp._fields[self.comp._modeled_field(call.func.value)]}"
        none = self._xfer(fr, Xfer.RAISE, "w", env, [], exc='"AttributeError"')
        return f"(CASE {th} = 0 -> {none}\n  [] OTHER -> {ready(th)})"

    def _join_thread(self, fr: _Frame, call: ast.Call, label: str, nxt: Jump) -> None:
        env = self._env(fr.method, "w")
        bad = self._xfer(fr, Xfer.RAISE, "w", env, [], exc='"RuntimeError"')

        def ready(th: str) -> str:
            return f"IF w.pc[{th}] = {_NEW} THEN {bad} ELSE (w.pc[{th}] = {_DONE} /\\ {nxt('w', env, [])})"

        self._step(label, self._thread_cases(fr, call, env, ready))

    def _start(self, fr: _Frame, call: ast.Call, label: str, nxt: Jump) -> None:
        env = self._env(fr.method, "w")
        bad = self._xfer(fr, Xfer.RAISE, "w", env, [], exc='"RuntimeError"')

        def ready(th: str) -> str:
            arms = [
                f"{th} = {tid} -> {nxt('w', env, [f'!.pc[{tid}] = {chr(34)}{self._entry(m)}{chr(34)}', f'!.tret[{tid}][{chr(34)}{m}{chr(34)}] = {chr(34)}thread{chr(34)}'])}"
                for m, tid in self.targets.items()
            ]
            return f"IF w.pc[{th}] # {_NEW} THEN {bad} ELSE (CASE " + "\n  [] ".join(arms) + ")"

        self._step(label, self._thread_cases(fr, call, env, ready))

    def _s_annassign(self, fr: _Frame, st: ast.AnnAssign, label: str, nxt: Jump) -> None:
        if st.value is None:
            self._s_plain(fr, st, label, nxt)
            return
        plain = ast.Assign(
            targets=[st.target], value=st.value, lineno=st.lineno, col_offset=st.col_offset
        )
        self._s_assign(fr, plain, label, nxt)

    def _s_assign(self, fr: _Frame, st: ast.Assign, label: str, nxt: Jump) -> None:
        value, target = st.value, st.targets[0]
        fld = self.comp._modeled_field(target)
        func = ast.unparse(value.func) if isinstance(value, ast.Call) else ""
        env = self._env(fr.method, "w")
        if func == _THREAD_CTOR and fld:
            tid = self.targets[self._thread_target(value)]
            self._step(label, nxt("w", env, [f"!.s.{self.comp._fields[fld]} = {tid}"]))
            return
        if func in _LOCK_CTORS and fld:
            self._step(label, nxt("w", env, []))
            return
        if func in _QUEUE_CTORS and fld:
            self._step(label, nxt("w", env, [f"!.s.{self.comp._fields[fld]} = 0"]))
            return
        call = self._proc_call(value)
        if call is not None and not isinstance(value, ast.UnaryOp):

            def bind(w, env_r, rv) -> str:
                return self.comp._bind(target, rv, w, env_r, self._cont(fr, nxt))

            self._call(fr, call, label, bind)
            return
        self._s_plain(fr, st, label, nxt)

    # ── module ─────────────────────────────────────────────────────

    def _compile(self) -> None:
        self.steps, self.fins, self._n = [], [], 0
        for m in sorted(self.procs):
            fr = _Frame(m)
            end = f"{m.strip('_') or 'm'}_end"
            env = self._env(m, "w")
            self._step(end, self._xfer(fr, Xfer.RETURN, "w", env, [], val=self.x.NONE_VAL))
            first = self._block(fr, self.methods[m].body, self._goto(fr, end))
            self._step(self._entry(m), first("w", self._env(m, "w"), []))

    def _default(self, kind) -> str:
        vk = kind[0]
        if vk is self.x.VKind.BOOL:
            return "FALSE"
        return self.x._NONE_STR if vk is self.x.VKind.STR else "0"

    def _fn(self, table: dict, start: dict | None = None) -> str:
        """A TLC function literal ("k" :> v @@ ...) over the storable slots."""
        start = start or {}
        parts = [
            f"({k} :> {start.get(k, self._default(v))})" for k, v in sorted(table.items()) if v
        ]
        return " @@ ".join(parts) if parts else "[x \\in {} |-> 0]"

    def _initial(self) -> str:
        slots = {self._slot(m, n): k for (m, n), k in self.lkind.items()}
        pinned = {self._slot(m, n): v.tla for (m, n), v in self.pinned.items()}
        rvs = {f'"{m}"': k for m, k in self.rkind.items()}
        pc0 = f'"{self._entry("__init__")}"' if self.init else _IDLE
        ret0 = f'[m \\in {self.x._tla_set(self.procs)} |-> IF t = {MAIN} /\\ m = "__init__" THEN "init" ELSE "none"]'
        fins = self.x._tla_set(self.fins)
        locks = self.x._tla_set(self.locks)
        conds = self.x._tla_set(self.conds)
        return (
            "TW == W0 @@ [\n"
            f"  pc |-> [t \\in 1..NT |-> IF t = {MAIN} THEN {pc0} ELSE {_NEW}],\n"
            f"  loc |-> [t \\in 1..NT |-> {self._fn(slots, pinned)}],\n"
            f"  tret |-> [t \\in 1..NT |-> {ret0}],\n"
            f"  rv |-> [t \\in 1..NT |-> {self._fn(rvs)}],\n"
            '  rstat |-> [t \\in 1..NT |-> "ok"], exc |-> [t \\in 1..NT |-> ""],\n'
            f'  fin |-> [t \\in 1..NT |-> [f \\in {fins} |-> [k |-> "none", e |-> ""]]],\n'
            f"  lock |-> [l \\in {locks} |-> 0], waiters |-> [c \\in {conds} |-> {{}}],\n"
            f"  thcrash |-> FALSE, bound |-> FALSE, inited |-> {'FALSE' if self.init else 'TRUE'}]"
        )

    def _calls(self) -> list[str]:
        out = []
        for m, es in self.entries.items():
            if es.phase is not tla.Phase.CALL:
                continue
            guard = f" /\\ {self.comp.pure(es.requires)}" if es.requires else ""
            start = (
                f'!.pc[{MAIN}] = "{self._entry(m)}", !.tret[{MAIN}]["{m}"] = "entry", '
                f'!.calls = @ + 1, !.last = "{m}"'
            )
            out.append(
                f"Call_{m} ==\n  ~w.bound /\\ w.pc[{MAIN}] = {_IDLE} /\\ w.inited /\\ w.calls < MaxCalls{guard} /\\\n"
                f"  w' = [w EXCEPT {start}]\n"
            )
        return out

    def _invariants(self, step_names: list[str]) -> list[str]:
        base = [i for i in self.em._invariants() if not i.startswith(tuple(self.spec.invariants))]
        steps = " \\/ ".join(f"{s}(t)" for s in step_names)
        own = [
            f"{n} == w.pc[{MAIN}] = {_IDLE} => {self.comp.pure(src)}"
            for n, src in self.spec.invariants.items()
        ]
        return [
            *base,
            # The caller is mid-call and no thread can take a step: the call never returns.
            f"NoDeadlock == ~w.bound /\\ w.pc[{MAIN}] \\notin {{{_IDLE}, {_DEAD}}} => ENABLED (\\E t \\in 1..NT : {steps})",
            "NoThreadCrash == ~w.thcrash",
            f"NoHeldLock == w.pc[{MAIN}] = {_IDLE} => \\A l \\in DOMAIN w.lock : w.lock[l] # {MAIN}",
            *own,
        ]

    def emit(self):
        self._compile()  # first pass: learn which locals / returns are storable
        self._compile()
        name = self.comp.cls.__name__
        step_names = [s.split("(", 1)[0] for s in self.steps]
        calls = self._calls()
        invariants = self._invariants(step_names)
        nxt = " \\/ ".join(
            [f"(\\E t \\in 1..NT : {s}(t))" for s in step_names]
            + [c.split(" ==")[0] for c in calls]
            + ["Die"]
        )
        text = "\n".join(
            [
                f"---- MODULE {name} ----",
                f"\\* Generated by tools/tla_extract.py (thread model) from {self.comp.cls.__module__}.{name}. Do not edit.",
                "EXTENDS Naturals, TLC, Syscalls",
                "CONSTANT MaxCalls",
                "VARIABLE w",
                f"NT == {self.nt}",
                self.em._initial(),
                self._initial(),
                *self.steps,
                *calls,
                "Init == w = TW",
                'Die == \\E w2 \\in Died(w) : w\' = [w2 EXCEPT !.last = "<loader died>"]',
                f"Next == {nxt}",
                *invariants,
                "====",
            ]
        )
        names = [i.split(" ==", 1)[0] for i in invariants]
        return self.x.Model(name, text, names, list(self.comp.havoc))
