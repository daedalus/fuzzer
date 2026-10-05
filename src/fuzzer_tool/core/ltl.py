"""LTL property monitor (``--ltl``), after LTL-Fuzzer (ICSE '22).

The target reports events (``__fuzz_event``); this module runs the Buchi
automaton of the *negated* property over each event trace. Acceptance means
the property is violated.

    target --events--> read_events --> Monitor.run --> Run
                                                        |
                              LtlFrontier.update <------+--> violation?
                                    |
                              LtlFrontier.pick --> prefix + fresh tail

Automata come from ``ltl2tgba -B -H`` (state-based acceptance, one Inf set,
explicit labels). Event ``eN`` is a one-hot valuation: AP ``eN`` is true
exactly when the target emitted event N, every other AP is false.

Two violation kinds, with different strength:

* ``TRAP``: an accepting state every continuation stays in. Sound for the
  observed prefix.
* ``LASSO``: an accepting state revisited with the same program-state hash
  while the events in between close a walk through the automaton. A
  *candidate*: the hash covers only the enum variables that ``state_vars``
  instruments, not the whole program state.
"""

from __future__ import annotations

import enum
import re
import struct
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

NO_OFFSET = 0xFFFFFFFF  # event reported without an input offset
ANY_OTHER = -1  # "no AP true": an event the property does not name

MAX_STATES = 4096
MAX_TRANSITIONS = 65536
MAX_FRONTIER_STATES = 1024
TAIL_MAX = 32  # longest fresh tail appended to a stored prefix

_REC = struct.Struct("<III")
_ACC_RE = re.compile(r"^1\s+Inf\(0\)$")
_STATE_RE = re.compile(r'^State:\s*(\d+)\s*(?:"[^"]*")?\s*(?:\{0\})?\s*$')
_TRANS_RE = re.compile(r"^\[([^\]]*)\]\s*(\d+)\s*$")
_TOKEN_RE = re.compile(r"\s*(\d+|[tf!&|()])")
_EVENT_AP_RE = re.compile(r"^e?(\d+)$")


class Record(NamedTuple):
    """One target event: id, input bytes consumed so far, state-var hash."""

    event: int
    offset: int
    state_hash: int


class Kind(enum.Enum):
    TRAP = "trap"
    LASSO = "lasso"


class Violation(NamedTuple):
    kind: Kind
    index: int  # record index at which the monitor accepted


class Trans(NamedTuple):
    dst: int
    sat: frozenset[int]  # AP indices (or ANY_OTHER) that enable it


# ── label expressions ────────────────────────────────────────────────


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    pos = 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise ValueError(f"bad label {text!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


class _Label:
    """Recursive-descent parser: ``|`` < ``&`` < ``!`` < atom."""

    def __init__(self, text: str, n_ap: int):
        self._tok = _tokens(text)
        self._n_ap = n_ap
        self._i = 0

    def parse(self) -> frozenset[int]:
        """AP indices (plus ANY_OTHER) under which the label holds."""
        tree = self._or()
        if self._i != len(self._tok):
            raise ValueError("trailing label tokens")
        return frozenset(a for a in range(ANY_OTHER, self._n_ap) if self._eval(tree, a))

    def _peek(self) -> str | None:
        return self._tok[self._i] if self._i < len(self._tok) else None

    def _take(self) -> str:
        tok = self._peek()
        if tok is None:
            raise ValueError("label ends early")
        self._i += 1
        return tok

    def _or(self):
        node = self._and()
        while self._peek() == "|":
            self._take()
            node = ("|", node, self._and())
        return node

    def _and(self):
        node = self._not()
        while self._peek() == "&":
            self._take()
            node = ("&", node, self._not())
        return node

    def _not(self):
        if self._peek() == "!":
            self._take()
            return ("!", self._not())
        return self._atom()

    def _atom(self):
        tok = self._take()
        if tok == "(":
            node = self._or()
            if self._take() != ")":
                raise ValueError("missing )")
            return node
        if tok in ("t", "f"):
            return tok
        if not tok.isdigit() or int(tok) >= self._n_ap:
            raise ValueError(f"bad AP {tok!r}")
        return int(tok)

    def _eval(self, node, true_ap: int) -> bool:
        if node == "t":
            return True
        if node == "f":
            return False
        if isinstance(node, int):
            return node == true_ap
        if node[0] == "!":
            return not self._eval(node[1], true_ap)
        left = self._eval(node[1], true_ap)
        if node[0] == "&":
            return left and self._eval(node[2], true_ap)
        return left or self._eval(node[2], true_ap)


# ── automaton ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Automaton:
    """Buchi automaton, one-hot event alphabet."""

    init: tuple[int, ...]
    accepting: frozenset[int]
    trans: dict[int, tuple[Trans, ...]]
    ap_of: dict[int, int]  # event id -> AP index

    def step(self, states: frozenset[int], event: int) -> frozenset[int]:
        ap = self.ap_of.get(event, ANY_OTHER)
        out: set[int] = set()
        for s in states:
            for tr in self.trans.get(s, ()):
                if ap in tr.sat:
                    out.add(tr.dst)
        return frozenset(out)

    @cached_property
    def _n_aps(self) -> int:
        return len(self.ap_of)

    @cached_property
    def traps(self) -> frozenset[int]:
        """Accepting states with a self-loop that every event enables."""
        full = frozenset(range(ANY_OTHER, self._n_aps))
        return frozenset(
            s
            for s in self.accepting
            if any(t.dst == s and t.sat == full for t in self.trans.get(s, ()))
        )

    @cached_property
    def dist(self) -> dict[int, int]:
        """Edges to the nearest accepting state that sits on a cycle."""
        live = self._live_edges()
        targets = [s for s in self.accepting if self._on_cycle(s, live)]
        back: dict[int, list[int]] = {}
        for src, dsts in live.items():
            for d in dsts:
                back.setdefault(d, []).append(src)

        dist = {s: 0 for s in targets}
        frontier = list(targets)
        while frontier:
            nxt: list[int] = []
            for d in frontier:
                for src in back.get(d, ()):
                    if src in dist:
                        continue
                    dist[src] = dist[d] + 1
                    nxt.append(src)
            frontier = nxt
        return dist

    def _live_edges(self) -> dict[int, set[int]]:
        return {s: {t.dst for t in ts if t.sat} for s, ts in self.trans.items()}

    @staticmethod
    def _on_cycle(state: int, live: dict[int, set[int]]) -> bool:
        seen: set[int] = set()
        stack = list(live.get(state, ()))
        while stack:
            s = stack.pop()
            if s == state:
                return True
            if s in seen:
                continue
            seen.add(s)
            stack.extend(live.get(s, ()))
        return False


def _ap_events(header: dict[str, str]) -> dict[int, int]:
    line = header.get("AP")
    if line is None:
        raise ValueError("missing AP")
    count, _, rest = line.partition(" ")
    names = re.findall(r'"([^"]*)"', rest)
    if not count.isdigit() or int(count) != len(names):
        raise ValueError("bad AP line")

    ap_of: dict[int, int] = {}
    for i, name in enumerate(names):
        m = _EVENT_AP_RE.match(name)
        if m is None:
            raise ValueError(f"AP {name!r} is not an event id (eN or N)")
        event = int(m.group(1))
        if event in ap_of:
            raise ValueError(f"event {event} named twice")
        ap_of[event] = i
    return ap_of


def _header(lines: list[str]) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in lines:
        key, sep, val = line.partition(":")
        if not sep:
            continue
        if key == "Start" and "Start" in header:
            header["Start"] += " " + val.strip()
            continue
        header[key.strip()] = val.strip()
    if not _ACC_RE.match(header.get("Acceptance", "")):
        raise ValueError("need `Acceptance: 1 Inf(0)` (ltl2tgba -B)")
    return header


def _body(lines: list[str], n_ap: int) -> tuple[dict[int, list[Trans]], set[int]]:
    trans: dict[int, list[Trans]] = {}
    accepting: set[int] = set()
    cur: int | None = None
    count = 0
    for line in lines:
        line = line.strip()
        if not line or line == "--END--":
            continue
        m = _STATE_RE.match(line)
        if m:
            cur = int(m.group(1))
            trans.setdefault(cur, [])
            if line.rstrip().endswith("{0}"):
                accepting.add(cur)
            continue
        t = _TRANS_RE.match(line)
        if t is None or cur is None:
            raise ValueError(f"bad body line {line!r} (need explicit labels, state acc)")
        count += 1
        if count > MAX_TRANSITIONS:
            raise ValueError("too many transitions")
        trans[cur].append(Trans(int(t.group(2)), _Label(t.group(1), n_ap).parse()))
    return trans, accepting


def parse_hoa(text: str) -> Automaton:
    """Parse a state-acc, explicit-label, single-Inf HOA automaton.

    Raises:
        ValueError: anything outside that subset.
    """
    head, sep, rest = text.partition("--BODY--")
    if not sep:
        raise ValueError("missing --BODY--")

    header = _header(head.splitlines())
    ap_of = _ap_events(header)
    trans, accepting = _body(rest.splitlines(), len(ap_of))
    if len(trans) > MAX_STATES:
        raise ValueError("too many states")

    init = tuple(int(x) for x in re.split(r"\s+", header.get("Start", "")) if x.isdigit())
    if not init or any(s not in trans for s in init):
        raise ValueError("bad Start")
    for ts in trans.values():
        if any(t.dst not in trans for t in ts):
            raise ValueError("transition to undefined state")

    return Automaton(init, frozenset(accepting), {s: tuple(ts) for s, ts in trans.items()}, ap_of)


# ── monitor ──────────────────────────────────────────────────────────


@dataclass
class Run:
    """Outcome of one trace."""

    final: frozenset[int]
    covered: set[tuple[int, int]] = field(default_factory=set)  # (src, transition idx)
    entry: dict[int, int] = field(default_factory=dict)  # state -> offset entered at
    violation: Violation | None = None
    checked: int = 0  # lasso closures replayed


class Monitor:
    """Runs the automaton over event records."""

    MAX_EVENTS = 65536
    MAX_LASSO_CHECKS = 32
    MAX_LASSO_SPAN = 4096

    def __init__(self, aut: Automaton):
        self._aut = aut

    def run(self, records: list[Record]) -> Run:
        aut = self._aut
        states = frozenset(aut.init)
        run = Run(states, entry={s: 0 for s in states})
        if states & aut.traps:
            run.violation = Violation(Kind.TRAP, -1)
            return run

        recs = records[: self.MAX_EVENTS]
        seen: dict[tuple[int, int], int] = {}
        for i, rec in enumerate(recs):
            states = self._advance(states, rec, run)
            run.final = states
            if states & aut.traps:
                run.violation = Violation(Kind.TRAP, i)
                return run
            if self._lasso(recs, i, states, seen, run):
                run.violation = Violation(Kind.LASSO, i)
                return run
        return run

    def _advance(self, states: frozenset[int], rec: Record, run: Run) -> frozenset[int]:
        aut = self._aut
        ap = aut.ap_of.get(rec.event, ANY_OTHER)
        nxt: set[int] = set()
        for s in states:
            for k, tr in enumerate(aut.trans.get(s, ())):
                if ap not in tr.sat:
                    continue
                run.covered.add((s, k))
                nxt.add(tr.dst)
        for s in nxt:
            run.entry.setdefault(s, rec.offset)
        return frozenset(nxt)

    def _lasso(
        self,
        recs: list[Record],
        i: int,
        states: frozenset[int],
        seen: dict[tuple[int, int], int],
        run: Run,
    ) -> bool:
        h = recs[i].state_hash
        if not h:
            return False  # no state vars reported: repetition proves nothing

        for s in states & self._aut.accepting:
            prev = seen.get((h, s))
            seen[(h, s)] = i
            if prev is None or run.checked >= self.MAX_LASSO_CHECKS:
                continue
            run.checked += 1
            if self._closes(recs, prev, i, s):
                return True
        return False

    def _closes(self, recs: list[Record], i: int, j: int, state: int) -> bool:
        if j - i > self.MAX_LASSO_SPAN:
            return False
        cur = frozenset({state})
        for rec in recs[i + 1 : j + 1]:
            cur = self._aut.step(cur, rec.event)
        return state in cur


def read_events(path: str) -> list[Record]:
    """Parse the target's event file; a torn trailing record is dropped."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(Monitor.MAX_EVENTS * _REC.size)
    except FileNotFoundError:
        return []
    raw = raw[: len(raw) - len(raw) % _REC.size]
    return [Record(*r) for r in _REC.iter_unpack(raw)]


# ── frontier ─────────────────────────────────────────────────────────


class LtlFrontier:
    """Shortest input prefix reaching each automaton state, plus the set of
    transitions ever fired.

    A state whose outgoing transitions have not all fired is a frontier:
    keep its prefix, mutate only the tail. States nearer an accepting cycle
    are drawn more often.
    """

    def __init__(self, aut: Automaton, cap: int = MAX_FRONTIER_STATES):
        self._aut = aut
        self._cap = cap
        self._prefix: dict[int, bytes] = {}
        self._seen: set[tuple[int, int]] = set()

    def update(self, run: Run, data: bytes) -> bool:
        """Fold one run in. True when it fired a transition for the first time."""
        novel = not run.covered <= self._seen
        self._seen |= run.covered
        for state, off in run.entry.items():
            self._offer(state, off, data)
        return novel

    def _offer(self, state: int, off: int, data: bytes) -> None:
        if off == NO_OFFSET or off > len(data):
            return
        cur = self._prefix.get(state)
        if cur is None and len(self._prefix) >= self._cap:
            return
        if cur is None or off < len(cur):
            self._prefix[state] = data[:off]

    def prefix(self, state: int) -> bytes | None:
        return self._prefix.get(state)

    def size(self) -> int:
        return len(self._prefix)

    def forget_transitions(self) -> None:
        self._seen.clear()

    def _open(self, state: int) -> bool:
        ts = self._aut.trans.get(state, ())
        return any(t.sat and (state, k) not in self._seen for k, t in enumerate(ts))

    def pick(self, rng) -> bytes | None:
        """Prefix of a frontier state, weighted 1/(1+distance to accept)."""
        states = sorted(s for s in self._prefix if self._open(s))
        roll = rng.random()
        if not states:
            return None

        far = len(self._aut.trans)
        weights = [1.0 / (1 + self._aut.dist.get(s, far)) for s in states]
        target = roll * sum(weights)
        acc = 0.0
        for s, w in zip(states, weights, strict=True):
            acc += w
            if target < acc:
                return self._prefix[s]
        return self._prefix[states[-1]]


# ── channel ──────────────────────────────────────────────────────────


class Observation(NamedTuple):
    novel: bool
    violation: Violation | None
    marker: str  # stderr line for the crash signature; "" when no violation


_MARKERS = {
    Kind.TRAP: "LTL-VIOLATION: trap",
    Kind.LASSO: "LTL-VIOLATION: lasso (candidate)",
}
_MARKER_RE = re.compile(r"LTL-VIOLATION: (trap|lasso)\b")


def violation_signature(stderr: str) -> str | None:
    """Crash-bucket signature (``ltl:trap`` / ``ltl:lasso``) from a marker line."""
    m = _MARKER_RE.search(stderr)
    return f"ltl:{m.group(1)}" if m else None


class LtlChannel:
    """Per-fuzzer glue: event file, monitor, frontier."""

    def __init__(self, aut: Automaton, events_path: str):
        self.events_path = events_path
        self._monitor = Monitor(aut)
        self._frontier = LtlFrontier(aut)

    @classmethod
    def from_file(cls, hoa_path: str, events_path: str) -> LtlChannel:
        with open(hoa_path, encoding="utf-8") as fh:
            return cls(parse_hoa(fh.read()), events_path)

    def clear(self) -> None:
        """Empty the event file before an execution."""
        with open(self.events_path, "wb"):
            pass

    def collect(self, data: bytes) -> Observation:
        return self.observe(read_events(self.events_path), data)

    def observe(self, records: list[Record], data: bytes) -> Observation:
        run = self._monitor.run(records)
        novel = self._frontier.update(run, data)
        v = run.violation
        return Observation(novel, v, _MARKERS[v.kind] if v else "")

    def splice(self, rng, max_len: int) -> bytes | None:
        """A stored prefix plus a fresh random tail, or None."""
        prefix = self._frontier.pick(rng)
        if prefix is None:
            return None

        room = max_len - len(prefix)
        if room <= 0:
            return None

        n = rng.randint(1, min(TAIL_MAX, room))
        return prefix + rng.randbytes(n)[:n]
