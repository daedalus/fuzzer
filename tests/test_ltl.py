"""``--ltl``: Buchi-automaton monitor over target event traces.

Automata are the negated property, as emitted by ``ltl2tgba -B -H`` (state
based acceptance, one Inf set). Event ``eN`` is a one-hot valuation: AP ``eN``
is true exactly when the target emitted event N.
"""

from __future__ import annotations

import struct

import pytest

from fuzzer_tool.core.ltl import (
    NO_OFFSET,
    Automaton,
    Kind,
    LtlChannel,
    LtlFrontier,
    Monitor,
    Record,
    parse_hoa,
    read_events,
)
from tests.support.scripted_rng import ScriptedRng

# Negation of G !e1 ("e1 never happens"): F e1. State 1 is an accepting trap.
NEVER_E1 = """HOA: v1
States: 2
Start: 0
AP: 2 "e1" "e2"
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc
--BODY--
State: 0
[!0] 0
[0] 1
State: 1 {0}
[t] 1
--END--
"""

# Negation of GF e1: FG !e1. Accepting state 1 loops on !e1 only.
GF_E1 = """HOA: v1
States: 2
Start: 0
AP: 1 "e1"
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc
--BODY--
State: 0
[t] 0
[!0] 1
State: 1 {0}
[!0] 1
--END--
"""

# Negation of G(e1 -> F e2): F(e1 & G !e2). Chain 0 -e1-> 1 -e1-> 2(acc).
CHAIN = """HOA: v1
States: 3
Start: 0
AP: 2 "e1" "e2"
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc
--BODY--
State: 0
[t] 0
[0] 1
State: 1
[0&!1] 2
State: 2 {0}
[!1] 2
--END--
"""


def _rec(event, offset=0, h=0):
    return Record(event, offset, h)


# ── parsing ──────────────────────────────────────────────────────────


def test_parse_basic():
    a = parse_hoa(NEVER_E1)
    assert a.init == (0,)
    assert a.accepting == frozenset({1})
    assert a.step(frozenset({0}), 2) == frozenset({0})
    assert a.step(frozenset({0}), 1) == frozenset({1})
    assert a.step(frozenset({1}), 7) == frozenset({1})


def test_label_grammar():
    text = NEVER_E1.replace("[0] 1", "[(0|1)&!(0&1)] 1").replace("[!0] 0", "[f] 0")
    a = parse_hoa(text)
    assert a.step(frozenset({0}), 1) == frozenset({1})
    assert a.step(frozenset({0}), 2) == frozenset({1})
    assert a.step(frozenset({0}), 9) == frozenset()


def test_ap_names_e_prefix_or_decimal():
    text = NEVER_E1.replace('"e1" "e2"', '"5" "e6"')
    a = parse_hoa(text)
    assert a.step(frozenset({0}), 5) == frozenset({1})
    assert a.ap_of == {5: 0, 6: 1}
    assert a.step(frozenset({0}), 6) == frozenset({0})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("Acceptance: 1 Inf(0)", "Acceptance: 2 Inf(0)&Inf(1)"),
        lambda t: t.replace('"e1" "e2"', '"open" "e2"'),
        lambda t: t.replace("[0] 1", "1"),
        lambda t: t.replace("State: 1 {0}\n[t] 1", "State: 1\n[t] 1 {0}"),
        lambda t: t.replace("--BODY--", ""),
        lambda t: t.replace("[0] 1", "[0&] 1"),
        lambda t: t.replace("[0] 1", "[7] 1"),
    ],
)
def test_rejects_unsupported(mutate):
    with pytest.raises(ValueError):
        parse_hoa(mutate(NEVER_E1))


# ── automaton graph facts ────────────────────────────────────────────


def test_distance_to_accept():
    a = parse_hoa(CHAIN)
    assert a.dist[2] == 0
    assert a.dist[1] == 1
    assert a.dist[0] == 2


def test_distance_ignores_accepting_state_off_a_cycle():
    text = CHAIN.replace("State: 2 {0}\n[!1] 2", "State: 2 {0}\n[f] 2")
    a = parse_hoa(text)
    assert a.dist == {}


def test_trap_states():
    assert parse_hoa(NEVER_E1).traps == frozenset({1})
    assert parse_hoa(GF_E1).traps == frozenset()


# ── monitor ──────────────────────────────────────────────────────────


def test_trap_violation():
    run = Monitor(parse_hoa(NEVER_E1)).run([_rec(2), _rec(2), _rec(1)])
    assert run.violation is not None
    assert run.violation.kind is Kind.TRAP
    assert run.violation.index == 2


def test_no_violation_when_property_holds():
    run = Monitor(parse_hoa(NEVER_E1)).run([_rec(2)] * 50)
    assert run.violation is None
    assert run.final == frozenset({0})


def test_empty_trace():
    run = Monitor(parse_hoa(NEVER_E1)).run([])
    assert run.violation is None
    assert run.final == frozenset({0})
    assert run.entry == {0: 0}


def test_entry_offsets_record_the_event_that_entered_the_state():
    run = Monitor(parse_hoa(CHAIN)).run([_rec(2, 3), _rec(1, 5), _rec(1, 9)])
    assert run.entry[0] == 0
    assert run.entry[1] == 5
    assert run.entry[2] == 9


def test_lasso_needs_accepting_cycle_and_repeated_state_hash():
    aut = parse_hoa(GF_E1)
    loop = [_rec(2, i, 77) for i in range(4)]
    run = Monitor(aut).run(loop)
    assert run.violation is not None
    assert run.violation.kind is Kind.LASSO


def test_lasso_not_reported_without_state_hash():
    run = Monitor(parse_hoa(GF_E1)).run([_rec(2, i, 0) for i in range(6)])
    assert run.violation is None


def test_lasso_not_reported_when_state_keeps_changing():
    run = Monitor(parse_hoa(GF_E1)).run([_rec(2, i, 100 + i) for i in range(6)])
    assert run.violation is None


def test_lasso_not_reported_when_loop_leaves_the_accepting_cycle():
    recs = [_rec(2, 0, 5), _rec(1, 1, 6), _rec(2, 2, 5)]
    run = Monitor(parse_hoa(GF_E1)).run(recs)
    assert run.violation is None


def test_covered_transitions():
    aut = parse_hoa(CHAIN)
    run = Monitor(aut).run([_rec(2), _rec(1)])
    assert (0, 0) in run.covered
    assert (0, 1) in run.covered
    assert (1, 0) not in run.covered


def test_adversarial_long_trace_is_bounded():
    # e2,e1 alternating at a constant hash: the accepting state is re-entered
    # every other event but the closure never holds, so every check fails.
    recs = [_rec(2 - i % 2, i, 5) for i in range(200_000)]
    run = Monitor(parse_hoa(GF_E1)).run(recs)
    assert run.violation is None
    assert run.checked == Monitor.MAX_LASSO_CHECKS


def test_nondeterministic_state_set():
    text = NEVER_E1.replace("[!0] 0", "[t] 0")
    run = Monitor(parse_hoa(text)).run([_rec(2)])
    assert run.final == frozenset({0})
    run = Monitor(parse_hoa(text)).run([_rec(1)])
    assert run.final == frozenset({0, 1})


# ── event file ───────────────────────────────────────────────────────


def _pack(*recs):
    return b"".join(struct.pack("<III", *r) for r in recs)


def test_read_events(tmp_path):
    p = tmp_path / "ev"
    p.write_bytes(_pack((1, 0, 0), (2, 5, 9)))
    assert read_events(str(p)) == [Record(1, 0, 0), Record(2, 5, 9)]


def test_read_events_drops_torn_tail(tmp_path):
    p = tmp_path / "ev"
    p.write_bytes(_pack((1, 0, 0)) + b"\x01\x02\x03")
    assert read_events(str(p)) == [Record(1, 0, 0)]


def test_read_events_missing_file(tmp_path):
    assert read_events(str(tmp_path / "nope")) == []


def test_read_events_caps_records(tmp_path):
    p = tmp_path / "ev"
    p.write_bytes(_pack(*[(1, 0, 0)] * (Monitor.MAX_EVENTS + 10)))
    assert len(read_events(str(p))) == Monitor.MAX_EVENTS


# ── frontier ─────────────────────────────────────────────────────────


def test_frontier_keeps_shortest_prefix_per_state():
    aut = parse_hoa(CHAIN)
    fr = LtlFrontier(aut)
    mon = Monitor(aut)
    fr.update(mon.run([_rec(1, 8)]), b"AAAAAAAAAAAA")
    fr.update(mon.run([_rec(1, 3)]), b"BBBBBBBBBBBB")
    fr.update(mon.run([_rec(1, 6)]), b"CCCCCCCCCCCC")
    assert fr.prefix(1) == b"BBB"


def test_frontier_ignores_unknown_offsets():
    aut = parse_hoa(CHAIN)
    fr = LtlFrontier(aut)
    fr.update(Monitor(aut).run([_rec(1, NO_OFFSET)]), b"AAAA")
    assert fr.prefix(1) is None


def test_frontier_fresh_flags_only_new_transitions():
    aut = parse_hoa(CHAIN)
    fr = LtlFrontier(aut)
    mon = Monitor(aut)
    assert fr.update(mon.run([_rec(2)]), b"x") is True
    assert fr.update(mon.run([_rec(2)]), b"x") is False
    assert fr.update(mon.run([_rec(1)]), b"x") is True


def test_pick_prefers_states_with_unvisited_transitions():
    aut = parse_hoa(CHAIN)
    fr = LtlFrontier(aut)
    fr.update(Monitor(aut).run([_rec(1, 4)]), b"abcdefgh")
    # state 0 (prefix b"") and state 1 (b"abcd") stored; state 0's two
    # transitions are both covered, state 1's is not.
    rng = ScriptedRng(randoms=[0.0])
    assert fr.pick(rng) == b"abcd"


def test_pick_none_when_nothing_unvisited():
    aut = parse_hoa(GF_E1)
    fr = LtlFrontier(aut)
    fr.update(Monitor(aut).run([_rec(2, 1), _rec(2, 2)]), b"abc")
    assert fr.pick(ScriptedRng(randoms=[0.0])) is None


def test_pick_weights_nearer_states_higher():
    aut = parse_hoa(CHAIN)
    fr = LtlFrontier(aut)
    fr.update(Monitor(aut).run([_rec(1, 4)]), b"abcdefgh")
    fr.forget_transitions()
    # weights: state 0 -> 1/(1+2), state 1 -> 1/(1+1); roll above 0.4 picks 1.
    assert fr.pick(ScriptedRng(randoms=[0.9])) == b"abcd"
    assert fr.pick(ScriptedRng(randoms=[0.0])) == b""


def test_frontier_state_cap():
    aut = parse_hoa(NEVER_E1)
    fr = LtlFrontier(aut, cap=1)
    mon = Monitor(aut)
    fr.update(mon.run([_rec(1, 2)]), b"abcd")
    assert fr.size() <= 1


# ── channel ──────────────────────────────────────────────────────────


def test_channel_observe_flags_novelty_and_violation(tmp_path):
    ch = LtlChannel(parse_hoa(NEVER_E1), str(tmp_path / "ev"))
    obs = ch.observe([_rec(2)], b"data")
    assert obs.novel is True
    assert obs.violation is None
    obs = ch.observe([_rec(2)], b"data")
    assert obs.novel is False
    obs = ch.observe([_rec(1)], b"data")
    assert obs.violation is not None
    assert "LTL-VIOLATION" in obs.marker


def test_channel_clear_truncates(tmp_path):
    p = tmp_path / "ev"
    p.write_bytes(b"junk")
    LtlChannel(parse_hoa(NEVER_E1), str(p)).clear()
    assert p.read_bytes() == b""


def test_channel_marker_distinguishes_kinds(tmp_path):
    ch = LtlChannel(parse_hoa(GF_E1), str(tmp_path / "ev"))
    obs = ch.observe([_rec(2, i, 7) for i in range(3)], b"d")
    assert obs.violation is not None
    assert "lasso" in obs.marker
    assert "candidate" in obs.marker


def test_automaton_is_immutable_value():
    a = parse_hoa(NEVER_E1)
    assert isinstance(a, Automaton)
    with pytest.raises(AttributeError):
        a.init = (1,)  # type: ignore[misc]
