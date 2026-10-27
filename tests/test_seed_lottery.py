"""SeedLotteryScheduler: compensated tickets, currencies, parent transfer."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_lottery import (
    CREDIT_EPS,
    REFRESH_PICKS,
    TRANSFER_CAP,
    SeedLotteryScheduler,
)
from fuzzer_tool.services.seed_picker import FRESH_GROUP, SeedPicker

NAN = float("nan")
INF = float("inf")


def _sched(**kw):
    return SeedLotteryScheduler(rng=RandPool(seed=kw.pop("seed", 7)), **kw)


def _share(picks, key):
    return picks.count(key) / len(picks)


def _run(s, keys, n=6000, **fns):
    return [s.select_seed(keys, **fns) for _ in range(n)]


# --- flat and weighted ---------------------------------------------------------


def test_flat_is_a_uniform_random_pick():
    """Falsification: flat weight and cost -> every seed equally likely."""
    keys = ["a", "b", "c", "d"]
    picks = _run(_sched(), keys, 8000)

    assert all(_share(picks, k) == pytest.approx(0.25, abs=0.025) for k in keys)


def test_share_follows_weight():
    w = {"a": 3.0, "b": 1.0}
    picks = _run(_sched(), ["a", "b"], 8000, weight_fn=w.get)

    assert _share(picks, "a") == pytest.approx(0.75, abs=0.025)


def test_compensation_gives_equal_time_not_equal_picks():
    """Equal weight, cost 1:4 -> picks 4:1 (tickets inflated by 1 / cost)."""
    c = {"a": 1.0, "b": 4.0}
    picks = _run(_sched(), ["a", "b"], 8000, cost_fn=c.get)

    assert _share(picks, "a") == pytest.approx(0.8, abs=0.025)


@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF])
def test_garbage_weight_and_cost_are_neutral(bad):
    """Adversarial: garbage counts as 1 -- never starved, never dominant."""
    for fns in ({"weight_fn": {"a": bad, "b": 1.0}.get}, {"cost_fn": {"a": bad, "b": 1.0}.get}):
        picks = _run(_sched(), ["a", "b"], 4000, **fns)

        assert _share(picks, "a") == pytest.approx(0.5, abs=0.04)


def test_empty_and_single():
    s = _sched()

    assert s.select_seed([]) == ""
    assert s.select_seed(["x"]) == "x"


# --- membership ------------------------------------------------------------------


def test_departed_never_picked_and_joiner_is_served():
    s = _sched()
    _run(s, ["a", "b", "c"], 50)
    after = _run(s, ["a", "c", "d"], 600)

    assert "b" not in after
    assert {"a", "c", "d"} <= set(after)


def test_membership_change_does_not_wait_for_a_refresh():
    """A joiner is drawable on the very next pick (O(log n) admit, not a rebuild)."""
    s = _sched()
    _run(s, ["a", "b"], 5)
    keys = ["a", "b", "new"]
    seen = {s.select_seed(keys) for _ in range(300)}

    assert "new" in seen
    assert REFRESH_PICKS > 5


def test_refresh_rereads_a_changed_weight():
    """Between refreshes a seed keeps its tickets; the refresh picks up the change."""
    w = {"a": 1.0, "b": 1.0}
    s = _sched()
    keys = ["a", "b"]
    _run(s, keys, 10, weight_fn=lambda k: w[k])
    w["a"] = 99.0
    _run(s, keys, REFRESH_PICKS, weight_fn=lambda k: w[k])  # crosses a refresh
    later = _run(s, keys, 4000, weight_fn=lambda k: w[k])

    assert _share(later, "a") > 0.95


# --- currencies ------------------------------------------------------------------

BIG = [f"big{i}" for i in range(20)]
KEYS = [*BIG, "lone"]


def _group(k):
    return "B" if k == "lone" else "A"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("off", 1 / 21), ("equal", 1 / 2), ("sqrt", 1 / (1 + 20**0.5))],
)
def test_currency_backing(mode, expected):
    """One-seed group vs a 20-seed group: its draw share follows the backing rule."""
    s = _sched(currency=mode)
    picks = _run(s, KEYS, 12000, group_fn=_group)

    assert _share(picks, "lone") == pytest.approx(expected, abs=0.025)


def test_currency_off_ignores_group_fn_entirely():
    def boom(_k):
        raise AssertionError("group_fn must not be read when currency is off")

    s = _sched(currency="off")

    assert s.select_seed(KEYS, group_fn=boom) in KEYS
    assert s.groups == 1


def test_single_group_equals_flat_currency():
    """Falsification: one group under 'equal' is the flat pool."""
    keys = ["a", "b", "c", "d"]
    picks = _run(_sched(currency="equal"), keys, 8000, group_fn=lambda _k: "only")

    assert all(_share(picks, k) == pytest.approx(0.25, abs=0.025) for k in keys)


def test_emptied_group_is_dropped():
    s = _sched(currency="equal")
    _run(s, KEYS, 20, group_fn=_group)
    assert s.groups == 2
    after = _run(s, BIG, 300, group_fn=_group)

    assert s.groups == 1
    assert "lone" not in after


def test_tickets_within_a_group_still_apply():
    w = {"x": 9.0, "y": 1.0, "z": 1.0}
    grp = {"x": "A", "y": "A", "z": "B"}
    picks = _run(
        _sched(currency="equal"), ["x", "y", "z"], 12000, weight_fn=w.get, group_fn=grp.get
    )

    assert _share(picks, "z") == pytest.approx(0.5, abs=0.025)
    assert _share(picks, "x") == pytest.approx(0.45, abs=0.025)


# --- transfer --------------------------------------------------------------------


def _parent_map(k):
    return {"child": "parent"}.get(k)


def _primed(**kw):
    s = _sched(**kw)
    keys = ["parent", "child", "other"]
    s.select_seed(keys, flow_fn=_parent_map)
    return s, keys


def test_transfer_credits_the_parent_on_success():
    """Three successes at transfer 1.0 -> parent tickets 1 * (1 + 3)."""
    s, _keys = _primed(transfer=1.0)
    for _ in range(3):
        s.record("child", success=True)

    assert s._tickets("parent") == pytest.approx(4.0)
    assert s._tickets("child") == pytest.approx(1.0)


def test_transfer_raises_the_parents_draw_share():
    """Credit applied, then a short window (credit halves at each refresh by design)."""
    s, keys = _primed(transfer=1.0)
    for _ in range(3):
        s.record("child", success=True)
    picks = _run(s, keys, REFRESH_PICKS - 10, flow_fn=_parent_map)

    # parent 4 of (4 + 1 + 1)
    assert _share(picks, "parent") == pytest.approx(4 / 6, abs=0.08)


def test_transfer_off_is_inert():
    s, _keys = _primed(transfer=0.0)
    for _ in range(5):
        s.record("child", success=True)

    assert s._tickets("parent") == pytest.approx(1.0)
    assert s._credit == {}


def test_failure_transfers_nothing():
    s, _keys = _primed(transfer=1.0)
    for _ in range(5):
        s.record("child", success=False)

    assert s._credit == {}


def test_transfer_is_capped():
    s, _keys = _primed(transfer=10.0)
    for _ in range(50):
        s.record("child", success=True)

    assert s._tickets("parent") == pytest.approx(1.0 + TRANSFER_CAP)


def test_credit_halves_at_each_refresh():
    s, keys = _primed(transfer=1.0)
    for _ in range(4):
        s.record("child", success=True)
    assert s._tickets("parent") == pytest.approx(5.0)
    _run(s, keys, REFRESH_PICKS, flow_fn=_parent_map)  # crosses exactly one refresh

    assert s._tickets("parent") == pytest.approx(3.0)


def test_credit_below_eps_is_forgotten():
    s, keys = _primed(transfer=CREDIT_EPS)
    s.record("child", success=True)
    _run(s, keys, REFRESH_PICKS, flow_fn=_parent_map)

    assert s._credit == {}


def test_orphan_self_and_unknown_parent_are_ignored():
    s = _sched(transfer=1.0)
    keys = ["a", "b"]
    parents = {"a": None, "b": "b"}
    s.select_seed(keys, flow_fn=parents.get)
    s.record("a", success=True)
    s.record("b", success=True)  # self-parent
    s.record("never_seen", success=True)

    assert s._credit == {}


def test_departed_parent_credit_is_dropped():
    s, keys = _primed(transfer=1.0)
    s.record("child", success=True)
    _run(s, ["child", "other"], 5, flow_fn=_parent_map)
    _run(s, ["child", "other"], REFRESH_PICKS, flow_fn=_parent_map)

    assert "parent" not in s._credit


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "flat", "SQRT", None])
def test_bad_currency_is_rejected(bad):
    with pytest.raises(ValueError, match="currency"):
        _sched(currency=bad)


@pytest.mark.parametrize("bad", [-1.0, NAN, INF])
def test_bad_transfer_is_rejected(bad):
    with pytest.raises(ValueError, match="transfer"):
        _sched(transfer=bad)


def test_deterministic_under_a_fixed_seed():
    keys = ["a", "b", "c"]

    assert _run(_sched(seed=5), keys, 200) == _run(_sched(seed=5), keys, 200)


# --- picker group signal ----------------------------------------------------------

# 30 seeds: seeds 2i and 2i+1 share edge 1000+i (held by 2 of 30, under the 10%
# strip line), and every seed also holds an edge of its own.
SEEDS = {f"k{i}": bytes([i + 1]) * 4 for i in range(30)}
EDGES = {f"k{i}": {i, 1000 + i // 2} for i in range(30)}


def _picker(corpus_size=30, exec_count=0, tracker=True):
    f = SimpleNamespace(
        exec_count=exec_count,
        _edge_tracker=SimpleNamespace(attributed_edges=lambda k: EDGES[k]) if tracker else None,
        _rng=RandPool(seed=1),
        seed_meta={},
        _favored=set(),
        mean_exec_time=lambda: 0.001,
    )
    return SeedPicker(f), f, dict(list(SEEDS.items())[:corpus_size])


def test_component_labels_group_seeds_that_share_an_edge():
    sp, _f, key_to_seed = _picker()
    labels = sp._component_labels(key_to_seed)

    assert labels["k0"] == labels["k1"]
    assert labels["k28"] == labels["k29"]
    assert len(set(labels.values())) == 15


def test_component_labels_are_memoized_until_drift_or_age():
    sp, f, key_to_seed = _picker()
    first = sp._component_labels(key_to_seed)

    assert sp._component_labels(key_to_seed) is first
    f.exec_count = 10**6

    assert sp._component_labels(key_to_seed) is not first


def test_component_labels_refresh_when_the_corpus_drifts():
    sp, _f, key_to_seed = _picker(corpus_size=10)
    first = sp._component_labels(key_to_seed)
    bigger = dict(list(SEEDS.items())[:30])

    assert sp._component_labels(bigger) is not first


def test_small_drift_keeps_the_memo():
    """Under 10% growth the labels are reused; the newcomer is a fresh currency."""
    sp, _f, key_to_seed = _picker(corpus_size=28)
    first = sp._component_labels(key_to_seed)
    plus_one = dict(list(SEEDS.items())[:29])

    assert sp._component_labels(plus_one) is first
    assert sp._seed_signals(plus_one)["group"]("k28") == FRESH_GROUP


def test_no_tracker_means_one_currency():
    sp, _f, key_to_seed = _picker(tracker=False)

    assert sp._component_labels(key_to_seed) == {}
    assert sp._seed_signals(key_to_seed)["group"]("k0") == FRESH_GROUP


def test_labels_are_plain_ints():
    sp, _f, key_to_seed = _picker()

    assert all(isinstance(v, int) for v in sp._component_labels(key_to_seed).values())
    assert isinstance(np.int64(1).item(), int)


# --- CLI flags ---------------------------------------------------------------


def _fuzz_kwargs(monkeypatch, tmp_path, extra):
    target = tmp_path / "t"
    target.write_bytes(b"\x7fELF")
    target.chmod(0o755)
    captured = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    monkeypatch.setattr(
        sys, "argv", ["fuzzer-tool", "fuzz", str(target), "-d", str(tmp_path / "c"), *extra]
    )
    commands.main()
    return captured


def test_lottery_flags_reach_fuzzer(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(
        monkeypatch,
        tmp_path,
        [
            "--seed-lottery-scheduler",
            "--seed-lottery-currency",
            "sqrt",
            "--seed-lottery-transfer",
            "0.5",
        ],
    )

    assert kw["seed_lottery_scheduler"] is True
    assert kw["seed_lottery_currency"] == "sqrt"
    assert kw["seed_lottery_transfer"] == 0.5


def test_lottery_flags_default_off(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, [])

    assert kw["seed_lottery_scheduler"] is False
    assert kw["seed_lottery_currency"] == "off"
    assert kw["seed_lottery_transfer"] == 0.0


def test_bad_currency_choice_is_an_argparse_error(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _fuzz_kwargs(monkeypatch, tmp_path, ["--seed-lottery-currency", "bogus"])
