"""Tests for EdgeLedger (core/edge_ledger.py)."""

import pytest

from fuzzer_tool.core.edge_ledger import (
    FAMILY_SHIFT_DEFAULT,
    OCC_CAP,
    PROLOGUE_FRAC,
    EdgeLedger,
    Res,
    Trust,
)
from fuzzer_tool.core.novelty_confirm import confirm
from fuzzer_tool.core.rand_pool import RandPool

SHIFT = 4  # 16 tags per family keeps occupancy tests small


def _e(fam: int, tag: int, shift: int = SHIFT) -> int:
    return (fam << shift) | tag


def _ledger(**kw) -> EdgeLedger:
    return EdgeLedger(ctx_bits=kw.pop("ctx_bits", SHIFT), **kw)


class TestShift:
    def test_ctx_bits_is_family_shift(self):
        assert _ledger().shift == SHIFT

    @pytest.mark.parametrize("ctx", [0, None])
    def test_ctx_free_falls_back(self, ctx):
        assert EdgeLedger(ctx_bits=ctx).shift == FAMILY_SHIFT_DEFAULT


class TestObserve:
    def test_novelty_reports_new_edges_and_families(self):
        led = _ledger()
        nov = led.observe("s1", frozenset({_e(1, 0), _e(1, 1), _e(2, 0)}))
        assert nov.edges == frozenset({_e(1, 0), _e(1, 1), _e(2, 0)})
        assert nov.families == frozenset({1, 2})
        assert nov.level is Res.TAG

    def test_repeat_is_not_novel(self):
        led = _ledger()
        led.observe("s1", frozenset({_e(1, 0)}))
        nov = led.observe("s2", frozenset({_e(1, 0)}))
        assert nov.edges == frozenset()
        assert nov.families == frozenset()

    def test_new_tag_in_known_family_is_tag_novelty(self):
        led = _ledger()
        led.observe("s1", frozenset({_e(1, 0)}))
        nov = led.observe("s2", frozenset({_e(1, 1)}))
        assert nov.families == frozenset({1})
        assert nov.level is Res.TAG

    def test_owner_counts_count_seeds_once(self):
        led = _ledger()
        led.observe("s1", frozenset({_e(1, 0)}))
        led.observe("s1", frozenset({_e(1, 0), _e(1, 1)}))
        led.observe("s2", frozenset({_e(1, 0)}))
        assert led.owner(_e(1, 0)) == 2
        assert led.owner(_e(1, 1)) == 1
        assert led.family_owner(1) == 2
        assert led.n_seeds == 2

    def test_seeds_in_keeps_insertion_order(self):
        led = _ledger()
        for k in ("c", "a", "b"):
            led.observe(k, frozenset({_e(3, 0)}))
        assert led.seeds_in(3) == ["c", "a", "b"]

    def test_edges_in(self):
        led = _ledger()
        led.observe("s", frozenset({_e(1, 0), _e(1, 5), _e(2, 0)}))
        assert sorted(led.edges_in("s", 1)) == [_e(1, 0), _e(1, 5)]
        assert led.edges_in("missing", 1) == []


class TestFrontier:
    def test_prologue_families_excluded(self):
        led = _ledger()
        # family 0 in every seed (prologue), family k only in seed k.
        for k in range(4):
            led.observe(f"s{k}", frozenset({_e(0, 0), _e(k + 1, 0)}))
        assert led.frontier() == [1, 2, 3, 4]

    def test_boundary_is_strict(self):
        # owner/n == PROLOGUE_FRAC exactly is prologue, not frontier.
        led = _ledger()
        led.observe("a", frozenset({_e(1, 0)}))
        led.observe("b", frozenset({_e(2, 0)}))
        assert led.family_owner(1) / led.n_seeds == PROLOGUE_FRAC
        assert led.frontier() == []

    def test_rarest_family(self):
        led = _ledger()
        for k in range(3):
            led.observe(f"s{k}", frozenset({_e(0, 0), _e(9, 0)}))
        led.observe("x", frozenset({_e(0, 0), _e(7, 0)}))
        assert led.rarest_family("x") == 7
        assert led.rarest_family("nope") is None


class TestResolution:
    def test_trust_flip_switches_every_family_without_touching_owners(self):
        led = _ledger()
        for k in range(3):
            led.observe(f"s{k}", frozenset({_e(k, 0), _e(k, 1)}))
        owners = {f: led.family_owner(f) for f in range(3)}
        assert all(led.res(f) is Res.TAG for f in range(3))
        led.set_trust(Trust.UNSTABLE)
        assert all(led.res(f) is Res.FAMILY for f in range(3))
        assert {f: led.family_owner(f) for f in range(3)} == owners

    def test_unstable_new_tag_in_known_family_not_novel(self):
        led = _ledger()
        led.set_trust(Trust.UNSTABLE)
        led.observe("s1", frozenset({_e(1, 0)}))
        nov = led.observe("s2", frozenset({_e(1, 1)}))
        assert nov.edges == frozenset({_e(1, 1)})
        assert nov.families == frozenset()
        nov = led.observe("s3", frozenset({_e(2, 0)}))
        assert nov.families == frozenset({2})
        assert nov.level is Res.FAMILY

    def test_occupancy_cap_forces_family(self):
        led = _ledger()
        tags = 1 << SHIFT
        need = int(OCC_CAP * tags + 0.999999)
        led.observe("s", frozenset(_e(5, t) for t in range(need - 1)))
        assert led.res(5) is Res.TAG
        led.observe("s", frozenset({_e(5, need - 1)}))
        assert led.occupancy(5) >= OCC_CAP
        assert led.res(5) is Res.FAMILY


class TestForget:
    def test_forget_releases_ownership(self):
        led = _ledger()
        led.observe("a", frozenset({_e(1, 0)}))
        led.observe("b", frozenset({_e(1, 0), _e(2, 0)}))
        led.forget("b")
        assert led.owner(_e(1, 0)) == 1
        assert led.family_owner(2) == 0
        assert led.seeds_in(2) == []
        assert led.n_seeds == 1
        # Seen-ness survives: forgetting a seed is not rediscovery.
        assert led.observe("c", frozenset({_e(2, 0)})).edges == frozenset()

    def test_forget_unknown_is_noop(self):
        led = _ledger()
        led.forget("ghost")
        assert led.n_seeds == 0


class TestRelabeling:
    def test_family_outputs_invariant_under_tag_bijection(self):
        # "The id axis is blocked" as an executable statement: permuting tags
        # within each family changes no family-level output.
        rng = RandPool(seed=5)
        tags = list(range(1 << SHIFT))
        perm = {}
        for fam in range(6):
            shuffled = tags[:]
            rng.shuffle(shuffled)
            perm[fam] = dict(zip(tags, shuffled, strict=True))

        def relabel(ids):
            return frozenset(_e(e >> SHIFT, perm[e >> SHIFT][e & 15]) for e in ids)

        stream = []
        for k in range(12):
            fams = [k % 6, (k * 5 + 1) % 6]
            stream.append(
                (f"s{k}", frozenset(_e(f, (k * 3 + j) % 16) for f in fams for j in range(3)))
            )

        a, b = _ledger(), _ledger()
        for key, ids in stream:
            na = a.observe(key, ids)
            nb = b.observe(key, relabel(ids))
            assert na.families == nb.families
            assert len(na.edges) == len(nb.edges)
        assert a.frontier() == b.frontier()
        for fam in range(6):
            assert a.seeds_in(fam) == b.seeds_in(fam)
            assert a.family_owner(fam) == b.family_owner(fam)
            assert a.occupancy(fam) == b.occupancy(fam)
            for key, _ in stream:
                oa = sorted(a.owner(e) for e in a.edges_in(key, fam))
                ob = sorted(b.owner(e) for e in b.edges_in(key, fam))
                assert oa == ob
        assert a.eff_edges() == pytest.approx(b.eff_edges())


class TestPhantoms:
    def test_confirm_gate_restores_phantom_free_ledger(self):
        clean = [(f"s{k}", frozenset({_e(0, 0), _e(k % 3 + 1, k % 4)})) for k in range(9)]
        phantom = {k: frozenset({_e(10 + k, 0)}) for k in range(0, 9, 2)}  # singletons

        def run(gate: bool) -> EdgeLedger:
            led = _ledger()
            for k, (key, ids) in enumerate(clean):
                observed = ids | phantom.get(k, frozenset())
                if gate:
                    c = confirm(True, observed, False, observed, ids)  # rerun lacks phantom
                    observed = c.edge_ids
                led.observe(key, observed)
            return led

        ref = _ledger()
        for key, ids in clean:
            ref.observe(key, ids)

        gated, raw = run(True), run(False)
        assert gated.frontier() == ref.frontier()
        assert gated.eff_edges() == ref.eff_edges()
        # Falsification: without the gate the phantoms enter the frontier.
        assert raw.frontier() != ref.frontier()


class TestPersistence:
    def test_roundtrip(self):
        led = _ledger()
        led.set_trust(Trust.STABLE)
        led.observe("a", frozenset({_e(1, 0), _e(2, 3)}))
        led.observe("b", frozenset({_e(1, 0)}))
        back = EdgeLedger.from_dict(led.to_dict())
        assert back.shift == led.shift
        assert back.trust is Trust.STABLE
        assert back.frontier() == led.frontier()
        assert back.seeds_in(1) == led.seeds_in(1)
        assert back.owner(_e(1, 0)) == 2
        assert back.observe("c", frozenset({_e(2, 3)})).edges == frozenset()

    def test_from_dict_rejects_garbage(self):
        with pytest.raises((KeyError, TypeError, ValueError)):
            EdgeLedger.from_dict({"shift": "x"})


class TestAdversarial:
    def test_empty_ledger(self):
        led = _ledger()
        assert led.frontier() == []
        assert led.eff_edges() == 0.0
        assert led.n_seeds == 0

    def test_empty_observation(self):
        led = _ledger()
        nov = led.observe("s", frozenset())
        assert nov.edges == frozenset() and nov.families == frozenset()
        assert led.n_seeds == 0

    def test_single_family_all_prologue(self):
        led = _ledger()
        for k in range(5):
            led.observe(f"s{k}", frozenset({_e(1, k)}))
        assert led.frontier() == []
