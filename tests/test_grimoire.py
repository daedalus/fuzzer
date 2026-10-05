"""Grimoire: grammar-free structure inference (``--grimoire``).

Covers core/grimoire.py (generalization + the three mutators),
services/grimoire.py (novelty bookkeeping, once-per-seed stage) and the
operator wiring. A *probe* is ``candidate -> still reaches the novelties``.
"""

import pytest

from fuzzer_tool.core import grimoire as gr
from fuzzer_tool.core.grimoire import GAP, GrimoireBook, strip
from fuzzer_tool.services.grimoire import GrimoireStage
from tests.support.scripted_rng import ScriptedRng


def _contains(token):
    return lambda cand: token in cand


def _counting(probe):
    calls = []

    def run(cand):
        calls.append(cand)
        return probe(cand)

    run.calls = calls
    return run


def _gen(data, probe, max_execs=10_000):
    return gr._Gen(data, probe, max_execs)


def _alive(g):
    return bytes(i for i, a in enumerate(g.alive) if a)


class TestStrip:
    def test_drops_gaps_keeps_order(self):
        assert strip((GAP, b"ab", GAP, b"cd", GAP)) == b"abcd"

    def test_all_gaps_is_empty(self):
        assert strip((GAP, GAP)) == b""


class TestGeneralize:
    def test_dead_bytes_become_gaps(self):
        probe = _contains(b"if(x)")
        out = gr.generalize(b"junk;if(x);more", probe, max_execs=1000)

        assert out is not None
        assert out.items == (GAP, b"if(x)", GAP)

    def test_kept_bytes_alone_reproduce_novelty(self):
        probe = _contains(b"if(x)")
        out = gr.generalize(b"junk;if(x);more", probe, max_execs=1000)

        assert probe(strip(out.items))

    def test_everything_live_keeps_whole_input_between_end_gaps(self):
        data = b"abcdef"
        out = gr.generalize(data, lambda c: c == data, max_execs=1000)

        assert out.items == (GAP, data, GAP)

    def test_unstable_seed_is_rejected(self):
        run = _counting(lambda c: False)

        assert gr.generalize(b"abc", run, max_execs=100) is None
        assert len(run.calls) == 1

    def test_empty_input_runs_nothing(self):
        run = _counting(lambda c: True)

        assert gr.generalize(b"", run, max_execs=100) is None
        assert run.calls == []

    def test_zero_budget_runs_nothing(self):
        run = _counting(lambda c: True)

        assert gr.generalize(b"abc", run, max_execs=0) is None
        assert run.calls == []

    def test_budget_is_a_hard_cap(self):
        run = _counting(_contains(b"k"))
        out = gr.generalize(b"a" * 300 + b"k" + b"b" * 300, run, max_execs=40)

        assert len(run.calls) <= 40
        assert out.execs == len(run.calls)

    def test_exhausted_budget_stays_conservative(self):
        # Only the verification run fits: nothing is proven dead.
        data = b"junk;if(x);more"
        out = gr.generalize(data, _contains(b"if(x)"), max_execs=1)

        assert out.items == (GAP, data, GAP)

    def test_oversize_input_is_declined(self):
        run = _counting(lambda c: True)

        assert gr.generalize(b"a" * (gr.GENERALIZE_MAX_LEN + 1), run, max_execs=100) is None
        assert run.calls == []

    def test_probe_is_never_asked_about_the_empty_input(self):
        run = _counting(lambda c: True)
        out = gr.generalize(b"abcdefgh", run, max_execs=1000)

        assert b"" not in run.calls
        # Probe accepts anything non-empty: one byte must survive.
        assert len(strip(out.items)) >= 1

    def test_adjacent_gaps_merge(self):
        out = gr.generalize(b"xxkxx", _contains(b"k"), max_execs=1000)

        assert out.items == (GAP, b"k", GAP)
        assert all(
            not (a is GAP and b is GAP) for a, b in zip(out.items, out.items[1:], strict=False)
        )


class TestStages:
    def test_offsets_drops_in_chunks(self):
        g = _gen(b"aaaaaaaaKaaaaaaaa", _contains(b"K"))
        gr._offsets(g)

        assert _alive(g) == bytes([8])

    def test_delims_drops_up_to_each_delimiter(self):
        g = _gen(b"aa;bbbb;cc", lambda c: c.startswith(b"aa;"))
        gr._delims(g)

        assert _alive(g) == bytes([0, 1, 2])

    def test_delims_skips_a_delimiter_free_input_without_spending(self):
        g = _gen(b"abcdef", lambda c: True)
        gr._delims(g)

        # No delimiter present: only the tail-after-last-delimiter window runs
        # once per delimiter class that occurs, i.e. never.
        assert g.execs == 0

    def test_brackets_drop_whole_closure_furthest_first(self):
        data = b"f(a(b)c)Z"
        g = _gen(data, lambda c: c.startswith(b"f") and c.endswith(b"Z"))
        gr._brackets(g)

        assert _alive(g) == bytes([0, 8])

    def test_brackets_fall_back_to_nearer_closer(self):
        # Removing through the far ")" breaks the probe; the near one works.
        data = b"f(ab)X)Z"
        g = _gen(data, _contains(b"X)Z"))
        gr._brackets(g)

        assert strip_alive(g) == b"fX)Z" or strip_alive(g) == b"X)Z"
        assert b"X)Z" in strip_alive(g)

    def test_quotes_pair_on_the_same_byte(self):
        data = b'say "hi there" end'
        g = _gen(data, lambda c: c.startswith(b"say ") and c.endswith(b" end"))
        gr._brackets(g)

        assert strip_alive(g) == b"say  end"

    def test_unmatched_open_bracket_spends_nothing(self):
        g = _gen(b"((((", lambda c: True)
        gr._brackets(g)

        assert g.execs == 0

    def test_closer_tries_are_bounded(self):
        data = b"(" + b")" * 100
        g = _gen(data, lambda c: False)
        gr._brackets(g)

        assert g.execs <= gr.MAX_CLOSER_TRIES * (len(data))


def strip_alive(g):
    return bytes(b for b, a in zip(g.data, g.alive, strict=True) if a)


def _book(*pairs, max_len=4096):
    book = GrimoireBook(max_len=max_len)
    for data, items in pairs:
        book.add(data, items)
    return book


K1 = (GAP, b"AB", GAP)
K2 = (GAP, b"CD", GAP)


class TestExtend:
    def test_appends_another_inputs_tokens(self):
        book = _book((b"k1", K1), (b"k2", K2))
        rng = ScriptedRng(choice_idxs=[1], randoms=[0.9])

        assert book.extend(b"xyz", rng) == b"xyzCD"

    def test_prepends_on_low_draw(self):
        book = _book((b"k1", K1), (b"k2", K2))
        rng = ScriptedRng(choice_idxs=[1], randoms=[0.1])

        assert book.extend(b"xyz", rng) == b"CDxyz"

    def test_empty_book_declines(self):
        assert GrimoireBook(max_len=64).extend(b"xyz", ScriptedRng()) is None

    def test_all_gap_donor_declines(self):
        book = _book((b"k", (GAP,)))

        assert book.extend(b"xyz", ScriptedRng(choice_idxs=[0], randoms=[0.9])) is None

    def test_clamped_to_max_len(self):
        book = _book((b"k1", (GAP, b"A" * 50, GAP)), max_len=8)
        out = book.extend(b"xyz", ScriptedRng(choice_idxs=[0], randoms=[0.9]))

        assert len(out) == 8
        assert out.startswith(b"xyz")


class TestRecurse:
    def test_gap_replaced_by_another_generalized_input(self):
        book = _book((b"k", (GAP, b"<", GAP, b">", GAP)), (b"j", (GAP, b"x", GAP)))
        rng = ScriptedRng(randints=[1], choice_idxs=[1, 1])
        # depth 1; gap #1 (the middle one); donor j.

        assert book.recurse(b"k", rng) == b"<x>"

    def test_donor_gaps_allow_deeper_nesting(self):
        book = _book((b"k", (GAP, b"<", GAP, b">", GAP)), (b"j", (GAP, b"x", GAP)))
        # depth 2. Step 1 leaves gaps at items [0, 2, 4, 6]; picking list
        # index 2 (item 4, the donor's own trailing gap) and the donor again
        # nests it: "<xx>".
        rng = ScriptedRng(randints=[2], choice_idxs=[1, 1, 2, 1])

        assert book.recurse(b"k", rng) == b"<xx>"

    def test_unknown_parent_declines(self):
        book = _book((b"k", K1))

        assert book.recurse(b"other", ScriptedRng()) is None

    def test_parent_without_gap_declines(self):
        book = _book((b"k", (b"AB",)))

        assert book.recurse(b"k", ScriptedRng(randints=[1])) is None

    def test_growth_is_bounded_by_max_len(self):
        book = _book((b"k", (GAP, b"A" * 30, GAP)), max_len=64)
        idx = [0, 0] * gr.MAX_DEPTH
        out = book.recurse(b"k", ScriptedRng(randints=[gr.MAX_DEPTH], choice_idxs=idx))

        assert len(out) <= 64


class TestStringReplace:
    def test_first_occurrence_swapped(self):
        book = _book((b"k", (GAP, b"GET", GAP, b"POST", GAP)))
        # strings: [GET, POST]; pick GET (present), then POST; coin 0.1 = first only.
        rng = ScriptedRng(choice_idxs=[0, 1], randoms=[0.1])

        assert book.replace(b"GET GET /", rng) == b"POST GET /"

    def test_all_occurrences_swapped(self):
        book = _book((b"k", (GAP, b"GET", GAP, b"POST", GAP)))
        rng = ScriptedRng(choice_idxs=[0, 1], randoms=[0.9])

        assert book.replace(b"GET GET /", rng) == b"POST POST /"

    def test_absent_strings_decline_after_bounded_tries(self):
        book = _book((b"k", (GAP, b"GET", GAP, b"POST", GAP)))
        rng = ScriptedRng(choice_idxs=[0] * gr.REPLACE_TRIES)

        assert book.replace(b"nothing here", rng) is None

    def test_same_string_twice_declines(self):
        book = _book((b"k", (GAP, b"GET", GAP, b"POST", GAP)))
        rng = ScriptedRng(choice_idxs=[0, 0], randoms=[0.9])

        assert book.replace(b"GET /", rng) is None

    def test_needs_two_strings(self):
        book = _book((b"k", (GAP, b"GET", GAP)))

        assert book.replace(b"GET /", ScriptedRng(choice_idxs=[0, 0], randoms=[0.9])) is None

    def test_result_clamped_to_max_len(self):
        book = _book((b"k", (GAP, b"ab", GAP, b"X" * 40, GAP)), max_len=10)
        rng = ScriptedRng(choice_idxs=[0, 1], randoms=[0.9])

        assert len(book.replace(b"ab ab ab", rng)) == 10


class TestBookBounds:
    def test_seed_table_is_lru_bounded(self):
        book = GrimoireBook(max_len=64)
        for i in range(gr.MAX_SEEDS + 20):
            book.add(b"seed%d" % i, (GAP, b"t%d" % i, GAP))

        assert book.seeds <= gr.MAX_SEEDS
        assert book.items_of(b"seed0") is None
        assert book.items_of(b"seed%d" % (gr.MAX_SEEDS + 19)) is not None

    def test_string_pool_is_bounded_and_deduplicated(self):
        book = GrimoireBook(max_len=64)
        for i in range(gr.MAX_STRINGS + 50):
            book.add(b"s%d" % i, (GAP, b"tok%d" % i, GAP))
        book.add(b"dup", (GAP, b"tok1", GAP, b"tok1", GAP))

        assert book.string_count <= gr.MAX_STRINGS

    def test_short_tokens_are_not_pooled(self):
        book = GrimoireBook(max_len=64)
        book.add(b"k", (GAP, b"a", GAP))

        assert book.string_count == 0


class _FakeTarget:
    """Edge set of one run: the set of 'features' present in the input."""

    FEATURES = {b"if(x)": 1, b"while": 2, b"{}": 3}

    def __init__(self):
        self.runs = 0

    def __call__(self, cand):
        self.runs += 1
        return {e for tok, e in self.FEATURES.items() if tok in cand}


class TestStage:
    def test_generalizes_against_noted_novelty(self):
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=500, max_len=4096)
        data = b"junk;if(x);while;more"
        stage.note(data, {1})

        stage.generalize(data)

        assert stage.book.items_of(data) == (GAP, b"if(x)", GAP)

    def test_without_novelty_uses_the_seeds_full_edge_set(self):
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=500, max_len=4096)
        data = b"junk;if(x);while;more"

        stage.generalize(data)

        strip_ = strip(stage.book.items_of(data))
        assert b"if(x)" in strip_ and b"while" in strip_

    def test_seed_with_no_edges_is_skipped(self):
        # An empty target set is vacuously preserved by deleting everything.
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=500, max_len=4096)

        stage.generalize(b"no features at all")

        assert stage.book.items_of(b"no features at all") is None

    def test_empty_novelty_note_is_ignored(self):
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=500, max_len=4096)
        data = b"if(x) while"
        stage.note(data, set())

        stage.generalize(data)

        assert b"if(x)" in strip(stage.book.items_of(data))

    def test_tried_table_is_bounded_and_keeps_old_seeds_out(self):
        from fuzzer_tool.services import grimoire as svc

        stage = GrimoireStage(_FakeTarget(), max_execs=10, max_len=4096)
        for i in range(svc._TRIED_MAX + 50):
            stage.generalize(b"s%d" % i)

        assert stage.tried <= svc._TRIED_MAX

    def test_runs_once_per_seed(self):
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=500, max_len=4096)
        data = b"junk;if(x);more"
        stage.note(data, {1})
        stage.generalize(data)
        before = target.runs

        assert stage.generalize(data) == 0
        assert target.runs == before

    def test_unstable_seed_is_remembered_not_retried(self):
        runs = []

        def flaky(cand):
            runs.append(cand)
            return set()

        stage = GrimoireStage(flaky, max_execs=500, max_len=4096)
        stage.note(b"abc", {1})
        stage.generalize(b"abc")
        stage.generalize(b"abc")

        assert len(runs) == 1
        assert stage.book.items_of(b"abc") is None

    def test_oversize_seed_is_not_noted_or_run(self):
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=500, max_len=4096)
        big = b"a" * (gr.GENERALIZE_MAX_LEN + 1)
        stage.note(big, {1})

        assert stage.generalize(big) == 0
        assert target.runs == 0

    def test_returns_execs_spent(self):
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=500, max_len=4096)
        data = b"junk;if(x);more"
        stage.note(data, {1})

        assert stage.generalize(data) == target.runs

    def test_budget_caps_the_stage(self):
        target = _FakeTarget()
        stage = GrimoireStage(target, max_execs=10, max_len=4096)
        data = b"a" * 200 + b"if(x)" + b"b" * 200
        stage.note(data, {1})

        assert stage.generalize(data) <= 10

    def test_novelty_table_is_bounded(self):
        stage = GrimoireStage(_FakeTarget(), max_execs=10, max_len=4096)
        for i in range(gr.MAX_SEEDS * 3):
            stage.note(b"s%d" % i, {1})

        assert stage.noted <= gr.MAX_SEEDS * 2


class TestWiring:
    def test_ops_registered_and_gated(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        names = {"grimoire_extend", "grimoire_recurse", "grimoire_string"}
        assert names <= set(REGISTRY.names())
        assert {REGISTRY.category_of(n) for n in names} == {"structural"}

    def test_ops_unavailable_without_flag(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        class F:
            _grimoire = None

        avail = set(REGISTRY.available(F(), b"x"))

        assert not {"grimoire_extend", "grimoire_recurse", "grimoire_string"} & avail

    def test_ops_available_when_book_populated(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        stage = GrimoireStage(_FakeTarget(), max_execs=10, max_len=64)
        stage.book.add(b"p", (GAP, b"if", GAP, b"while", GAP))

        class F:
            _grimoire = stage

        avail = set(REGISTRY.available(F(), b"p"))

        assert {"grimoire_extend", "grimoire_recurse", "grimoire_string"} <= avail

    def test_recurse_needs_a_generalized_parent(self):
        from fuzzer_tool.core.operator_registry import REGISTRY

        stage = GrimoireStage(_FakeTarget(), max_execs=10, max_len=64)
        stage.book.add(b"p", (GAP, b"if", GAP, b"while", GAP))

        class F:
            _grimoire = stage

        avail = set(REGISTRY.available(F(), b"unseen"))

        assert "grimoire_recurse" not in avail
        assert "grimoire_extend" in avail

    @pytest.mark.parametrize("op", ["grimoire_extend", "grimoire_recurse", "grimoire_string"])
    def test_ops_are_delocalised(self, op):
        from fuzzer_tool.services.operators import _DELOCALISED_OPS

        assert op in _DELOCALISED_OPS

    def test_flag_builds_stage_and_default_does_not(self):
        from tests.test_regression_analyzer_registry import _build_fuzzer

        on = _build_fuzzer(grimoire=True)
        off = _build_fuzzer()

        assert type(on._grimoire).__name__ == "GrimoireStage"
        assert off._grimoire is None

    def test_cli_flag_present_and_in_hail_mary(self):
        from fuzzer_tool.cli import commands

        assert "grimoire" in commands._HAIL_MARY_FLAGS


class TestRoundHooks:
    @staticmethod
    def _round(fuzz_count, stage):
        from unittest.mock import MagicMock

        from fuzzer_tool.services.fuzz_round import FuzzRound

        f = MagicMock()
        f._grimoire = stage
        f.seed_meta = {b"seed": {"fuzz_count": fuzz_count}}
        r = FuzzRound(f, b"seed")
        r._begin()
        return f, r

    @pytest.mark.parametrize("fuzz_count", [0, 1, 2, 7])
    def test_every_round_offers_the_seed_to_the_stage(self, fuzz_count):
        # Other paths bump fuzz_count before a first FuzzRound, so the round
        # must not gate on it; the stage owns once-per-seed.
        from unittest.mock import MagicMock

        stage = MagicMock()
        _, r = self._round(fuzz_count, stage)
        r._generalize()

        stage.generalize.assert_called_once_with(b"seed")

    def test_seed_without_meta_is_not_generalized(self):
        from unittest.mock import MagicMock

        from fuzzer_tool.services.fuzz_round import FuzzRound

        stage = MagicMock()
        f = MagicMock()
        f._grimoire = stage
        f.seed_meta = {}
        r = FuzzRound(f, b"seed")
        r._begin()
        r._generalize()

        stage.generalize.assert_not_called()

    def test_round_without_stage_is_a_noop(self):
        _, r = self._round(0, None)
        r._generalize()
        r._note_novelty()

    def test_novelty_noted_only_on_new_coverage(self):
        from unittest.mock import MagicMock

        stage = MagicMock()
        f, r = self._round(0, stage)
        f._last_new_edge_ids = [3, 4]
        r._mutated = b"mutant"

        r._has_new_coverage = False
        r._note_novelty()
        stage.note.assert_not_called()

        r._has_new_coverage = True
        r._note_novelty()
        stage.note.assert_called_once_with(b"mutant", [3, 4])
