"""--advisors: one zero-cost proposal per advisor per stage (soksolv advisors.go)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.cond_stmt import LazyConds
from fuzzer_tool.services.advisors import (
    MAX_ADVISORS,
    SEEN_MAX,
    AdvisorCtx,
    AdvisorStage,
    CondAdvisor,
    I2SAdvisor,
    TokenAdvisor,
    default_advisors,
)
from fuzzer_tool.services.seed_stage import PERF_NEUTRAL, SeedStage
from tests import test_commands_extended

SEED = b"00AB00CD"


def _ctx(meta=None, dictionary=(), conds=(), fresh=lambda _c: True, max_len=64):
    return AdvisorCtx(
        seed_meta={SEED: meta} if meta is not None else {},
        dictionary=list(dictionary),
        conds=lambda: conds,
        fresh=fresh,
        max_len=max_len,
    )


class _Fixed:
    """Advisor proposing a scripted sequence, one value per call."""

    def __init__(self, name: str, *proposals):
        self.name = name
        self._it = iter(proposals)
        self.calls = 0

    def propose(self, seed, ctx):
        self.calls += 1
        return next(self._it, None)


# ── advisors ────────────────────────────────────────────────────────


def test_i2s_replays_first_holding_match():
    meta = {"redqueen_matches": [(2, b"AB", b"XY"), (6, b"CD", b"ZZ")]}
    assert I2SAdvisor().propose(SEED, _ctx(meta)) == b"00XY00CD"


def test_i2s_skips_stale_and_tried_matches():
    """Falsification: a match whose bytes moved, or a tried result, is passed over."""
    meta = {"redqueen_matches": [(0, b"AB", b"QQ"), (2, b"AB", b"XY"), (6, b"CD", b"ZZ")]}
    tried = {b"00XY00CD"}
    out = I2SAdvisor().propose(SEED, _ctx(meta, fresh=lambda c: c not in tried))
    assert out == b"00AB00ZZ"


def test_i2s_no_meta_no_proposal():
    assert I2SAdvisor().propose(SEED, _ctx()) is None
    assert I2SAdvisor().propose(SEED, _ctx({})) is None


def test_token_writes_newest_token_at_cmp_offset():
    meta = {"redqueen_offsets": [2]}
    out = TokenAdvisor().propose(SEED, _ctx(meta, dictionary=[b"old", b"KK"]))
    assert out == b"00KK00CD"


def test_token_clips_at_seed_end():
    """Adversarial: a token past the end overwrites what fits, length kept."""
    meta = {"redqueen_offsets": [7]}
    out = TokenAdvisor().propose(SEED, _ctx(meta, dictionary=[b"LONG"]))
    assert out == b"00AB00CL"


def test_token_offset_past_end_declines():
    meta = {"redqueen_offsets": [len(SEED)]}
    assert TokenAdvisor().propose(SEED, _ctx(meta, dictionary=[b"KK"])) is None


def test_cond_swaps_operand_found_in_seed():
    conds = LazyConds([(b"ZZ", b"YY"), (b"CD", b"EF")], {}, {}).unsolved_view()
    assert CondAdvisor().propose(SEED, _ctx(conds=conds)) == b"00AB00EF"


def test_cond_without_pool_declines():
    assert CondAdvisor().propose(SEED, _ctx(conds=())) is None
    assert CondAdvisor().propose(b"", _ctx(conds=())) is None


# ── stage ───────────────────────────────────────────────────────────


def test_plan_queues_one_per_advisor_in_order():
    stage = AdvisorStage([_Fixed("a", b"A1"), _Fixed("b", b"B1")])
    names = stage.plan(SEED, _ctx(fresh=stage.fresh))
    assert names == ["a", "b"]
    assert [stage.pop(), stage.pop(), stage.pop()] == [b"A1", b"B1", None]


def test_plan_dedups_identical_proposals():
    """Two advisors on the same move run it once; a later stage skips it too."""
    stage = AdvisorStage([_Fixed("a", b"SAME", b"SAME"), _Fixed("b", b"SAME", b"NEW")])
    assert stage.plan(SEED, _ctx(fresh=stage.fresh)) == ["a"]
    assert stage.plan(SEED, _ctx(fresh=stage.fresh)) == ["b"]
    assert [stage.pop(), stage.pop(), stage.pop()] == [b"SAME", b"NEW", None]


def test_plan_rejects_seed_itself():
    """Falsification: proposing the parent is no move."""
    stage = AdvisorStage([_Fixed("a", SEED)])
    assert stage.plan(SEED, _ctx(fresh=stage.fresh)) == []
    assert stage.pop() is None


def test_plan_caps_advisors():
    advisors = [_Fixed(f"a{i}", bytes([i])) for i in range(MAX_ADVISORS + 2)]
    stage = AdvisorStage(advisors)
    assert len(stage.plan(SEED, _ctx(fresh=stage.fresh))) == MAX_ADVISORS
    assert advisors[-1].calls == 0


def test_seen_set_is_bounded():
    """Adversarial: endless distinct proposals keep memory at SEEN_MAX."""
    proposals = [i.to_bytes(4, "little") for i in range(SEEN_MAX + 5)]
    stage = AdvisorStage([_Fixed("a", *proposals)])
    for _ in proposals:
        stage.plan(SEED, _ctx(fresh=stage.fresh))
        stage.pop()
    assert stage.seen_size() == SEEN_MAX
    assert stage.fresh(proposals[0])  # evicted first
    assert not stage.fresh(proposals[-1])


def test_credit_and_stats():
    stage = AdvisorStage([_Fixed("a", b"A1"), _Fixed("b", None)])
    stage.plan(SEED, _ctx(fresh=stage.fresh))
    stage.credit("a", True)
    stats = stage.stats
    assert stats["stages"] == 1
    assert stats["execs"] == 1
    assert stats["a"] == (1, 1)
    assert stats["b"] == (0, 0)


def test_default_advisors_names():
    assert [a.name for a in default_advisors()] == ["i2s", "token", "cond"]


# ── wiring ──────────────────────────────────────────────────────────


@pytest.fixture
def fuzzer():
    from fuzzer_tool.services.fuzzer import Fuzzer

    with tempfile.TemporaryDirectory(prefix="advisors_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
                max_len=256,
                timeout=1,
                advisors=True,
            )
        yield f


def test_fuzzer_builds_stage_only_when_asked(fuzzer):
    from fuzzer_tool.services.fuzzer import Fuzzer

    assert isinstance(fuzzer._advisors, AdvisorStage)
    with tempfile.TemporaryDirectory(prefix="advisors_off_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            off = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
            )
        assert off._advisors is None


def test_advise_runs_each_proposal_free(fuzzer):
    """Each proposal is one fuzz_one of the seed; the stage's energy is untouched."""
    fuzzer._advisors = AdvisorStage([_Fixed("a", b"P1"), _Fixed("b", b"P2")])
    fuzzer._stage = SeedStage(4)
    fuzzer._stage.begin(SEED, PERF_NEUTRAL)
    left = fuzzer._stage._left
    ran = []

    def fake_one(data):
        ran.append((data, bytes(fuzzer.mutate(data))))
        return data == SEED and len(ran) == 1

    with patch.object(fuzzer, "fuzz_one", side_effect=fake_one):
        fuzzer._advise(SEED)

    assert ran == [(SEED, b"P1"), (SEED, b"P2")]
    assert fuzzer._stage._left == left
    assert fuzzer._advisors.stats["a"] == (1, 1)
    assert fuzzer._advisors.stats["b"] == (1, 0)


def test_mutate_drains_proposal_without_op_credit(fuzzer):
    fuzzer._advisors = AdvisorStage([_Fixed("a", b"PROPOSAL")])
    fuzzer._advisors.plan(SEED, _ctx(fresh=fuzzer._advisors.fresh))
    fuzzer._last_ops_used = ["stale_op"]

    assert bytes(fuzzer.mutate(SEED)) == b"PROPOSAL"
    assert fuzzer._last_ops_used == []
    assert fuzzer._advisors.pop() is None


def test_advise_ctx_reads_seed_state(fuzzer):
    """The real ctx carries the seed's redqueen matches to the i2s advisor."""
    fuzzer.seed_meta[SEED] = {"redqueen_matches": [(2, b"AB", b"XY")], "fuzz_count": 0}
    seen = []
    with patch.object(fuzzer, "fuzz_one", side_effect=lambda d: seen.append(fuzzer.mutate(d))):
        fuzzer._advise(SEED)
    assert bytes(seen[0]) == b"00XY00CD"


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.advisors = flag
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["advisors"] is flag


def test_report_lines():
    from fuzzer_tool.services.report import _advisor_lines

    f = MagicMock()
    f._advisors = None
    assert _advisor_lines(f) == []

    stage = AdvisorStage([_Fixed("a", b"A1")])
    stage.plan(SEED, _ctx(fresh=stage.fresh))
    stage.credit("a", True)
    f._advisors = stage
    assert "Advisors: 1 execs / 1 stages, proposed/admitted a=1/1" in "\n".join(_advisor_lines(f))
