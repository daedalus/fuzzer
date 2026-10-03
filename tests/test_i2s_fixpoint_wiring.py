"""--i2s-fixpoint wiring: CLI -> Fuzzer -> first-fuzz search -> mutate drain."""

import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector
from fuzzer_tool.core.i2s_fixpoint import Outcome
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.i2s_fixpoint import QUEUE_MAX, I2SFixpoint
from tests import test_commands_extended
from tests.test_i2s_fixpoint import _sum_target, _toggle_target, _valid_sum

SEED = b"MG\x00\x09\x00\x00payload"
ITERS = 8


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="i2s_fixpoint_") as tmp:
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
                i2s_fixpoint=True,
                i2s_fixpoint_iters=ITERS,
            )
        yield f


# ── service ──────────────────────────────────────────────────────────


def test_search_queues_consistent_input():
    fx = I2SFixpoint(_sum_target, ITERS)

    result = fx.search(SEED)

    assert result.outcome is Outcome.FIXED
    cand = fx.pop()
    assert cand is not None and _valid_sum(cand)
    assert fx.pop() is None
    assert fx.stats["execs"] == result.execs
    assert fx.stats[Outcome.FIXED.value] == 1


def test_queue_is_bounded():
    """Adversarial: a flood of cycles cannot grow the queue past QUEUE_MAX."""
    fx = I2SFixpoint(_toggle_target, ITERS)
    for i in range(QUEUE_MAX + 5):
        fx.search(struct.pack(">H", 2 * i) + b"zz")
    assert fx.queued == QUEUE_MAX


def test_blind_search_queues_nothing():
    """Falsification: no cmplog reading, no candidate."""
    fx = I2SFixpoint(lambda _x: None, ITERS)
    fx.search(SEED)
    assert fx.pop() is None
    assert fx.stats[Outcome.BLIND.value] == 1


# ── Fuzzer ───────────────────────────────────────────────────────────


def test_probe_reads_only_its_own_run(fuzzer):
    """Records left over from earlier runs are drained, not reported."""
    stale, fresh = [(b"OLD1", b"OLD2")], [(b"NEW1", b"NEW2")]
    cmplog = MagicMock()
    cmplog.last_pairs = []
    ran = []

    def collect():
        cmplog.last_pairs = fresh if ran else stale
        return []

    cmplog.collect_tokens.side_effect = collect
    fuzzer._cmplog = cmplog
    runner = MagicMock()
    runner.run_target.side_effect = lambda data: ran.append(data) or (0, "")
    fuzzer._runner = runner
    execs = fuzzer.exec_count

    with patch.object(fuzzer, "_reset_cmplog"):
        pairs = fuzzer._i2s_probe(b"input")

    assert pairs == fresh
    assert ran == [b"input"]
    assert fuzzer.exec_count == execs + 1


def test_probe_without_cmplog_is_blind(fuzzer):
    fuzzer._cmplog = None
    assert fuzzer._i2s_probe(b"input") is None


def test_fuzzer_builds_service_only_when_asked(fuzzer):
    assert isinstance(fuzzer._i2s_fixpoint, I2SFixpoint)
    with tempfile.TemporaryDirectory(prefix="i2s_fixpoint_off_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            off = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
            )
        assert off._i2s_fixpoint is None


def test_mutate_drains_queue_without_op_credit(fuzzer):
    """A queued candidate is executed as-is and credits no operator."""
    fuzzer._i2s_fixpoint = I2SFixpoint(_sum_target, ITERS)
    fuzzer._i2s_fixpoint.search(SEED)
    fuzzer._last_ops_used = ["stale_op"]

    out = fuzzer.mutate(b"any seed")

    assert _valid_sum(bytes(out))
    assert fuzzer._last_ops_used == []
    assert fuzzer._i2s_fixpoint.pop() is None


# ── first fuzz ───────────────────────────────────────────────────────


def _begin_round(f, fuzz_count: int) -> None:
    f._i2s_fixpoint = MagicMock()
    f.seed_meta = {b"seed": {"fuzz_count": fuzz_count}}
    r = FuzzRound(f, b"seed")
    r._begin()
    r._search_fixpoint()


def test_first_fuzz_is_searched():
    f = MagicMock()
    _begin_round(f, fuzz_count=0)
    f._i2s_fixpoint.search.assert_called_once_with(b"seed")


def test_later_fuzz_is_not_searched():
    """Falsification: one search per seed, not one per round."""
    f = MagicMock()
    _begin_round(f, fuzz_count=1)
    f._i2s_fixpoint.search.assert_not_called()


def test_seed_without_meta_is_not_searched():
    """Adversarial: no metadata means no fuzz count; never search blindly."""
    f = MagicMock()
    f._i2s_fixpoint = MagicMock()
    f.seed_meta = {}
    r = FuzzRound(f, b"seed")
    r._begin()
    r._search_fixpoint()
    f._i2s_fixpoint.search.assert_not_called()


def test_round_without_service():
    f = MagicMock()
    f._i2s_fixpoint = None
    f.seed_meta = {b"seed": {"fuzz_count": 0}}
    r = FuzzRound(f, b"seed")
    r._begin()
    r._search_fixpoint()


# ── cmplog / CLI / report ────────────────────────────────────────────


def test_last_pairs_is_one_drain():
    """Replaced per drain: a later drain never reports an earlier run's pairs."""
    c = CmplogCollector(max_pairs=50, max_tokens=10_000)
    a, b = b"AAAA", b"BBBB"
    c._parse_lines([f"CMP {a.hex()} {b.hex()} 0 {len(a)}"])
    assert (a, b) in c.last_pairs
    c._parse_lines([])
    assert c.last_pairs == []


def test_empty_drain_clears_last_pairs(tmp_path):
    """Adversarial: a drain that returns early must not keep old pairs."""
    c = CmplogCollector(max_pairs=50, max_tokens=10_000)
    c.fifo_sink = False
    c.log_path = str(tmp_path / "missing.log")
    c.last_pairs = [(b"OLD1", b"OLD2")]
    with patch.object(c, "collect_counts"):
        c.collect_tokens()
    assert c.last_pairs == []


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.i2s_fixpoint = flag
    args.i2s_fixpoint_iters = 3
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["i2s_fixpoint"] is flag
    assert seen["i2s_fixpoint_iters"] == 3


def test_report_lines():
    from fuzzer_tool.services.report import _i2s_fixpoint_lines

    f = MagicMock()
    f._i2s_fixpoint = None
    assert _i2s_fixpoint_lines(f) == []

    fx = I2SFixpoint(_sum_target, ITERS)
    fx.search(SEED)
    f._i2s_fixpoint = fx
    text = "\n".join(_i2s_fixpoint_lines(f))
    assert "I2S fixpoint" in text
    assert f"{Outcome.FIXED.value}=1" in text


def test_first_round_feeds_next_mutate(fuzzer):
    """End to end: a seed's first round queues its fixed point for the next."""
    fuzzer._i2s_fixpoint = I2SFixpoint(_sum_target, ITERS)
    fuzzer.save_to_corpus(SEED)
    with (
        patch.object(fuzzer, "_dedup_mutate", return_value=b"boring"),
        patch.object(fuzzer, "_run_target", return_value=(0, "")),
        patch.object(fuzzer, "_is_crash", return_value=False),
        patch.object(fuzzer, "_is_interesting", return_value=False),
    ):
        fuzzer.fuzz_one(SEED)

    assert fuzzer._i2s_fixpoint.queued == 1
    assert _valid_sum(bytes(fuzzer.mutate(SEED)))


def test_regression_probe_tokens_reach_dictionary(fuzzer):
    """Tokens the probe drains are marked known by cmplog; keep them in the dictionary."""
    cmplog = MagicMock()
    cmplog.last_pairs = []
    cmplog.collect_tokens.side_effect = [[b"STALE_TOK"], [b"FRESH_TOK"]]
    fuzzer._cmplog = cmplog
    fuzzer._runner = MagicMock()

    with patch.object(fuzzer, "_reset_cmplog"):
        fuzzer._i2s_probe(b"input")

    assert b"STALE_TOK" in fuzzer.dictionary
    assert b"FRESH_TOK" in fuzzer.dictionary
