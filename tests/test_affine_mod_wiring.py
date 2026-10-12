"""--affine-mod: service, Fuzzer, first-round search, mutate drain, CLI, report.

Targets are Python models of ``if ((a*x + b) % m == t)`` over a field of the
input; the probe reports the compare as cmplog would, ``(v, t)`` little-endian.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.services.affine_mod import MAX_SEARCH_EXECS, QUEUE_MAX, AffineMod
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from tests import test_commands_extended

SEED = bytes(range(1, 17))


def _target(a, b, m, t, off, width, order="little", mapping=None):
    """Probe of a target comparing ``f(field) % m`` to ``t``."""

    def probe(data: bytes):
        if len(data) < off + width:
            return []
        x = int.from_bytes(data[off : off + width], order)
        v = mapping(x) % m if mapping else (a * x + b) % m
        return [(v.to_bytes(width, "little"), t.to_bytes(width, "little"))]

    return probe


def _holds(data, a, b, m, t, off, width, order="little"):
    x = int.from_bytes(data[off : off + width], order)
    return (a * x + b) % m == t


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="affine_mod_") as tmp:
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
                affine_mod=True,
                affine_mod_execs=MAX_SEARCH_EXECS,
            )
        yield f


# ── service ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "m", "t", "off", "width", "order"),
    [
        (37, 11, 65521, 12345, 3, 2, "little"),
        (37, 11, 65521, 12345, 3, 2, "big"),
        (1103515245, 12345, 2**31 - 1, 777777, 1, 4, "little"),
        (7, 3, 251, 200, 5, 1, "little"),
        (31, 0, 10007, 4242, 4, 4, "big"),
    ],
)
def test_search_queues_a_satisfying_input(a, b, m, t, off, width, order):
    probe = _target(a, b, m, t, off, width, order)
    svc = AffineMod(probe)

    svc.search(SEED)

    cand = svc.pop()
    assert cand is not None
    assert _holds(cand, a, b, m, t, off, width, order)
    assert cand[:off] == SEED[:off]
    assert svc.stats["solved"] == 1


def test_operands_already_in_the_input_are_left_to_redqueen():
    """Falsification: Redqueen patches ``v`` when it is spelled in the input."""
    t = 777
    seed = b"\x00\x00" + t.to_bytes(2, "little") + b"\x00"
    probe = _target(1, 0, 65521, t + 1, 2, 2)

    svc = AffineMod(probe)
    svc.search(seed)

    assert svc.queued == 0
    assert svc.stats["execs"] == 1


def test_blind_probe_queues_nothing():
    svc = AffineMod(lambda _data: None)
    svc.search(SEED)
    assert svc.pop() is None


def test_input_that_moves_nothing_queues_nothing():
    """Falsification: a compare no input byte feeds has no field to find."""
    svc = AffineMod(lambda _data: [(b"\x01\x00", b"\x02\x00")])
    svc.search(SEED)
    assert svc.queued == 0


def test_non_affine_target_queues_nothing():
    """Adversarial: ``x*x % m`` must fail closed, not queue a wrong field."""
    probe = _target(0, 0, 65521, 12345, 3, 2, mapping=lambda x: x * x)
    svc = AffineMod(probe)

    svc.search(SEED)

    assert svc.queued == 0
    assert "solved" not in svc.stats
    assert svc.stats["bad_fit"] >= 1


def test_search_respects_its_exec_budget():
    probe = _target(37, 11, 65521, 12345, 3, 2)
    svc = AffineMod(probe, max_execs=10)

    svc.search(SEED)

    assert svc.stats["execs"] <= 10


def test_queue_is_bounded():
    """Adversarial: a flood of seeds cannot grow the queue past QUEUE_MAX."""
    probe = _target(37, 11, 65521, 12345, 3, 2)
    svc = AffineMod(probe)
    for i in range(QUEUE_MAX + 5):
        svc.search(SEED[:3] + bytes([i % 200 + 1, 7]) + SEED[5:])
    assert svc.queued <= QUEUE_MAX


# ── Fuzzer ───────────────────────────────────────────────────────────


def test_fuzzer_builds_service_only_when_asked(fuzzer):
    assert isinstance(fuzzer._affine_mod, AffineMod)
    with tempfile.TemporaryDirectory(prefix="affine_mod_off_") as tmp:
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            off = Fuzzer(
                target="/bin/true",
                corpus_dir=str(Path(tmp) / "corpus"),
                crashes_dir=str(Path(tmp) / "crashes"),
            )
        assert off._affine_mod is None


def test_mutate_drains_queue_without_op_credit(fuzzer):
    """A queued candidate is executed as-is and credits no operator."""
    probe = _target(37, 11, 65521, 12345, 3, 2)
    fuzzer._affine_mod = AffineMod(probe)
    fuzzer._affine_mod.search(SEED)
    fuzzer._last_ops_used = ["stale_op"]

    out = fuzzer.mutate(b"any seed")

    assert _holds(bytes(out), 37, 11, 65521, 12345, 3, 2)
    assert fuzzer._last_ops_used == []
    assert fuzzer._affine_mod.pop() is None


# ── first round ──────────────────────────────────────────────────────


def _begin_round(f, fuzz_count):
    f._affine_mod = MagicMock()
    f.seed_meta = {b"seed": {"fuzz_count": fuzz_count}}
    r = FuzzRound(f, b"seed")
    r._begin()
    r._search_affine()


def test_first_fuzz_is_searched():
    f = MagicMock()
    _begin_round(f, fuzz_count=0)
    f._affine_mod.search.assert_called_once_with(b"seed")


def test_later_fuzz_is_not_searched():
    """Falsification: one search per seed, not one per round."""
    f = MagicMock()
    _begin_round(f, fuzz_count=1)
    f._affine_mod.search.assert_not_called()


def test_seed_without_meta_is_not_searched():
    """Adversarial: no metadata means no fuzz count; never search blindly."""
    f = MagicMock()
    f._affine_mod = MagicMock()
    f.seed_meta = {}
    r = FuzzRound(f, b"seed")
    r._begin()
    r._search_affine()
    f._affine_mod.search.assert_not_called()


def test_round_without_service():
    f = MagicMock()
    f._affine_mod = None
    f.seed_meta = {b"seed": {"fuzz_count": 0}}
    r = FuzzRound(f, b"seed")
    r._begin()
    r._search_affine()


# ── CLI / report ─────────────────────────────────────────────────────


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.affine_mod = flag
    args.affine_mod_execs = 123
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["affine_mod"] is flag
    assert seen["affine_mod_execs"] == 123


def test_hail_mary_arms_the_flag():
    from fuzzer_tool.cli.commands import _HAIL_MARY_FLAGS

    assert "affine_mod" in _HAIL_MARY_FLAGS


def test_report_lines():
    from fuzzer_tool.services.report import _affine_mod_lines

    f = MagicMock()
    f._affine_mod = None
    assert _affine_mod_lines(f) == []

    svc = AffineMod(_target(37, 11, 65521, 12345, 3, 2))
    svc.search(SEED)
    f._affine_mod = svc
    text = "\n".join(_affine_mod_lines(f))
    assert "Affine mod" in text
    assert "solved=1" in text


def test_regression_help_renders_the_percent_sign():
    """argparse %-formats help text: a bare ``% m`` made ``fuzz --help`` raise."""
    import subprocess
    import sys

    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fuzzer_tool.cli.commands import main; main()",
            "fuzz",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert r.returncode == 0, r.stderr
    assert "--affine-mod" in r.stdout
    assert "(a*x + b) % m == t" in r.stdout
