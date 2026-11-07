"""A collect round must not wipe the exec-boundary per-site cmplog vector.

``_execute`` drains the counters (``collect_counts``); ``_mine_cmplog`` then
calls ``collect_tokens``, which drained them a second time. That second drain
reset ``last_site_asserted`` to ``{}`` with no exec in between, so
``_cmp_progress`` read an empty per-site vector on every collect round.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.core.cmplog import CmplogCollector
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer

_SITE_A = ("trace_cmp4", 0x401000)
_SITE_B = ("trace_cmp8", 0x402000)


def _cns(site: tuple[str, int], fired: int, asserted: int) -> str:
    return f"CNS {site[0]} 0x{site[1]:x} {fired} {asserted}\n"


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="collect_site_") as tmp:
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
            )

        # Real collector reading a sites file; no shim to rewind.
        f._cmplog.stop()
        c = CmplogCollector(site_counts=True, workdir=tmp)
        c.sites_path = str(Path(tmp) / "sites")
        f._cmplog = c
        with patch.object(f, "_rewind_cmplog_shim"):
            yield f
        c.stop()


def _append(f, line: str) -> None:
    with open(f._cmplog.sites_path, "a") as fh:
        fh.write(line)


def _exec_drain(f) -> None:
    # What FuzzRound._execute does on the execution boundary.
    f._last_cmp_fired, f._last_cmp_asserted = f._cmplog.collect_counts()


def test_regression_collect_tokens_keeps_site_vector(fuzzer):
    """_cmp_progress sees this exec's per-site vector on a collect round."""
    _append(fuzzer, _cns(_SITE_A, 3, 2))
    _exec_drain(fuzzer)

    r = FuzzRound(fuzzer, b"seed")
    r._collect_tokens()
    assert r._collect_now  # pool < 500: every round collects

    seen = []
    with patch.object(fuzzer, "_record_cmp_progress", lambda a, s: seen.append(dict(s))):
        r._cmp_progress()
    assert seen == [{_SITE_A: 2}]


def test_late_deltas_reach_next_exec_drain(fuzzer):
    """Adversarial: deltas written after the exec drain are not lost by skipping."""
    _append(fuzzer, _cns(_SITE_A, 1, 1))
    _exec_drain(fuzzer)

    # Written between the exec drain and the collect (e.g. a shim flush).
    _append(fuzzer, _cns(_SITE_B, 4, 3))
    FuzzRound(fuzzer, b"seed")._collect_tokens()

    _exec_drain(fuzzer)
    assert fuzzer._cmplog.last_site_asserted == {_SITE_B: 3}
    assert fuzzer._cmplog.site_asserted == {_SITE_A: 1, _SITE_B: 3}


def test_default_collect_tokens_still_drains_counts(fuzzer):
    """Falsification: callers without an exec-boundary drain keep the combined drain."""
    _append(fuzzer, _cns(_SITE_A, 2, 1))
    fuzzer._cmplog.collect_tokens()
    assert fuzzer._cmplog.last_site_asserted == {_SITE_A: 1}
