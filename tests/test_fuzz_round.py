"""FuzzRound: the per-iteration class ``Fuzzer.fuzz_one`` delegates to.

One instance per round, so a round's outcome flags cannot leak into the next.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer

_SEGV = (139, "segv")
_CLEAN = (0, "")


@pytest.fixture
def fuzzer():
    with tempfile.TemporaryDirectory(prefix="fuzz_round_") as tmp:
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
                mutations_per_input=2,
            )
        yield f


def _round(f, mutant: bytes, result: tuple[int, str], crash: bool) -> bool:
    with (
        patch.object(f, "_dedup_mutate", return_value=mutant),
        patch.object(f, "_run_target", return_value=result),
        patch.object(f, "_is_crash", return_value=crash),
        patch.object(f, "_is_interesting", return_value=False),
    ):
        return f.fuzz_one(f.corpus[0])


def test_fuzz_one_delegates(fuzzer):
    """Falsification: fuzz_one's answer is FuzzRound.run's, for this data."""
    seen = []

    def fake_run(self):
        seen.append(self._data)
        return "sentinel"

    with patch.object(FuzzRound, "run", fake_run):
        assert fuzzer.fuzz_one(b"SEED") == "sentinel"
    assert seen == [b"SEED"]


def test_crash_does_not_leak_into_next_round(fuzzer):
    """Adversarial: a crash round followed by a clean one must read as clean."""
    crashes_before = fuzzer.crash_count

    assert _round(fuzzer, b"CRASHY01", _SEGV, crash=True) is True
    assert fuzzer.crash_count == crashes_before + 1

    assert _round(fuzzer, b"BORING01", _CLEAN, crash=False) is False
    assert fuzzer.crash_count == crashes_before + 1


def test_round_rejects_stray_state(fuzzer):
    """Adversarial: per-round state is declared, not grown ad hoc."""
    rnd = FuzzRound(fuzzer, b"SEED")
    with pytest.raises(AttributeError):
        rnd.undeclared = 1
