"""Regression test: the original invocation survives --resume.

`fuzzer.invocation` (assigned in cmd_fuzz from argv) was never saved, so a
report produced by a resumed session showed only the `--resume` command and
the command that actually started the campaign was gone. save_state now
persists it and load_state restores it onto `original_invocation` -- a
separate attribute, because cmd_fuzz overwrites `invocation` with the current
argv right after the Fuzzer is constructed, i.e. after load_state has run.

Verifies:
1. save_state writes the invocation into the state section
2. load_state restores it onto original_invocation, leaving invocation alone
3. the original survives a CHAIN of resumes (resume-of-a-resume)
4. a state written before this change loads as "" rather than raising
5. the report prints "Started as" only when the two differ
6. shlex.join'd argv round-trips back through shlex.split
"""

from __future__ import annotations

import shlex
import tempfile
from array import array
from pathlib import Path
from types import SimpleNamespace

from fuzzer_tool.core.state_store import StateStore
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.operators import OperatorEngine
from fuzzer_tool.services.report import _run_summary


class _StubTracker:
    def to_dict(self):
        return {}

    def from_dict(self, data):
        pass


class _StubSaveLoad:
    def save(self):
        return {}

    def load(self, data):
        pass


def _make_fuzzer(tmp_path, **overrides):
    f = SimpleNamespace(
        corpus_dir=tmp_path,
        _state_store=StateStore(tmp_path),
        corpus=[],
        seed_meta={},
        map_size=8192,
        resume=False,
        _use_elo=False,
        _edge_tracker=_StubTracker(),
        _sensitivity=_StubSaveLoad(),
        _crash_mi=_StubSaveLoad(),
        exec_count=0,
        crash_count=0,
        timeout_count=0,
        crash_sigs={},
        crash_frames={},
        crash_min_sizes={},
        op_counts={},
        op_success={},
        op_edges={},
        _operators=OperatorEngine(None),
        _corpus_size_history=array("I"),
    )
    for k, v in overrides.items():
        setattr(f, k, v)
    return f


ORIGINAL = "fuzzer-tool fuzz -c targets/png_read_afl.so"
RESUMED = "fuzzer-tool fuzz --resume -c targets/png_read_afl.so"


def test_save_state_persists_invocation():
    """The command that started the run reaches the state file."""
    with tempfile.TemporaryDirectory() as d:
        f = _make_fuzzer(Path(d), invocation=ORIGINAL)
        CorpusManager(f).save_state()
        assert f._state_store.get("corpus")["invocation"] == ORIGINAL


def test_load_restores_onto_original_not_invocation():
    """load_state must not write f.invocation -- cmd_fuzz clobbers it."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        CorpusManager(_make_fuzzer(tmp, invocation=ORIGINAL)).save_state()

        # Second process: state loaded during __init__, argv assigned after.
        g = _make_fuzzer(tmp, resume=True)
        CorpusManager(g).load_state()
        assert g.original_invocation == ORIGINAL
        g.invocation = RESUMED  # what cmd_fuzz does, a moment later
        assert g.original_invocation == ORIGINAL


def test_original_survives_a_chain_of_resumes():
    """Resuming a resumed session still reports the FIRST command."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        CorpusManager(_make_fuzzer(tmp, invocation=ORIGINAL)).save_state()

        second = _make_fuzzer(tmp, resume=True)
        cm2 = CorpusManager(second)
        cm2.load_state()
        second.invocation = RESUMED
        cm2.save_state()

        third = _make_fuzzer(tmp, resume=True)
        CorpusManager(third).load_state()
        assert third.original_invocation == ORIGINAL


def test_state_without_the_key_loads_as_empty():
    """States written before this change must still resume."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        f = _make_fuzzer(tmp, invocation=ORIGINAL)
        cm = CorpusManager(f)
        cm.save_state()

        # Simulate an older state file: drop the key entirely.
        section = f._state_store.get("corpus")
        del section["invocation"]
        f._state_store.set("corpus", section)
        f._state_store.save()

        g = _make_fuzzer(tmp, resume=True)
        CorpusManager(g).load_state()
        assert g.original_invocation == ""


def test_report_shows_started_as_only_when_it_differs():
    """No "Started as" line on a fresh run, where the two are equal."""
    from tests.test_report import _make_mock_fuzzer

    fresh = _make_mock_fuzzer(invocation=ORIGINAL, original_invocation=ORIGINAL)
    assert "Started as" not in _run_summary(fresh)

    resumed = _make_mock_fuzzer(invocation=RESUMED, original_invocation=ORIGINAL)
    out = _run_summary(resumed)
    assert "Started as" in out
    assert ORIGINAL in out


def test_shlex_join_round_trips_argv_with_spaces():
    """The recorded command is re-runnable, which " ".join was not."""
    argv = ["fuzzer-tool", "fuzz", "--target-args", "-i @@ -o /tmp/out dir", "t.so"]
    assert shlex.split(shlex.join(argv)) == argv
    assert shlex.split(" ".join(argv)) != argv  # the bug this avoids
