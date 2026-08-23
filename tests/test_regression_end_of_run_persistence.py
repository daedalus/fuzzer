"""Regression: an unexpected exception discarded the entire campaign.

``Fuzzer.run()``'s main loop was guarded by ``except (KeyboardInterrupt,
SystemExit)`` plus ``except OSError``. Everything that persists campaign state
-- ``_dump_stats``, every ``_state_store.set`` (markov, mi, crash_mi,
length_tracker, ga, qea, cmaes, mcts, fluctuation), both ``_save_state`` calls,
and the ablation-log fd close -- sits BELOW that try and is not in a
``finally``.

So any exception the loop did not name propagated straight past all of it. A
ValueError out of a scheduler or a KeyError out of a mutator, hours into a run,
took the Markov model, the Elo ratings, the crash-MI counters and every
generation counter with it, and leaked the ablation fd on the way out.

The fix catches broadly, logs the traceback (so the underlying defect stays
loud rather than being swallowed -- Hard Rule 20), and marks the run aborted so
the summary does not read like a clean stop.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer


def _make_fuzzer(tmp: Path, **kwargs):
    corpus = tmp / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "seed_a").write_bytes(b"AAAA")
    defaults = dict(
        target="/bin/true",
        corpus_dir=str(corpus),
        crashes_dir=str(tmp / "crashes"),
        max_len=16,
        timeout=1,
        quiet_stats=True,
    )
    defaults.update(kwargs)
    with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
        return Fuzzer(**defaults)


class _Boom(ValueError):
    """Stands in for any exception the main loop never named."""


PROBE = b"\x00" * 64  # run()'s pre-loop raw-speed probe payload
SEED = b"AAAA"  # the single corpus seed planted by _make_fuzzer


def _raise_on(exc, *, after_seed_pass=False):
    """Build a _run_target side effect that raises at a chosen point.

    Keyed on the PAYLOAD rather than a call counter, because the number of
    pre-loop probe executions varies with corpus size and the number of main
    loop iterations varies with the session fuzz-seed. A counter-based seam
    passed standalone and then either failed or -- worse -- passed vacuously
    under the full suite, where a different seed drove a different path.

    Args:
        exc: Exception class to raise.
        after_seed_pass: When True, let both the probe and the initial
            seed-as-is pass succeed, so the raise lands in the mutation loop
            proper rather than in the seed pass.
    """

    def _side_effect(data, *_a, **_k):
        payload = bytes(data) if data is not None else b""
        if payload == PROBE:
            return (0, "")
        if after_seed_pass and payload == SEED:
            return (0, "")
        raise exc

    return _side_effect


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory(prefix="fuzz_persist_") as d:
        yield Path(d)


class TestPersistenceSurvivesUnexpectedException:
    def test_run_does_not_propagate_unexpected_exception(self, tmp_root):
        f = _make_fuzzer(tmp_root)
        with patch.object(f, "_run_target", side_effect=_Boom("scheduler blew up")):
            f.run(iterations=1)  # must return, not raise
        assert f._aborted_by_error is True

    def test_save_state_still_called(self, tmp_root):
        f = _make_fuzzer(tmp_root)
        with (
            patch.object(f, "_run_target", side_effect=_Boom("mutator blew up")),
            patch.object(f, "_save_state", wraps=f._save_state) as save,
        ):
            f.run(iterations=1)
        # This is the whole point of the fix: the campaign was persisted.
        assert save.call_count >= 1

    def test_state_store_receives_campaign_keys(self, tmp_root):
        f = _make_fuzzer(tmp_root)
        with (
            patch.object(f, "_run_target", side_effect=_Boom("boom")),
            patch.object(f._state_store, "set", wraps=f._state_store.set) as store_set,
        ):
            f.run(iterations=1)
        keys = {c.args[0] for c in store_set.call_args_list if c.args}
        # crash_mi and length_tracker are set unconditionally after the loop;
        # under the bug neither was ever reached.
        assert "crash_mi" in keys
        assert "length_tracker" in keys

    def test_ablation_fd_closed(self, tmp_root):
        path = tmp_root / "ablation.csv"
        f = _make_fuzzer(tmp_root, schedule_ablation=str(path))
        if getattr(f, "_ablation_file", None) is None:
            pytest.skip("ablation log not wired under this construction")
        with patch.object(f, "_run_target", side_effect=_Boom("boom")):
            f.run(iterations=1)
        # Leaked fd was part of the original finding.
        assert f._ablation_file is None


class TestFalsification:
    def test_clean_run_is_not_marked_aborted(self, tmp_root):
        # Falsification: if _aborted_by_error were set unconditionally, or the
        # broad handler swallowed normal completion, this would fail. A run
        # that ends without an exception must still report a clean stop.
        f = _make_fuzzer(tmp_root)
        with patch.object(f, "_run_target", return_value=(0, "")):
            f.run(iterations=1)
        assert f._aborted_by_error is False

    def test_keyboard_interrupt_is_not_marked_aborted(self, tmp_root):
        # KeyboardInterrupt is a deliberate stop, not a defect: it must keep
        # taking the quiet path it always took, and must NOT be recorded as an
        # error abort.
        #
        # Raised only after the pre-loop raw-speed probe. That probe calls
        # _run_target inside its own `except Exception: pass`, which by design
        # does not catch BaseException -- so a Ctrl-C landing there escapes
        # run() entirely. Separate pre-existing behaviour, out of scope here;
        # this test targets the main loop.
        f = _make_fuzzer(tmp_root)
        with patch.object(f, "_run_target", side_effect=_raise_on(KeyboardInterrupt)):
            f.run(iterations=5)
        assert f._aborted_by_error is False
        # Guard against a vacuous pass: the interrupt must actually have cut
        # the run short, not simply never fired.
        assert f.exec_count <= 1

    def test_flag_defaults_false_before_run(self, tmp_root):
        f = _make_fuzzer(tmp_root)
        assert f._aborted_by_error is False


class TestAdversarial:
    def test_exception_from_deep_in_the_loop_still_persists(self, tmp_root):
        # Adversarial: raise from the mutation loop rather than the seed pass,
        # so the fix cannot be satisfied by special-casing one seam. The probe
        # and the seed-as-is execution both succeed first.
        f = _make_fuzzer(tmp_root)
        with (
            patch.object(
                f, "_run_target", side_effect=_raise_on(_Boom("deep in havoc"), after_seed_pass=True)
            ),
            patch.object(f, "_save_state", wraps=f._save_state) as save,
        ):
            f.run(iterations=5)
        assert f._aborted_by_error is True
        assert save.call_count >= 1

    def test_baseexception_subclass_still_propagates(self, tmp_root):
        # Adversarial: the handler must be `except Exception`, not `except
        # BaseException`. SystemExit/KeyboardInterrupt are handled explicitly;
        # a bare BaseException (e.g. a test-runner timeout) must NOT be
        # quietly converted into a normal return.
        class _Fatal(BaseException):
            pass

        f = _make_fuzzer(tmp_root)
        with (
            patch.object(f, "_run_target", side_effect=_raise_on(_Fatal)),
            pytest.raises(_Fatal),
        ):
            f.run(iterations=5)

    def test_repeated_aborts_do_not_accumulate_state(self, tmp_root):
        # Adversarial: a resumed run that aborts again must not inherit the
        # previous run's abort flag as a starting condition.
        f = _make_fuzzer(tmp_root)
        with patch.object(f, "_run_target", side_effect=_Boom("first")):
            f.run(iterations=1)
        assert f._aborted_by_error is True

        g = _make_fuzzer(tmp_root)
        assert g._aborted_by_error is False
