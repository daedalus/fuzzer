"""FormatSeedGenerator wired into the live run loop.

Before: ``Fuzzer._format_seed_generator`` was set to None and never assigned,
so the generator never ran and the report's "Format Seed Generator" section
was always empty. Now ``_refill_format_seeds`` (stats tick) queues field
variants of the last fuzzed seed, and ``OperatorEngine.mutate`` drains one
per round through the normal exec/coverage/save path.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.core.analyzers.analyzer_format_learner import FieldHypothesis
from fuzzer_tool.services.fuzzer import FORMAT_SEED_BUDGET, FORMAT_SEED_EVERY_EXECS, Fuzzer
from fuzzer_tool.services.report import _format_learning

TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")
BASE = b"HDR\x00\x10payload-bytes-here"


def _hyp(offset, width, field_type):
    return FieldHypothesis(
        offset=offset,
        width=width,
        field_type=field_type,
        confidence=0.9,
        observations=20,
        controlled_edges={1, 2},
    )


@pytest.fixture
def fuzzer():
    tmp = tempfile.mkdtemp()
    corpus = Path(tmp) / "corpus"
    crashes = Path(tmp) / "crashes"
    (corpus / "seeds").mkdir(parents=True)
    crashes.mkdir()
    (corpus / "seeds" / "seed1").write_bytes(BASE)
    f = Fuzzer(
        target=TARGET,
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        learn_format=True,
    )
    f._last_parent_seed = BASE
    f.exec_count = FORMAT_SEED_EVERY_EXECS
    yield f
    shutil.rmtree(tmp, ignore_errors=True)


def test_regression_format_seed_generator_runs(fuzzer):
    """Refill queues variants and the report section is no longer empty."""
    assert fuzzer._format_learner is not None
    fuzzer._format_learner.hypotheses[:] = [_hyp(3, 2, "length")]

    fuzzer._refill_format_seeds()

    queued = list(fuzzer._format_seed_queue)
    assert 0 < len(queued) <= FORMAT_SEED_BUDGET
    assert all(len(q) == len(BASE) and q[:3] == BASE[:3] and q[5:] == BASE[5:] for q in queued)
    assert fuzzer._format_seed_generator.generator_stats["total_seeds_generated"] == len(queued)
    assert "Format Seed Generator:" in _format_learning(fuzzer)


def test_mutate_drains_queue_first(fuzzer):
    """mutate() hands out the queued seed verbatim and credits no operator."""
    first, second = b"queued-one", b"queued-two"
    fuzzer._format_seed_queue.extend([first, second])

    out = fuzzer.mutate(BASE)

    assert out == first
    assert list(fuzzer._format_seed_queue) == [second]
    assert fuzzer._last_ops_used == []


def test_cadence_gates_refill(fuzzer):
    """No refill before the interval elapses, nor while the queue is non-empty."""
    fuzzer._format_learner.hypotheses[:] = [_hyp(3, 2, "length")]
    fuzzer.exec_count = FORMAT_SEED_EVERY_EXECS - 1
    fuzzer._refill_format_seeds()
    assert not fuzzer._format_seed_queue

    fuzzer.exec_count = FORMAT_SEED_EVERY_EXECS
    fuzzer._format_seed_queue.append(b"pending")
    fuzzer._refill_format_seeds()
    assert list(fuzzer._format_seed_queue) == [b"pending"]


def test_falsify_no_learner_no_generation(fuzzer):
    """Falsification: without --learn-format nothing is generated."""
    fuzzer._format_learner = None
    fuzzer._refill_format_seeds()
    assert fuzzer._format_seed_generator is None
    assert not fuzzer._format_seed_queue


def test_adversarial_hostile_hypotheses(fuzzer, monkeypatch):
    """Magic-only, out-of-range and oversize fields: no crash, max_len honoured."""
    fuzzer._format_learner.hypotheses[:] = [_hyp(0, 3, "magic"), _hyp(10_000, 4, "length")]
    fuzzer._refill_format_seeds()
    assert not fuzzer._format_seed_queue

    fuzzer.max_len = 4
    fuzzer.exec_count += FORMAT_SEED_EVERY_EXECS
    fuzzer._format_learner.hypotheses[:] = [_hyp(3, 2, "length")]
    fuzzer._refill_format_seeds()
    assert fuzzer._format_seed_queue
    assert all(len(q) <= fuzzer.max_len for q in fuzzer._format_seed_queue)


def test_refill_tracks_new_hypotheses(fuzzer):
    """A second refill uses the learner's current fields, and stats accumulate."""
    fuzzer._format_learner.hypotheses[:] = [_hyp(3, 2, "length")]
    fuzzer._refill_format_seeds()
    n_first = len(fuzzer._format_seed_queue)
    gen = fuzzer._format_seed_generator
    fuzzer._format_seed_queue.clear()

    fuzzer.exec_count += FORMAT_SEED_EVERY_EXECS
    fuzzer._format_learner.hypotheses[:] = [_hyp(5, 1, "crc")]
    fuzzer._refill_format_seeds()

    assert fuzzer._format_seed_generator is gen
    assert {q[5] for q in fuzzer._format_seed_queue} <= {0x00, 0xFF, 0x01, 0x80}
    assert gen.generator_stats["total_seeds_generated"] == n_first + len(fuzzer._format_seed_queue)
