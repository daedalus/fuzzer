"""``_cull_queue`` read ``seed_meta`` by hash key; production keys it by bytes.

Every lookup missed, so every seed cost the corpus mean x 1 byte and the
favored set ignored exec time and size. The old fixtures keyed both maps by
the same string, hiding it.
"""

import shutil
import tempfile
from unittest.mock import patch

import pytest

from fuzzer_tool.services.fuzzer import Fuzzer

SMALL = b"s"
BIG = b"B" * 1000


@pytest.fixture
def fuzzer():
    tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        f = Fuzzer(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=2048,
            timeout=1,
            mutations_per_input=2,
        )
    yield f
    shutil.rmtree(tmpdir, ignore_errors=True)


def _load(f: Fuzzer, order: list[bytes]) -> None:
    """Both seeds cover edge 1; same exec time; only size differs."""
    f.corpus = list(order)
    f.seed_meta = {
        s: {"total_time": 0.001, "cost_samples": 1, "fuzz_count": 1, "input_size": len(s)}
        for s in order
    }
    f._edge_tracker.seed_edges = {f._seed_key(s): {1} for s in order}


@pytest.mark.parametrize("order", [[BIG, SMALL], [SMALL, BIG]])
def test_regression_cull_queue_meta_key(fuzzer, order):
    _load(fuzzer, order)
    fuzzer._cull_queue()
    assert fuzzer._favored == {fuzzer._seed_key(SMALL)}


def test_cull_queue_seed_left_corpus(fuzzer):
    """Adversarial: a tracked key with no corpus seed costs the mean, no raise."""
    _load(fuzzer, [SMALL])
    fuzzer._edge_tracker.seed_edges["gone"] = {2}
    fuzzer._cull_queue()
    assert fuzzer._favored == {fuzzer._seed_key(SMALL), "gone"}
