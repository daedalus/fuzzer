"""A timed-out run earns no coverage credit (wtf ``RevokeLastNewCoverage``).

A hang's edges are real but truncated: admitting the input seeds the corpus
with something that reproduces only as a hang, and rewards the operators that
made it. The edges stay unseen, so the first clean input reaching them is
credited instead.
"""

import ctypes
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.services.fuzzer import Fuzzer

_TIMEOUT = (-1, "")
_CLEAN = (0, "")
_EDGES = {11: 1, 99: 1}
_PATH_STEP = 7919


@pytest.fixture
def fuzzer():
    shm = ShmCoverage(size=4096)
    with tempfile.TemporaryDirectory(prefix="timeout_cov_") as tmp:
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
        f.shm_cov = shm
        yield f
    shm.cleanup()


def _put(cov, edges: dict[int, int]) -> None:
    # One execution's table as the C shim leaves it; record_edge() would
    # mark the ids seen itself and never report novelty.
    cov.reset_edge_map()
    gen = cov.read_generation()
    for slot, (edge, count) in enumerate(edges.items()):
        cov._entries[slot].edge_id = edge
        cov._entries[slot].count = (gen << 24) | count
    ctypes.c_uint64.from_address(cov._ptr + 16).value = len(edges)
    path = ctypes.c_uint64.from_address(cov._ptr + 8)
    path.value += _PATH_STEP


def _round(f, mutant: bytes, result: tuple[int, str]) -> bool:
    def run(_data):
        _put(f.shm_cov, _EDGES)
        return result

    with (
        patch.object(f, "_dedup_mutate", return_value=mutant),
        patch.object(f, "_run_target", side_effect=run),
        patch.object(f, "_confirm_hang", return_value=None),
        patch.object(f, "_is_crash", return_value=False),
        patch.object(f, "_is_interesting", return_value=False),
    ):
        return f.fuzz_one(f.corpus[0])


def test_clean_new_edges_admitted(fuzzer):
    """Falsification: the same edges from a clean run are credited."""
    before = fuzzer.shm_cov.cumulative_edges

    assert _round(fuzzer, b"CLEAN001", _CLEAN) is True
    assert b"CLEAN001" in fuzzer.corpus
    assert fuzzer.shm_cov.cumulative_edges == before + len(_EDGES)


def test_regression_timeout_coverage_not_credited(fuzzer):
    """A timeout reaching new edges is neither admitted nor counted."""
    before = fuzzer.shm_cov.cumulative_edges

    assert _round(fuzzer, b"HANG0001", _TIMEOUT) is False
    assert b"HANG0001" not in fuzzer.corpus
    assert fuzzer.shm_cov.cumulative_edges == before


def test_timeout_edges_left_for_clean_input(fuzzer):
    """Adversarial: a timeout must not swallow the edges it reached."""
    _round(fuzzer, b"HANG0001", _TIMEOUT)

    assert _round(fuzzer, b"CLEAN001", _CLEAN) is True
    assert b"CLEAN001" in fuzzer.corpus
