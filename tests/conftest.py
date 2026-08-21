"""Shared test fixtures and environment guards.

Several suites need a toolchain component that is not a declared project
dependency -- clang, or the optional ``smt`` extra (z3). When it is absent
those tests errored out rather than skipping, so a machine without clang
reported 28 failures that said nothing about the code under test. CI hits
this too: ``pip install -e ".[dev]"`` does not pull in z3.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import zlib

import pytest


def _has(tool: str) -> bool:
    return shutil.which(tool) is not None


requires_clang = pytest.mark.skipif(not _has("clang"), reason="clang not installed")
requires_gcc = pytest.mark.skipif(not _has("gcc"), reason="gcc not installed")
requires_z3 = pytest.mark.skipif(
    importlib.util.find_spec("z3") is None,
    reason="z3-solver not installed (optional 'smt' extra)",
)


@pytest.fixture(scope="session", autouse=True)
def _fuzz_loader_built():
    """Build ``fuzz_loader`` once, before any test runs.

    W3: in a fresh clone the binary does not exist (it is gitignored), and
    the suite *hangs* rather than fails -- a test that starts a
    ForkserverRunner blocks on a reply from a loader that was never built,
    and the join only unblocks after the runner's grace period, per test.
    Building it up front is a one-off ~0.3s and turns the failure mode from
    "CI wedged" into "these tests skip".

    Deliberately best-effort: no compiler is a legitimate environment (the
    tests that need the loader already skip on ``_ensure_compiled() is
    None``), so a failure here must not abort collection.
    """
    try:
        from fuzzer_tool.adapters.forkserver import _ensure_compiled

        _ensure_compiled()
    except Exception:  # pragma: no cover - environment-dependent
        pass


# ---------------------------------------------------------------------------
# Session seed
# ---------------------------------------------------------------------------
#
# Randomised tests across the suite hardcode their seeds (Random(1), seed=42,
# Random(7), ...), so every run explores the same handful of points forever.
# The fix is not to make them deterministic-with-42, it is to make the seed a
# session-level input: fixed when reproducing, random when accumulating
# coverage, and *always* recoverable.
#
# The seed is generated at configure time and printed by
# pytest_report_header -- i.e. before collection, not inside the test. That
# placement is the whole point. A test that segfaults or hangs (both of which
# this suite has done: see docs/handover/suite_segfault_z3_finalization_
# 2026-08-16.md and docs/handover/test_shm_hang_2026-08-14.md) never gets to
# print anything itself, and a seed you cannot recover from a CI log is the
# same as no seed at all.


def pytest_addoption(parser):
    parser.addoption(
        "--fuzz-seed",
        action="store",
        default=None,
        help="Session seed for randomised tests (int, 0x-prefixed accepted). "
        "Default: random per session, printed in the header.",
    )


def pytest_configure(config):
    raw = config.getoption("--fuzz-seed")
    config.fuzz_seed = int(raw, 0) if raw is not None else int.from_bytes(os.urandom(8), "little")


def pytest_report_header(config):
    return f"fuzz-seed: reproduce with --fuzz-seed=0x{config.fuzz_seed:016x}"


@pytest.fixture
def random_seed(request) -> int:
    """A per-test seed derived from the session seed.

    Derived from the node id rather than handed out raw so that deselecting or
    reordering tests does not change what any individual test sees -- a test
    that fails under ``-k convergence`` must fail identically under a full run
    with the same ``--fuzz-seed``.
    """
    node = zlib.crc32(request.node.nodeid.encode())
    return (request.config.fuzz_seed ^ (node * 0x9E3779B97F4A7C15)) & 0xFFFF_FFFF_FFFF_FFFF


def pytest_make_parametrize_id(config, val):
    """Shorten long byte-string parametrize IDs to keep test output readable."""
    if isinstance(val, bytes) and len(val) > 32:
        return f"bytes[{len(val)}]"
    return None
