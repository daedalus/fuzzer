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


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------
#
# Production code mutates os.environ process-globally and does not always put
# it back: CmplogCollector.setup_env_for_run() sets LD_PRELOAD and
# _CMPLOG_OUT, and the SHM paths set __AFL_SHM_ID / __AFL_DIST_SHM_ID /
# AFL_MAP_SIZE. Inside one pytest process those leak *forward into unrelated
# tests*, and the failure is silent rather than loud: an ASAN test whose
# subprocess inherits a leaked cmplog preload runs to completion at full speed
# and reports zero crashes, because the shim resolves the coverage symbols
# ASAN wanted. It passes in isolation and fails in a full run, which reads as
# flakiness rather than as contamination.
#
# Restoring after every test is the cheap half. Reporting *which* test leaked
# is the half that stops it coming back -- otherwise the next leak is found
# the same way this one was, by bisecting a suite.

_ENV_OWNED = ("LD_PRELOAD", "_CMPLOG_OUT", "__AFL_SHM_ID", "__AFL_DIST_SHM_ID", "AFL_MAP_SIZE")


@pytest.fixture(autouse=True)
def _env_isolation(request):
    """Restore the keys production code is known to mutate, after every test.

    Not a full os.environ snapshot: a test that deliberately exports something
    for a subprocess it spawns is doing normal work, and reverting everything
    would fight it. This reverts only the five keys whose leakage has actually
    caused misdiagnosed failures.

    Set ``-p no:cacheprovider`` aside; to see leaks instead of silently fixing
    them, run with ``--env-leak-strict``.
    """
    before = {k: os.environ.get(k) for k in _ENV_OWNED}
    yield
    leaked = [k for k in _ENV_OWNED if os.environ.get(k) != before[k]]
    for k in leaked:
        if before[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = before[k]
    if leaked and request.config.getoption("--env-leak-strict"):
        pytest.fail(f"test leaked os.environ keys: {', '.join(sorted(leaked))}", pytrace=False)


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
    parser.addoption(
        "--env-leak-strict",
        action="store_true",
        default=False,
        help="Fail any test that leaves LD_PRELOAD/_CMPLOG_OUT/__AFL_SHM_ID/"
        "__AFL_DIST_SHM_ID/AFL_MAP_SIZE changed, instead of quietly restoring them.",
    )


def pytest_configure(config):
    raw = config.getoption("--fuzz-seed")
    config.fuzz_seed = int(raw, 0) if raw is not None else int.from_bytes(os.urandom(8), "little")

    # Default per-test timeout. A bare `pytest` could previously wedge forever:
    # tests/test_structural_constraints.py once sat >9 min inside Z3's
    # solver.add(), which SIGTERM could not interrupt because it was stuck in
    # native code -- only SIGKILL worked.
    #
    # The method is "signal", not "thread", and that is a deliberate trade.
    # "thread" is the only method that survives a native hang, but it arms a
    # threading.Timer for every test, which makes the pytest process
    # multi-threaded for the whole session. This suite forks constantly
    # (persistent.py, runner.py's ptrace launch, the inprocess loader), and
    # fork-from-a-multi-threaded-process is a real deadlock hazard, not a
    # style warning -- see docs/handover/test_shm_hang_2026-08-14.md. Measured:
    # arming the thread method makes CPython emit its multi-threaded-fork
    # DeprecationWarning on a test that is otherwise silent. Bounding the suite
    # is not worth making every fork in it riskier.
    #
    # "signal" costs no thread and bounds every hang that reaches a bytecode
    # boundary. The native-code cases opt into "thread" per module, below.
    #
    # Applied here rather than in addopts so that a dev environment without
    # pytest-timeout installed still runs the suite instead of dying on an
    # unrecognised argument. An explicit flag on the command line wins.
    pm = getattr(config, "pluginmanager", None)
    if pm is not None and pm.hasplugin("timeout"):
        if not config.getoption("timeout", None):
            config.option.timeout = 300
        if not config.getoption("timeout_method", None):
            config.option.timeout_method = "signal"


# Modules that can block inside a long native call, where SIGALRM is never
# delivered because the C code never returns to the interpreter. These opt
# into the thread method individually, so the rest of the suite keeps a
# single-threaded process to fork from. Z3 is the known case: its `timeout`
# parameter bounds check(), not assertion processing.
_NATIVE_HANG_MODULES = ("test_structural_constraints", "test_field_constraints")


def pytest_collection_modifyitems(config, items):
    pm = getattr(config, "pluginmanager", None)
    if pm is None or not pm.hasplugin("timeout"):
        return
    for item in items:
        if any(mod in item.nodeid for mod in _NATIVE_HANG_MODULES):
            item.add_marker(pytest.mark.timeout(method="thread"))


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
