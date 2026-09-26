"""``tools/edge_diagnostic.py`` runs from any clone (P2-4 of the edge-id handover).

The tool used to prepend one machine's ``/home/.../src`` to ``sys.path``, so
from a fresh clone every mode died with ``ModuleNotFoundError`` unless the
package happened to be installed -- and every reproduction command in
``handover_edge_id_axis_2026-09-18.md`` goes through it.  These tests run the
tool the way a reader of that handover would: as a script, from an unrelated
directory, with no ``PYTHONPATH``.

The local-only probe and memory modes cannot be made portable (they read
fixed builds), so the contract for them is weaker and explicit: a missing
input stops the mode with exit 2 and the ``EDGE_DIAG_*`` variable to set,
never a traceback.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "edge_diagnostic.py"


def _clean_env(**extra: str) -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.startswith("EDGE_DIAG_")
    }
    env.update(extra)
    return env


def _run(args: list[str], cwd: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=cwd,
        env=_clean_env(**env),
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("sub", ["matrix", "phantom"])
def test_portable_subcommands_start_from_a_foreign_cwd(sub, tmp_path):
    r = _run([sub, "--help"], cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr


def test_srcdir_is_this_checkout():
    spec = importlib.util.spec_from_file_location("edge_diagnostic_port", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert Path(mod.SRCDIR) == ROOT / "src"


def test_local_only_mode_names_the_missing_input(tmp_path):
    empty = tmp_path / "corpus"
    empty.mkdir()
    r = _run(
        ["per-input-sweep"],
        cwd=tmp_path,
        EDGE_DIAG_CTX_TARGET=str(tmp_path / "absent"),
        EDGE_DIAG_CORPUS=str(empty),
    )
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert "EDGE_DIAG_CTX_TARGET" in r.stderr
    assert "EDGE_DIAG_CORPUS" in r.stderr


def test_memory_mode_names_the_missing_target(tmp_path):
    r = _run(["mem-stats"], cwd=tmp_path, EDGE_DIAG_PROF_TARGET=str(tmp_path / "absent"))
    assert r.returncode == 2
    assert "EDGE_DIAG_PROF_TARGET" in r.stderr


def test_preflight_passes_when_inputs_exist(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(6):
        (corpus / f"in{i}").write_bytes(b"{}")
    target = tmp_path / "fuzzgoat_read"
    target.write_bytes(b"")
    monkeypatch.setenv("EDGE_DIAG_CORPUS", str(corpus))
    monkeypatch.setenv("EDGE_DIAG_CTX_TARGET", str(target))
    spec = importlib.util.spec_from_file_location("edge_diagnostic_port2", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.FILES) == 6
    reqs = mod._mode_requirements("per-input-sweep", None)
    assert reqs and all(ok for _, _, ok in reqs)


def test_recipe_mode_needs_nothing(tmp_path):
    r = _run(["fire-trace-build"], cwd=tmp_path, EDGE_DIAG_SCRATCH=str(tmp_path))
    assert r.returncode == 0
    assert str(tmp_path) in r.stdout


def test_no_python_tool_hardcodes_a_home_directory():
    """Same defect class as the SRCDIR above, across tools/: a repo root or
    a working directory pinned to one user's home.  bench_cache,
    debug_repro and the three profile_* scripts chdir'd there."""
    offenders = [
        f"{p.name}:{n}"
        for p in sorted((ROOT / "tools").glob("**/*.py"))
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
        if "/home/" in line and not line.lstrip().startswith("#")
    ]
    assert offenders == []
