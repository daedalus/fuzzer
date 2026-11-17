"""mutants-build / mutants-score CLI (docs/action_plan_mutant_kill_metric.md P4)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fuzzer_tool.adapters.mutant_build import asan_available
from fuzzer_tool.services.mutant_kill import KillScorer
from tests.support.mutant_toy import TOY_LIB, TOY_WRAP

needs_asan = pytest.mark.skipif(not asan_available(), reason="needs clang + ASAN runtime")

_INPUTS = [b"abc", b"abcd", b"abcdefg", b"ok", b"C"]


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", *args], capture_output=True, text=True
    )


def _summary(stdout: str) -> dict[str, int]:
    """The `key=value ...` summary line."""
    (line,) = [ln for ln in stdout.splitlines() if ln.startswith("kills=")]
    return {k: int(v) for k, v in (kv.split("=") for kv in line.split())}


def _corpus(root: Path, inputs: list[bytes]) -> Path:
    seeds = root / "seeds"
    seeds.mkdir(parents=True)
    for i, data in enumerate(inputs):
        (seeds / f"s{i}").write_bytes(data)
    return root


@pytest.fixture(scope="module")
def kill_set(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cli")
    (root / "toy.c").write_text(TOY_LIB)
    (root / "w.c").write_text(TOY_WRAP)
    res = _cli(
        "mutants-build",
        str(root / "toy.c"),
        "--wrapper",
        str(root / "w.c"),
        "-o",
        str(root / "set"),
        "-j",
        "4",
    )
    assert res.returncode == 0, res.stderr
    return root / "set"


@pytest.mark.parametrize("cmd", ["mutants-build", "mutants-score"])
def test_help(cmd: str) -> None:
    res = _cli(cmd, "--help")
    assert res.returncode == 0
    assert cmd in res.stdout


@needs_asan
def test_build_reports_counts(kill_set) -> None:
    from fuzzer_tool.adapters.mutant_build import MutantSet

    built = MutantSet.load(kill_set)
    assert len(built.sites) > 0
    assert (kill_set / "original").is_file()


@needs_asan
def test_score_matches_service(kill_set, tmp_path) -> None:
    corpus = _corpus(tmp_path / "corpus", _INPUTS)
    res = _cli("mutants-score", str(kill_set), "-d", str(corpus), "-j", "4")
    assert res.returncode == 0, res.stderr

    want = KillScorer.open(kill_set, jobs=4).score(corpus)
    got = _summary(res.stdout)
    assert got["kills"] == want.kills > 0
    assert (got["hangs"], got["flaky"], got["inputs"], got["clean_inputs"]) == (
        want.hangs,
        want.flaky,
        want.inputs,
        want.clean_inputs,
    )
    assert got["mutants"] == KillScorer.open(kill_set, jobs=1).mutant_count > 0


@needs_asan
def test_corpus_of_original_crashes_kills_nothing(kill_set, tmp_path) -> None:
    corpus = _corpus(tmp_path / "corpus", [b"C"])
    res = _cli("mutants-score", str(kill_set), "-d", str(corpus), "--confirm", "none")
    assert res.returncode == 0, res.stderr
    assert _summary(res.stdout)["kills"] == 0


def test_score_without_manifest_fails_cleanly(tmp_path) -> None:
    (tmp_path / "corpus" / "seeds").mkdir(parents=True)
    res = _cli("mutants-score", str(tmp_path / "nothing"), "-d", str(tmp_path / "corpus"))
    assert res.returncode == 1
    assert "Traceback" not in res.stderr
    assert "manifest" in res.stderr


def test_score_missing_corpus_fails_cleanly(tmp_path) -> None:
    res = _cli("mutants-score", str(tmp_path), "-d", str(tmp_path / "nope"))
    assert res.returncode == 1
    assert "Traceback" not in res.stderr


def test_build_missing_source_fails_cleanly(tmp_path) -> None:
    res = _cli(
        "mutants-build",
        str(tmp_path / "nope.c"),
        "--wrapper",
        str(tmp_path / "w.c"),
        "-o",
        str(tmp_path / "o"),
    )
    assert res.returncode == 1
    assert "Traceback" not in res.stderr


@pytest.mark.parametrize("cmd", ["mutants-build", "mutants-score"])
def test_zero_jobs_rejected(cmd: str, tmp_path) -> None:
    args = (
        ["x.c", "--wrapper", "w.c", "-o", str(tmp_path)]
        if cmd == "mutants-build"
        else [str(tmp_path), "-d", "c"]
    )
    res = _cli(cmd, *args, "-j", "0")
    assert res.returncode == 2
    assert "-j/--jobs" in res.stderr
