"""Uninitialized-memory probe: same input, two malloc fill bytes.

lcamtuf's browser/libjpeg disclosure bugs: a decoder that emits heap bytes
it never wrote produces different output on identical input. Filling fresh
allocations with 0xAA on one run and 0x55 on another turns that into a
deterministic diff, without an MSAN rebuild. A repeat run with the *same*
fill is the control: a target whose output differs on identical runs
(timestamps, pids) is NOISY, never LEAK.
"""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.services.uninit_probe import (
    FILL_A,
    FILL_B,
    UninitProbe,
    Verdict,
    fill_env,
    probe,
)
from tests import test_commands_extended


def _fake(outputs):
    """Runner returning scripted (rc, stdout) per call; records fills used."""
    calls = []
    it = iter(outputs)

    def run(data, env):
        calls.append(env["MALLOC_PERTURB_"])
        return next(it)

    return run, calls


class TestVerdict:
    def test_identical_outputs_are_stable(self):
        run, _ = _fake([(0, b"x"), (0, b"x"), (0, b"x")])
        assert probe(run, b"in") is Verdict.STABLE

    def test_fill_dependent_output_is_leak(self):
        run, _ = _fake([(0, b"\xaa"), (0, b"\xaa"), (0, b"\x55")])
        assert probe(run, b"in") is Verdict.LEAK

    def test_fill_dependent_exit_code_is_leak(self):
        run, _ = _fake([(0, b""), (0, b""), (1, b"")])
        assert probe(run, b"in") is Verdict.LEAK

    def test_runs_control_before_treatment(self):
        run, calls = _fake([(0, b""), (0, b""), (0, b"")])
        probe(run, b"in")
        a, b = (fill_env({}, f)["MALLOC_PERTURB_"] for f in (FILL_A, FILL_B))
        assert calls == [a, a, b]


class TestFalsification:
    def test_noisy_target_is_never_leak(self):
        # Control fails: output differs on identical runs. The fill diff
        # carries no information then.
        run, _ = _fake([(0, b"t=1"), (0, b"t=2"), (0, b"t=3")])
        assert probe(run, b"in") is Verdict.NOISY

    def test_noisy_short_circuits(self):
        # No third run once the control has failed.
        run, calls = _fake([(0, b"1"), (0, b"2")])
        assert probe(run, b"in") is Verdict.NOISY
        assert len(calls) == 2


class TestAdversarial:
    def test_timeout_is_not_a_verdict(self):
        run, _ = _fake([(-1, b""), (-1, b""), (-1, b"")])
        assert probe(run, b"in") is Verdict.TIMEOUT

    def test_timeout_on_treatment_only(self):
        # A fill that makes the target hang is suspicious but not a proven
        # leak; report TIMEOUT rather than a diff on a killed process.
        run, _ = _fake([(0, b"x"), (0, b"x"), (-1, b"")])
        assert probe(run, b"in") is Verdict.TIMEOUT


class TestFillEnv:
    def test_appends_to_existing_asan_options(self):
        env = fill_env({"ASAN_OPTIONS": "abort_on_error=1"}, FILL_A)
        opts = env["ASAN_OPTIONS"].split(":")
        assert opts[0] == "abort_on_error=1"
        assert f"malloc_fill_byte={FILL_A}" in opts

    def test_does_not_mutate_input(self):
        base = {"ASAN_OPTIONS": "x=1"}
        fill_env(base, FILL_B)
        assert base == {"ASAN_OPTIONS": "x=1"}

    def test_perturb_yields_the_fill(self):
        # glibc fills malloc'd memory with MALLOC_PERTURB_ ^ 0xff.
        for fill in (FILL_A, FILL_B):
            assert int(fill_env({}, fill)["MALLOC_PERTURB_"]) ^ 0xFF == fill

    def test_fills_differ(self):
        assert FILL_A != FILL_B


# ── real processes (glibc MALLOC_PERTURB_; no ASAN runtime needed) ──────

_LEAKY = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    unsigned char *p = malloc(64);
    memset(p, 'A', 32);            /* second half never written */
    fwrite(p, 1, 64, stdout);
    return 0;
}
"""

_CLEAN = r"""
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    unsigned char *p = calloc(64, 1);
    fwrite(p, 1, 64, stdout);
    return 0;
}
"""

_NOISY = r"""
#include <stdio.h>
#include <time.h>
int main(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    printf("%ld\n", t.tv_nsec);
    return 0;
}
"""


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    d = tmp_path_factory.mktemp("uninit")

    def _build(name: str, src: str) -> str:
        c = d / f"{name}.c"
        c.write_text(src)
        exe = d / name
        r = subprocess.run(["clang", "-O0", "-o", str(exe), str(c)], capture_output=True)
        if r.returncode != 0:
            pytest.skip(r.stderr.decode()[:300])
        return str(exe)

    return _build


@pytest.mark.parametrize(
    ("src", "verdict"),
    [(_LEAKY, Verdict.LEAK), (_CLEAN, Verdict.STABLE), (_NOISY, Verdict.NOISY)],
)
def test_real_target(build, tmp_path, src, verdict):
    exe = build(verdict.name.lower(), src)
    p = UninitProbe(exe, timeout=5.0, out_dir=tmp_path)
    assert p.check(b"input") is verdict


def test_leak_is_saved_once(build, tmp_path):
    exe = build("leaky2", _LEAKY)
    p = UninitProbe(exe, timeout=5.0, out_dir=tmp_path)
    p.check(b"input")
    p.check(b"input")
    saved = list(tmp_path.glob("uninit_*"))
    assert len(saved) == 1
    assert saved[0].read_bytes() == b"input"
    assert p.leaks == 1


def test_file_mode_substitutes_path(build, tmp_path):
    exe = build("leaky3", _LEAKY)
    p = UninitProbe(exe, timeout=5.0, out_dir=tmp_path, target_args=["{file}"])
    assert p.check(b"input") is Verdict.LEAK


def test_shared_object_target_is_skipped(tmp_path):
    # Adversarial: an in-process .so cannot be exec'd.
    p = UninitProbe(str(tmp_path / "lib.so"), timeout=1.0, out_dir=tmp_path)
    assert p.check(b"x") is Verdict.SKIPPED


# ── wiring ───────────────────────────────────────────────────────────────


def test_admitted_input_is_probed(monkeypatch):
    from fuzzer_tool.services.fuzz_round import FuzzRound

    f = MagicMock()
    f.corpus = []
    r = FuzzRound(f, b"seed")
    r._mutated = b"new"
    monkeypatch.setattr(FuzzRound, "_feed_population", lambda self: None)
    r._admit()
    f._uninit_probe.check.assert_called_once_with(b"new")


def test_admit_without_probe(monkeypatch):
    from fuzzer_tool.services.fuzz_round import FuzzRound

    f = MagicMock()
    f.corpus = []
    f._uninit_probe = None
    r = FuzzRound(f, b"seed")
    r._mutated = b"new"
    monkeypatch.setattr(FuzzRound, "_feed_population", lambda self: None)
    assert r._admit()


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.uninit_probe = flag
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["uninit_probe"] is flag


TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


def test_fuzzer_builds_probe_only_when_asked(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    built = []
    for i, flag in enumerate((False, True)):
        corpus = tmp_path / f"c{i}"
        (corpus / "seeds").mkdir(parents=True)
        (corpus / "seeds" / "s").write_bytes(b"x")
        f = Fuzzer(
            target=TARGET,
            corpus_dir=str(corpus),
            crashes_dir=str(tmp_path / f"k{i}"),
            uninit_probe=flag,
        )
        built.append(f._uninit_probe)
    assert built[0] is None
    assert isinstance(built[1], UninitProbe)
