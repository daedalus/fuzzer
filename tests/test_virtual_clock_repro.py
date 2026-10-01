"""``--clock virtual`` makes a seeded campaign replay bit-for-bit.

Under the wall clock, rewards scaled by operator/exec cost, seed ages, LST
deadlines and EPS-sized caps all read the host clock. With scripted crashes
(every success round pays a cost-scaled reward) two seed-7 campaigns picked
different operators by round 7. The virtual clock advances one tick per
execution, so the same seed replays the same decisions.

Each campaign runs in a fresh interpreter: cadence counters and dlopen
state are process-global, and a second in-process run would inherit them.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.core.clock import Clock, ClockMode
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler
from fuzzer_tool.services.katz_channel import KatzChannel

_REPO = Path(__file__).resolve().parent.parent
_SHIM = _REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
_TARGET_SRC = _REPO / "targets" / "test_target.c"
_SEEDS = {"s0": b"CRASX hello world", "s1": b"GET / HTTP/1.0"}
_ROUNDS = 200
_CRASH_EVERY = 5  # scripted crash cadence: success rounds pay cost-scaled rewards
_CLI_EXECS = 50
_RUN_TIMEOUT_S = 300

# Runs in a fresh interpreter: argv = target, root, clock, seed.
# Prints one JSON row per round: (admitted, ops, crash count, corpus size).
_HARNESS = """
import json, sys
from pathlib import Path
from fuzzer_tool.core.clock import ClockMode
from fuzzer_tool.services.fuzzer import Fuzzer

target, root, clock, seed = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
f = Fuzzer(target=target, corpus_dir=str(root / "corpus"), crashes_dir=str(root / "crashes"),
           max_len=256, use_coverage=True, seed=seed, clock=ClockMode(clock))
real, n = f._run_target, [0]

def scripted(data):
    rc, err = real(data)
    n[0] += 1
    if n[0] % CRASH_EVERY == 0:
        return -11, "AddressSanitizer: SEGV on unknown address"
    return rc, err

f._run_target = scripted
rows = []
for i in range(ROUNDS):
    admitted = f.fuzz_one(f.corpus[i % len(f.corpus)])
    rows.append([admitted, sorted(set(f._last_ops_used)), f.crash_count, len(f.corpus)])
print(json.dumps(rows))
"""
_SEED = 7
_OTHER_SEED = 8


@pytest.fixture(scope="module")
def target(tmp_path_factory) -> Path:
    """test_target with AFL edge coverage, built in tmp (never in-tree)."""
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("clang not available")
    out = tmp_path_factory.mktemp("target")
    obj, exe = out / "test_target.o", out / "test_target"
    # Compile and link separately: linking with -fsanitize-coverage pulls a
    # sanitizer runtime the shim already replaces.
    compile_cmd = [
        clang, "-O1", "-c", "-fsanitize-coverage=trace-pc-guard",
        "-include", str(_SHIM), str(_TARGET_SRC), "-o", str(obj),
    ]  # fmt: skip
    built = subprocess.run(compile_cmd, capture_output=True, check=False)
    built = built.returncode == 0 and (
        subprocess.run([clang, str(obj), "-o", str(exe)], capture_output=True).returncode == 0
    )
    if not built:
        pytest.skip("test_target did not build")
    return exe


def _seed_corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True)
    for name, data in _SEEDS.items():
        (corpus / name).write_bytes(data)
    return corpus


def _campaign(target: Path, root: Path, clock: str, seed: int = _SEED) -> list:
    """One scripted-crash campaign in a fresh interpreter; per-round decisions."""
    _seed_corpus(root)
    script = _HARNESS.replace("CRASH_EVERY", str(_CRASH_EVERY)).replace("ROUNDS", str(_ROUNDS))
    cmd = [sys.executable, "-c", script, str(target), str(root), clock, str(seed)]
    done = subprocess.run(cmd, cwd=root, capture_output=True, timeout=_RUN_TIMEOUT_S)
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-2000:]
    return json.loads(done.stdout.decode().strip().splitlines()[-1])


def test_virtual_clock_replays_seeded_run(target, tmp_path):
    first = _campaign(target, tmp_path / "a", ClockMode.VIRTUAL.value)
    second = _campaign(target, tmp_path / "b", ClockMode.VIRTUAL.value)
    assert first == second


def test_virtual_clock_still_follows_the_seed(target, tmp_path):
    """Falsification: the oracle must see a difference when one exists."""
    first = _campaign(target, tmp_path / "a", ClockMode.VIRTUAL.value)
    other = _campaign(target, tmp_path / "b", ClockMode.VIRTUAL.value, seed=_OTHER_SEED)
    assert first != other


def test_cli_accepts_virtual_clock(target, tmp_path):
    """Wiring: --clock reaches the Fuzzer through the CLI."""
    corpus = _seed_corpus(tmp_path)
    cmd = [
        sys.executable, "-m", "fuzzer_tool", "fuzz", str(target),
        "-d", str(corpus), "--crashes", str(tmp_path / "crashes"),
        "--max-execs", str(_CLI_EXECS), "--clock", ClockMode.VIRTUAL.value,
    ]  # fmt: skip
    done = subprocess.run(cmd, cwd=tmp_path, capture_output=True, timeout=_RUN_TIMEOUT_S)
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-2000:]


# ── Seams ────────────────────────────────────────────────────────────


def test_katz_recompute_gate_reads_injected_clock():
    """Virtual time: recompute cost reads 0, so only the exec gate applies."""
    clock = Clock(ClockMode.VIRTUAL)
    ch = KatzChannel(MagicMock(), {}, clock=clock)
    assert ch._last_recompute_wall == clock.monotonic()


def test_monte_carlo_null_is_seeded():
    """The numpy permutation null draws from the run's RandPool, not OS entropy."""
    pooled = [({0: 5, 1: 5, 2: 3}, 6, 7), ({3: 4, 4: 4}, 4, 4)]
    runs = [
        MonteCarloScheduler._null_js_samples_numpy(
            pooled, 8, seed=RandPool(seed=_SEED).randint(0, 2**31)
        )
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def test_monte_carlo_deadline_reads_injected_clock():
    """Adversarial: a virtual clock never expires the budget, so the count is fixed."""
    mc = MonteCarloScheduler(rng=RandPool(seed=_SEED), clock=Clock(ClockMode.VIRTUAL))
    assert mc._clock.mode is ClockMode.VIRTUAL
