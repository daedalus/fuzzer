"""bench_paired: per-target seed corpora (eval_set.SEED_DIRS).

Every cell started from an empty corpus; a crash-reward A/B on FFmpeg then
saw no crash at 20k execs. A target listed in SEED_DIRS starts from its
seeds (copied, so cells never share or alter them).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

import bench_paired  # noqa: E402
import eval_set  # noqa: E402


@pytest.fixture
def seen(monkeypatch):
    """Captures the work dir's seeds at the moment the campaign starts."""
    got: dict[str, list[str]] = {}

    def fake_run(cmd, **kw):
        workdir = Path(cmd[cmd.index("-d") + 1])
        got["seeds"] = sorted(p.name for p in (workdir / "seeds").glob("*"))
        if got["seeds"]:
            (workdir / "seeds" / "scribble").write_bytes(b"x")  # the run writes its corpus
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(bench_paired.subprocess, "run", fake_run)
    return got


@pytest.fixture
def seed_dir(tmp_path, monkeypatch):
    src = tmp_path / "pack"
    (src / "seeds").mkdir(parents=True)
    for name in ("a.bin", "b.bin"):
        (src / "seeds" / name).write_bytes(name.encode())
    monkeypatch.setitem(eval_set.SEED_DIRS, "/t/target.so", src)
    return src


def test_seeded_target_starts_from_seeds(seen, seed_dir) -> None:
    bench_paired.run_cell("baseline", "/t/target.so", "", 1, 10, 10)

    assert seen["seeds"] == ["a.bin", "b.bin"]


def test_source_seeds_untouched(seen, seed_dir) -> None:
    # Adversarial: a campaign writing into its corpus must not reach the pack.
    bench_paired.run_cell("baseline", "/t/target.so", "", 1, 10, 10)

    assert sorted(p.name for p in (seed_dir / "seeds").iterdir()) == ["a.bin", "b.bin"]


def test_unlisted_target_starts_empty(seen, seed_dir) -> None:
    # Falsification: other targets keep the empty-corpus protocol.
    bench_paired.run_cell("baseline", "/t/other.so", "", 1, 10, 10)

    assert seen["seeds"] == []


def test_missing_pack_fails_loudly(seen, tmp_path, monkeypatch) -> None:
    # Edge: a listed pack that is absent must not silently run unseeded.
    monkeypatch.setitem(eval_set.SEED_DIRS, "/t/target.so", tmp_path / "absent")

    with pytest.raises(FileNotFoundError):
        bench_paired.run_cell("baseline", "/t/target.so", "", 1, 10, 10)
