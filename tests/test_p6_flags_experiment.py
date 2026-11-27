"""``tools/p6_flags_experiment.py``: seed set and arm table."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "p6_flags_experiment.py"
_SPEC = importlib.util.spec_from_file_location("p6_flags_experiment", _TOOL)
exp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exp)


def test_control_arm_has_no_planted_bug():
    assert exp.ARMS["control"] == 0
    assert all(v > 0 for k, v in exp.ARMS.items() if k != "control")


def test_seeds_one_record_per_op(tmp_path):
    exp.seeds(tmp_path / "s")
    files = sorted((tmp_path / "s").iterdir())

    assert len(files) == exp.NOPS
    assert {f.stat().st_size for f in files} == {12}
    assert [f.read_bytes()[0] for f in sorted(files, key=lambda f: int(f.name[2:]))] == list(
        range(exp.NOPS)
    )


def test_seeds_refuses_existing_dir(tmp_path):
    (tmp_path / "s").mkdir()

    with pytest.raises(FileExistsError):
        exp.seeds(tmp_path / "s")


@pytest.mark.skipif(not shutil.which("cc"), reason="needs a C compiler")
def test_seeds_pass_control_target(tmp_path):
    exe = tmp_path / "ctl"
    subprocess.run(["cc", "-O1", "-o", str(exe), str(exp.SRC)], check=True)
    exp.seeds(tmp_path / "s")

    for f in (tmp_path / "s").iterdir():
        assert subprocess.run([str(exe), "fuzz", str(f)]).returncode == 0
