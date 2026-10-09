"""build_targets.sh stages vendored sources with copy-on-write ``cp --reflink=auto``.

The helper is sliced out of the script and run against temp trees; a ``cp``
shim on PATH records its argv, then defers to the real ``cp``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "tools" / "build_targets.sh"
BASH = shutil.which("bash") or "bash"
HELPER = "cow_copy_tree"
REFLINK_FLAG = "--reflink=auto"
SKIPPED = (".git", ".forgejo", "presets")


def _helper_src() -> str:
    """The helper's definition, sliced verbatim from the script."""
    m = re.search(rf"^{HELPER}\(\) \{{.*?^\}}", SCRIPT.read_text(), re.S | re.M)
    assert m, f"{HELPER} not defined in {SCRIPT.name}"
    return m.group(0)


def _run(tmp_path: Path, src: Path, dst: Path) -> list[str]:
    """Run the helper; return the recorded cp argv lines."""
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    log = tmp_path / "cp.log"
    real_cp = shutil.which("cp")
    shim = shim_dir / "cp"
    shim.write_text(f'#!/bin/sh\necho "$*" >> "{log}"\nexec {real_cp} "$@"\n')
    shim.chmod(0o755)

    script = f'{_helper_src()}\n{HELPER} "$1" "$2" {" ".join(SKIPPED)}\n'
    env = {**os.environ, "PATH": f"{shim_dir}:{os.environ['PATH']}"}
    subprocess.run([BASH, "-c", script, "_", str(src), str(dst)], check=True, env=env)
    return log.read_text().splitlines()


def _vendored(tmp_path: Path) -> Path:
    """Vendored tree: sources plus the heavy dirs the stage must skip."""
    src = tmp_path / "vendor"
    (src / "libavcodec").mkdir(parents=True)
    (src / "libavcodec" / "h264.c").write_text("v1")
    (src / "configure").write_text("#!/bin/sh\n")
    for name in SKIPPED:
        (src / name).mkdir()
        (src / name / "blob").write_text("x")
    return src


def test_regression_cow_stage_copy_reflinks(tmp_path: Path) -> None:
    """Falsification: every cp is CoW, skipped dirs stay out, objects survive."""
    src = _vendored(tmp_path)
    dst = tmp_path / "stage"
    (dst / "libavcodec").mkdir(parents=True)
    (dst / "libavcodec" / "h264.o").write_text("obj")

    calls = _run(tmp_path, src, dst)

    assert calls and all(REFLINK_FLAG in c for c in calls)
    assert (dst / "libavcodec" / "h264.c").read_text() == "v1"
    assert (dst / "configure").exists()
    assert (dst / "libavcodec" / "h264.o").read_text() == "obj"
    for name in SKIPPED:
        assert not (dst / name).exists()


def test_regression_cow_stage_copy_update(tmp_path: Path) -> None:
    """Adversarial: odd names copy, and an upstream edit reaches the stage."""
    src = _vendored(tmp_path)
    (src / "with space").write_text("s")
    (src / "-rf").write_text("d")
    dst = tmp_path / "stage"
    _run(tmp_path, src, dst)

    (src / "libavcodec" / "h264.c").write_text("v2")
    shutil.rmtree(tmp_path / "shim")
    (tmp_path / "cp.log").unlink()
    _run(tmp_path, src, dst)

    assert (dst / "libavcodec" / "h264.c").read_text() == "v2"
    assert (dst / "with space").read_text() == "s"
    assert (dst / "-rf").read_text() == "d"


def test_regression_cow_stage_no_rsync() -> None:
    """The FFmpeg stage goes through the helper, not rsync."""
    text = SCRIPT.read_text()
    assert not re.search(r"^\s*rsync\b", text, re.M)
    assert re.search(rf'^\s*{HELPER} "\$SRC_DIR" "\$STAGE_DIR"', text, re.M)
