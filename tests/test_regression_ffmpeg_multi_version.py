"""vendor_ffmpeg.sh --top=N: fetch the newest N FFmpeg release lines side by side.

Resolution runs against a local git repo standing in for upstream
(FFMPEG_GIT_URL), and the expected source trees are pre-created so nothing
is downloaded. FFMPEG_SRC points at a missing file: a wrongly resolved
version has no tree, falls through to the fetch, and fails the run.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "tools" / "vendor_ffmpeg.sh"
BUILD_SCRIPT = SCRIPT.parent / "build_targets.sh"
BASH = shutil.which("bash") or "bash"

pytestmark = [
    pytest.mark.skipif(shutil.which("clang") is None, reason="clang required"),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
]

# Upstream-shaped tags: dev tags, a bare major.minor, and 8.0.10 > 8.0.3
# (a lexical sort would pick 8.0.3).
TAGS = ["n7.1.5", "n8.0", "n8.0.3", "n8.0.10", "n8.1", "n8.1.3", "n9.0", "n9.0.2", "n9.1-dev"]
TOP3 = ["9.0.2", "8.1.3", "8.0.10"]
MIRROR_MARK = "FROM_MIRROR"


def _upstream(tmp_path: Path) -> Path:
    """Local git repo carrying TAGS and a stub configure (clonable as a mirror)."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "configure").write_text("#!/bin/sh\nexit 0\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / MIRROR_MARK).write_text("")
    subprocess.run([*git, "add", "configure", MIRROR_MARK], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    for tag in TAGS:
        subprocess.run([*git, "tag", tag], check=True)
    return repo


def _tree(vendor: Path, ver: str) -> None:
    """Pre-existing source tree, so fetch_source skips the download."""
    d = vendor / f"ffmpeg-{ver}"
    d.mkdir(parents=True)
    (d / "configure").write_text("#!/bin/sh\nexit 0\n")


def _run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    full_env = {
        **os.environ,
        "HOME": str(tmp_path),
        "FUZZ_VENDOR_ROOT": str(tmp_path / "vendor"),
        "FFMPEG_SRC": f"file://{tmp_path}/missing.tar",
        **env,
    }
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_top_picks_newest_release_lines(tmp_path):
    """Falsification: --top=3 resolves 9.0.2 / 8.1.3 / 8.0.10, fetches only those."""
    repo = _upstream(tmp_path)
    for ver in TOP3:
        _tree(tmp_path / "vendor", ver)

    r = _run(tmp_path, "--top=3", FFMPEG_GIT_URL=str(repo))

    assert r.returncode == 0, r.stdout + r.stderr
    present = re.findall(r"Source present at .*/ffmpeg-(\S+)", r.stdout)
    assert present == TOP3
    assert "[3/5]" not in r.stdout


def test_top_wrong_version_fails(tmp_path):
    """Adversarial: only the lexical pick (8.0.3) exists -> 8.0.10 must be fetched and fail."""
    repo = _upstream(tmp_path)
    for ver in ["9.0.2", "8.1.3", "8.0.3"]:
        _tree(tmp_path / "vendor", ver)

    r = _run(tmp_path, "--top=3", FFMPEG_GIT_URL=str(repo))

    assert r.returncode != 0
    assert "ffmpeg-8.0.10" in r.stdout + r.stderr


def test_versions_override_skips_resolution(tmp_path):
    """FFMPEG_VERSIONS wins over the remote: an unreachable URL is never read."""
    for ver in ["1.0", "2.0"]:
        _tree(tmp_path / "vendor", ver)

    r = _run(tmp_path, "--top=2", FFMPEG_VERSIONS="2.0 1.0", FFMPEG_GIT_URL=str(tmp_path / "nope"))

    assert r.returncode == 0, r.stdout + r.stderr
    assert re.findall(r"Source present at .*/ffmpeg-(\S+)", r.stdout) == ["2.0", "1.0"]


@pytest.mark.parametrize("arg", ["--top=0", "--top=abc", "--top="])
def test_top_rejects_bad_count(tmp_path, arg):
    """Adversarial: non-positive / non-numeric counts exit 2 before any fetch."""
    r = _run(tmp_path, arg)

    assert r.returncode == 2
    assert "Source present" not in r.stdout


def test_top_unreachable_remote_fails(tmp_path):
    """Adversarial: resolution yielding nothing must fail, not succeed with zero versions."""
    r = _run(tmp_path, "--top=3", FFMPEG_GIT_URL=str(tmp_path / "nope"))

    assert r.returncode != 0


def test_build_script_links_each_version():
    """build_targets.sh builds one ffmpeg_read per vendored ffmpeg-<ver> tree."""
    script = BUILD_SCRIPT.read_text()

    assert re.search(r'\$VENDOR"?/ffmpeg-\*', script)
    assert re.search(
        r'build_target\s+"\$\{TARGETS_SRC:-\$TARGETS\}/ffmpeg_read\.c"\s+"\$TARGETS/ffmpeg_read_\$\{?ver',
        script,
    )


def test_top_fetches_from_configured_mirror(tmp_path):
    """FFMPEG_GIT_URL serves the sources too, not only the tag list."""
    repo = _upstream(tmp_path)

    r = _run(tmp_path, "--top=1", FFMPEG_GIT_URL=str(repo), FFMPEG_SRC="")

    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "vendor" / "ffmpeg-9.0.2" / MIRROR_MARK).is_file()


def test_top_mirror_missing_tag_fails(tmp_path):
    """Adversarial: a mirror without the tag must fail, not fall back to another source."""
    repo = _upstream(tmp_path)

    r = _run(tmp_path, "--top=1", FFMPEG_GIT_URL=str(repo), FFMPEG_SRC="", FFMPEG_VERSIONS="6.6.6")

    assert r.returncode != 0
    assert not (tmp_path / "vendor" / "ffmpeg-6.6.6").exists()


def test_top_needs_no_compiler(tmp_path):
    """--top is sources-only: a PATH without clang must still vendor."""
    _tree(tmp_path / "vendor", "1.0")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ["mkdir", "dirname", "git", "rm", "cat"]:
        (bindir / tool).symlink_to(shutil.which(tool))

    r = _run(tmp_path, "--top=1", FFMPEG_VERSIONS="1.0", PATH=str(bindir))

    assert r.returncode == 0, r.stdout + r.stderr
    assert "Source present" in r.stdout
