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


def test_version_link_line_derives_extralibs():
    """Adversarial: no hardcoded -llzma/-lX11 on the per-version link; ffmpeg_extralibs decides."""
    body = BUILD_SCRIPT.read_text().split("build_ffmpeg_versions() {", 1)[1].split("\n}\n", 1)[0]

    assert "ffmpeg_extralibs" in body
    assert not re.search(r"-l(lzma|X11|atomic|bz2)\b", body)


def _extralibs(tmp_path: Path, cc: str) -> str:
    """Run build_targets.sh's ffmpeg_extralibs on a fake tree needing -lm -lz."""
    mak = tmp_path / "root" / "ffbuild" / "config.mak"
    mak.parent.mkdir(parents=True)
    mak.write_text("EXTRALIBS-avutil=-lm -lz\n")
    fn = re.search(r"^ffmpeg_extralibs\(\) \{.*?^\}", BUILD_SCRIPT.read_text(), re.M | re.S)
    assert fn
    script = f'{fn.group(0)}\nDEFAULT_CC="{cc}"\nffmpeg_extralibs "{tmp_path}/root"\n'
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)
    return r.stdout


def test_regression_extralibs_multiword_cc(tmp_path):
    """DEFAULT_CC="ccache clang" made every link probe fail: all libs silently dropped."""
    out = _extralibs(tmp_path, "env clang")

    assert "-lm" in out.split()
    assert "-lz" in out.split()


def test_regression_extralibs_bogus_lib_dropped(tmp_path):
    """Adversarial: the probe still drops a library that does not exist."""
    mak = tmp_path / "x" / "ffbuild" / "config.mak"
    mak.parent.mkdir(parents=True)
    mak.write_text("EXTRALIBS-avutil=-lm -lnosuchlib_fuzz\n")
    fn = re.search(r"^ffmpeg_extralibs\(\) \{.*?^\}", BUILD_SCRIPT.read_text(), re.M | re.S)
    script = f'{fn.group(0)}\nDEFAULT_CC="env clang"\nffmpeg_extralibs "{tmp_path}/x"\n'
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)

    assert "-lm" in r.stdout.split()
    assert "-lnosuchlib_fuzz" not in r.stdout


def _versions(
    tmp_path: Path, variant: str, stale: str = "", opts: str = ""
) -> list[str]:
    """Run build_ffmpeg_versions for one variant against stubbed builders; return their calls."""
    vendor, build = tmp_path / "vendor", tmp_path / "build"
    _tree(vendor, "9.0.2")
    for root in ["ffmpeg-9.0.2", "ffmpeg-9.0.2_asan"]:
        (build / root / "libavformat").mkdir(parents=True)
        (build / root / "libavformat" / "libavformat.a").write_text("")
    if stale:
        (build / stale / ".stale").write_text("")

    text = BUILD_SCRIPT.read_text()
    fn = re.search(r"^build_ffmpeg_versions\(\) \{.*?^\}", text, re.M | re.S)
    want = re.search(r"^ffmpeg_opt_wanted\(\) \{.*?^\}", text, re.M | re.S)
    assert fn and want
    stubs = "\n".join(
        f'{name}() {{ echo "{name} $*"; }}'
        for name in [
            "build_vendored_ffmpeg_sancov",
            "build_target",
            "build_so_target",
            "warn_failed",
        ]
    )
    script = (
        f"{stubs}\nffmpeg_extralibs() {{ :; }}\n{want.group(0)}\n{fn.group(0)}\n"
        f'VENDOR="{vendor}"; FUZZ_BUILD_ROOT="{build}"; TARGETS="{build}"; DEFAULT_CC=clang\n'
        f'FFMPEG_OPTS="{opts}"\n'
        f"build_ffmpeg_versions {variant}\n"
    )
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_regression_versions_build_noasan_and_so(tmp_path):
    """Per-version build emitted only the ASAN exe: no _noasan exe, no .so."""
    asan = "\n".join(_versions(tmp_path / "a", '"_asan" "-fsanitize=address"'))
    noasan = "\n".join(_versions(tmp_path / "n", '"_noasan" ""'))

    assert "build_vendored_ffmpeg_sancov _asan ffmpeg-9.0.2" in asan
    assert re.search(
        r"build_target \S+ \S+/ffmpeg_read_9\.0\.2_asan .*-I\S+/ffmpeg-9\.0\.2_asan", asan
    )
    assert re.search(
        r"build_so_target \S+ \S+/ffmpeg_read_9\.0\.2_asan\.so .*-I\S+/ffmpeg-9\.0\.2_asan", asan
    )

    assert "build_vendored_ffmpeg_sancov  ffmpeg-9.0.2" in noasan
    assert re.search(
        r"build_target \S+ \S+/ffmpeg_read_9\.0\.2_noasan .*-I\S+/ffmpeg-9\.0\.2$", noasan, re.M
    )
    assert re.search(
        r"build_so_target \S+ \S+/ffmpeg_read_9\.0\.2_noasan\.so .*-I\S+/ffmpeg-9\.0\.2$",
        noasan,
        re.M,
    )
    assert "_asan" not in noasan
    assert "-fsanitize" not in noasan


def test_regression_versions_noasan_stale_links_nothing(tmp_path):
    """Adversarial: stale nosan archives -> no noasan link, old binaries removed."""
    old = [tmp_path / "build" / f"ffmpeg_read_9.0.2_noasan{ext}" for ext in ["", ".so"]]
    for f in old:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("old")

    calls = _versions(tmp_path, '"_noasan" ""', stale="ffmpeg-9.0.2")

    assert not any(c.startswith(("build_target", "build_so_target")) for c in calls)
    assert any(c.startswith("warn_failed") for c in calls)
    assert not any(f.exists() for f in old)


def test_regression_versions_wired_into_both_passes():
    """build_ffmpeg_versions runs in the ASAN and the no-ASAN pass."""
    main = BUILD_SCRIPT.read_text().split("# ── Main ──", 1)[1]

    assert re.search(r'build_ffmpeg_versions "_asan" "\$ASAN_CFLAGS"', main)
    assert re.search(r'build_ffmpeg_versions "_noasan" ""', main)


def test_regression_versions_ubsan_and_ngram_variants(tmp_path):
    """ubsan / ng<k> link the coverage-only tree and carry their own flags, exe and .so."""
    ubsan = "\n".join(_versions(tmp_path / "u", '"_ubsan" "-fsanitize=undefined" "clang"'))
    ng = "\n".join(
        _versions(
            tmp_path / "g",
            '"_ng2" "-D__AFL_NGRAM_K=2 -fsanitize-coverage=trace-pc" "clang"',
        )
    )

    # coverage-only tree (no _asan suffix): ASAN archives would leave __asan_* undefined
    assert "build_vendored_ffmpeg_sancov  ffmpeg-9.0.2" in ubsan
    assert re.search(
        r"build_target \S+ \S+/ffmpeg_read_9\.0\.2_ubsan .*-fsanitize=undefined clang -I\S+/ffmpeg-9\.0\.2$",
        ubsan,
        re.M,
    )
    assert re.search(r"build_so_target \S+ \S+/ffmpeg_read_9\.0\.2_ubsan\.so ", ubsan)

    assert re.search(r"build_target \S+ \S+/ffmpeg_read_9\.0\.2_ng2 .*__AFL_NGRAM_K=2", ng)
    assert re.search(r"build_so_target \S+ \S+/ffmpeg_read_9\.0\.2_ng2\.so .*__AFL_NGRAM_K=2", ng)
    assert "_asan" not in ng


def test_regression_versions_opts_filter_skips_unlisted(tmp_path):
    """Adversarial: FFMPEG_OPTS excludes a variant -> no tree build, no link; a listed one still runs."""
    skipped = _versions(tmp_path / "s", '"_noasan" ""', opts="asan,ng2")
    kept = _versions(tmp_path / "k", '"_noasan" ""', opts="asan,noasan")

    assert skipped == []
    assert any(c.startswith("build_target") for c in kept)


def test_regression_versions_matrix_wired_in_main():
    """ubsan rides the ASAN pass, ng<k> the --ngram pass; --ffmpeg-opts= is parsed."""
    text = BUILD_SCRIPT.read_text()
    main = text.split("# ── Main ──", 1)[1]

    assert re.search(r'build_ffmpeg_versions "_asan" "\$ASAN_CFLAGS"\n\s+build_ffmpeg_version_ubsan', main)
    assert re.search(r"build_ngram_so_targets\n\s+build_ffmpeg_version_ngram", main)
    assert "--ffmpeg-opts=*) FFMPEG_OPTS=" in text
