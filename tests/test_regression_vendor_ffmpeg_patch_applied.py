"""Regression: vendor_ffmpeg.sh warned on an already-applied patch.

FFmpeg 9.0.2 ships the vpk guard and the demux.c ch_layout copy upstream.
The already-applied test grepped a forward `git apply --check` for
"Check patch", which git never prints, so every 9.0.2 vendor run said
"WARN: Patch may not apply cleanly".

The fake tree holds each patched file rebuilt from the patch's own hunks,
before (context + removed lines) or after (context + added lines).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "tools" / "vendor_ffmpeg.sh"
PATCH = ROOT / "patches" / "ffmpeg-vpk-divide-by-zero.patch"

pytestmark = [
    pytest.mark.skipif(not SCRIPT.exists(), reason="vendor_ffmpeg.sh not present"),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(shutil.which("clang") is None, reason="clang required"),
]

CONFIGURE_FAIL = "#!/bin/sh\nexit 1\n"
HUNK = re.compile(r"^@@ -(\d+),(\d+) \+\d+,(\d+) @@")
APPLIED = "OK: Patch already applied"
FRESH = "OK: Patch applied successfully"
WARN = "WARN: Patch may not apply cleanly"


def _files(side: str) -> dict[str, str]:
    """Rebuild each patched file: `side` '-' is before the patch, '+' after.

    Blank context lines are stored bare (whitespace stripped), so a hunk is
    read by its header's line counts, not by line prefixes alone.
    """
    out: dict[str, list[str]] = {}
    name = ""
    left = (0, 0)  # (old, new) lines still owed by the current hunk
    for line in PATCH.read_text().splitlines():
        if left != (0, 0):
            tag, text = (line[0], line[1:]) if line else (" ", "")
            if tag in (" ", side):
                out[name].append(text)
            left = (left[0] - (tag != "+"), left[1] - (tag != "-"))
            continue

        if line.startswith("+++ b/"):
            name = line[len("+++ b/") :]
            out[name] = []
            continue

        m = HUNK.match(line)
        if not m:
            continue

        # Filler up to the hunk's start line keeps git's positions honest.
        body = out[name]
        body += [f"/* filler {i} */" for i in range(len(body), int(m.group(1)) - 1)]
        left = (int(m.group(2)), int(m.group(3)))
    return {k: "\n".join(v) + "\n" for k, v in out.items()}


def _run(tmp_path: Path, files: dict[str, str]) -> str:
    tree = tmp_path / "ffmpeg"
    for rel, text in files.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)
    stub = tree / "configure"
    stub.write_text(CONFIGURE_FAIL)
    stub.chmod(0o755)

    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "FUZZ_VENDOR_ROOT": str(tmp_path / "vendor"),
        "FFMPEG_DIR": str(tree),
    }
    r = subprocess.run(
        ["bash", str(SCRIPT), "--nosan", "--minimal"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.stdout


def test_regression_already_applied_is_recognised(tmp_path):
    out = _run(tmp_path, _files("+"))

    assert APPLIED in out
    assert WARN not in out


def test_unpatched_tree_gets_patched(tmp_path):
    """Falsification: a pre-patch tree must still be patched, not reported applied."""
    out = _run(tmp_path, _files("-"))

    assert FRESH in out
    assert APPLIED not in out
    assert (tmp_path / "ffmpeg" / "libavformat" / "vpk.c").read_text() == _files("+")[
        "libavformat/vpk.c"
    ]


def test_unrelated_tree_still_warns(tmp_path):
    """Adversarial: neither side matches -- must warn, not claim applied."""
    junk = {name: "int unrelated;\n" for name in _files("+")}

    out = _run(tmp_path, junk)

    assert WARN in out
    assert APPLIED not in out
