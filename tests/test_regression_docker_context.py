"""The Docker build context must ship the scripts the build sources (2026-11-24 review).

``.dockerignore`` excluded ``tools/lib/`` while ``tools/build_targets.sh``
unconditionally sourced ``tools/lib/ffmpeg_config.sh`` and ``asan_flags.sh``
from it (and compiled ``frontier_stub.c``), so any docker build that reached
a target-build stage failed on a fresh checkout. This gate parses both files
so the exclusion cannot come back silently.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _dockerignore_excluded() -> set[str]:
    """Path prefixes excluded from the Docker context (globs kept verbatim)."""
    out: set[str] = set()
    for line in (REPO / ".dockerignore").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.rstrip("/"))
    return out


def test_every_lib_path_the_build_uses_is_in_the_context():
    """Files build_targets.sh sources or compiles from tools/lib/ must ship."""
    src = (REPO / "tools" / "build_targets.sh").read_text()
    used = set(re.findall(r"\$REPO_ROOT/(tools/lib/[A-Za-z0-9_./-]+)", src))
    assert used, "no tools/lib references found — update this test's pattern"

    excluded = _dockerignore_excluded()
    missing = [u for u in sorted(used) if any(u.startswith(e) for e in excluded)]
    assert not missing, (
        f"tools/build_targets.sh uses {missing} but .dockerignore excludes "
        "them from the Docker context — every target-build stage fails on "
        "a fresh checkout"
    )
