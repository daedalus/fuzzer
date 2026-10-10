"""Touched-slot scan is on by default for large maps (fast path), off for small.

``_live_entries`` walks the whole edge table every exec: 0.14 ms at 65,536
entries, 0.71 ms at 262,144 (FFmpeg auto-sizes there), 1.7 ms at 1M. The
bitmap scan (``--touched-scan``) was measured to net positive from 65,536
entries up and slower at 8,192, but was opt-in. Default is now automatic by
map size; ``--touched-scan`` / ``--no-touched-scan`` force it either way.
"""

import pytest

from fuzzer_tool.adapters.shm import TOUCHED_SCAN_MIN_ENTRIES
from fuzzer_tool.services.fuzzer import resolve_touched_scan


@pytest.mark.parametrize(
    "requested, map_size, expected",
    [
        (None, TOUCHED_SCAN_MIN_ENTRIES * 4, True),  # FFmpeg-sized: fast path
        (None, TOUCHED_SCAN_MIN_ENTRIES, True),  # boundary is inclusive
        (None, TOUCHED_SCAN_MIN_ENTRIES - 1, False),  # just below: full scan
        (None, 8192, False),  # measured slower here
    ],
)
def test_auto_by_map_size(requested, map_size, expected):
    """Falsification: auto picks the measured-faster read per map size."""
    assert resolve_touched_scan(requested, map_size) is expected


@pytest.mark.parametrize("requested", [True, False])
@pytest.mark.parametrize("map_size", [8192, 1 << 20])
def test_explicit_choice_wins(requested, map_size):
    """Adversarial: an explicit flag overrides the size rule both ways."""
    assert resolve_touched_scan(requested, map_size) is requested


@pytest.mark.parametrize(
    "argv, expected",
    [([], None), (["--touched-scan"], True), (["--no-touched-scan"], False)],
)
def test_cli_tristate(monkeypatch, tmp_path, argv, expected):
    """Falsification: the CLI forwards auto (None) by default, flags force on/off."""
    from fuzzer_tool.cli import commands

    seen = {}

    class _Stop(Exception):
        pass

    def fake_fuzzer(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(commands, "Fuzzer", fake_fuzzer)
    target = tmp_path / "t"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    monkeypatch.setattr(
        "sys.argv", ["fuzzer-tool", "fuzz", str(target), "-d", str(tmp_path / "c"), *argv]
    )
    with pytest.raises(_Stop):
        commands.main()
    assert seen["touched_scan"] is expected
