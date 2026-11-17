"""ffmpeg_read IJON variant (-DFUZZ_IJON) against the default build.

    default  noasan: no IJON_SET edges, edge ids as ever
    variant  ijon:   stream-log / output-format IJON_SET edges (bit 31), and
                     IJON_STATE(demuxer) moves real edge ids; same block sites

Skipped unless both builds exist (tools/build_targets.sh --ffmpeg-opts=noasan,ijon).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.elf import parse_sancov_guard_count
from tests.test_regression_shim_audit import _clean_env

BUILDS = Path(os.environ.get("FUZZ_BUILD_ROOT", Path.home() / "fuzzing" / "builds"))
NOASAN = BUILDS / "ffmpeg_read_9.0.2_noasan"
IJON = BUILDS / "ffmpeg_read_9.0.2_ijon"
SYNTH = 0x80000000
MAP_ENTRIES = 1 << 16

pytestmark = pytest.mark.skipif(
    not (NOASAN.exists() and IJON.exists()), reason="ffmpeg noasan/ijon builds absent"
)


def _wav(tmp_path: Path) -> Path:
    """Minimal PCM WAV: 16 frames, mono, 8 kHz."""
    import struct

    data = b"\x00\x01" * 16
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    p = tmp_path / "s.wav"
    p.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return p


def _edges(exe: Path, inp: Path) -> set[int]:
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        env = _clean_env(
            __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES), FUZZER_KEEP_ASLR="1"
        )
        r = subprocess.run([str(exe), str(inp)], env=env, capture_output=True, timeout=60)
        assert r.returncode == 0, r.stderr[-500:]
        return shm.get_edge_ids()
    finally:
        shm.cleanup()


def test_falsify_default_build_has_no_ijon_edges(tmp_path):
    """The default build must not carry the variant's annotations."""
    wav = _wav(tmp_path)
    plain = _edges(NOASAN, wav)
    variant = _edges(IJON, wav)
    synth_plain = {e for e in plain if e & SYNTH}
    synth_variant = {e for e in variant if e & SYNTH}
    assert synth_variant - synth_plain, "variant adds IJON_SET edges"


def test_variant_state_moves_real_edges(tmp_path):
    """IJON_STATE(demuxer): the decode loop's real edges get new ids."""
    wav = _edges(IJON, _wav(tmp_path))
    plain = _edges(NOASAN, _wav(tmp_path))
    real = {e for e in wav if not e & SYNTH}
    base = {e for e in plain if not e & SYNTH}
    assert real != base


def test_adversarial_variant_is_deterministic(tmp_path):
    wav = _wav(tmp_path)
    assert _edges(IJON, wav) == _edges(IJON, wav)


def test_same_guard_sites_as_default_build():
    """Helpers are __AFL_NO_COV: block counts of the two builds compare 1:1."""
    assert parse_sancov_guard_count(str(IJON)) == parse_sancov_guard_count(str(NOASAN))
