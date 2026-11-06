"""Regression: verse nodes ordered their samples by PYTHONHASHSEED.

`_WsNode` / `_AlphaNumNode` / `_NumNode` deduped with `list(set(samples))`.
Bytes hash per interpreter, so `rng.choice(self._samples)` picked a
different sample from the same draw: two `--seed 1 --clock virtual` fuzzgoat
runs split before 8k execs, and matched through 12k with the hash seed pinned.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
SAMPLES = [b"zeta", b"alpha", b" ", b"zeta", b"\t", b"42", b"alpha", b"0x1f"]
NODES = ("_WsNode", "_AlphaNumNode", "_NumNode")

_PROBE = f"""
import json
from fuzzer_tool.core.mutations import generic
samples = {SAMPLES!r}
print(json.dumps([[s.hex() for s in getattr(generic, n)(samples)._samples] for n in {NODES!r}]))
"""


def _orders(hash_seed: str) -> list[list[str]]:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": SRC}
    r = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _first_seen(items: list[bytes]) -> list[str]:
    out: list[str] = []
    for s in items:
        if s.hex() not in out:
            out.append(s.hex())
    return out


def test_regression_samples_ignore_hash_seed():
    a, b = _orders("0"), _orders("1")

    assert a == b


def test_samples_keep_first_seen_order():
    """Falsification: dedup must still drop repeats, in input order."""
    want = _first_seen(SAMPLES)

    for got in _orders("0"):
        assert got == want


def test_empty_samples_stay_empty():
    """Adversarial: no samples, no crash, empty list."""
    probe = (
        "from fuzzer_tool.core.mutations import generic;"
        f"print([getattr(generic, n)([])._samples for n in {NODES!r}])"
    )
    env = {**os.environ, "PYTHONPATH": SRC}
    r = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)

    assert r.stdout.strip() == "[[], [], []]", r.stderr
