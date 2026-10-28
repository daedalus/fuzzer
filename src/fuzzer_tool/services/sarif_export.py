"""Export a crash directory as a SARIF log (``sarif`` subcommand, ``fuzz --sarif``).

SARIF is JSON by spec, so this is the one place crash data leaves as JSON
rather than through the pickle machinery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fuzzer_tool.adapters.filesystem import load_crash_sidecars
from fuzzer_tool.core.sarif import build_sarif


def export_sarif(crashes_dir: str | Path, out_path: str | Path, src_root: str) -> int:
    """Write the SARIF log of *crashes_dir* to *out_path*; return the result count.

    The file is written to a dot-temp and renamed, so a reader never sees a
    half-written log.
    """
    root = os.path.abspath(src_root) if src_root else ""
    doc = build_sarif(load_crash_sidecars(Path(crashes_dir)), root)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, out)
    return len(doc["runs"][0]["results"])
