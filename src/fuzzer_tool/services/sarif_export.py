"""Export a crash directory as a SARIF log (``sarif`` subcommand, ``fuzz --sarif``).

SARIF is JSON by spec, so this is the one place crash data leaves as JSON
rather than through the pickle machinery.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fuzzer_tool.adapters.filesystem import CRASH_SIDECAR_GLOB, iter_crash_sidecars
from fuzzer_tool.core.sarif import build_sarif


def export_sarif(crashes_dir: str | Path, out_path: str | Path, src_root: str) -> int:
    """Write the SARIF log of *crashes_dir* to *out_path*; return the result count.

    The file is written to a dot-temp and renamed, so a reader never sees a
    half-written log.

    Raises:
        ValueError: *out_path* names a crash sidecar in *crashes_dir*.
    """
    out = Path(out_path)

    # Writing over crash_*.json would replace that crash's metadata.
    # os.replace swaps the name itself, so match the unresolved name.
    in_dir = out.absolute().parent.resolve() == Path(crashes_dir).resolve()
    if in_dir and out.match(CRASH_SIDECAR_GLOB):
        raise ValueError(f"SARIF output would overwrite crash sidecar {out}")

    root = os.path.abspath(src_root) if src_root else ""
    doc = build_sarif(iter_crash_sidecars(Path(crashes_dir)), root)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, out)
    return len(doc["runs"][0]["results"])
