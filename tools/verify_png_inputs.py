#!/usr/bin/env python3
"""Pipe each id_* file in a directory to targets/png_read and report stdout/stderr/returncode.

Usage: verify_png_inputs.py <directory>
"""

import glob
import subprocess
import sys
from pathlib import Path

TARGET = Path("targets/png_read")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(glob.glob(str(input_dir / "id_*")))
    if not files:
        print(f"No id_* files found in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"{'FILE':<50} {'RC':>4}  {'STDERR'}")
    print("-" * 120)

    for fpath in files:
        fname = Path(fpath).name
        try:
            result = subprocess.run(
                [str(TARGET)],
                stdin=open(fpath, "rb"),
                capture_output=True,
                timeout=5,
            )
            stdout = result.stdout.decode("utf-8", errors="replace").strip()
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            rc = result.returncode
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "(timeout)"
            rc = -1
        except Exception as e:
            stdout = ""
            stderr = str(e)
            rc = -2

        if rc == 0 and not stderr:
            tag = "(ok)"
        else:
            tag = ""

        print(f"{fname:<50} {rc:>4}  {stderr or stdout or tag}")


if __name__ == "__main__":
    main()
