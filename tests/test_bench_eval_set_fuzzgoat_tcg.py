"""Bench set ``fuzzgoat_tcg``: the calibration target with trace-cmp in its library object.

The default fuzzgoat .so logs no compares (``cmplog: 0t 0p``), so cmplog A/Bs
need this build (``build_tracecmp_targets`` in ``tools/build_targets.sh``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools" / "lib"))

from eval_set import TARGET_SETS  # noqa: E402

_SET = "fuzzgoat_tcg"


def test_targets_trace_cmp_build():
    targets = [t for t, _ in TARGET_SETS[_SET]]
    assert targets and all(t.endswith("fuzzgoat_read_tcg.so") for t in targets)


def test_falsify_only_build_differs_from_fuzzgoat():
    """Same flags as ``fuzzgoat``: an A/B across the sets measures the build alone."""
    (_, tcg_flags), *_ = TARGET_SETS[_SET]
    (_, base_flags), *_ = TARGET_SETS["fuzzgoat"]
    assert tcg_flags == base_flags


def test_adversarial_name_matches_build_script():
    """A rename on either side would point the set at a file nobody builds."""
    script = (_ROOT / "tools" / "build_targets.sh").read_text()
    built = set(re.findall(r'\$TARGETS/(fuzzgoat_read_tcg\.so)"', script))
    (target, _), *_ = TARGET_SETS[_SET]
    assert Path(target).name in built
