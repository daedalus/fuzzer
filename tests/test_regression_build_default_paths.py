"""Regression: build_targets.sh defaulted TAILSLAYER to one developer's home
(/home/<user>/code/tailslayer). Library defaults must derive from $VENDOR."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "tools" / "build_targets.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="build_targets.sh not present")

_DEFAULT_RE = re.compile(r'^\s*[A-Z_]+="\$\{[A-Z_]+_DIR:-([^}]*)\}"', re.M)


def test_regression_tailslayer_default_under_vendor():
    text = SCRIPT.read_text()
    assert 'TAILSLAYER="${TAILSLAYER_DIR:-$VENDOR/tailslayer}"' in text


def test_regression_no_home_default_paths():
    """Adversarial: no *_DIR default may be an absolute /home path."""
    defaults = _DEFAULT_RE.findall(SCRIPT.read_text())
    assert defaults, "pattern matched nothing; test is broken"
    assert [d for d in defaults if d.startswith("/home/")] == []
