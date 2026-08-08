"""Guard the mypy ratchet: the exemption list may shrink, never grow.

``[tool.mypy] strict = true`` reported 1724 errors across 114 of 131 source
files, so ``mypy src/`` could never go green. A gate that is permanently red
is a gate nobody reads, which is how 47 ruff errors and 28 test failures
survived until the 2026-08-08 review.

The ``[[tool.mypy.overrides]]`` block in ``pyproject.toml`` exempts exactly
those 114 modules and lets the other 17 be checked strictly. New modules are
strict automatically because they are absent from the list. These tests keep
the list honest: entries must name real modules, the list must not grow, and
nothing may be added to buy silence for a new file.
"""

import os
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")

# Ceiling recorded at the ratchet's introduction. Lower it as modules are
# fixed; never raise it.
EXEMPT_CEILING = 114


def _exempt_modules() -> list[str]:
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    modules: list[str] = []
    for override in cfg["tool"]["mypy"].get("overrides", []):
        if override.get("ignore_errors"):
            mods = override["module"]
            modules.extend([mods] if isinstance(mods, str) else mods)
    return modules


def _module_path(module: str) -> str:
    return os.path.join(SRC, module.replace(".", os.sep) + ".py")


def test_strict_is_still_the_target():
    """The exemptions are a ratchet, not a decision to stop type checking."""
    with open(os.path.join(REPO, "pyproject.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["tool"]["mypy"]["strict"] is True


def test_every_exempt_module_exists():
    """A stale entry silently exempts nothing and hides a rename."""
    missing = [m for m in _exempt_modules() if not os.path.exists(_module_path(m))]
    assert not missing, (
        f"exempt modules with no source file: {missing} — delete them from "
        "the [[tool.mypy.overrides]] block in pyproject.toml"
    )


def test_exemption_list_does_not_grow():
    exempt = _exempt_modules()
    assert len(exempt) <= EXEMPT_CEILING, (
        f"{len(exempt)} modules exempted from mypy, ceiling is "
        f"{EXEMPT_CEILING}. New code is strict by default — fix the types "
        "rather than adding a line here."
    )


def test_no_duplicate_entries():
    exempt = _exempt_modules()
    dupes = {m for m in exempt if exempt.count(m) > 1}
    assert not dupes, f"duplicate exemptions: {sorted(dupes)}"


def test_ceiling_tracks_the_list():
    """Once the list shrinks, the ceiling comes down with it."""
    exempt = _exempt_modules()
    assert EXEMPT_CEILING - len(exempt) <= 10, (
        f"{EXEMPT_CEILING - len(exempt)} modules have been fixed without "
        f"lowering EXEMPT_CEILING; set it to {len(exempt)}"
    )
