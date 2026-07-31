# mutation-module-reorg: Move format mutation modules to core/mutations/ subpackage

**Date:** 2026-07-31
**Context:** fuzzer-tool repo, `src/fuzzer_tool/core/` reorganization

## Problem
Format-specific mutation modules (`png_mutations.py`, `jpeg_mutations.py`, etc.) were flat in `core/` alongside the generic `mutations.py`. The task was to reorganize them into a `core/mutations/` subpackage as `core/mutations/[format].py`.

## Rejected
- **Keep format files flat, just rename** — didn't address the organizational goal and left `core/` cluttered.
- **Create `mutations/` package without moving `mutations.py`** — caused a naming conflict where the `mutations/` directory shadowed `mutations.py`, breaking all `from fuzzer_tool.core.mutations import ...` imports across the codebase.

## Approach
1. Created `core/mutations/` package with `__init__.py`
2. Moved each `core/[format]_mutations.py` → `core/mutations/[format].py` (8 files)
3. Moved `core/mutations.py` → `core/mutations/generic.py` (the generic mutation operators)
4. `__init__.py` re-exports from `generic.py` via `from fuzzer_tool.core.mutations.generic import *` plus explicit private-name re-exports (`_FUNNY_UNICODE`, `_divisor_sizes`) needed by tests
5. Updated all imports in `operators.py`, `png.py` (self-referencing docstring), and 6 test files
6. Updated module path strings in `test_operator_smoke.py` and `test_rng_threading.py`

## Key insight
The naming conflict between `mutations.py` (generic module) and `mutations/` (new package) was the critical obstacle. Python resolves packages before modules with the same name, so `from fuzzer_tool.core.mutations import MUTATIONS` would find the package's `__init__.py` instead of the `.py` file. Moving `mutations.py` into the package as `generic.py` and re-exporting via `__init__.py` resolved this cleanly.

## Verification
- All 2798 tests pass (8 skipped) after the move
- `ruff format` and `ruff check` clean on changed files
- No stale references to old module paths remain in `src/` or `tests/`

## Generalizes to
This pattern (flat format-specific modules → subpackage) applies whenever a codebase accumulates many `*_mutations.py` files in a single directory. The key pitfall is always the naming conflict between a module and a package of the same name — the generic module must be renamed or moved into the package to avoid shadowing.
