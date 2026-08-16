"""Keep z3's global context out of interpreter finalization.

``z3.Context.__del__`` calls ``Z3_del_context()``. For the singleton context
that every bare ``z3.Solver()`` shares (``z3.main_ctx()``), that destructor
runs during interpreter shutdown, against z3's own module globals -- several
hundred ``Elementaries`` wrappers, the error handler, the loaded shared
library -- which CPython clears in an order z3 does not control. When the
context loses that race, ``Z3_del_context`` faults inside the native library:

    Fatal Python error: Segmentation fault
    Current thread ...:
      File ".../z3/z3core.py", line 1684 in Z3_del_context
      File ".../z3/z3.py", line 224 in __del__

Observed as an intermittent ``Segmentation fault`` on a full test suite run
-- roughly one run in eight, *after* every test had already passed, so the
only visible symptom was a run that ended without its summary line. See
``docs/handover/suite_segfault_z3_finalization_2026-08-16.md``.

The fix is to not destroy the context at shutdown at all. ``__del__`` is
guarded on ``self.owner``, so clearing that flag from an ``atexit`` hook --
which runs while the interpreter is still fully alive, before module globals
are torn down -- makes the destructor a no-op.

That deliberately leaks the context, and that is the correct trade: the
process is exiting, so the OS reclaims the memory either way, and the only
thing ``Z3_del_context`` buys at that point is the opportunity to crash. It
is scoped to the singleton, so a caller that builds its own ``z3.Context()``
still gets ordinary refcounted teardown while the process is running.

Registration is lazy and z3-free: the hook resolves z3 out of
``sys.modules`` at exit, so importing this module never imports z3 and does
nothing at all on a machine without the optional ``smt`` extra.
"""

from __future__ import annotations

import atexit
import contextlib
import sys

_registered = False


def _disown_main_context() -> None:
    """Clear ``owner`` on z3's singleton context, if one was ever created."""
    z3_mod = sys.modules.get("z3.z3")
    if z3_mod is None:
        return
    ctx = getattr(z3_mod, "_main_ctx", None)
    if ctx is None:
        return
    with contextlib.suppress(Exception):  # z3 internals are not ours
        ctx.owner = False


def guard_z3_shutdown() -> None:
    """Arm the shutdown guard. Idempotent, and safe to call before z3 exists.

    Call this from every site that imports z3. Calling it early is fine --
    the hook looks the context up at exit, not now -- so a site that ends up
    never constructing a solver costs one no-op atexit callback.
    """
    global _registered
    if _registered:
        return
    _registered = True
    atexit.register(_disown_main_context)
