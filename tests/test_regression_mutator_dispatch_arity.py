"""Every entry in a mutator's dispatch list must accept the dispatch call.

The format mutators all share one shape: build a list of bound methods,
draw an index, call ``mutators[op](...)`` with a fixed argument tuple. That
makes the list and the call site a contract, and nothing checked it.

Seven of the eight mutators added in b9baf25 broke it the same way: the
generator went into the list alongside four editors called
``(data, parsed, max_len)``, but the generators are declared
``(self, max_len=..., rng=...)``. So the last index raised TypeError every
time it came up -- 20-25% of mutations on *parseable* input, for av1_rtp,
cfhd, dvbsub, rasc, shorten, tiff and magicyuv.

Nothing caught it because the operator smoke tests mutate whatever they are
given and most seeds do not parse, which takes the early-return branch and
never reaches the list. tests/test_exhaustive_pool.py did surface it, as
four failures that read as an operator-enumeration problem rather than an
arity one.

Checked statically. The alternative -- drive each mutator until the index
comes up -- needs a parseable seed per format and silently stops proving
anything the day a seed stops parsing.
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
from importlib import import_module

import pytest

import fuzzer_tool.core.mutations as mutations_pkg


def _dispatch_sites():
    """Yield (module, class, method, entries, call_site_arg_names).

    Finds ``mutators = [...]`` paired with a ``mutators[...](...)`` call in
    the same function body. Discovered by walking the package, so a mutator
    added later is covered without editing this file.
    """
    for info in pkgutil.iter_modules(mutations_pkg.__path__):
        mod = import_module(f"{mutations_pkg.__name__}.{info.name}")
        try:
            src = inspect.getsource(mod)
        except OSError:  # pragma: no cover - namespace/compiled modules
            continue
        tree = ast.parse(src)
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                entries = None
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Assign)
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "mutators"
                        and isinstance(node.value, ast.List)
                    ):
                        entries = node.value.elts
                if entries is None:
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Subscript)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "mutators"
                    ):
                        arg_names = [a.id if isinstance(a, ast.Name) else None for a in node.args]
                        yield info.name, cls.name, fn.name, entries, arg_names


_SITES = list(_dispatch_sites())


def test_discovery_found_the_dispatch_sites():
    """Guard: an empty list would make every assertion below vacuous."""
    assert len(_SITES) >= 20, f"only found {len(_SITES)} dispatch sites"


@pytest.mark.parametrize(
    "site", _SITES, ids=lambda s: f"{s[0]}.{s[1]}.{s[2]}" if isinstance(s, tuple) else str(s)
)
def test_every_dispatch_entry_accepts_the_dispatch_call(site):
    mod_name, cls_name, fn_name, entries, arg_names = site
    n_args = len(arg_names)
    mod = import_module(f"{mutations_pkg.__name__}.{mod_name}")
    cls = getattr(mod, cls_name)

    for idx, entry in enumerate(entries):
        # A lambda in the list declares its own arity at the call site;
        # count its parameters directly.
        if isinstance(entry, ast.Lambda):
            declared = len(entry.args.args)
            assert declared == n_args, (
                f"{mod_name}.{cls_name}.{fn_name} entry {idx} is a lambda taking "
                f"{declared} args but the dispatch passes {n_args}"
            )
            continue

        # Otherwise it is `self._method`; resolve it on the class and count
        # the parameters it can take positionally, excluding `self`.
        assert isinstance(entry, ast.Attribute), (
            f"{mod_name}.{cls_name}.{fn_name} entry {idx} is neither a lambda nor "
            f"an attribute: {ast.unparse(entry)}"
        )
        method = getattr(cls, entry.attr, None)
        assert method is not None, (
            f"{mod_name}.{cls_name}.{fn_name} entry {idx} names {entry.attr}, "
            "which the class does not define"
        )

        params = list(inspect.signature(method).parameters.values())[1:]  # drop self
        positional = [
            p
            for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        ]
        takes_varargs = any(p.kind is p.VAR_POSITIONAL for p in positional)
        assert takes_varargs or len(positional) >= n_args, (
            f"{mod_name}.{cls_name}.{fn_name} dispatches with {n_args} positional "
            f"args but {entry.attr} accepts at most {len(positional)} -- that index "
            "raises TypeError every time it is drawn"
        )

        # Matching the count is not enough. jpeg2000's generator accepted two
        # positional args and was dispatched with two, so it type-checked and
        # still died: `(markers, max_len)` landed on `(max_len, rng)` and it
        # called `rng.randbytes` on an int. The parameter names cannot be
        # compared in general -- the vestigial first slots are deliberately
        # named `_boxes`, `_words`, `info_or_max` -- but the *last* argument
        # carries its role in its name at every one of these call sites, so
        # it must land on a parameter of that name.
        last = arg_names[-1] if arg_names else None
        if last is not None and not takes_varargs:
            landed = positional[n_args - 1].name
            assert landed == last, (
                f"{mod_name}.{cls_name}.{fn_name} passes `{last}` as argument "
                f"{n_args} but {entry.attr} receives it as `{landed}` -- the count "
                "matches and the meaning does not"
            )
