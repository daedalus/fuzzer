"""Hot-path shim globals bind locally in a ``.so`` built without -Bsymbolic.

Default visibility left ``__afl_area`` and ``__afl_prev_loc`` preemptible, so
every edge read them through the GOT (one extra load each). Protected
visibility keeps them exported (``nm`` detection, ctypes ``in_dll``) but
binds the shim's own references directly.
"""

from __future__ import annotations

import ctypes
import subprocess

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build

HOT_GLOBALS = ("__afl_area", "__afl_prev_loc")

_LIB = """
uintptr_t area_addr(void) { return (uintptr_t)&__afl_area; }
uintptr_t prev_addr(void) { return (uintptr_t)&__afl_prev_loc; }
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("nogot"), _LIB, "-fsanitize-coverage=trace-pc-guard",
                  shared=True)  # fmt: skip


def _dynsyms(path):
    out = subprocess.run(["readelf", "-W", "--dyn-syms", str(path)], capture_output=True, text=True)
    return out.stdout


def _relocs(path):
    out = subprocess.run(["readelf", "-W", "-r", str(path)], capture_output=True, text=True)
    return out.stdout


@requires_clang
class TestHotGlobalsBindLocally:
    @pytest.mark.parametrize("name", HOT_GLOBALS)
    def test_no_got_slot(self, lib, name):
        hits = [ln for ln in _relocs(lib).splitlines() if "GLOB_DAT" in ln and f" {name}" in ln]
        assert hits == []

    @pytest.mark.parametrize("name", HOT_GLOBALS)
    def test_falsify_still_exported(self, lib, name):
        """Hiding them would pass the GOT test and break detection / in_dll."""
        assert any(ln.split()[-1] == name for ln in _dynsyms(lib).splitlines() if ln.strip())

    def test_adversarial_dlsym_sees_the_shim_storage(self, lib):
        """A copy the shim does not use would read NULL after attach."""
        so = ctypes.CDLL(str(lib))
        so.area_addr.restype = so.prev_addr.restype = ctypes.c_size_t
        assert ctypes.addressof(ctypes.c_void_p.in_dll(so, "__afl_area")) == so.area_addr()
        assert ctypes.addressof(ctypes.c_uint32.in_dll(so, "__afl_prev_loc")) == so.prev_addr()
