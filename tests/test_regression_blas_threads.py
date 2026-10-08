"""OpenBLAS ran the fuzzer's tiny matmuls/reductions on a thread pool.

Thread handoff costs more than it saves on 28-arm and 64-tile arrays, and
parallel campaigns oversubscribe the cores: 3 parallel 3k-exec --hail-mary
runs took 95-108 s, 29-43 s with one BLAS thread. fuzzer_tool defaults
the BLAS pools to one thread at import; an explicit setting still wins.
"""

import os
import subprocess
import sys

_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _env_after_import(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _VARS}
    env.update(extra)
    code = f"import os, fuzzer_tool; print(*(os.environ.get(v) for v in {_VARS!r}))"
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return dict(zip(_VARS, out.stdout.split(), strict=True))


def test_regression_blas_threads():
    assert _env_after_import({}) == dict.fromkeys(_VARS, "1")


def test_user_setting_wins():
    """Adversarial: an operator who asks for 8 BLAS threads gets 8."""
    got = _env_after_import({"OPENBLAS_NUM_THREADS": "8"})
    assert got["OPENBLAS_NUM_THREADS"] == "8"
    assert got["MKL_NUM_THREADS"] == "1"
