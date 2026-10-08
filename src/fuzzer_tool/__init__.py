"""fuzzer-tool: Coverage-guided binary fuzzer."""

__version__ = "0.1.0"
__all__ = [
    "MarkovChain",
    "MonteCarloScheduler",
    "SanitizerReport",
    "Fuzzer",
    "load_dictionary",
    "parse_dict_line",
]

import os as _os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# One BLAS thread unless the operator says otherwise. The fuzzer's numpy
# work is thousands of tiny matmuls/reductions per exec, where OpenBLAS's
# thread handoff costs more than it saves, and parallel campaigns
# oversubscribe the cores: 3 parallel 3k-exec --hail-mary runs went from
# 95-108 s to 29-43 s. Must run before anything imports numpy.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
