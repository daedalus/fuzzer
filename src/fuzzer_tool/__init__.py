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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
