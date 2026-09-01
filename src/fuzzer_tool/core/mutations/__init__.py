"""Re-export generic mutation operators from the subpackage.

This module re-exports all public names from :mod:`fuzzer_tool.core.mutations.generic`
so that existing imports of ``fuzzer_tool.core.mutations`` continue to work.
"""

from fuzzer_tool.core.mutations.generic import *  # noqa: F401,F403
from fuzzer_tool.core.mutations.generic import _FUNNY_UNICODE, _divisor_sizes  # noqa: F401

# Fractal jittered Voronoi spatial meta-mutator (self-registers on import)
from fuzzer_tool.core.mutations.fractal_voronoi import FractalVoronoiMutator  # noqa: F401
