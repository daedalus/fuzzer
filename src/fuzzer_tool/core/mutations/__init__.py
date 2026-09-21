"""Re-export generic mutation operators from the subpackage.

This module re-exports all public names from :mod:`fuzzer_tool.core.mutations.generic`
so that existing imports of ``fuzzer_tool.core.mutations`` continue to work.
"""

# Pairwise covering-array PNG IHDR field mutation (self-registers)
from fuzzer_tool.core.mutations.covering_array_mutate import (  # noqa: F401
    PngCoveringArrayMutator,
)

# Fractal jittered Voronoi spatial meta-mutator (self-registers on import)
from fuzzer_tool.core.mutations.fractal_voronoi import FractalVoronoiMutator  # noqa: F401
from fuzzer_tool.core.mutations.generic import *  # noqa: F401,F403
from fuzzer_tool.core.mutations.generic import (  # noqa: F401
    _FUNNY_UNICODE,
    _big_int_squared,
    _divisor_sizes,
)

# Perlin/gradient noise coherent-perturbation mutator (self-registers on import)
from fuzzer_tool.core.mutations.perlin_noise import PerlinNoiseMutator  # noqa: F401

# Learned-adjacency WFC chunk reordering, isobmff/webp/riff/gif (self-registers)
from fuzzer_tool.core.wfc_chunks import WfcChunkMutator  # noqa: F401
