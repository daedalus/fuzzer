"""Core domain logic for fuzzer-tool."""

from fuzzer_tool.core.chi_squared import (
    ContingencyTable,
    chi_squared_goodness_of_fit,
    chi_squared_homogeneity,
    chi_squared_independence,
    chi_squared_pvalue,
    cramers_v,
)
from fuzzer_tool.core.count_class import classify_counts, classify_single, new_bits
from fuzzer_tool.core.critical_slowing import (
    CoverageHomogeneityDetector,
    CriticalSlowingDown,
)
from fuzzer_tool.core.markov import MarkovChain
from fuzzer_tool.core.mi import MutualInformationTracker
from fuzzer_tool.core.mutations import (
    DICT_MUTATIONS,
    INTERESTING_8,
    INTERESTING_16,
    INTERESTING_32,
    MAGIC_TABLE,
    MUTATIONS,
    SPECIAL_STRINGS,
    ascii_num_arithmetic,
    load_dictionary,
    parse_dict_line,
)
from fuzzer_tool.core.rate_distortion import RateDistortionCorpus
from fuzzer_tool.core.renyi import CoverageSpectrumAnalyzer, RenyiEntropy
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.core.schedulers import (
    EpsilonGreedyScheduler,
    Exp3Scheduler,
    GPUCBScheduler,
    HierarchicalBanditScheduler,
    MonteCarloScheduler,
    MOptScheduler,
    ReplicatorScheduler,
)
from fuzzer_tool.core.seed_quality import BayesianSeedQuality
from fuzzer_tool.core.shapley import ShapleyAttribution
from fuzzer_tool.core.transfer_entropy import TransferEntropy

__all__ = [
    "ContingencyTable",
    "chi_squared_goodness_of_fit",
    "chi_squared_homogeneity",
    "chi_squared_independence",
    "chi_squared_pvalue",
    "cramers_v",
    "classify_counts",
    "classify_single",
    "new_bits",
    "BayesianSeedQuality",
    "MarkovChain",
    "MonteCarloScheduler",
    "MOptScheduler",
    "ReplicatorScheduler",
    "EpsilonGreedyScheduler",
    "Exp3Scheduler",
    "HierarchicalBanditScheduler",
    "GPUCBScheduler",
    "ShapleyAttribution",
    "MutualInformationTracker",
    "RenyiEntropy",
    "CoverageSpectrumAnalyzer",
    "RateDistortionCorpus",
    "TransferEntropy",
    "SanitizerReport",
    "CriticalSlowingDown",
    "CoverageHomogeneityDetector",
    "INTERESTING_8",
    "INTERESTING_16",
    "INTERESTING_32",
    "MUTATIONS",
    "DICT_MUTATIONS",
    "MAGIC_TABLE",
    "SPECIAL_STRINGS",
    "ascii_num_arithmetic",
    "parse_dict_line",
    "load_dictionary",
]
