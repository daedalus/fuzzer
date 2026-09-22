"""Operator-selection schedulers (bandit algorithms) and seed-selection schedulers."""

from fuzzer_tool.core.schedulers.op_bo_gp_ucb import BOGPUCBScheduler
from fuzzer_tool.core.schedulers.op_c2ucb import C2UCBScheduler
from fuzzer_tool.core.schedulers.op_canary import CanaryScheduler
from fuzzer_tool.core.schedulers.op_cmaes import CMAESScheduler
from fuzzer_tool.core.schedulers.op_consolidated import ConsolidatedScheduler
from fuzzer_tool.core.schedulers.op_contextual import ContextualLinUCBScheduler
from fuzzer_tool.core.schedulers.op_corral import CorralScheduler
from fuzzer_tool.core.schedulers.op_cucb import CUCBScheduler
from fuzzer_tool.core.schedulers.op_cusum_ucb import CUSUM_UCBScheduler
from fuzzer_tool.core.schedulers.op_ducb import DUCBScheduler
from fuzzer_tool.core.schedulers.op_epsilon_greedy import EpsilonGreedyScheduler
from fuzzer_tool.core.schedulers.op_exp3 import Exp3Scheduler
from fuzzer_tool.core.schedulers.op_exp4 import Exp4Scheduler
from fuzzer_tool.core.schedulers.op_fewa import FEWAScheduler
from fuzzer_tool.core.schedulers.op_fpl import FPLScheduler
from fuzzer_tool.core.schedulers.op_gp_ucb import GPUCBScheduler
from fuzzer_tool.core.schedulers.op_gradient import GradientBanditScheduler
from fuzzer_tool.core.schedulers.op_hierarchical import HierarchicalBanditScheduler
from fuzzer_tool.core.schedulers.op_kl_ducb import KL_DUCBScheduler
from fuzzer_tool.core.schedulers.op_kl_swucb import KL_SWUCBScheduler
from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler
from fuzzer_tool.core.schedulers.op_mopt import MOptScheduler
from fuzzer_tool.core.schedulers.op_moss import MOSSScheduler
from fuzzer_tool.core.schedulers.op_replicator import ReplicatorScheduler
from fuzzer_tool.core.schedulers.op_round_robin import RoundRobinScheduler
from fuzzer_tool.core.schedulers.op_softmax import SoftmaxScheduler
from fuzzer_tool.core.schedulers.op_successive_elim import SuccessiveEliminationScheduler
from fuzzer_tool.core.schedulers.op_swucb import SWUCBScheduler
from fuzzer_tool.core.schedulers.op_topk import TopKScheduler
from fuzzer_tool.core.schedulers.op_whittle import WhittleIndexScheduler
from fuzzer_tool.core.schedulers.seed_canary import SeedCanaryScheduler
from fuzzer_tool.core.schedulers.seed_kruskal_count import KruskalCountSeedStrategy
from fuzzer_tool.core.schedulers.seed_mcts import AlphaBetaMCTSSeedScheduler, MCTSSeedScheduler
from fuzzer_tool.core.schedulers.seed_round_robin import SeedRoundRobinScheduler
from fuzzer_tool.core.schedulers.seed_tang import TangRecommendationScheduler

__all__ = [
    "CorralScheduler",
    "WhittleIndexScheduler",
    "FPLScheduler",
    "CMAESScheduler",
    "ConsolidatedScheduler",
    "C2UCBScheduler",
    "MonteCarloScheduler",
    "MOptScheduler",
    "MOSSScheduler",
    "ReplicatorScheduler",
    "Exp3Scheduler",
    "Exp4Scheduler",
    "EpsilonGreedyScheduler",
    "GradientBanditScheduler",
    "HierarchicalBanditScheduler",
    "BOGPUCBScheduler",
    "GPUCBScheduler",
    "MCTSSeedScheduler",
    "AlphaBetaMCTSSeedScheduler",
    "ContextualLinUCBScheduler",
    "CUCBScheduler",
    "CUSUM_UCBScheduler",
    "FEWAScheduler",
    "DUCBScheduler",
    "SWUCBScheduler",
    "KL_DUCBScheduler",
    "KL_SWUCBScheduler",
    "RoundRobinScheduler",
    "CanaryScheduler",
    "SuccessiveEliminationScheduler",
    "SoftmaxScheduler",
    "TopKScheduler",
    "TangRecommendationScheduler",
    "SeedCanaryScheduler",
    "SeedRoundRobinScheduler",
    "KruskalCountSeedStrategy",
]
