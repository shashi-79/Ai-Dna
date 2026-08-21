"""
Validation Experiment Suites (Phases 1 to 6).
"""

from .exp1_svd_hypothesis import run_experiment_1
from .exp2_transfer_curve import run_experiment_2
from .exp3_cppn_encoding import run_experiment_3
from .exp4_regeneration import run_experiment_4
from .exp5_multigen_evolution import run_experiment_5
from .exp6_multi_parent_fusion import run_experiment_6
from .exp7_official_benchmarks import run_experiment_7_official_benchmarks

__all__ = [
    "run_experiment_1",
    "run_experiment_2",
    "run_experiment_3",
    "run_experiment_4",
    "run_experiment_5",
    "run_experiment_6",
    "run_experiment_7_official_benchmarks",
]
