"""
Multi-Objective Evolutionary Fitness Evaluator and Generational Scaling Tracker.
Implements idea.md Sections 27 and 28:
- F(D) = eta * S_E - beta * C_compute - lambda_D * |D| - delta * L_forgetting
- N_D(n) ~ N_R * e^{-kappa*n}
- kappa_model = lambda_S * E_k * ln(C_R)
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from ..dna.structure import Genotype


class EvolutionaryFitnessEvaluator:
    """
    Computes global multi-objective fitness for evolved genotypes (Section 27).
    """
    def __init__(
        self,
        weight_sample_efficiency: float = 2.0,   # eta
        weight_compute_cost: float = 0.05,       # beta
        weight_dna_size: float = 1e-5,           # lambda_D
        weight_forgetting: float = 1.0,          # delta
    ):
        self.eta = weight_sample_efficiency
        self.beta = weight_compute_cost
        self.lambda_d = weight_dna_size
        self.delta = weight_forgetting

    def compute_fitness(
        self,
        genotype: Genotype,
        sample_efficiency: float = 1.0,
        compute_cost: float = 10.0,
        forgetting_loss: float = 0.0,
    ) -> float:
        """
        F(D) = eta * S_E - beta * C_compute - lambda_D * |D| - delta * L_forgetting
        """
        d_size = float(genotype.total_parameters())

        fitness = (
            self.eta * sample_efficiency
            - self.beta * compute_cost
            - self.lambda_d * d_size
            - self.delta * forgetting_loss
        )

        genotype.fitness_history["overall_fitness"] = fitness
        genotype.fitness_history["sample_efficiency"] = sample_efficiency
        genotype.fitness_history["compute_cost"] = compute_cost
        genotype.fitness_history["forgetting_loss"] = forgetting_loss

        return fitness


class GenerationalScalingTracker:
    """
    Tracks and models generational scaling behavior (Section 28):
    N_D(n) ~ N_R * e^{-kappa*n}

    Where:
    - N_D(n): Training steps needed to reach P* from DNA generation n
    - N_R: Baseline training steps (random initialization)
    - kappa: Transfer efficiency coefficient
    - kappa_model = lambda_S * E_k * ln(C_R)

    Monitors whether successive DNA generations learn faster (exponential improvement).
    """

    def __init__(self, n_baseline: Optional[int] = None):
        self.n_baseline = n_baseline  # N_R: random init baseline steps
        self.generation_data: List[Dict[str, Any]] = []

    def record_generation(
        self,
        generation: int,
        steps_to_target: int,
        sample_efficiency: float,
        retained_energy: float = 0.0,
        compression_ratio: float = 1.0,
    ):
        """
        Records training data for a generation.

        Args:
            generation: Generation number n.
            steps_to_target: N_D(n) — steps needed to reach target performance.
            sample_efficiency: S_E = N_R / N_D(n).
            retained_energy: E_k — mean energy retained.
            compression_ratio: C_R — true compression ratio.
        """
        self.generation_data.append({
            "generation": generation,
            "steps_to_target": steps_to_target,
            "sample_efficiency": sample_efficiency,
            "retained_energy": retained_energy,
            "compression_ratio": compression_ratio,
        })

        if self.n_baseline is None and generation == 0:
            self.n_baseline = steps_to_target

    def fit_exponential_model(self) -> Dict[str, float]:
        """
        Fits N_D(n) ~ N_R * e^{-kappa*n} to observed generation data.

        Uses log-linear regression: ln(N_D) = ln(N_R) - kappa*n

        Returns:
            Dict with fitted parameters: n_baseline, kappa, r_squared
        """
        if len(self.generation_data) < 2:
            return {"n_baseline": float(self.n_baseline or 0), "kappa": 0.0, "r_squared": 0.0}

        n_vals = [d["generation"] for d in self.generation_data]
        y_vals = [math.log(max(d["steps_to_target"], 1)) for d in self.generation_data]

        # Simple linear regression on log-transformed data
        n = len(n_vals)
        sum_x = sum(n_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(n_vals, y_vals))
        sum_x2 = sum(x * x for x in n_vals)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-12:
            return {"n_baseline": float(self.n_baseline or 0), "kappa": 0.0, "r_squared": 0.0}

        slope = (n * sum_xy - sum_x * sum_y) / denom  # -kappa
        intercept = (sum_y - slope * sum_x) / n  # ln(N_R)

        kappa = -slope
        fitted_n_baseline = math.exp(intercept)

        # R-squared
        mean_y = sum_y / n
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(n_vals, y_vals))
        ss_tot = sum((y - mean_y) ** 2 for y in y_vals) + 1e-12
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)

        return {
            "n_baseline": fitted_n_baseline,
            "kappa": kappa,
            "r_squared": r_squared,
        }

    def compute_transfer_coefficient(
        self,
        lambda_s: float = 1.0,
    ) -> Dict[str, float]:
        """
        Computes theoretical transfer coefficient (Section 28):
        kappa_model = lambda_S * E_k * ln(C_R)

        Uses mean E_k and C_R across all generations.
        Returns fitted kappa and theoretical kappa_model for comparison.
        """
        fitted = self.fit_exponential_model()

        if not self.generation_data:
            return {**fitted, "kappa_model": 0.0, "kappa_ratio": 0.0}

        mean_ek = sum(d["retained_energy"] for d in self.generation_data) / len(self.generation_data)
        mean_cr = sum(d["compression_ratio"] for d in self.generation_data) / len(self.generation_data)

        kappa_model = lambda_s * mean_ek * math.log(max(mean_cr, 1.01))

        kappa_ratio = 0.0
        if abs(kappa_model) > 1e-9:
            kappa_ratio = fitted["kappa"] / kappa_model

        return {
            **fitted,
            "kappa_model": kappa_model,
            "kappa_ratio": kappa_ratio,
            "mean_retained_energy": mean_ek,
            "mean_compression_ratio": mean_cr,
        }

    def predict_steps(self, generation: int) -> Optional[float]:
        """Predicts N_D(n) for a given generation using the fitted model."""
        fitted = self.fit_exponential_model()
        if fitted["kappa"] <= 0 or fitted["n_baseline"] <= 0:
            return None
        return fitted["n_baseline"] * math.exp(-fitted["kappa"] * generation)

    def get_improvement_trend(self) -> List[float]:
        """Returns S_E improvement per generation: Delta_S_E(n) = S_E(n+1) - S_E(n)."""
        if len(self.generation_data) < 2:
            return []
        se_values = [d["sample_efficiency"] for d in self.generation_data]
        return [se_values[i+1] - se_values[i] for i in range(len(se_values) - 1)]


class MapElitesArchive:
    """
    Multi-dimensional Quality-Diversity (QD) MAP-Elites Evolutionary Archive.
    Maintains a 2D behavioral niche grid (e.g., Reasoning Capability vs Parameter Compression Ratio).
    Preserves diverse elite genotypes across behavioral dimensions to prevent evolutionary collapse.
    """
    def __init__(
        self,
        dim_x_bins: int = 10,
        dim_y_bins: int = 10,
        x_range: Tuple[float, float] = (0.0, 1.0),   # e.g. Reasoning / Task Accuracy
        y_range: Tuple[float, float] = (1.0, 10.0),  # e.g. Compression Ratio C_R
    ):
        self.dim_x_bins = dim_x_bins
        self.dim_y_bins = dim_y_bins
        self.x_range = x_range
        self.y_range = y_range
        self.grid: Dict[Tuple[int, int], Tuple[Genotype, float]] = {}

    def _get_cell_coords(self, behavior_x: float, behavior_y: float) -> Tuple[int, int]:
        """Maps continuous behavior coordinates into discrete grid cell indices."""
        norm_x = (behavior_x - self.x_range[0]) / max(1e-6, self.x_range[1] - self.x_range[0])
        norm_y = (behavior_y - self.y_range[0]) / max(1e-6, self.y_range[1] - self.y_range[0])
        cell_x = int(math.floor(norm_x * self.dim_x_bins))
        cell_y = int(math.floor(norm_y * self.dim_y_bins))
        cell_x = max(0, min(self.dim_x_bins - 1, cell_x))
        cell_y = max(0, min(self.dim_y_bins - 1, cell_y))
        return (cell_x, cell_y)

    def add_or_replace(self, genotype: Genotype, fitness: float, behavior_x: float, behavior_y: float) -> bool:
        """
        Attempts to insert genotype into its behavioral niche cell.
        Replaces occupant only if candidate fitness is superior.
        Returns True if genotype was added/replaced as new cell elite.
        """
        cell = self._get_cell_coords(behavior_x, behavior_y)
        if cell not in self.grid or fitness > self.grid[cell][1]:
            self.grid[cell] = (genotype, fitness)
            return True
        return False

    def get_elites(self) -> List[Genotype]:
        """Returns all current elite genotypes stored across the behavioral grid."""
        return [g for g, _ in self.grid.values()]

    def coverage(self) -> float:
        """Returns archive niche coverage percentage in [0.0, 1.0]."""
        total_cells = self.dim_x_bins * self.dim_y_bins
        return len(self.grid) / max(1, total_cells)

