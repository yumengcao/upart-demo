"""Public demo utilities for uncertainty-guided level-set estimation."""

from .gp_model import fit_gp_model, predict_gp
from .partition import Region
from .sampling import initial_design, select_next_sample, select_region_for_refinement
from .scoring import compute_level_set_score
from .synthetic_function import make_prediction_grid, synthetic_level_set_function

__all__ = [
    "Region",
    "compute_level_set_score",
    "fit_gp_model",
    "initial_design",
    "make_prediction_grid",
    "predict_gp",
    "select_next_sample",
    "select_region_for_refinement",
    "synthetic_level_set_function",
]
