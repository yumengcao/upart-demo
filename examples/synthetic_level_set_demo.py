"""Run a synthetic uncertainty-guided level-set estimation demo.

Execute from the repository root:

    python examples/synthetic_level_set_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gp_model import fit_gp_model, predict_gp
from src.partition import Region
from src.sampling import (
    initial_design,
    select_next_sample,
    select_region_for_refinement,
)
from src.synthetic_function import make_prediction_grid, synthetic_level_set_function
from src.visualization import plot_iteration


def run_demo(
    n_iterations: int = 6,
    threshold: float = 0.15,
    grid_resolution: int = 90,
    random_state: int = 13,
) -> None:
    """Run the synthetic adaptive sampling demo and save iteration figures.

    Parameters
    ----------
    n_iterations:
        Number of adaptive sampling iterations.
    threshold:
        Level-set threshold used to define the boundary of interest.
    grid_resolution:
        Number of prediction grid points per dimension.
    random_state:
        Seed used for the initial design and GP optimizer restarts.
    """
    figures_dir = PROJECT_ROOT / "figures"
    grid_x, grid_y, candidates = make_prediction_grid(resolution=grid_resolution)
    true_values = synthetic_level_set_function(candidates).reshape(grid_x.shape)

    sample_points = initial_design(n_points=10, random_state=random_state)
    sample_values = synthetic_level_set_function(sample_points)

    active_regions = [
        Region(bounds=((0.0, 1.0), (0.0, 1.0)), region_id="R", depth=0)
    ]
    max_partition_depth = 3

    for iteration in range(1, n_iterations + 1):
        model = fit_gp_model(
            sample_points=sample_points,
            sample_values=sample_values,
            random_state=random_state + iteration,
        )
        posterior_mean, predictive_std = predict_gp(model, candidates)

        next_point, scores = select_next_sample(
            candidates=candidates,
            posterior_mean=posterior_mean,
            predictive_std=predictive_std,
            threshold=threshold,
            existing_points=sample_points,
            alpha=0.75,
        )
        selected_region = select_region_for_refinement(active_regions, candidates, scores)

        # Tie adaptive sampling to the selected partition region when possible.
        next_point, scores = select_next_sample(
            candidates=candidates,
            posterior_mean=posterior_mean,
            predictive_std=predictive_std,
            threshold=threshold,
            existing_points=sample_points,
            region=selected_region,
            alpha=0.75,
        )

        output_path = figures_dir / f"iteration_{iteration}.png"
        plot_iteration(
            grid_x=grid_x,
            grid_y=grid_y,
            true_values=true_values,
            posterior_mean=posterior_mean.reshape(grid_x.shape),
            predictive_std=predictive_std.reshape(grid_x.shape),
            scores=scores.reshape(grid_x.shape),
            threshold=threshold,
            sample_points=sample_points,
            next_point=next_point,
            regions=active_regions,
            selected_region=selected_region,
            output_path=output_path,
            iteration=iteration,
        )

        sample_points = np.vstack([sample_points, next_point])
        sample_values = synthetic_level_set_function(sample_points)

        if selected_region.depth < max_partition_depth:
            active_regions = [
                region for region in active_regions if region.region_id != selected_region.region_id
            ]
            active_regions.extend(selected_region.split())

        print(
            f"Iteration {iteration}: selected x = "
            f"({next_point[0]:.3f}, {next_point[1]:.3f}); saved {output_path}"
        )


if __name__ == "__main__":
    run_demo()
