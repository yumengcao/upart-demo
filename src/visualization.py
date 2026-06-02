"""Visualization helpers for the synthetic level-set demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np

from .partition import Region


def plot_iteration(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    true_values: np.ndarray,
    posterior_mean: np.ndarray,
    predictive_std: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sample_points: np.ndarray,
    next_point: np.ndarray,
    regions: list[Region],
    selected_region: Region,
    output_path: Path,
    iteration: int,
) -> None:
    """Save a four-panel visualization for one adaptive sampling iteration.

    Parameters
    ----------
    grid_x:
        Meshgrid x-coordinates.
    grid_y:
        Meshgrid y-coordinates.
    true_values:
        True synthetic function values on the grid.
    posterior_mean:
        GP posterior mean on the grid.
    predictive_std:
        GP predictive standard deviation on the grid.
    scores:
        Acquisition scores on the grid.
    threshold:
        Level-set threshold of interest.
    sample_points:
        Current observed sample locations.
    next_point:
        Next point selected by the adaptive sampling rule.
    regions:
        Active rectangular regions.
    selected_region:
        Region selected for refinement in the current iteration.
    output_path:
        Path where the figure should be saved.
    iteration:
        One-based iteration number used in the figure title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], true_values, "True synthetic function", "viridis"),
        (axes[0, 1], posterior_mean, "GP posterior mean", "viridis"),
        (axes[1, 0], predictive_std, "GP predictive uncertainty", "magma"),
        (axes[1, 1], scores, "Uncertainty-guided score", "cividis"),
    ]

    for ax, values, title, cmap in panels:
        contour = ax.contourf(grid_x, grid_y, values, levels=28, cmap=cmap)
        fig.colorbar(contour, ax=ax, shrink=0.86)
        _draw_level_set(ax, grid_x, grid_y, values, threshold)
        _draw_regions(ax, regions, selected_region)
        _draw_samples(ax, sample_points, next_point)
        _format_axis(ax, title)

    fig.suptitle(
        f"UPART Demo: iteration {iteration}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_level_set(
    ax: Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> None:
    """Draw the threshold contour if it is present in the plotted range."""
    if np.min(values) <= threshold <= np.max(values):
        ax.contour(
            grid_x,
            grid_y,
            values,
            levels=[threshold],
            colors="white",
            linewidths=2.0,
        )
        ax.contour(
            grid_x,
            grid_y,
            values,
            levels=[threshold],
            colors="black",
            linewidths=0.8,
            linestyles="--",
        )


def _draw_samples(ax: Axes, sample_points: np.ndarray, next_point: np.ndarray) -> None:
    """Draw observed samples and the next selected sample."""
    ax.scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        s=32,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        label="Observed samples",
    )
    ax.scatter(
        next_point[0],
        next_point[1],
        s=125,
        marker="*",
        c="#ffcf33",
        edgecolors="black",
        linewidths=0.9,
        label="Next sample",
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def _draw_regions(
    ax: Axes,
    regions: list[Region],
    selected_region: Region,
) -> None:
    """Draw active region boundaries and emphasize the selected region."""
    for region in regions:
        (x_min, x_max), (y_min, y_max) = region.bounds
        edge_color = "#ffcf33" if region.region_id == selected_region.region_id else "white"
        line_width = 2.0 if region.region_id == selected_region.region_id else 0.9
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=0.85,
            )
        )


def _format_axis(ax: Axes, title: str) -> None:
    """Apply consistent axis formatting for demo plots."""
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
