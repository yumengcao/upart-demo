"""Synthetic benchmark functions for the public UPART demo."""

from __future__ import annotations

import numpy as np


def synthetic_level_set_function(points: np.ndarray) -> np.ndarray:
    """Evaluate a smooth two-dimensional synthetic function on ``[0, 1]^2``.

    Parameters
    ----------
    points:
        Array with shape ``(n_points, 2)`` or a single point with shape ``(2,)``.

    Returns
    -------
    np.ndarray
        Function values with shape ``(n_points,)``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 2:
        raise ValueError("Expected points with shape (n_points, 2).")

    x = points[:, 0]
    y = points[:, 1]

    return (
        0.65 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        + 0.25 * np.sin(4.0 * np.pi * (x + 0.2 * y))
        + 0.15 * np.cos(3.0 * np.pi * y)
    )


def make_prediction_grid(resolution: int = 90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a regular two-dimensional prediction grid on ``[0, 1]^2``.

    Parameters
    ----------
    resolution:
        Number of grid points per dimension.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Meshgrid arrays ``grid_x`` and ``grid_y``, plus flattened candidate
        points with shape ``(resolution * resolution, 2)``.
    """
    axis = np.linspace(0.0, 1.0, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)
    candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return grid_x, grid_y, candidates
