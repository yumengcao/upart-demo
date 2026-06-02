"""Simple axis-aligned region partitioning for the public demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

Bounds = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangular region in a two-dimensional input space.

    Parameters
    ----------
    bounds:
        Region bounds as ``((x_min, x_max), (y_min, y_max))``.
    region_id:
        Human-readable identifier for the region.
    parent_id:
        Optional identifier of the parent region.
    depth:
        Depth in the partition tree, where the root region has depth 0.
    """

    bounds: Bounds
    region_id: str
    parent_id: Optional[str] = None
    depth: int = 0

    def center(self) -> np.ndarray:
        """Return the geometric center of the region."""
        (x_min, x_max), (y_min, y_max) = self.bounds
        return np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])

    def contains(self, points: np.ndarray) -> np.ndarray | bool:
        """Check whether one or more points lie inside the region.

        Parameters
        ----------
        points:
            Array with shape ``(n_points, 2)`` or one point with shape ``(2,)``.

        Returns
        -------
        np.ndarray | bool
            Boolean mask for multiple points, or a single boolean for one point.
        """
        points = np.asarray(points, dtype=float)
        is_single_point = points.ndim == 1
        if is_single_point:
            points = points.reshape(1, -1)
        if points.shape[1] != 2:
            raise ValueError("Expected points with shape (n_points, 2).")

        (x_min, x_max), (y_min, y_max) = self.bounds
        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
        )
        return bool(mask[0]) if is_single_point else mask

    def split(self) -> list["Region"]:
        """Split the region into four equal rectangular child regions."""
        (x_min, x_max), (y_min, y_max) = self.bounds
        x_mid, y_mid = self.center()
        child_depth = self.depth + 1

        child_bounds: list[Bounds] = [
            ((x_min, x_mid), (y_min, y_mid)),
            ((x_mid, x_max), (y_min, y_mid)),
            ((x_min, x_mid), (y_mid, y_max)),
            ((x_mid, x_max), (y_mid, y_max)),
        ]

        return [
            Region(
                bounds=bounds,
                region_id=f"{self.region_id}.{index}",
                parent_id=self.region_id,
                depth=child_depth,
            )
            for index, bounds in enumerate(child_bounds)
        ]
