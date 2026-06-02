"""Adaptive sampling helpers for the synthetic level-set demo."""

from __future__ import annotations

import numpy as np

from .partition import Region
from .scoring import compute_level_set_score, score_regions


def initial_design(n_points: int = 10, random_state: int = 13) -> np.ndarray:
    """Create a small reproducible initial design in ``[0, 1]^2``.

    Parameters
    ----------
    n_points:
        Number of initial sample points.
    random_state:
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        Initial sample locations with shape ``(n_points, 2)``.
    """
    if n_points < 4:
        raise ValueError("Use at least four initial points for this demo.")

    rng = np.random.default_rng(random_state)
    corners = np.array(
        [
            [0.05, 0.05],
            [0.05, 0.95],
            [0.95, 0.05],
            [0.95, 0.95],
        ]
    )
    random_points = rng.uniform(0.05, 0.95, size=(n_points - len(corners), 2))
    return np.vstack([corners, random_points])


def select_region_for_refinement(
    regions: list[Region],
    candidates: np.ndarray,
    scores: np.ndarray,
) -> Region:
    """Select the active region with the highest uncertainty-guided score."""
    region_scores = score_regions(regions, candidates, scores)
    best_region_id = max(region_scores, key=region_scores.get)
    return next(region for region in regions if region.region_id == best_region_id)


def select_next_sample(
    candidates: np.ndarray,
    posterior_mean: np.ndarray,
    predictive_std: np.ndarray,
    threshold: float,
    existing_points: np.ndarray,
    region: Region | None = None,
    alpha: float = 0.75,
    min_distance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the next adaptive sample point from candidate scores.

    Parameters
    ----------
    candidates:
        Candidate points with shape ``(n_candidates, 2)``.
    posterior_mean:
        GP posterior mean at the candidate points.
    predictive_std:
        GP predictive standard deviation at the candidate points.
    threshold:
        Level-set threshold of interest.
    existing_points:
        Previously sampled points.
    region:
        Optional active region used to restrict the next sample.
    alpha:
        Penalty weight for distance from the estimated threshold boundary.
    min_distance:
        Minimum Euclidean distance from previously sampled points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The selected next point and the full candidate score array.
    """
    scores = compute_level_set_score(
        posterior_mean=posterior_mean,
        predictive_std=predictive_std,
        threshold=threshold,
        alpha=alpha,
    )

    valid_mask = np.ones(len(candidates), dtype=bool)
    if region is not None:
        valid_mask &= region.contains(candidates)

    ranked_indices = np.argsort(scores)[::-1]
    for index in ranked_indices:
        if not valid_mask[index]:
            continue
        distances = np.linalg.norm(existing_points - candidates[index], axis=1)
        if np.all(distances > min_distance):
            return candidates[index], scores

    fallback_index = ranked_indices[0]
    return candidates[fallback_index], scores
