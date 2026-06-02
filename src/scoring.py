"""Uncertainty-guided scoring utilities for level-set estimation."""

from __future__ import annotations

import numpy as np

from .partition import Region


def compute_level_set_score(
    posterior_mean: np.ndarray,
    predictive_std: np.ndarray,
    threshold: float,
    alpha: float = 0.75,
    normalize: bool = True,
) -> np.ndarray:
    """Score candidate points by uncertainty and closeness to the threshold.

    The unnormalized score follows the interpretable form
    ``predictive_std - alpha * abs(posterior_mean - threshold)``. By default,
    the two components are scaled to ``[0, 1]`` first so the tradeoff remains
    stable across iterations.

    Parameters
    ----------
    posterior_mean:
        GP posterior mean at candidate points.
    predictive_std:
        GP predictive standard deviation at candidate points.
    threshold:
        Level-set threshold of interest.
    alpha:
        Penalty weight for distance from the estimated level-set boundary.
    normalize:
        Whether to scale uncertainty and threshold distance before scoring.

    Returns
    -------
    np.ndarray
        Acquisition scores with higher values indicating preferred samples.
    """
    posterior_mean = np.asarray(posterior_mean, dtype=float)
    predictive_std = np.asarray(predictive_std, dtype=float)
    distance_to_threshold = np.abs(posterior_mean - threshold)

    if not normalize:
        return predictive_std - alpha * distance_to_threshold

    uncertainty_component = _scale_to_unit_interval(predictive_std)
    distance_component = _scale_to_unit_interval(distance_to_threshold)
    return uncertainty_component - alpha * distance_component


def score_regions(
    regions: list[Region],
    candidates: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute one aggregate score per active region.

    The region score is the maximum candidate score inside that region. This
    keeps the demo simple while still tying partition refinement to the same
    uncertainty-guided acquisition logic used for sampling.
    """
    region_scores: dict[str, float] = {}
    for region in regions:
        mask = region.contains(candidates)
        region_scores[region.region_id] = float(np.max(scores[mask])) if np.any(mask) else -np.inf
    return region_scores


def _scale_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Scale numeric values to ``[0, 1]`` with a zero-safe denominator."""
    values = np.asarray(values, dtype=float)
    value_min = np.min(values)
    value_range = np.max(values) - value_min
    if value_range < 1e-12:
        return np.zeros_like(values)
    return (values - value_min) / value_range
