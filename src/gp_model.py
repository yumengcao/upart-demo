"""Gaussian Process surrogate model utilities."""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


def fit_gp_model(
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    random_state: int = 0,
) -> GaussianProcessRegressor:
    """Fit a Gaussian Process surrogate model to observed function values.

    Parameters
    ----------
    sample_points:
        Training inputs with shape ``(n_samples, 2)``.
    sample_values:
        Observed function values with shape ``(n_samples,)``.
    random_state:
        Seed used by the optimizer restarts inside scikit-learn.

    Returns
    -------
    GaussianProcessRegressor
        A fitted Gaussian Process model.
    """
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
        * RBF(length_scale=[0.25, 0.25], length_scale_bounds=(0.03, 2.0))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=random_state,
    )
    model.fit(np.asarray(sample_points, dtype=float), np.asarray(sample_values, dtype=float))
    return model


def predict_gp(
    model: GaussianProcessRegressor,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict posterior mean and standard deviation at candidate points.

    Parameters
    ----------
    model:
        Fitted Gaussian Process model.
    candidates:
        Candidate points with shape ``(n_candidates, 2)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Posterior mean and predictive standard deviation.
    """
    posterior_mean, predictive_std = model.predict(
        np.asarray(candidates, dtype=float),
        return_std=True,
    )
    return posterior_mean, predictive_std
