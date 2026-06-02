# UPART Demo: Uncertainty-Guided Partitioning for Level-Set Estimation

## Overview

This repository contains a simplified public demo of uncertainty-guided partitioning for level-set estimation. The demo uses a two-dimensional synthetic function on `[0, 1]^2` and adaptively samples points near an estimated threshold boundary using a Gaussian Process surrogate model.

The goal is to provide a clean, portfolio-quality example of algorithm design, Gaussian Process modeling, adaptive sampling, uncertainty scoring, simple hierarchical partitioning, and iteration-level visualization.

This repository does **not** include unpublished UPART research code, project-specific data, private datasets, or full experimental results. It is intentionally lightweight and uses only synthetic toy examples suitable for public release.

## Research Context

Level-set estimation asks where an unknown function is above or below a target threshold. In expensive black-box settings, the function may only be evaluated at a limited number of input locations, so sampling decisions should focus on points that are both informative and relevant to the threshold boundary.

This demo illustrates a simplified version of that idea:

- Fit a Gaussian Process surrogate to observed samples.
- Estimate predictive mean and uncertainty over a dense candidate grid.
- Score candidate points by combining uncertainty with closeness to the level-set threshold.
- Refine simple rectangular regions around high-scoring areas.
- Visualize the true function, GP posterior, uncertainty, samples, and estimated boundary at each iteration.

The partitioning logic shown here is a public-facing teaching example and does not reproduce a full research implementation.

## Features

- Synthetic 2D benchmark function with a configurable level-set threshold.
- Gaussian Process surrogate modeling with `scikit-learn`.
- Interpretable uncertainty-guided acquisition score.
- Adaptive sampling near uncertain level-set boundaries.
- Simple hierarchical region partitioning with axis-aligned rectangles.
- Iteration-level visualizations saved to `figures/`.
- Modular source files with readable docstrings and comments.

## Project Structure

```text
upart-demo/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- synthetic_function.py
|   |-- gp_model.py
|   |-- partition.py
|   |-- sampling.py
|   |-- scoring.py
|   `-- visualization.py
|-- examples/
|   `-- synthetic_level_set_demo.py
`-- figures/
    `-- .gitkeep
```

## Installation

```bash
git clone https://github.com/yumengcao/upart-demo.git
cd upart-demo
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Demo

Run the synthetic level-set estimation demo from the repository root:

```bash
python examples/synthetic_level_set_demo.py
```

The script will:

- Generate initial samples from a synthetic 2D function.
- Fit a Gaussian Process surrogate model.
- Score candidate points using uncertainty and threshold proximity.
- Refine rectangular regions around high-scoring areas.
- Save iteration figures to `figures/`.

## Example Outputs

Each iteration figure includes:

- The true synthetic function and true level-set boundary.
- The GP posterior mean and estimated level-set boundary.
- The GP posterior predictive uncertainty.
- The acquisition score used for adaptive sampling.
- Current sampled points, the next selected sample, and the active region partition.

Generated figures are saved as:

```text
figures/iteration_1.png
figures/iteration_2.png
...
```

The generated image files are ignored by Git to keep the public repository small and reproducible.

## Author

Yumeng Cao

This repository is intended as a public demo of research engineering skills in Gaussian Process modeling, uncertainty-aware adaptive sampling, partitioning logic, and scientific visualization.
