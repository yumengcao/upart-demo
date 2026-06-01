# UPART Demo: Uncertainty-Guided Partitioning for Level-Set Estimation

This repository contains a simplified Python demo of an uncertainty-guided partitioning workflow for level-set estimation.

The demo uses a synthetic test function and is intended to illustrate the main algorithmic ideas behind uncertainty-guided refinement:

- Gaussian Process surrogate modeling
- uncertainty-based scoring
- adaptive sampling
- hierarchical region partitioning
- iteration-level visualization

This repository does **not** include unpublished research code, project-specific data, or full experimental results from the NSF-supported UPART project.

## Research Context

UPART is an uncertainty-guided partitioning framework for level-set estimation. The goal is to classify regions of an input space relative to a target threshold while using model uncertainty to guide refinement and sampling decisions.

This simplified demo is designed for portfolio and educational purposes.

## Installation

```bash
git clone https://github.com/yumengcao/upart-demo.git
cd upart-demo
pip install -r requirements.txt
