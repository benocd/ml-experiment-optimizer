# Hybrid Machine Learning for Experimental Optimization

This repository contains a Python notebook demonstrating a **hybrid machine learning framework** for guiding experimental design in biological systems. Specifically, it focuses on optimizing the growth conditions (phosphate and temperature) for a model organism using data-efficient, uncertainty-aware sampling.

## Overview

The workflow combines four main components:

1. **Quadratic Response Surface Model (OLS):**  
   Provides a smooth global estimate of the response surface and a prediction of the maximum growth rate.

2. **Gaussian Process Regression (GPR):**  
   Captures local uncertainty and enables computation of Expected Improvement (EI) across the experimental space.

3. **Expected Improvement (EI):**  
   Used as an acquisition function to identify high-potential experimental conditions.

4. **K-means Clustering:**  
   Ensures **diversity** in suggested experiments by selecting a representative from each cluster among the top EI-ranked candidates.

The system iteratively selects **5 new experimental conditions per cycle**, appends them to a growing dataset, and refits models to continue the search for optimal conditions. This loop can be manually repeated until the desired improvement threshold is met.

## Features

- Latin Hypercube Sampling (initial exploration)
- Quadratic model fitting and contour visualization
- Gaussian Process with Matern kernel (ν=2.5)
- EI-based sampling strategy
- Diversity via clustering (KMeans)
- Fully reproducible via provided notebook
- Automatic result logging and PNG export of each cycle

## Benchmarking

The hybrid framework was benchmarked against standard optimization techniques using the same stopping criteria (0.01% threshold improvement):

| Algorithm      | Cycles         | Growth Rate     | Phosphate       | Temperature     | Accuracy         |
|----------------|----------------|------------------|------------------|------------------|------------------|
| **Hybrid**     | 4.4 ± 0.55     | 1.09 ± 0.03       | 15.7 ± 2.8       | 23.2 ± 0.9       | 2.7 ± 2.2        |
| Bayesian Opt.  | 4.4 ± 0.89     | 1.15 ± 0.01       | 16.4 ± 0.6       | 25.0 ± 0.7       | 3.9 ± 0.7        |
| TPE            | 16.8 ± 4.44    | 1.12 ± 0.04       | 13.8 ± 5.5       | 25.8 ± 1.9       | 5.6 ± 3.6        |

## File Structure

- `OLS Temp and Phosphate-EI.ipynb`: Main notebook containing the full optimization pipeline.
- `CurrentResults.csv`: Accumulated dataset across cycles.
- `DataCube.csv`: Full experimental space and ground-truth values.
- PNG outputs: Saved plots per cycle showing response surfaces and EI heatmaps.

## Dependencies

Tested with:
- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `statsmodels`

## How to Use

1. Clone the repository.
2. Install dependencies (you can use a virtualenv or conda environment).
3. Run the notebook in Jupyter or VS Code.
4. Review suggested experimental points.
5. Append real or simulated measurements to `CurrentResults.csv`.
6. Re-run the notebook for the next cycle.

## Citation

If using this work in a publication, please cite as:

> (in review)
