# ParetoOptimization

This repository contains Python scripts for generating Latin Hypercube Sampling (LHS), building surrogate models using SMT (Surrogate Modeling Toolbox), and performing multi-objective optimization using NSGA-II and Pareto front analysis.

## Features

- 🧪 **Latin Hypercube Sampling**: Generate 50 sample points in the input design space (X1, X2, X3).
- 🧠 **Surrogate Modeling (SMT)**: Use Kriging (KRG) models for predicting `STH` and `Height`.
- 🔁 **Leave-One-Out Cross-Validation (LOOCV)**: Assess training accuracy of surrogate models and plot the historical error trend.
- 🧬 **NSGA-II Optimization**: Perform multi-objective optimization to maximize both STH and Height.
- 📊 **Visualization**: Save plots for LOOCV error trend and the resulting Pareto front.

## Structure

```
ParetoOptimization/
├── lhs_samples_saved.xlsx              # LHS design and simulation results
├── pareto_optimal_nsga2_krg.xlsx       # Optimal input/output solutions from NSGA-II
├── Figure/
│   ├── krg_loocv_errors.png            # Plot of LOOCV RMSE over training samples
│   └── pareto_front.png                # Plot of Pareto front for the surrogate-based optimization
├── surrogate_nsga2_krg_loocv.py        # Main script: KRG model + LOOCV + NSGA-II optimization
├── requirements.txt                    # Python dependencies
```

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the main script:
```bash
python surrogate_nsga2_krg_loocv.py
```

## Input/Output
- **Input**: `lhs_samples_saved.xlsx` with columns `X1`, `X2`, `X3`, `STH`, `Height`
- **Output**:
  - `pareto_optimal_nsga2_krg.xlsx`: Contains the Pareto-optimal solutions
  - `Figure/*.png`: Visual representations of surrogate model training and optimization front

## References
- NSGA-II in Pymoo: https://medium.com/analytics-vidhya/optimization-modelling-in-python-multiple-objectives-760b9f1f26ee
- SMT Surrogate Models: https://smt.readthedocs.io/en/latest/
- Scipy LHS sampling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html
- OAPackage Pareto: https://oapackage.readthedocs.io/en/latest/examples/example_pareto.html

---

🔬 *This project supports the research on "Effect of Selected Process Parameters in CDA-110 Copper T-Shaped Tube Hydroforming".*
