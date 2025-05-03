# ParetoOptimization

This repository contains Python scripts for generating Latin Hypercube Sampling (LHS), building surrogate models using SMT (Surrogate Modeling Toolbox), and performing multi-objective optimization using Pareto front analysis.

## Features

- 🔁 **Latin Hypercube Sampling**: Generates 50 LHS samples for 3 variables `X1`, `X2`, `X3` with specified bounds.
- 🧠 **Surrogate Modeling (SMT)**: KRG models for predicting STH and Height based on simulation data.
- 🔍 **Pareto Front Optimization**: Visualizes and exports non-dominated solutions (STH & Height) using `pymoo` (NSGA-II).
- 📈 **Visualization**: Includes 2D plots Pareto front, 3D scatter of input space.

## Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

Ensure [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) is installed to build SMT models.

## Files

- `lhs_generator.py`: Generate and export LHS samples.
- `nsga2_krg.py`: Perform NSGA-II optimization using KRG surrogate model.
- `lhs_samples_saved.xlsx`: 50 simulation samples.
- `pareto_optimal_nsga2_krg.xlsx`: Optimized Pareto front results.
- `requirements.txt`: List of Python packages.

## Example Output

- 📊 Pareto Front with simulation data  
- 🤖 Surrogate model-based Pareto Front with NSGA-II  
- 📁 Exported CSV/Excel of optimal input variables

## Citations

Some techniques or code snippets were referenced from:

- SMT Library: https://smt.readthedocs.io/en/latest/
- OAPackage Pareto Example: https://oapackage.readthedocs.io/en/latest/examples/example_pareto.html
- LHS Sampling (Scipy): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html
- NSGA-II with pymoo: https://medium.com/analytics-vidhya/optimization-modelling-in-python-multiple-objectives-760b9f1f26ee
