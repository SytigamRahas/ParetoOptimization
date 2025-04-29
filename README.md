# Pareto Optimization of Hydroforming Parameters

This repository contains scripts and data related to the multi-objective optimization of T-tube hydroforming process using Latin Hypercube Sampling (LHS), finite element simulations, and Pareto Front analysis.

## 📁 Project Structure

- `lhs_generator.py` – Script for generating LHS samples for 3 input variables (X1, X2, X3)
- `test_pareto.py` – Pareto front analysis using `oapackage`
- `lhs_samples.csv` / `lhs_samples_saved.xlsx` – LHS input data
- `pareto_optimal_results.csv` – Output results that are Pareto-optimal
- `Figure_*.png` – Visualization outputs (3D scatter, Pareto front, etc.)

## 🔧 Tools & Libraries

- Python 3.13+
- `numpy`, `matplotlib`, `pandas`
- `scipy.stats.qmc` for LHS
- `oapackage` for Pareto front filtering

## 🎯 Objectives

The goal is to maximize two key performance outputs:
- **STH**: Minimum wall thickness
- **Height**: Branch height in hydroformed T-tube

Given three process variables:
- X1: Axial Displacement
- X2: Pressure Amplification Factor
- X3: Internal Pressure

## 📊 Methodology

1. Generate 50 LHS samples over given bounds.
2. Run FEM simulations (external) to obtain STH and Height.
3. Apply Pareto optimization to identify non-dominated solutions.
4. Visualize results and extract optimal designs.

## 📚 References

- [1] OAPackage Pareto Front Example: https://oapackage.readthedocs.io/en/latest/examples/example_pareto.html  
- [2] Scipy Latin Hypercube Sampling: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html  

## 🛡 License

This repository is under development and currently **private** for research purposes. Please contact the author before redistribution.

---

🧑‍💻 Author: [Your Name]  
📧 Contact: [your.email@example.com]
