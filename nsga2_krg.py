import pandas as pd
import numpy as np
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# Đọc dữ liệu
file_path = "lhs_samples_saved.xlsx"
df = pd.read_excel(file_path)
X = df[['X1', 'X2', 'X3']].values
Y = df[['STH', 'Height']].values

# Chuẩn hóa output
scaler_sth = MinMaxScaler()
scaler_height = MinMaxScaler()

Y_sth_scaled = scaler_sth.fit_transform(Y[:, 0].reshape(-1, 1)).flatten()
Y_height_scaled = scaler_height.fit_transform(Y[:, 1].reshape(-1, 1)).flatten()

# Huấn luyện surrogate model (KRG)
model_sth = KRG(print_global=False)
model_sth.set_training_values(X, Y_sth_scaled)
model_sth.train()

model_height = KRG(print_global=False)
model_height.set_training_values(X, Y_height_scaled)
model_height.train()

# Định nghĩa bài toán tối ưu hóa với pymoo
class SurrogateProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=np.array([10, 0.3, 40]), xu=np.array([15, 0.9, 60]))

    def _evaluate(self, X, out, *args, **kwargs):
        sth_pred_scaled = model_sth.predict_values(X).flatten()
        height_pred_scaled = model_height.predict_values(X).flatten()

        # Trả về giá trị thật
        sth_pred = scaler_sth.inverse_transform(sth_pred_scaled.reshape(-1, 1)).flatten()
        height_pred = scaler_height.inverse_transform(height_pred_scaled.reshape(-1, 1)).flatten()

        # Tối đa hoá cả hai mục tiêu => pymoo mặc định là minimize => negate
        out["F"] = -np.column_stack([sth_pred, height_pred])

# Khởi tạo thuật toán NSGA-II
algorithm = NSGA2(
    pop_size=200,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)

termination = get_termination("n_gen", 200)

# Tối ưu
res = minimize(
    SurrogateProblem(),
    algorithm,
    termination,
    seed=42,
    save_history=True,
    verbose=True,
)

# Giá trị thật của Pareto Front
F_actual = -res.F

# Vẽ Pareto front
plt.figure(figsize=(8, 6))
plt.scatter(F_actual[:, 0], F_actual[:, 1], c='red', label='Pareto Optimal')
plt.xlabel("STH (mm)")
plt.ylabel("Height (mm)")
plt.title("Pareto Front from NSGA-II with Kriging Surrogate Model (Maximization)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Lấy các điểm input tối ưu
X_optimal = res.X

# Dự đoán lại output thực
STH_scaled = model_sth.predict_values(X_optimal).flatten()
Height_scaled = model_height.predict_values(X_optimal).flatten()

STH_optimal = scaler_sth.inverse_transform(STH_scaled.reshape(-1, 1)).flatten()
Height_optimal = scaler_height.inverse_transform(Height_scaled.reshape(-1, 1)).flatten()

# Xuất ra Excel
df_optimal = pd.DataFrame(X_optimal, columns=["X1", "X2", "X3"])
df_optimal["STH (mm)"] = STH_optimal
df_optimal["Height (mm)"] = Height_optimal
df_optimal.to_excel("pareto_optimal_nsga2_krg.xlsx", index=False)
print("✅ Pareto optimal results have been saved to 'pareto_optimal_nsga2_krg.xlsx'")
