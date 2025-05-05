import os
import pandas as pd
import numpy as np
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# Ensure output directory exists
os.makedirs("Figure", exist_ok=True)

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

# Hàm tính LOOCV error with theta0
def compute_loocv_errors(X, Y_scaled, scaler, model_type=KRG):
    n_samples = len(X)
    errors = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Tạo tập train và test
        train_indices = np.arange(n_samples) != i
        X_train, Y_train = X[train_indices], Y_scaled[train_indices]
        X_test, Y_test = X[i].reshape(1, -1), Y_scaled[i].reshape(1, -1)

        # Khởi tạo model với theta0
        model = model_type(print_global=False, theta0=[1e-2]*X.shape[1])
        model.set_training_values(X_train, Y_train)
        model.train()

        Y_pred_scaled = model.predict_values(X_test).flatten()
        Y_pred_actual = scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
        Y_test_actual = scaler.inverse_transform(Y_test).flatten()

        errors[i] = np.sqrt(mean_squared_error(Y_test_actual, Y_pred_actual))
    return errors

# Tính lỗi LOOCV
sth_loocv_errors = compute_loocv_errors(X, Y_sth_scaled, scaler_sth)
height_loocv_errors = compute_loocv_errors(X, Y_height_scaled, scaler_height)

# Vẽ đồ thị lỗi LOOCV
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(sth_loocv_errors) + 1), sth_loocv_errors, label='STH LOOCV RMSE', color='blue')
plt.plot(range(1, len(height_loocv_errors) + 1), height_loocv_errors, label='Height LOOCV RMSE', color='red')
plt.xlabel('Sample Index', fontsize=18)
plt.ylabel('LOOCV RMSE', fontsize=18)
plt.title('LOOCV Training Errors for KRG Surrogate Models', fontsize=18)
plt.tick_params(axis='both', labelsize=18)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('Figure/krg_loocv_errors.png')
plt.close()

# Huấn luyện mô hình trên toàn bộ tập dữ liệu với theta0
model_sth = KRG(print_global=False, theta0=[1e-2]*X.shape[1])
model_sth.set_training_values(X, Y_sth_scaled)
model_sth.train()

model_height = KRG(print_global=False, theta0=[1e-2]*X.shape[1])
model_height.set_training_values(X, Y_height_scaled)
model_height.train()

# Định nghĩa bài toán tối ưu
class SurrogateProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=np.array([10, 0.3, 40]), xu=np.array([15, 0.9, 60]))

    def _evaluate(self, X, out, *args, **kwargs):
        sth_pred_scaled = model_sth.predict_values(X).flatten()
        height_pred_scaled = model_height.predict_values(X).flatten()

        sth_pred = scaler_sth.inverse_transform(sth_pred_scaled.reshape(-1, 1)).flatten()
        height_pred = scaler_height.inverse_transform(height_pred_scaled.reshape(-1, 1)).flatten()

        out["F"] = -np.column_stack([sth_pred, height_pred])

# Cấu hình thuật toán NSGA-II
algorithm = NSGA2(
    pop_size=200,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)

termination = get_termination("n_gen", 200)

# Thực hiện tối ưu hóa
res = minimize(
    SurrogateProblem(),
    algorithm,
    termination,
    seed=42,
    save_history=True,
    verbose=True,
)

# Phục hồi giá trị thực từ kết quả tối ưu
F_actual = -res.F
plt.figure(figsize=(8, 6))
plt.scatter(F_actual[:, 0], F_actual[:, 1], c='red', label='Pareto Optimal')
plt.xlabel("STH (mm)", fontsize=18)
plt.ylabel("Height (mm)", fontsize=18)
plt.title("Pareto Front from NSGA-II with Kriging Surrogate Model", fontsize=16)
plt.tick_params(axis='both', labelsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure/pareto_front.png")
plt.close()

# Xuất dữ liệu đầu ra tối ưu
X_optimal = res.X
STH_scaled = model_sth.predict_values(X_optimal).flatten()
Height_scaled = model_height.predict_values(X_optimal).flatten()
STH_optimal = scaler_sth.inverse_transform(STH_scaled.reshape(-1, 1)).flatten()
Height_optimal = scaler_height.inverse_transform(Height_scaled.reshape(-1, 1)).flatten()

df_optimal = pd.DataFrame(X_optimal, columns=["X1", "X2", "X3"])
df_optimal["STH (mm)"] = STH_optimal
df_optimal["Height (mm)"] = Height_optimal
df_optimal.to_excel("pareto_optimal_nsga2_krg.xlsx", index=False)
print("Pareto optimal results have been saved to 'pareto_optimal_nsga2_krg.xlsx'")

# Thêm phân tích độ nhạy (Sensitivity Analysis)
def compute_sensitivity(model, X, scaler, output_name):
    n_samples = X.shape[0]
    sensitivities = np.zeros((n_samples, X.shape[1]))
    
    for i in range(n_samples):
        X_perturb = X[i].reshape(1, -1)
        for j in range(X.shape[1]):
            X_perturb_delta = X_perturb.copy()
            delta = 0.01 * (X[:, j].max() - X[:, j].min())  # Nhỏ perturbation
            X_perturb_delta[0, j] += delta
            Y_base = model.predict_values(X_perturb).flatten()
            Y_perturb = model.predict_values(X_perturb_delta).flatten()
            Y_base_actual = scaler.inverse_transform(Y_base.reshape(-1, 1)).flatten()
            Y_perturb_actual = scaler.inverse_transform(Y_perturb.reshape(-1, 1)).flatten()
            sensitivities[i, j] = np.abs((Y_perturb_actual - Y_base_actual) / delta)
    
    return np.mean(sensitivities, axis=0)

# Tính độ nhạy cho STH và Height
sensitivity_sth = compute_sensitivity(model_sth, X, scaler_sth, "STH")
sensitivity_height = compute_sensitivity(model_height, X, scaler_height, "Height")

# Vẽ đồ thị độ nhạy
params = ['X1', 'X2', 'X3']
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(params)) - 0.2, sensitivity_sth, 0.4, label='STH Sensitivity', color='blue')
plt.bar(np.arange(len(params)) + 0.2, sensitivity_height, 0.4, label='Height Sensitivity', color='red')
plt.xlabel('Parameters', fontsize=18)
plt.ylabel('Average Sensitivity (mm/unit)', fontsize=18)
plt.title('Sensitivity Analysis of Parameters on STH and Height', fontsize=18)
plt.xticks(np.arange(len(params)), params, fontsize=18)
plt.tick_params(axis='both', labelsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('Figure/sensitivity_analysis.png')
plt.close()

print(f"Sensitivity Analysis Results:")
print(f"STH Sensitivity: X1={sensitivity_sth[0]:.4f}, X2={sensitivity_sth[1]:.4f}, X3={sensitivity_sth[2]:.4f}")
print(f"Height Sensitivity: X1={sensitivity_height[0]:.4f}, X2={sensitivity_height[1]:.4f}, X3={sensitivity_height[2]:.4f}")