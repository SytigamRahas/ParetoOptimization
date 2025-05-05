import os
import pandas as pd
import numpy as np
from smt.surrogate_models import KRG
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

# --- 1. Thiết lập thư mục đầu ra
os.makedirs("Figure", exist_ok=True)

# --- 2. Đọc dữ liệu gốc
df = pd.read_excel("lhs_samples_saved.xlsx")
X_raw = df[['X1','X2','X3']].values   # mm, –, MPa
Y       = df[['STH','Height']].values # mm, mm

# --- 3. Chuẩn hóa X về [0,1] và Y về [0,1]
x_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)

y1_scaler = MinMaxScaler()
y2_scaler = MinMaxScaler()
Y1_scaled = y1_scaler.fit_transform(Y[:,0].reshape(-1,1)).flatten()
Y2_scaled = y2_scaler.fit_transform(Y[:,1].reshape(-1,1)).flatten()

# --- 4. Hàm tính LOOCV error trên X_scaled
def compute_loocv_errors(X, Y_scaled, y_scaler):
    n = len(X)
    errs = np.zeros(n)
    for i in range(n):
        # chia train/test
        mask = np.arange(n) != i
        X_tr, X_te = X[mask], X[i].reshape(1, -1)
        Y_tr, Y_te = Y_scaled[mask], Y_scaled[i].reshape(1, -1)

        # huấn luyện model với theta0
        model = KRG(print_global=False, theta0=[1e-2]*X.shape[1])
        model.set_training_values(X_tr, Y_tr)
        model.train()

        # dự đoán và tính RMSE trên giá trị gốc
        y_pred_s = model.predict_values(X_te).flatten()
        y_pred   = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).flatten()
        y_true   = y_scaler.inverse_transform(Y_te).flatten()
        errs[i]  = np.sqrt(mean_squared_error(y_true, y_pred))
    return errs

# Tính và vẽ LOOCV
err_loocv_1 = compute_loocv_errors(X_scaled, Y1_scaled, y1_scaler)
err_loocv_2 = compute_loocv_errors(X_scaled, Y2_scaled, y2_scaler)

plt.figure(figsize=(8,6))
plt.plot(err_loocv_1, label='STH LOOCV RMSE', color='blue')
plt.plot(err_loocv_2, label='Height LOOCV RMSE', color='red')
plt.xlabel('Sample Index', fontsize=18)
plt.ylabel('LOOCV RMSE (mm)', fontsize=18)
plt.title('LOOCV Training Errors for KRG Surrogates', fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('Figure/krg_loocv_errors.png')
plt.close()

# --- 5. Huấn luyện KRG trên toàn bộ dữ liệu
model1 = KRG(print_global=False, theta0=[1e-2]*3)
model2 = KRG(print_global=False, theta0=[1e-2]*3)
model1.set_training_values(X_scaled, Y1_scaled); model1.train()
model2.set_training_values(X_scaled, Y2_scaled); model2.train()

# --- 6. Định nghĩa bài toán NSGA-II trên [0,1]^3
class SurrogateProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=np.zeros(3), xu=np.ones(3))
    def _evaluate(self, X, out, *args, **kwargs):
        y1_s = model1.predict_values(X).flatten()
        y2_s = model2.predict_values(X).flatten()
        y1   = y1_scaler.inverse_transform(y1_s.reshape(-1,1)).flatten()
        y2   = y2_scaler.inverse_transform(y2_s.reshape(-1,1)).flatten()
        out["F"] = -np.column_stack([y1, y2])

algo = NSGA2(
    pop_size=200,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)
term = get_termination("n_gen", 200)

res = minimize(
    SurrogateProblem(), algo, term,
    seed=42, save_history=True, verbose=False
)

# --- 7. Vẽ Pareto front
Pareto = -res.F  # [STH, Height]
plt.figure(figsize=(8,6))
plt.scatter(Pareto[:,0], Pareto[:,1], c='red', label='Pareto Optimal')
plt.xlabel('STH (mm)', fontsize=18)
plt.ylabel('Height (mm)', fontsize=18)
plt.title('Pareto Front from NSGA-II with KRG', fontsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('Figure/pareto_front.png')
plt.close()

# --- 8. Xuất bảng Pareto với X1–X3 về giá trị gốc
Xp_scaled = res.X
Xp_orig   = x_scaler.inverse_transform(Xp_scaled)
df_pareto = pd.DataFrame(Xp_orig, columns=['X1','X2','X3'])
df_pareto['STH (mm)']    = Pareto[:,0]
df_pareto['Height (mm)'] = Pareto[:,1]
df_pareto.to_excel("pareto_optimal_nsga2_krg.xlsx", index=False)

# --- 9. Tính sensitivity trên thang normalize [0,1] và vẽ
def compute_sensitivity_norm(model, X, y_scaler):
    N, d = X.shape
    S = np.zeros((N,d))
    delta = 0.01  # 1% step
    for i in range(N):
        x0 = X[i].reshape(1,-1)
        for j in range(d):
            xp = x0.copy(); xp[0,j] += delta
            y0_s = model.predict_values(x0).flatten()
            yp_s = model.predict_values(xp).flatten()
            # trích scalar
            y0 = y_scaler.inverse_transform(y0_s.reshape(-1,1)).flatten()[0]
            yp = y_scaler.inverse_transform(yp_s.reshape(-1,1)).flatten()[0]
            S[i,j] = abs((yp - y0)/delta)
    return S.mean(axis=0)

sens1 = compute_sensitivity_norm(model1, X_scaled, y1_scaler)
sens2 = compute_sensitivity_norm(model2, X_scaled, y2_scaler)

# Vẽ biểu đồ độ nhạy
params = ['X1','X2','X3']
x = np.arange(len(params))
plt.figure(figsize=(8,6))
plt.bar(x-0.2, sens1, 0.4, label='STH Sensitivity')
plt.bar(x+0.2, sens2, 0.4, label='Height Sensitivity')
plt.xlabel('Parameters', fontsize=18)
plt.ylabel('ΔY per 1% ΔX_scaled', fontsize=18)
plt.title('Normalized Sensitivity Analysis', fontsize=16)
plt.xticks(x, params, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig('Figure/sensitivity_analysis.png')
plt.close()

print("Normalized sensitivity (per 1% change in X_scaled):")
print(f"STH:   X1={sens1[0]:.4f}, X2={sens1[1]:.4f}, X3={sens1[2]:.4f}")
print(f"Height:X1={sens2[0]:.4f}, X2={sens2[1]:.4f}, X3={sens2[2]:.4f}")
