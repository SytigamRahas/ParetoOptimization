import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smt.surrogate_models import RBF
from smt.sampling_methods import LHS
from sklearn.preprocessing import MinMaxScaler
import oapackage

# Đọc dữ liệu từ file Excel
df = pd.read_excel("lhs_samples_saved.xlsx")

# Tách input (X) và output (Y)
X = df[['X1', 'X2', 'X3']].values
Y = df[['STH', 'Height']].values

# Chuẩn hoá output về [0, 1] cho RBF
scaler_sth = MinMaxScaler()
scaler_height = MinMaxScaler()

Y_sth = scaler_sth.fit_transform(Y[:, 0].reshape(-1, 1)).flatten()
Y_height = scaler_height.fit_transform(Y[:, 1].reshape(-1, 1)).flatten()

# Huấn luyện surrogate model
model_sth = RBF(print_global=False)
model_sth.set_training_values(X, Y_sth)
model_sth.train()

model_height = RBF(print_global=False)
model_height.set_training_values(X, Y_height)
model_height.train()

# Sinh 5000 mẫu LHS trong khoảng giá trị X1, X2, X3
sampling = LHS(xlimits=np.array([[10, 15], [0.3, 0.9], [40, 60]]))
X_pred = sampling(5000)

# Dự đoán đã CHUẨN HOÁ
STH_pred_norm = model_sth.predict_values(X_pred).flatten()
Height_pred_norm = model_height.predict_values(X_pred).flatten()

# Chuyển lại giá trị thật
STH_pred = scaler_sth.inverse_transform(STH_pred_norm.reshape(-1, 1)).flatten()
Height_pred = scaler_height.inverse_transform(Height_pred_norm.reshape(-1, 1)).flatten()

# Tạo tập dữ liệu cho Pareto
datapoints = np.vstack([STH_pred, Height_pred])
pareto = oapackage.ParetoDoubleLong()
for i in range(datapoints.shape[1]):
    vec = oapackage.doubleVector([datapoints[0, i], datapoints[1, i]])
    pareto.addvalue(vec, i)

# Lấy các điểm Pareto
optimal_indices = list(pareto.allindices())
optimal_data = datapoints[:, optimal_indices]

# Vẽ Pareto front
plt.figure(figsize=(8, 6))
plt.plot(datapoints[0, :], datapoints[1, :], '.b', markersize=4, label="Tất cả điểm")
plt.plot(optimal_data[0, :], optimal_data[1, :], '.r', markersize=6, label="Pareto tối ưu")
plt.xlabel("STH")
plt.ylabel("Height")
plt.title("Pareto Front từ mô hình surrogate (scaled output)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Xuất input và output tương ứng với Pareto point
X_optimal = X_pred[optimal_indices]
df_optimal = pd.DataFrame(X_optimal, columns=['X1', 'X2', 'X3'])
df_optimal['STH'] = STH_pred[optimal_indices]
df_optimal['Height'] = Height_pred[optimal_indices]
df_optimal.to_excel("pareto_optimal_results_scaled.xlsx", index=False)
print("✅ Kết quả Pareto từ mô hình surrogate đã được lưu thành công!")
