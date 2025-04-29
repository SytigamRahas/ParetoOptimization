import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import oapackage

# Đọc dữ liệu từ file Excel
df = pd.read_excel("lhs_samples_saved.xlsx")

# Trích 2 cột mục tiêu
data = df[['STH', 'Height']].values.T  # Transpose để đưa về (2, n)

# Khởi tạo Pareto Front
pareto = oapackage.ParetoDoubleLong()
for ii in range(data.shape[1]):
    point = oapackage.doubleVector((data[0, ii], data[1, ii]))
    pareto.addvalue(point, ii)

# Lấy chỉ số Pareto optimal
pareto_indices = pareto.allindices()
optimal_data = data[:, pareto_indices]

# Trích bảng tương ứng
pareto_indices = list(pareto.allindices())  # Ép về list
pareto_df = df.iloc[pareto_indices].copy()
pareto_df.reset_index(drop=True, inplace=True)

# Lưu ra file
pareto_df.to_csv("pareto_optimal_results.csv", index=False)
pareto_df.to_excel("pareto_optimal_results.xlsx", index=False)

print("✅ Đã xuất bảng Pareto optimal thành công!")


# Vẽ kết quả
plt.figure(figsize=(8, 6))
plt.plot(data[0, :], data[1, :], '.b', markersize=12, label='All Points')
plt.plot(optimal_data[0, :], optimal_data[1, :], '.r', markersize=14, label='Pareto Optimal')
plt.xlabel('STH', fontsize=14)
plt.ylabel('Height', fontsize=14)
plt.title('Pareto Front from LHS-based Simulation', fontsize=16)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

#Vẽ Đồ thị

# Ví dụ: X1 vs STH
plt.figure()
plt.scatter(pareto_df['X1'], pareto_df['STH'], color='blue')
plt.xlabel('X1')
plt.ylabel('STH')
plt.title('X1 vs STH (Pareto Optimal)')
plt.grid(True)
plt.show()

# X2 vs Height
plt.figure()
plt.scatter(pareto_df['X2'], pareto_df['Height'], color='green')
plt.xlabel('X2')
plt.ylabel('Height')
plt.title('X2 vs Height (Pareto Optimal)')
plt.grid(True)
plt.show()

# X3 vs STH

plt.figure()
plt.scatter(pareto_df['X3'], pareto_df['STH'], color='purple')
plt.xlabel('X3')
plt.ylabel('STH')
plt.title('X3 vs STH (Pareto Optimal)')
plt.grid(True)
plt.show()
