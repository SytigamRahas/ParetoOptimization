import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from mpl_toolkits.mplot3d import Axes3D  # Import cần thiết để vẽ 3D scatter

# Thiết lập số mẫu cần sinh
n_samples = 50

# Khoảng biến [min, max] theo từng biến
bounds = np.array([
    [10, 15],    # X1
    [0.3, 0.9],  # X2
    [40, 60]     # X3
])

# Tạo đối tượng Latin Hypercube
sampler = qmc.LatinHypercube(d=3)

# Sinh mẫu
sample = sampler.random(n=n_samples)

# Scale mẫu về đúng khoảng giá trị
lhs_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

# In kết quả
print("Latin Hypercube Samples (X1, X2, X3):")
print(lhs_samples)

# Lưu ra CSV
df = pd.DataFrame(lhs_samples, columns=["X1", "X2", "X3"])
df.to_csv("lhs_samples.csv", index=False)
print("\nĐã lưu file 'lhs_samples.csv' thành công!")

# Vẽ scatter 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(lhs_samples[:, 0], lhs_samples[:, 1], lhs_samples[:, 2], c='r', marker='o')

ax.set_xlabel('X1 (10 → 15)')
ax.set_ylabel('X2 (0.3 → 0.9)')
ax.set_zlabel('X3 (40 → 60)')
ax.set_title('3D Scatter of LHS Samples')

plt.show()
