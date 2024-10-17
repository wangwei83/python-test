import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = np.random.rand(10, 10)

# 创建热力图
plt.imshow(data, cmap='hot', interpolation='nearest')

# 添加颜色条
plt.colorbar()

# 显示热力图
plt.show()