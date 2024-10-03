import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# 创建一个随机的 NumPy 数组
data = np.random.rand(100, 100)

# 保存数组到 output.tif 文件
tiff.imwrite('output.tif', data)

# 读取并可视化 output.tif 文件
image = tiff.imread('output.tif')
plt.imshow(image, cmap='gray')
plt.title('Visualization of output.tif')
plt.colorbar()
plt.show()