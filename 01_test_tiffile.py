
import tifffile as tiff
import numpy as np

# 创建一个随机的 NumPy 数组
data = np.random.rand(100, 100)

# 将 NumPy 数组保存为 TIFF 文件
tiff.imwrite('output.tif', data)

# 读取 TIFF 文件
image = tiff.imread('output.tif')

print(image.shape)
