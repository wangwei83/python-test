from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义随机选择的变换操作
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),  # 随机调整亮度
    transforms.ColorJitter(contrast=0.2),  # 随机调整对比度
    transforms.ColorJitter(saturation=0.2)  # 随机调整饱和度
])

# 应用随机选择的变换到图像上
img = Image.open('/home/wangwei83/Desktop/python-test/dataset/cifar/123.jpeg')
transformed_img = transform_ae(img)

# 可视化原始图像和变换后的图像
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 原始图像
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 变换后的图像
axes[1].imshow(transformed_img)
axes[1].set_title('Transformed Image')
axes[1].axis('off')

plt.show()