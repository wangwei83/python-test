from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义自定义的训练变换函数
def train_transform(img):
    # 例如：将图像转换为灰度图像
    return img.convert("L")

# 使用 transforms.Lambda 封装自定义的训练变换
transform = transforms.Lambda(train_transform)

# 加载图像并应用变换
img = Image.open('/home/wangwei83/Desktop/python-test/dataset/cifar/123.jpeg')
transformed_img = transform(img)

# 可视化原始图像和变换后的图像
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 显示原始图像
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

# 显示变换后的图像
ax[1].imshow(transformed_img, cmap='gray')
ax[1].set_title('Transformed Image')
ax[1].axis('off')

plt.show()