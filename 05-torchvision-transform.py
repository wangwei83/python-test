import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



# 检查是否可以使用 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA 可用")
# 定义一系列变换
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

# 打开图像
image = Image.open('./000000001155.jpg')

# 应用变换
transformed_image = transform(image)

# 可视化原始图像和变换后的图像
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# 显示原始图像
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

# 显示变换后的图像
# 需要将张量转换回PIL图像
transformed_image_pil = transforms.ToPILImage()(transformed_image)
axs[1].imshow(transformed_image_pil)
axs[1].set_title('Transformed Image')
axs[1].axis('off')

plt.show()

