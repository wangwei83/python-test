import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 加载数据集
dataset = ImageFolder(root='/home/wangwei83/Desktop/python-test/dataset/cifar/data', transform=transform)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 访问数据集中的第3个样本
img, label = dataset[2]
print(f'Image shape: {img.shape}, Label: {label}')

# 可视化图像
fig, axes = plt.subplots(1, 1, figsize=(3, 3))
axes.imshow(img.permute(1, 2, 0))  # 调整维度以适应 matplotlib 的显示
axes.set_title(f'Label: {label}')
axes.axis('off')

plt.show()